from unimol_tools.models.unimol import *

def shingling_architecture():
    args = argparse.ArgumentParser()
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.2)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.backbone = getattr(args, "backbone", "transformer")
    args.kernel = getattr(args, "kernel", "gaussian")
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    return args


class RxnShinglingModel(nn.Module):
    """
    Attributes:
        - output_dim: The dimension of the output layer.
        - remove_hs: Flag to indicate whether hydrogen atoms are removed in molecular data.
        - pretrain_path: Path to the pretrained model weights.
        - mask_idx: Index of the mask token in the dictionary.
        - padding_idx: Index of the padding token in the dictionary.
        - encoder: Transformer encoder backbone of the model.
        - gbf_proj, gbf: Layers for Gaussian basis functions or numerical embeddings.
        - classification_head: The final classification head of the model.
    """
    def __init__(self, output_dim=2, path="", use_positional_encoding=True, dropout=0.1, **params):
        """
        :param output_dim: (int) The number of output dimensions (classes).
        :param use_positional_encoding: (bool) use or not.
        :param params: Additional parameters for model configuration.
        """
        super().__init__()
        self.args = shingling_architecture()
        self.output_dim = output_dim
        self.padding_idx = 0
        self.use_positional_encoding = use_positional_encoding
        self.encoder = BACKBONE[self.args.backbone](
            encoder_layers=self.args.encoder_layers,
            embed_dim=self.args.encoder_embed_dim,
            ffn_embed_dim=self.args.encoder_ffn_embed_dim,
            attention_heads=self.args.encoder_attention_heads,
            emb_dropout=self.args.emb_dropout,
            dropout=self.args.dropout,
            attention_dropout=self.args.attention_dropout,
            activation_dropout=self.args.activation_dropout,
            max_seq_len=self.args.max_seq_len,
            activation_fn=self.args.activation_fn,
            no_final_head_layer_norm=self.args.delta_pair_repr_norm_loss < 0,
        )
        self.cls_token = nn.Parameter(torch.randn(1, self.args.encoder_embed_dim))

        K = 128
        n_edge_type = 2 # intra and inter
        """shingling distance"""
        self.gbf_proj = NonLinearHead(
            K, self.args.encoder_attention_heads, self.args.activation_fn
        )
        if self.args.kernel == 'gaussian':
            self.gbf = GaussianLayer(K, n_edge_type)
        else:
            self.gbf = NumericalEmbed(K, n_edge_type)
        """shingling similarity"""
        self.gbf_proj_sim = NonLinearHead(
            K, self.args.encoder_attention_heads, self.args.activation_fn
        )
        if self.args.kernel == 'gaussian':
            self.gbf_sim = GaussianLayer(K, n_edge_type)
        else:
            self.gbf_sim = NumericalEmbed(K, n_edge_type)

        self.classification_head = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=self.args.encoder_embed_dim,
            num_classes=self.output_dim,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=dropout, # self.args.pooler_dropout,
        )
        self.load_pretrained_weights(path)

    def load_pretrained_weights(self, path):
        """
        Loads pretrained weights into the model.

        :param path: (str) Path to the pretrained weight file.
        """
        if path:
            # logger.info("Loading pretrained weights from {}".format(path))
            # print("Loading pretrained weights from {}".format(path))
            ckpt = torch.load(path, map_location=lambda storage, loc: storage)
            self.load_state_dict(ckpt['state_dict'], strict=False)

    @classmethod
    def build_model(cls, args):
        """
        Class method to build a new instance of the UniMolModel.

        :param args: Arguments for model configuration.
        :return: An instance of UniMolModel.
        """
        return cls(args)

    def forward(
        self,
        emb,
        padding_mask,
        src_distance,
        src_similairty,
        src_edge_type,
        return_repr=False,
        return_atomic_reprs=False,
        **kwargs
    ):
        """
        Defines the forward pass of the model.

        :param emb: embeddings of shinglings.
        :param padding_mask: padding_mask for reactions.
        :param src_distance: Additional molecular features.
        :param src_similarity: Additional molecular features.
        :param src_edge_type: Additional molecular features.
        :param return_repr: Flags to return intermediate representations.
        :param return_atomic_reprs: Flags to return intermediate representations.

        :return: Output logits or requested intermediate representations.
        """
        # add cls token
        emb = torch.cat((self.cls_token.expand(emb.shape[0], -1).unsqueeze(1), emb), dim=1)[:, :src_distance.shape[1], :]
        add_token_mask = torch.ones(padding_mask.shape[0], 1, dtype=torch.long, device=emb.device)
        padding_mask = torch.cat((add_token_mask, padding_mask), dim=1)[:, :src_distance.shape[1]]

        padding_mask = padding_mask.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = emb
        def get_dist_features(dist, sim, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            gbf_feature_sim = self.gbf_sim(sim, et)
            gbf_result_sim = self.gbf_proj_sim(gbf_feature_sim)
            graph_attn_bias = gbf_result + gbf_result_sim
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        graph_attn_bias = get_dist_features(src_distance, 1-src_similairty, src_edge_type) if self.use_positional_encoding else None
        (
            encoder_rep,
            _,
            _,
            _,
            _,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        cls_repr = encoder_rep[:, 0, :]  # CLS token repr
        all_repr = encoder_rep[:, :, :]  # all token repr
        if return_repr:
            return {"cls_repr": cls_repr, "all_repr": all_repr}

        logits = self.classification_head(cls_repr)
        return logits

    def batch_collate_fn(self, samples):
        """
        Custom collate function for batch processing non-MOF data.

        :param samples: A list of sample data.

        :return: A tuple containing a batch dictionary and labels.
        """
        batch = {}
        for k in samples[0][0].keys():
            if k == 'src_coord':
                v = pad_coords([torch.tensor(s[0][k]).float() for s in samples], pad_idx=0.0)
            elif k == 'src_edge_type':
                v = pad_2d([torch.tensor(s[0][k]).long() for s in samples], pad_idx=self.padding_idx)
            elif k == 'src_distance':
                v = pad_2d([torch.tensor(s[0][k]).float() for s in samples], pad_idx=0.0)
            elif k == 'src_tokens':
                v = pad_1d_tokens([torch.tensor(s[0][k]).long() for s in samples], pad_idx=self.padding_idx)
            batch[k] = v
        try:
            label = torch.tensor([s[1] for s in samples])
        except:
            label = None
        return batch, label
