from model.utils import *
from model.rxn_encoder import RxnShinglingModel, shingling_architecture, ClassificationHead
from model.mol_encoder import UniMolShingling, UniMolMol
from typing import Any, Dict
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import contextlib
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from torchmetrics.regression import R2Score, MeanAbsoluteError, MeanSquaredError
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

@dataclass
class RxnOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_cross_entropy: Optional[torch.FloatTensor] = None
    loss_100: Optional[torch.FloatTensor] = None
    loss_1000: Optional[torch.FloatTensor] = None
    loss_4000: Optional[torch.FloatTensor] = None
    loss_rmse: Optional[torch.FloatTensor] = None
    loss_mae: Optional[torch.FloatTensor] = None
    loss_r2: Optional[torch.FloatTensor] = None
    preds: Optional[torch.FloatTensor] = None
    labels: Optional[torch.FloatTensor] = None


class RxnTrainer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = argparse.Namespace(**args)
        self.args = args
        if args.pretraining:
            # logging.info("Using pretraining model for reaction encoding")
            self.model = RxnModelPretraining(args=args)
        else:
            if args.cross_attention:
                # logging.info("Using cross attention for reaction encoding")
                self.model = RxnModel_cross(args=args)
            else:
                # logging.info("Using shingling for reaction encoding")
                self.model = RxnModel(args=args)
    
        self.save_hyperparameters(args)

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)

        # make sure mol_encoder is not trainable
        for name, param in self.model.mol_encoder.named_parameters():
            param.requires_grad = False
        params_to_optimize = [param for param in self.parameters() if param.requires_grad]

        optimizer = optim.AdamW(params_to_optimize, lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
        elif self.args.scheduler == 'linear_warmup_step_lr':
            self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, warmup_steps)
        elif self.args.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    @torch.no_grad()
    def validation_step(self, batch, batch_idx=-1):
        # (...), rxn_types, labels_y = batch
        batch_size = batch[0][4].shape[0]  # ids
        output = self.model(batch)
        ###============== Overall Loss ===================###
        if self.args.pred_type == "classification":
            self.log("val/loss_cross_entropy", float(output.loss_cross_entropy), batch_size=batch_size, sync_dist=True)
        elif self.args.pred_type == "regression":
            self.log("val/loss_rmse", float(output.loss_rmse), batch_size=batch_size, sync_dist=True)
            self.log("val/loss_mae", float(output.loss_mae), batch_size=batch_size, sync_dist=True)
            self.log("val/loss_r2", float(output.loss_r2), batch_size=batch_size, sync_dist=True)
        self.log("val/loss", float(output.loss), batch_size=batch_size, sync_dist=True)

    def training_step(self, batch, batch_idx=-1):
        self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        batch_size = batch[0][4].shape[0]  # ids
        output = self.model(batch)
        ###============== Overall Loss ===================###
        if self.args.pred_type == "classification":
            self.log("train/loss_cross_entropy", float(output.loss_cross_entropy), batch_size=batch_size, sync_dist=True)
        elif self.args.pred_type == "regression":
            self.log("train/loss_rmse", float(output.loss_rmse), batch_size=batch_size, sync_dist=True)
            self.log("train/loss_mae", float(output.loss_mae), batch_size=batch_size, sync_dist=True)
            self.log("train/loss_r2", float(output.loss_r2), batch_size=batch_size, sync_dist=True)
        self.log("train/loss", float(output.loss), batch_size=batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
        return output.loss
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)

    def forward(self, batch):
        return self.model(batch)


class RxnModel(nn.Module):
    """
    encode reactions
    """
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.mae = MeanAbsoluteError()
        self.r2 = R2Score()

        self.mol_encoder = UniMolShingling()
        self.ln_graph = nn.LayerNorm(512)
        self.shingling_encoder = RxnShinglingModel(output_dim=args.output_dim, path=args.init_checkpoint, dropout=args.dropout)

        for name, param in self.mol_encoder.named_parameters():
            param.requires_grad = False
        self.mol_encoder = self.mol_encoder.eval()
        self.mol_encoder.train = disabled_train
        logging.info("Freeze mol encoder")

        if args.init_checkpoint:
            logging.info("Loading shingling encoder pretrained weights from {}".format(args.init_checkpoint))

    def forward(self, batch):
        (conf, nums_mol, nums_confs, total_atoms, ids, mol_num, part_num, shin_dist, shin_sim, shin_edge), types, labels_y = batch

        batch_node, batch_mask = self.mol_encoder(conf, nums_mol, nums_confs, total_atoms, ids, mol_num)
        batch_node, batch_mask = batch_node.to(labels_y.device), batch_mask.to(labels_y.device)
        batch_node = self.ln_graph(batch_node)
        # max len cut
        batch_node = batch_node[:, :self.args.shingling_max_len, :]
        batch_mask = batch_mask[:, :self.args.shingling_max_len]
        shin_dist = shin_dist[:, :self.args.shingling_max_len, :self.args.shingling_max_len]
        shin_sim = shin_sim[:, :self.args.shingling_max_len, :self.args.shingling_max_len]
        shin_edge = shin_edge[:, :self.args.shingling_max_len, :self.args.shingling_max_len]

        logits = self.shingling_encoder(batch_node, batch_mask, shin_dist, shin_sim, shin_edge)

        if self.args.pred_type == "classification":
            loss_y = F.cross_entropy(logits, labels_y.squeeze())
            return RxnOutput(
                loss=loss_y,
                loss_cross_entropy=loss_y,
                preds=logits,
                labels=labels_y
            )
        elif self.args.pred_type == "regression":
            loss_y = F.mse_loss(logits, labels_y)
            loss_rmse = torch.sqrt(loss_y)
            loss_mae = self.mae(logits, labels_y)
            loss_r2 = self.r2(logits, labels_y) if logits.shape[0] > 1 else torch.tensor(0.0)
            # cut to 0-1 after loss during training; recalculate metrics during test
            # logits = torch.clamp(logits, self.args.min_value, self.args.max_value)
            return RxnOutput(
                loss=loss_y,
                loss_rmse=loss_rmse,
                loss_mae=loss_mae,
                loss_r2=loss_r2,
                preds=logits,
                labels=labels_y
            )
        else:
            raise NotImplementedError


class RxnModelPretraining(RxnModel):
    def __init__(self, args=None):
        super().__init__(args)

        self.shingling_encoder.classification_head = None

        cls_args = shingling_architecture()
        self.cls_head_100 = ClassificationHead(
            input_dim=cls_args.encoder_embed_dim,
            inner_dim=cls_args.encoder_embed_dim,
            num_classes=100,
            activation_fn=cls_args.pooler_activation_fn,
            pooler_dropout=cls_args.pooler_dropout,
        )
        self.cls_head_1000 = ClassificationHead(
            input_dim=cls_args.encoder_embed_dim,
            inner_dim=cls_args.encoder_embed_dim,
            num_classes=1000,
            activation_fn=cls_args.pooler_activation_fn,
            pooler_dropout=cls_args.pooler_dropout,
        )
        self.cls_head_4000 = ClassificationHead(
            input_dim=cls_args.encoder_embed_dim,
            inner_dim=cls_args.encoder_embed_dim,
            num_classes=4000,
            activation_fn=cls_args.pooler_activation_fn,
            pooler_dropout=cls_args.pooler_dropout,
        )

    def forward(self, batch):
        (conf, nums_mol, nums_confs, total_atoms, ids, mol_num, part_num, shin_dist, shin_sim, shin_edge), types, labels_y = batch

        batch_node, batch_mask = self.mol_encoder(conf, nums_mol, nums_confs, total_atoms, ids, mol_num)
        batch_node, batch_mask = batch_node.to(labels_y.device), batch_mask.to(labels_y.device)
        batch_node = self.ln_graph(batch_node)
        cls_repr = self.shingling_encoder(batch_node, batch_mask, shin_dist, shin_sim, shin_edge, return_repr=True)["cls_repr"]

        loss_100 = F.cross_entropy(self.cls_head_100(cls_repr), types[:, 0])
        loss_1000 = F.cross_entropy(self.cls_head_1000(cls_repr), types[:, 1])
        loss_4000 = F.cross_entropy(self.cls_head_4000(cls_repr), types[:, 2])
        return RxnOutput(
            loss=loss_100+loss_1000+loss_4000,
            loss_cross_entropy=loss_100+loss_1000+loss_4000,
            loss_100=loss_100,
            loss_1000=loss_1000,
            loss_4000=loss_4000,
            labels=labels_y
        )
