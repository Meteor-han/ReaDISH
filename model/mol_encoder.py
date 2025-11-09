import torch
import torch.nn as nn
import numpy as np
from unimol_tools.models import UniMolModel
from unimol_tools.data import DataHub
from unimol_tools.predictor import MolDataset
from unimol_tools.tasks.trainer import NNDataLoader
from torch.nn.utils.rnn import pad_sequence

def decorate_torch_batch(batch, device="cpu", task="repr"):
    """
    Prepares a standard PyTorch batch of data for processing by the model. Handles tensor-based data structures.

    :param batch: The batch of tensor-based data to be processed.

    :return: A tuple of (net_input, net_target) for model processing.
    """
    net_input, net_target = batch
    if isinstance(net_input, dict):
        net_input, net_target = {
            k: v.to(device) for k, v in net_input.items()}, net_target.to(device)
    else:
        net_input, net_target = {'net_input': net_input.to(
            device)}, net_target.to(device)
    if task == 'repr':
        net_target = None
    elif task in ['classification', 'multiclass', 'multilabel_classification']:
        net_target = net_target.long()
    else:
        net_target = net_target.float()
    return net_input, net_target


class UniMolShingling(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = UniMolModel(output_dim=1, data_type='molecule', remove_hs=True)

    def forward(self, batch_input, num_mols, num_confs, total_atoms, ids, mol_num):
        datahub = DataHub(data=batch_input, task='repr', is_train=False, )
        dataset = MolDataset(datahub.data['unimol_input'])
        feature_name = None
        return_atomic_reprs = True
        dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=dataset,
            batch_size=128,
            shuffle=False,
            collate_fn=self.model.batch_collate_fn,
        )
        repr_dict = {"cls_repr": [], "atomic_coords": [], "atomic_reprs": [], "atomic_symbol": []}
        for batch in dataloader:
            net_input, _ = decorate_torch_batch(batch, device=next(self.model.parameters()).device)
            with torch.no_grad():
                outputs = self.model(**net_input,
                                return_repr=True,
                                return_atomic_reprs=return_atomic_reprs)
                assert isinstance(outputs, dict)
                # outputs["graph_attn_bias"]
                repr_dict["cls_repr"].append(outputs["cls_repr"])
                if return_atomic_reprs:
                    repr_dict["atomic_symbol"].extend(outputs["atomic_symbol"])
                    repr_dict['atomic_coords'].extend(outputs['atomic_coords'])
                    """we can cat in unimol.py, currently not"""
                    # # cls is important, may we cat to atomic_reprs
                    # for i in range(len(outputs['atomic_reprs'])):
                    #     outputs['atomic_reprs'][i] = torch.cat([torch.unsqueeze(outputs["cls_repr"][i], dim=0), outputs['atomic_reprs'][i]])
                    repr_dict['atomic_reprs'].extend(outputs['atomic_reprs'])
        repr_dict["cls_repr"] = torch.concat(repr_dict["cls_repr"])
        
        reps = pad_sequence(repr_dict['atomic_reprs'], batch_first=True, padding_value=0.)
        reps = reps.split(num_mols)
        # [batch size (reaction), num_mols, num_atoms, repr_dim]
        reps = pad_sequence(reps, batch_first=True, padding_value=0.)

        # [batch size (reaction), num_mols, num_shinglings, num_atoms, repr_dim]
        result = reps[torch.arange(reps.shape[0]).view(-1, 1, 1, 1), torch.arange(reps.shape[1]).view(1, -1, 1, 1), ids]
        valid_mask = (ids != -1).float().unsqueeze(-1).expand(-1, -1, -1, -1, result.shape[-1]).to(result.device)
        input_masked = result * valid_mask
        # do not exist / 0 condition, average over atoms in a shingling
        # [batch size (reaction), num_mols, num_shinglings, repr_dim]
        output = input_masked.mean(dim=-2)

        # [batch size (reaction), num_shinglings, repr_dim]
        ids = ids.view(ids.shape[0], ids.shape[1] * ids.shape[2], ids.shape[3])
        output = output.view(output.shape[0], output.shape[1] * output.shape[2], output.shape[3])
        keep_mask = (ids != -1).any(dim=-1)

        output_shinglings = pad_sequence([output[i][keep_mask[i]] for i in range(ids.shape[0])], batch_first=True, padding_value=0.)
        batch_mask = (torch.arange(max(mol_num), device=mol_num.device) < mol_num[:, None]).to(dtype=torch.int)

        return output_shinglings, batch_mask


class UniMolMol(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = UniMolModel(output_dim=1, data_type='molecule', remove_hs=True)

    def forward(self, batch_input, num_mols, num_confs, total_atoms, ids, mol_num):
        datahub = DataHub(data=batch_input, task='repr', is_train=False, )
        dataset = MolDataset(datahub.data['unimol_input'])
        feature_name = None
        return_atomic_reprs = True
        dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=dataset,
            batch_size=128,
            shuffle=False,
            collate_fn=self.model.batch_collate_fn,
        )
        repr_dict = {"cls_repr": [], "atomic_coords": [], "atomic_reprs": [], "atomic_symbol": []}
        for batch in dataloader:
            net_input, _ = decorate_torch_batch(batch, device=next(self.model.parameters()).device)
            with torch.no_grad():
                outputs = self.model(**net_input,
                                return_repr=True,
                                return_atomic_reprs=return_atomic_reprs)
                assert isinstance(outputs, dict)
                # outputs["graph_attn_bias"]
                repr_dict["cls_repr"].append(outputs["cls_repr"])
                if return_atomic_reprs:
                    repr_dict["atomic_symbol"].extend(outputs["atomic_symbol"])
                    repr_dict['atomic_coords'].extend(outputs['atomic_coords'])
                    """we can cat in unimol.py, currently not"""
                    # # cls is important, may we cat to atomic_reprs
                    # for i in range(len(outputs['atomic_reprs'])):
                    #     outputs['atomic_reprs'][i] = torch.cat([torch.unsqueeze(outputs["cls_repr"][i], dim=0), outputs['atomic_reprs'][i]])
                    repr_dict['atomic_reprs'].extend(outputs['atomic_reprs'])
        repr_dict["cls_repr"] = torch.concat(repr_dict["cls_repr"])
        # based on num_mols, split repr_dict["cls_repr"] and padding as a batch
        repr_dict["cls_repr"] = repr_dict["cls_repr"].split(num_mols)
        repr_dict["cls_repr"] = pad_sequence(repr_dict["cls_repr"], batch_first=True, padding_value=0.)
        # mask based on num_mols
        num_mols = torch.tensor(num_mols)
        # for [batch_size, seq_len, dim], the mask should be [batch_size, seq_len, 1] (suppose the query dim just 1)
        batch_mask = (torch.arange(max(num_mols)) < num_mols[:, None]).to(dtype=torch.bool)

        return repr_dict["cls_repr"], batch_mask
