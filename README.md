# ReaDISH

Data and codes for the paper "Reaction Prediction via Interaction Modeling of Symmetric Difference Shingle Sets" (NeurIPS 2025).

## Requirements

We implement our model on `Python 3.10`. These packages are mainly used:

```
rdkit                2024.3.3
torch                2.3.1
tensorboard          2.17.0
lightning            2.3.3
pytorch-lightning    2.3.3
unicore              0.0.1
unimol_tools         0.1.0.post1
rxnfp                0.1.0
```

## Datasets

### Pre-training dataset

We filtered reactions from USPTO (https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873) and CJHIF (https://github.com/jshmjs45/data_for_chem). The code to cluster reactions based on DRFP fingerprints and Kmeans is stored in `/data/cluster.py`.

### Downstream dataset

The downstream datasets (7 sources) and the generated coordinates (`/ds_confs`) are stored in `/downstream_datasets`.

## Shingle generation

The code for shingle generation is stored in `/model/generate_shingling.py` (for the pre-training dataset) and `/model/utils.py`. The pipeline is adapted from https://github.com/reymond-group/drfp.

## Experiments

### Pre-training

Run `pretraining.py` to pre-train ReaDISH. For example,

```
python pretraining.py --max_epochs 3 --batch_size 8 --init_lr 5e-5 --min_lr 5e-6 --devices 0,1,2,3,4,5,6,7
```

The pre-trained model is stored in `/checkpoint`. 

### Fine-tuning

Run `run_BH.py` to fine-tune ReaDISH on the BH dataset. We also provide an example to fine-tune on another given downstream dataset. For example,

``` 
python run_BH.py

python downstream.py --devices 0,1,2,3 --batch_size 4 --accumulate_grad_batches 4 --ds_name SM --repeat_times 10 --max_epochs 150 --check_val_every_n_epoch 1 --num_workers 8 --init_lr 1e-3 --min_lr 1e-4 --warmup_lr 1e-6 --warmup_steps 500 --pred_type regression --init_checkpoint checkpoint/last.ckpt --norm
```

## Citation

```
@inproceedings{shi2025reaction,
  title={Reaction Prediction via Interaction Modeling of Symmetric Difference Shingle Sets},
  author={Shi, Runhan and Chen, Letian and Yu, Gufeng and Yang, Yang},
  booktitle={Proceedings of the Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```
