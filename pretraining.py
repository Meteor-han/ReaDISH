from collections import defaultdict
from model.pylighting_trainer import *
from lightning import seed_everything
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from lightning.pytorch import loggers as pl_loggers
torch.set_float32_matmul_precision('medium')
from lightning.pytorch.profilers import AdvancedProfiler
# os.environ["CUDA_VISIBLE_DEVICES"]="1,"


if __name__ == '__main__':
    args = get_args()
    args.devices = eval(args.devices)
    args.pretraining = True
    seed_everything(args.seed)
    with open("/amax/data/reaction_mvp/data/data_pretraining_confs/label_map.pkl", "rb") as f:
        labelmap = pickle.load(f)
    args.num_type_token = len(labelmap)
    
    dirpath = f"/amax/data/reaction/models/pretraining/{args.init_lr}_{args.batch_size}"
    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath=dirpath, 
                                         monitor="val_loss", 
                                         filename='pretraining_{epoch:02d}-{step:08d}', 
                                         every_n_train_steps=5000, 
                                         save_top_k=1, 
                                         save_on_train_epoch_end=True,
                                         save_last=True))
    # profiler = AdvancedProfiler(filename="profile.log")
    # callbacks.append(LitProgressBar())
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=dirpath)
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        val_check_interval=0.7,
        callbacks=callbacks,
        logger=tb_logger,
        strategy=args.strategy_name, #"ddp_find_unused_parameters_true", "auto"
        enable_checkpointing=True,  # profiler=profiler,
        num_sanity_val_steps=0,
        )

    with open("/amax/data/group_0/yield_data/pretraining_data/reactions_with_multiple.pkl", "rb") as f:
        data_type = pickle.load(f)
    print(f"reactions nums: {len(data_type)}")
    class_index = defaultdict(list)
    num_ = 0
    for i in tqdm(range(len(data_type))):
        # just omit, too long
        if len(data_type[i][0]) > 250:
            continue
        l, r = data_type[i][0].split(">>")
        l_set = set(l.split("."))
        r_set = set(r.split("."))
        if l_set.issubset(r_set):
            continue
        if r_set.issubset(l_set):
            continue
        # -1, use 4000 type id to split
        class_index[data_type[i][1][-1]].append(i)
        num_ += 1
    print(f"total reactions nums: {num_}")  # 3728503
    train_idx = []
    test_idx = []
    for k, v in class_index.items():
        random.shuffle(v)
        split_ = int(len(v)*0.8)
        train_idx.extend(v[:split_])
        test_idx.extend(v[int(len(v)*0.95):])

    p_ = os.path.join("/amax/data/data_pretraining_confs", "smi2id.pkl")
    with open(p_, "rb") as f:
        smi2id = pickle.load(f)

    dataset_train = ReactionDataset([data_type[i] for i in train_idx], smi2id=smi2id, omit=True, conf_max_size=1, mol_max_len=8, load_shingling=True)
    dataset_val = ReactionDataset([data_type[i] for i in test_idx], smi2id=smi2id, omit=True, conf_max_size=1, mol_max_len=8, load_shingling=True)
    
    args.max_steps = int(len(dataset_train) / args.batch_size / len(args.devices) * args.max_epochs)
    print(f"approximate steps: {args.max_steps}")
    
    training_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                 collate_fn=MyCollater(), pin_memory=False, persistent_workers=False)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, 
                            collate_fn=MyCollater(), pin_memory=False, persistent_workers=False)

    model = RxnTrainer(args=args)
    trainer.fit(model=model, train_dataloaders=training_loader, val_dataloaders=val_loader)

    print()
