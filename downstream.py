from model.pylighting_trainer import *
from data.data_utils import *
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from pytorch_lightning.utilities import rank_zero
from lightning.pytorch import loggers as pl_loggers
# os.environ["CUDA_VISIBLE_DEVICES"]="1,"
torch.set_float32_matmul_precision('medium')
from lightning.pytorch import seed_everything
import pandas as pd
import json
from datetime import datetime


class Finetuner:
    def __init__(self, args):
        self.args = args
        self.radius = args.radius
        self.symmetric_id = args.symmetric_id
        args.devices = eval(args.devices)
        self.data_prefix = "/downstream_datasets"
        self.confs_prefix = "/downstream_datasets/ds_confs/id2confs"
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.r2 = R2Score()
        self.dirpath = os.path.join(f"/amax/data/reaction/models/finetuning/{args.ds_name}")
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        if rank_zero.rank_zero_only.rank == 0:
            self.logger = create_file_logger(os.path.join(self.dirpath, f"{args.batch_size}_{args.accumulate_grad_batches}_{args.max_epochs}_{args.init_lr}_{args.min_lr}_{args.warmup_steps}_{args.radius}_{args.symmetric_id}_{args.cross_attention}_{now}.txt"))
        else:
            self.logger = LoggerPrint()

        p_ = os.path.join("/downstream_datasets/ds_confs/", "smi2id.pkl")
        with open(p_, "rb") as f:
            self.smi2id = pickle.load(f)
        p_ = os.path.join("/amax/data/data_pretraining_confs", "smi2id.pkl")
        with open(p_, "rb") as f:
            self.smi2id.update(pickle.load(f))

        args.min_value, args.max_value = get_range(args.ds_name)

    def train(self, training_data, test_data, mean_, std_):
        model = RxnTrainer(args=self.args)
        callbacks = []
        callbacks.append(plc.ModelCheckpoint(dirpath=self.dirpath, 
                                             monitor="val/loss", 
                                             filename='finetuning_{epoch:03d}-{step:08d}', 
                                             save_top_k=1, 
                                             save_on_train_epoch_end=True))
        # callbacks.append(LitProgressBar())
        tb_logger = pl_loggers.WandbLogger(save_dir=self.dirpath, project="rxn_finetuning", name=f"{self.args.ds_name}_{self.args.batch_size}_{self.args.accumulate_grad_batches}_{args.init_lr}_{args.warmup_steps}_{args.radius}_{args.symmetric_id}_{args.cross_attention}")
        trainer = pl.Trainer(
            accelerator=self.args.accelerator,
            devices=self.args.devices,
            precision=self.args.precision,
            max_epochs=self.args.max_epochs,
            accumulate_grad_batches=self.args.accumulate_grad_batches,
            check_val_every_n_epoch=self.args.check_val_every_n_epoch,
            callbacks=callbacks,
            logger=tb_logger,
            strategy="auto", #"ddp_find_unused_parameters_true", "auto", args.strategy_name
            enable_checkpointing=True,
            )

        training_loader = DataLoader(training_data, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                    shuffle=True, collate_fn=MyCollater())
        test_loader = DataLoader(test_data, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                shuffle=False, collate_fn=MyCollater())
        
        trainer.fit(model=model, train_dataloaders=training_loader, val_dataloaders=test_loader)

        self.logger.info(trainer.checkpoint_callback.best_model_path)
        model = RxnTrainer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, strict=False, args=self.args)
        res_ = {}
        if trainer.global_rank == 0:
            trainer_test = pl.Trainer(devices=[self.args.devices[0]], accelerator="gpu", logger=False)
            p = trainer_test.predict(model=model, dataloaders=test_loader)
            all_preds, all_labels = [], []
            for one in p:
                preds = torch.clamp(one.preds*std_+mean_, args.min_value, args.max_value)
                all_preds.append(preds)
                all_labels.append(one.labels*std_+mean_)
            all_preds, all_labels = torch.cat(all_preds), torch.cat(all_labels)
            print(max(all_preds), min(all_preds), all_preds.shape)
            res_ = {"rmse": torch.sqrt(self.mse(all_preds, all_labels)), 
                    "mae": self.mae(all_preds, all_labels),
                    "r2": self.r2(all_preds, all_labels)}
            os.remove(trainer.checkpoint_callback.best_model_path)
        return res_

    def print_results(self, all_best_p, times=100):
        res_ = {"mae": [], "rmse": [], "r2": []}
        res_float = {"mae": [], "rmse": [], "r2": []}
        for one in all_best_p:
            for k in one:
                res_[k].append(f"{float(one[k]*times):.4f}")
                res_float[k].append(float(one[k]*times))
        for k in res_:
            self.logger.info(res_[k])
            self.logger.info(f"{k}\t{np.mean(res_float[k]):.4f}\t{np.std(res_float[k]):.4f}")

    def run_BH_or_SM(self):
        seed_everything(seed=self.args.seed)
        """add split ratio"""
        if self.args.ds_name == "BH":
            num_ = int(3955*self.args.split_ratio[0])
            name_split = [('FullCV_{:02d}'.format(i), num_) for i in range(1, 11)]
        else:
            num_ = int(5760*self.args.split_ratio[0])
            name_split = [('random_split_{}'.format(i), num_) for i in range(10)]

        all_best_p = []
        # 10 splits
        for i, (name, split) in enumerate(name_split):
            # for parameter selection
            if i >= self.args.repeat_times:
                break
            if self.args.ds_name == "BH":
                df_doyle = pd.read_excel(os.path.join(self.data_prefix, 'BH/Dreher_and_Doyle_input_data.xlsx'),
                                            sheet_name=name, engine='openpyxl')
                raw_dataset = generate_Buchwald_Hartwig_rxns(df_doyle, 0.01)
            else:
                df = pd.read_csv(os.path.join(self.data_prefix, 'SM/{}.tsv'.format(name)), sep='\t')
                raw_dataset = generate_Suzuki_Miyaura_rxns(df, 0.01)
            y_training = torch.as_tensor([one[-1] for one in raw_dataset[:split]])
            mean_ = torch.mean(y_training) if self.args.norm else 0.
            std_ = torch.std(y_training) if self.args.norm else 1.
            """three parts"""
            data_type = []
            for one in raw_dataset:
                data_type.append((one[0].replace(">", ".", 1).replace(">", ">>", 1) if ">>" not in one[0] else one[0], 0, (one[1]-mean_)/std_))

            training_data = ReactionDataset(data_type[:split], radius=self.radius, smi2id=self.smi2id, omit=False, mol_max_len=15, data_prefix=self.confs_prefix, symmetric_id=self.symmetric_id)
            test_data = ReactionDataset(data_type[split:], radius=self.radius, smi2id=self.smi2id, omit=False, mol_max_len=15, data_prefix=self.confs_prefix, symmetric_id=self.symmetric_id)

            self.logger.info(f"{i}\tmae\trmse\tr2")
            p = self.train(training_data, test_data, mean_, std_)
            if p:
                self.logger.info(f"{p['mae']}\t{p['rmse']}\t{p['r2']}")
                all_best_p.append(p)
        if all_best_p:
            self.print_results(all_best_p)

    def run_BH_reactant(self):
        df_BH = pd.read_csv(os.path.join(self.data_prefix, "BH/BH.csv"), sep=',')
        dataset_BH = generate_Buchwald_Hartwig_rxns(df_BH, 0.01)
        """three parts"""
        data_type = []
        for one in dataset_BH:
            data_type.append([one[0].replace(">", ".", 1).replace(">", ">>", 1) if ">>" not in one[0] else one[0], 0, one[1]])
        with open(os.path.join(self.data_prefix, "BH/reactant_split_idxs.pickle"), "rb") as f:
            train_test_idxs = pickle.load(f)
        all_best_p = []
        # 5 random initialization
        for seed in range(min(self.args.repeat_times, 5)):
            seed_everything(seed=seed)
            training_data = []
            test_data = []
            for j in train_test_idxs[self.args.ds_name]["train_idx"]:
                training_data.append(data_type[j])
            for j in train_test_idxs[self.args.ds_name]["test_idx"]:
                test_data.append(data_type[j])
            y_training = torch.as_tensor([one[-1] for one in training_data])
            mean_ = torch.mean(y_training) if self.args.norm else 0.
            std_ = torch.std(y_training) if self.args.norm else 1.
            
            for i in range(len(training_data)):
                training_data[i][-1] = (training_data[i][-1]-mean_)/std_
                training_data[i] = tuple(training_data[i])
            for i in range(len(test_data)):
                test_data[i][-1] = (test_data[i][-1]-mean_)/std_
                test_data[i] = tuple(test_data[i])

            self.logger.info(f"{seed}\tmae\trmse\tr2")
            p = self.train(ReactionDataset(training_data, radius=self.radius, smi2id=self.smi2id, omit=False, data_prefix=self.confs_prefix, symmetric_id=self.symmetric_id), 
                           ReactionDataset(test_data, radius=self.radius, smi2id=self.smi2id, omit=False, data_prefix=self.confs_prefix, symmetric_id=self.symmetric_id), mean_, std_)
            if p:
                self.logger.info(f"{p['mae']}\t{p['rmse']}\t{p['r2']}")
                all_best_p.append(p)
        if all_best_p:
            self.print_results(all_best_p)

    def run_SM_ligand(self):
        # test range
        name_split_dict = {"SM_test1": ('1', 4320, 5760), "SM_test2": ('2', 4320, 5760),
                           "SM_test3": ('3', 4320, 5760), "SM_test4": ('4', 4320, 5760)}

        (name, start, end) = name_split_dict[self.args.ds_name]
        self.args.name = name
        all_best_p = []
        # 5 random initialization
        for seed in range(self.args.repeat_times):
            seed_everything(seed=seed)
            df = pd.read_csv(os.path.join(self.data_prefix, 'SM/SM_Test_{}.tsv'.format(name)), sep='\t')
            raw_dataset = generate_Suzuki_Miyaura_rxns(df, 0.01)
            y_training = torch.as_tensor([one[-1] for one in raw_dataset[:start]])
            mean_ = torch.mean(y_training) if self.args.norm else 0.
            std_ = torch.std(y_training) if self.args.norm else 1.
            """three parts"""
            data_type = []
            for one in raw_dataset:
                data_type.append((one[0].replace(">", ".", 1).replace(">", ">>", 1) if ">>" not in one[0] else one[0], 0, (one[1]-mean_)/std_))

            training_data = ReactionDataset(data_type[:start], radius=self.radius, smi2id=self.smi2id, omit=False, data_prefix=self.confs_prefix, symmetric_id=self.symmetric_id)
            test_data = ReactionDataset(data_type[start:], radius=self.radius, smi2id=self.smi2id, omit=False, data_prefix=self.confs_prefix, symmetric_id=self.symmetric_id)

            self.logger.info(f"{seed}\tmae\trmse\tr2")
            p = self.train(training_data, test_data, mean_, std_)
            if p:
                self.logger.info(f"{p['mae']}\t{p['rmse']}\t{p['r2']}")
                all_best_p.append(p)
        if all_best_p:
            self.print_results(all_best_p)

    def run_ELN(self):
        with open(os.path.join(self.data_prefix, "az/raw", "az_reactions_data.json"), "r") as f:
            raw_data_az_BH = json.load(f)
        dataset_az_BH = generate_ELN_BH_rxns(raw_data_az_BH)
        """three parts"""
        data_type = []
        for one in dataset_az_BH:
            data_type.append([one[0].replace(">", ".", 1).replace(">", ">>", 1) if ">>" not in one[0] else one[0], 0, one[1]])

        with open(os.path.join(self.data_prefix, "az/processed-0", "train_test_idxs.pickle"), "rb") as f:
            train_test_idxs = pickle.load(f)

        all_best_p = []
        # 10 splits
        for i in range(10):
            # for parameter selection
            if i >= self.args.repeat_times:
                break
            training_data = []
            test_data = []
            for j in train_test_idxs["train_idx"][i+1]:
                training_data.append(data_type[j])
            for j in train_test_idxs["test_idx"][i+1]:
                test_data.append(data_type[j])

            y_training = torch.as_tensor([one[-1] for one in training_data])
            mean_ = torch.mean(y_training) if self.args.norm else 0.
            std_ = torch.std(y_training) if self.args.norm else 1.

            for i in range(len(training_data)):
                training_data[i][-1] = (training_data[i][-1]-mean_)/std_
                training_data[i] = tuple(training_data[i])
            for i in range(len(test_data)):
                test_data[i][-1] = (test_data[i][-1]-mean_)/std_
                test_data[i] = tuple(test_data[i])

            self.logger.info(f"{i}\tmae\trmse\tr2")
            p = self.train(ReactionDataset(training_data, radius=self.radius, smi2id=self.smi2id, omit=False, mol_max_len=15, data_prefix=self.confs_prefix, symmetric_id=self.symmetric_id), 
                           ReactionDataset(test_data, radius=self.radius, smi2id=self.smi2id, omit=False, mol_max_len=15, data_prefix=self.confs_prefix, symmetric_id=self.symmetric_id), mean_, std_)
            if p:
                self.logger.info(f"{p['mae']}\t{p['rmse']}\t{p['r2']}")
                all_best_p.append(p)
        if all_best_p:
            self.print_results(all_best_p)

    def run_NiColit(self):
        with open("/amax/data/group_0/yield_data/ni.pkl", "rb") as f:
            dataset_ni = pickle.load(f)
        start = int(len(dataset_ni)*0.8)
        """three parts"""
        data_type = []
        for one in dataset_ni:
            data_type.append([one[0].replace(">", ".", 1).replace(">", ">>", 1) if ">>" not in one[0] else one[0], 0, one[1]])

        all_best_p = []
        # 10 splits
        for i in range(10):
            seed_everything(seed=i)
            random.shuffle(data_type)
            # for parameter selection
            if i >= self.args.repeat_times:
                break
            training_data = data_type[:start]
            test_data = data_type[start:]

            y_training = torch.as_tensor([one[-1] for one in training_data])
            mean_ = torch.mean(y_training) if self.args.norm else 0.
            std_ = torch.std(y_training) if self.args.norm else 1.

            for i in range(len(training_data)):
                training_data[i][-1] = (training_data[i][-1]-mean_)/std_
                training_data[i] = tuple(training_data[i])
            for i in range(len(test_data)):
                test_data[i][-1] = (test_data[i][-1]-mean_)/std_
                test_data[i] = tuple(test_data[i])

            self.logger.info(f"{i}\tmae\trmse\tr2")
            p = self.train(ReactionDataset(training_data, radius=self.radius, smi2id=self.smi2id, omit=False, mol_max_len=15, data_prefix=self.confs_prefix, symmetric_id=self.symmetric_id), 
                           ReactionDataset(test_data, radius=self.radius, smi2id=self.smi2id, omit=False, mol_max_len=15, data_prefix=self.confs_prefix, symmetric_id=self.symmetric_id), mean_, std_)
            if p:
                self.logger.info(f"{p['mae']}\t{p['rmse']}\t{p['r2']}")
                all_best_p.append(p)
        if all_best_p:
            self.print_results(all_best_p)

    def run_NiColit_test(self):
        with open("/amax/data/group_0/yield_data/ni.pkl", "rb") as f:
            dataset_ni = pickle.load(f)
        with open("/amax/data/group_0/yield_data/ni_substrate_class.pkl", "rb") as f:
            substrate_class = pickle.load(f)
        oos_name = self.args.ds_name
        
        """three parts"""
        data_type = []
        for one in dataset_ni:
            data_type.append([one[0].replace(">", ".", 1).replace(">", ">>", 1) if ">>" not in one[0] else one[0], 0, one[1]])
        training_data, test_data = [], []
        for i in range(len(data_type)):
            if substrate_class[i] == oos_name:
                test_data.append(data_type[i])
            else:
                training_data.append(data_type[i])

        all_best_p = []
        # 10 splits
        for i in range(5):
            seed_everything(seed=i)
            random.shuffle(training_data)
            # for parameter selection
            if i >= self.args.repeat_times:
                break
            # training_data = data_type[:start]
            # test_data = data_type[start:]

            y_training = torch.as_tensor([one[-1] for one in training_data])
            mean_ = torch.mean(y_training) if self.args.norm else 0.
            std_ = torch.std(y_training) if self.args.norm else 1.

            for i in range(len(training_data)):
                training_data[i][-1] = (training_data[i][-1]-mean_)/std_
                training_data[i] = tuple(training_data[i])
            for i in range(len(test_data)):
                test_data[i][-1] = (test_data[i][-1]-mean_)/std_
                test_data[i] = tuple(test_data[i])

            self.logger.info(f"{i}\tmae\trmse\tr2")
            p = self.train(ReactionDataset(training_data, radius=self.radius, smi2id=self.smi2id, omit=False, mol_max_len=15, data_prefix=self.confs_prefix, symmetric_id=self.symmetric_id), 
                           ReactionDataset(test_data, radius=self.radius, smi2id=self.smi2id, omit=False, mol_max_len=15, data_prefix=self.confs_prefix, symmetric_id=self.symmetric_id), mean_, std_)
            if p:
                self.logger.info(f"{p['mae']}\t{p['rmse']}\t{p['r2']}")
                all_best_p.append(p)
        if all_best_p:
            self.print_results(all_best_p)

    def run_NS(self):
        """add split ratio"""
        name_split = [('FullCV_{:02d}'.format(i), 600) for i in range(1, 11)]
        all_best_p = []
        # 10 splits
        for i, (name, split) in enumerate(name_split):
            # for parameter selection
            if i >= self.args.repeat_times:
                break
            p_prefix = "/downstream_datasets/N,S-acetal Denmark"
            df = pd.read_excel(os.path.join(p_prefix, "Denmark_input_data.xlsx"), sheet_name=name).fillna("")
            df_p = pd.read_csv(os.path.join(p_prefix, "Denmark_data_product.csv"))
            rxns = generate_N_S_acetal_rxns(df, df_p)

            y_training = torch.as_tensor([one[-1] for one in rxns[:split]])
            mean_ = torch.mean(y_training) if self.args.norm else 0.
            std_ = torch.std(y_training) if self.args.norm else 1.
            """three parts"""
            data_type = []
            for one in rxns:
                data_type.append((one[0].replace(">", ".", 1).replace(">", ">>", 1) if ">>" not in one[0] else one[0], 0, (one[1]-mean_)/std_))

            training_data = ReactionDataset(data_type[:split], radius=self.radius, smi2id=self.smi2id, omit=False, data_prefix=self.confs_prefix, symmetric_id=self.symmetric_id)
            test_data = ReactionDataset(data_type[split:], radius=self.radius, smi2id=self.smi2id, omit=False, data_prefix=self.confs_prefix, symmetric_id=self.symmetric_id)

            self.logger.info(f"{i}\tmae\trmse\tr2")
            p = self.train(training_data, test_data, mean_, std_)
            if p:
                self.logger.info(f"{p['mae']}\t{p['rmse']}\t{p['r2']}")
                all_best_p.append(p)
        if all_best_p:
            self.print_results(all_best_p, times=1.)

    def run_NS_test(self):
        """add split ratio"""
        name_split_dict = {"NS_test_sub": ("test_sub", 385-1, 600), 
                           "NS_test_cat": ("test_cat", 385-1, 688), 
                           "NS_test_sub-cat": ("test_sub-cat", 385-1, 555)}
        (name, start, end) = name_split_dict[self.args.ds_name]
        all_best_p = []
        # 10 splits
        for i in range(5):
            # for parameter selection
            if i >= self.args.repeat_times:
                break
            p_prefix = "/downstream_datasets/N,S-acetal Denmark"
            df = pd.read_excel(os.path.join(p_prefix, "Denmark_input_data.xlsx"), sheet_name=name).fillna("")
            df_p = pd.read_csv(os.path.join(p_prefix, "Denmark_data_product.csv"))
            rxns = generate_N_S_acetal_rxns(df, df_p)

            # train_set, test_set = set(df["Catalyst"][:start]), set(df["Catalyst"][start:])

            y_training = torch.as_tensor([one[-1] for one in rxns[:start]+rxns[end:]])
            mean_ = torch.mean(y_training) if self.args.norm else 0.
            std_ = torch.std(y_training) if self.args.norm else 1.
            """three parts"""
            data_type = []
            for one in rxns:
                data_type.append((one[0].replace(">", ".", 1).replace(">", ">>", 1) if ">>" not in one[0] else one[0], 0, (one[1]-mean_)/std_))

            training_data = ReactionDataset(data_type[:start]+data_type[end:], radius=self.radius, smi2id=self.smi2id, omit=False, data_prefix=self.confs_prefix, symmetric_id=self.symmetric_id)
            test_data = ReactionDataset(data_type[start:end], radius=self.radius, smi2id=self.smi2id, omit=False, data_prefix=self.confs_prefix, symmetric_id=self.symmetric_id)

            self.logger.info(f"{i}\tmae\trmse\tr2")
            p = self.train(training_data, test_data, mean_, std_)
            if p:
                self.logger.info(f"{p['mae']}\t{p['rmse']}\t{p['r2']}")
                all_best_p.append(p)
        if all_best_p:
            self.print_results(all_best_p, times=1.)

    def run_Heteroatom(self):
        """add split ratio"""
        name_split = [(f'Blatt{i}', 1075) for i in range(1, 11)]
        all_best_p = []
        # 10 splits
        for i, (name, split) in enumerate(name_split):
            # for parameter selection
            if i >= self.args.repeat_times:
                break
            df = pd.read_excel(os.path.join("/downstream_datasets/C-Heteroatom", "Cernak_and_Dreher_input_data.xlsx"), sheet_name=name)
            rxns = generate_C_Heteroatom_rxns(df)

            y_training = torch.as_tensor([one[-1] for one in rxns[:split]])
            mean_ = torch.mean(y_training) if self.args.norm else 0.
            std_ = torch.std(y_training) if self.args.norm else 1.
            """three parts"""
            data_type = []
            for one in rxns:
                data_type.append((one[0].replace(">", ".", 1).replace(">", ">>", 1) if ">>" not in one[0] else one[0], 0, (one[1]-mean_)/std_))

            training_data = ReactionDataset(data_type[:split], radius=self.radius, smi2id=self.smi2id, omit=False, data_prefix=self.confs_prefix, symmetric_id=self.symmetric_id)
            test_data = ReactionDataset(data_type[split:], radius=self.radius, smi2id=self.smi2id, omit=False, data_prefix=self.confs_prefix, symmetric_id=self.symmetric_id)

            self.logger.info(f"{i}\tmae\trmse\tr2")
            p = self.train(training_data, test_data, mean_, std_)
            if p:
                self.logger.info(f"{p['mae']:.6f}\t{p['rmse']:.6f}\t{p['r2']:.6f}")
                all_best_p.append(p)
        if all_best_p:
            self.print_results(all_best_p)

    def run_Heteroatom_test(self):
        """add split ratio"""
        name, split = "OOS", 1153-1
        all_best_p = []
        # 10 splits
        for i in range(5):
            # for parameter selection
            if i >= self.args.repeat_times:
                break
            df = pd.read_excel(os.path.join("/downstream_datasets/C-Heteroatom", "Cernak_and_Dreher_input_data.xlsx"), sheet_name=name)
            rxns = generate_C_Heteroatom_rxns(df)

            # train_set, test_set = set(df["Catalyst"][:start]), set(df["Catalyst"][start:])

            y_training = torch.as_tensor([one[-1] for one in rxns[:split]])
            mean_ = torch.mean(y_training) if self.args.norm else 0.
            std_ = torch.std(y_training) if self.args.norm else 1.
            """three parts"""
            data_type = []
            for one in rxns:
                data_type.append((one[0].replace(">", ".", 1).replace(">", ">>", 1) if ">>" not in one[0] else one[0], 0, (one[1]-mean_)/std_))

            training_data = ReactionDataset(data_type[:split], radius=self.radius, smi2id=self.smi2id, omit=False, data_prefix=self.confs_prefix, symmetric_id=self.symmetric_id)
            test_data = ReactionDataset(data_type[split:], radius=self.radius, smi2id=self.smi2id, omit=False, data_prefix=self.confs_prefix, symmetric_id=self.symmetric_id)

            self.logger.info(f"{i}\tmae\trmse\tr2")
            p = self.train(training_data, test_data, mean_, std_)
            if p:
                self.logger.info(f"{p['mae']:.6f}\t{p['rmse']:.6f}\t{p['r2']:.6f}")
                all_best_p.append(p)
        if all_best_p:
            self.print_results(all_best_p)

    def run_Condition(self):
        df_test = pd.read_csv('/downstream_datasets/uspto_1k_TPL_test.tsv', sep='\t')
        df_train_valid = pd.read_csv('/downstream_datasets/uspto_1k_TPL_train_valid.tsv', sep='\t')
        df_train = df_train_valid.iloc[:400000]
        df_valid = df_train_valid.iloc[400000:400604]
        rxns_train, rxns_val, rxns_test = [], [], []
        for one_df, one_dataset in zip([df_train, df_valid, df_test], [rxns_train, rxns_val, rxns_test]):
            rxns, labels = one_df["canonical_rxn"].tolist(), one_df["labels"].tolist()
            for i in tqdm(range(len(one_df))):
                one_dataset.append((rxns[i], 0, labels[i]))

        all_best_p = []
        # 10 splits
        for i in range(10):
            seed_everything(seed=i)
            random.shuffle(rxns_train)
            # for parameter selection
            if i >= self.args.repeat_times:
                break

            training_data = ReactionDataset(rxns_train, radius=self.radius, smi2id=self.smi2id, omit=False, data_prefix=self.confs_prefix, symmetric_id=1)
            val_data = ReactionDataset(rxns_val, radius=self.radius, smi2id=self.smi2id, omit=False, data_prefix=self.confs_prefix, symmetric_id=1)
            test_data = ReactionDataset(rxns_test, radius=self.radius, smi2id=self.smi2id, omit=False, data_prefix=self.confs_prefix, symmetric_id=1)

            self.logger.info(f"{i}\tacc")
            p = self.train_cls(training_data, val_data, test_data)
            if p:
                self.logger.info(f"{p['acc']:.6f}")
                all_best_p.append(p)
        if all_best_p:
            self.print_results_cls(all_best_p)

    def train_cls(self, training_data, val_data, test_data):
        model = RxnTrainer(args=self.args)
        callbacks = []
        callbacks.append(plc.ModelCheckpoint(dirpath=self.dirpath, 
                                             monitor="val/loss", 
                                             filename='finetuning_{epoch:03d}-{step:08d}', 
                                             save_top_k=1, 
                                             save_on_train_epoch_end=True))
        # callbacks.append(LitProgressBar())
        tb_logger = pl_loggers.WandbLogger(save_dir=self.dirpath, project="rxn_finetuning", 
                                           name=f"{self.args.ds_name}_{self.args.batch_size}_{self.args.accumulate_grad_batches}_{args.init_lr}_{args.warmup_steps}")
        trainer = pl.Trainer(
            accelerator=self.args.accelerator,
            devices=self.args.devices,
            precision=self.args.precision,
            max_epochs=self.args.max_epochs,
            accumulate_grad_batches=self.args.accumulate_grad_batches,
            check_val_every_n_epoch=self.args.check_val_every_n_epoch,
            callbacks=callbacks,
            logger=tb_logger,
            strategy="auto", #"ddp_find_unused_parameters_true", "auto", args.strategy_name
            enable_checkpointing=True,
            )

        training_loader = DataLoader(training_data, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                    shuffle=True, collate_fn=MyCollater())
        val_loader = DataLoader(val_data, batch_size=self.args.batch_size, num_workers=self.args.num_workers, 
                                shuffle=False, collate_fn=MyCollater())
        test_loader = DataLoader(test_data, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                shuffle=False, collate_fn=MyCollater())
        
        trainer.fit(model=model, train_dataloaders=training_loader, val_dataloaders=val_loader)

        self.logger.info(trainer.checkpoint_callback.best_model_path)
        model = RxnTrainer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, strict=False, args=self.args)
        res_ = {}
        if trainer.global_rank == 0:
            trainer_test = pl.Trainer(devices=[self.args.devices[0]], accelerator="gpu", logger=False)
            p = trainer_test.predict(model=model, dataloaders=test_loader)
            all_preds, all_labels = [], []
            for one in p:
                all_preds.append(one.preds)
                all_labels.append(one.labels)
            all_preds, all_labels = torch.cat(all_preds), torch.cat(all_labels)
            print(all_preds.shape)
            _, predicted = torch.max(all_preds, 1)
            accuracy = (predicted == all_labels).float().mean()
            res_ = {"acc": accuracy, }
            os.remove(trainer.checkpoint_callback.best_model_path)
        return res_

    def print_results_cls(self, all_best_p, times=100):
        res_ = {"acc": []}
        res_float = {"acc": []}
        for one in all_best_p:
            for k in one:
                res_[k].append(f"{float(one[k]*times):.4f}")
                res_float[k].append(float(one[k]*times))
        for k in res_:
            self.logger.info(res_[k])
            self.logger.info(f"{k}\t{np.mean(res_float[k]):.4f}\t{np.std(res_float[k]):.4f}")


if __name__ == '__main__':
    args = get_args()

    runner = Finetuner(args)
    if runner.args.ds_name in ["BH", "SM"]:
        runner.run_BH_or_SM()
    elif runner.args.ds_name in ["pyridyl"]:
        runner.run_BH_reactant()
    elif runner.args.ds_name in ["SM_test2"]:
        runner.run_SM_ligand()
    elif runner.args.ds_name in ["ELN"]:
        runner.run_ELN()
    elif runner.args.ds_name in ["NiColit"]:
        runner.run_NiColit()
    elif runner.args.ds_name in ['OPiv']:
        runner.run_NiColit_test()
    elif runner.args.ds_name in ["NS_acetal"]:
        runner.run_NS()
    elif runner.args.ds_name in ["NS_test_cat"]:
        runner.run_NS_test()
    elif runner.args.ds_name in ["Heteroatom"]:
        runner.run_Heteroatom()
    elif runner.args.ds_name in ["Heteroatom_test"]:
        runner.run_Heteroatom_test()
    elif runner.args.ds_name in ["Condition"]:
        runner.run_Condition()
    else:
        raise NotImplementedError

    print()
