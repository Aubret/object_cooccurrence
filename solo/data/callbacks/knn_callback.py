from typing import Dict, Any, Tuple, List

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from solo.data.classification_dataloader import prepare_datasets, prepare_transforms
from solo.utils.knn import WeightedKNNClassifier
import torch.nn.functional as F


class KNNCallback(pl.Callback):
    def __init__(self, cfg: DictConfig, dir="knn" ):
        self.cfg = cfg
        self.dir = dir
        self.train_loader, self.test_loader = None, None

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        T_train, T_val = prepare_transforms(self.cfg.dataset, **(self.cfg.transform_kwargs or {}))

        train_dataset, val_dataset = prepare_datasets(
            self.cfg.dataset,
            T_train,
            T_val,
            train_data_path=self.cfg.train_path,
            val_data_path=self.cfg.val_path,
            data_format=self.cfg.format,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=DistributedSampler(train_dataset, shuffle=False)
        )
        self.test_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            sampler=DistributedSampler(val_dataset, shuffle=False)
        )

        if isinstance(self.cfg.perform_every_n_batches, float):
            print("Estimated stepping batches", trainer.estimated_stepping_batches)
            self.cfg.perform_every_n_batches = int(trainer.estimated_stepping_batches * self.cfg.perform_every_n_batches / trainer.max_epochs)

    # def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    #     if self.cfg.perform_on_validation and trainer.current_epoch >= self.cfg.delay_epochs and not trainer.current_epoch%self.cfg.freq_epochs:
    #         self._run(trainer, pl_module)

    # def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # if "vit" in self.cfg.backbone.name:
        # if self.cfg.perform_on_validation and trainer.current_epoch >= self.cfg.delay_epochs and not trainer.current_epoch%self.cfg.freq_epochs:
        #     self._run(trainer, pl_module)


    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.cfg.perform_on_validation and trainer.current_epoch >= self.cfg.delay_epochs and not trainer.current_epoch%self.cfg.freq_epochs and trainer.current_epoch != 0:
            self._run(trainer, pl_module)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.cfg.perform_on_validation and trainer.current_epoch >= self.cfg.delay_epochs and trainer.current_epoch == 0:
            self._run(trainer, pl_module)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.cfg.perform_on_test:
            self._run(trainer, pl_module)

    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any,
            batch_idx: int
    ) -> None:
        if self.cfg.perform_every_n_batches is not None and batch_idx % self.cfg.perform_every_n_batches == 0 and batch_idx != 0:
            self._run(trainer, pl_module)

    def _run(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if not trainer.sanity_checking and not trainer.fast_dev_run:
            torch.cuda.empty_cache()
            pl_module.eval()

            result = self.run(trainer, pl_module)

            torch.cuda.empty_cache()
            pl_module.train()

            for k, value in result.items():
                pl_module.log_dict(
                # if hasattr(trainer.logger, 'log_metrics'):
                #     trainer.logger.log_metrics(
                    {
                        f'{self.dir}/{self.cfg.dataset}_{k}_top1': value[0],
                        f'{self.dir}/{self.cfg.dataset}_{k}_top5': value[1]
                    # }, step=trainer.global_step)
                    }, sync_dist=True)
                # else:
                #     raise ValueError("Please use a logger that supports `log_metrics`")

    @torch.no_grad()
    def extract_features(self, loader: DataLoader, model: pl.LightningModule, mode: str = "train") -> Tuple[
        torch.Tensor, torch.Tensor]:
        bar = tqdm(loader, desc=f'{mode} KNN',
                   total=len(loader)) if self.cfg.verbose and model.local_rank == 0 else loader

        res_X, res_y = [], []
        for batch in bar:
            X, y = batch
            y = y.to(model.device, non_blocking=True)
            if not isinstance(X, List):
                X = X.to(model.device, non_blocking=True)
                outs = model(X)
                feats = outs["feats"] if self.cfg.distance_fx != "cosine" else F.normalize(outs["feats"])
            else:
                feats = 0
                for x in X:
                    x_dev = x.to(model.device, non_blocking=True)
                    feats += model(x_dev)["feats"] if self.cfg.distance_fx != "cosine" else F.normalize(model(x_dev)["feats"])
                feats = feats / len(X)

            # free, total = torch.cuda.mem_get_info(outs["feats"].device)
            # mem_used_mb = (total - free) / 1024 ** 2
            # print(outs["feats"].shape, mem_used_mb)


            if self.cfg.clone:
                #Memory leak with vit and global pool token without the clone
                feats = feats.clone().detach()
            else:
                feats = feats.detach()
            res_X.append(feats)
            res_y.append(y.detach())
        res_X = torch.cat(res_X)
        res_y = torch.cat(res_y)
        return res_X, res_y

    def run(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> Dict:
        # extract train and test features
        X_train, y_train = self.extract_features(self.train_loader, pl_module, mode="train")
        X_test, y_test = self.extract_features(self.test_loader, pl_module, mode="test")

        # barrier to make sure all features are extracted
        trainer.strategy.barrier()

        result = {}
        for k in self.cfg.k:
            knn = WeightedKNNClassifier(k=k, T=self.cfg.T, distance_fx=self.cfg.distance_fx)
            knn(X_train, y_train, X_test, y_test)
            val_knn_acc1, val_knn_acc5 = knn.compute(normalize=False)
            result[k] = (val_knn_acc1, val_knn_acc5)
            del knn

        return result
