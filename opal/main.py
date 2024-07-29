from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Sequence

import lightning as pl
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from opal.data import OsuDataModule
from opal.model.delta_model import DeltaModel


@dataclass
class Experiment:
    n_keys: int = 4
    lr: float = 1e-3
    sample_set: Literal["full", "1%", "10%", "1%_cached"] = "full"
    batch_size: int = 2**10
    p_test: float = 0
    n_min_support: int = 10
    n_acc_quantiles: int = 10000
    n_emb: int = 1
    n_epochs: int = 25
    n_patience_early_stopping: int = 2
    l1_loss_weight: float = 0
    l2_loss_weight: float = 0

    @cached_property
    def datamodule(self):
        return OsuDataModule(
            n_keys=self.n_keys,
            batch_size=self.batch_size,
            n_min_map_support=self.n_min_support,
            n_min_user_support=self.n_min_support,
            n_acc_quantiles=self.n_acc_quantiles,
            p_test=self.p_test,
        )

    @cached_property
    def model(self):
        return DeltaModel(
            one_cycle_total_steps=len(self.datamodule.ds_train)
            // self.batch_size
            * self.n_epochs,
            lr=self.lr,
            le_uid=self.datamodule.le_uid,
            le_mid=self.datamodule.le_mid,
            dt_uid_w=self.datamodule.dt_uid_w,
            dt_mid_w=self.datamodule.dt_mid_w,
            qt_acc=self.datamodule.qt_acc,
            n_emb_mean=self.n_emb,
            l1_loss_weight=self.l1_loss_weight,
            l2_loss_weight=self.l2_loss_weight,
        )

    @cached_property
    def trainer(self):
        return pl.Trainer(
            max_epochs=self.n_epochs,
            accelerator="cpu",
            callbacks=[
                ModelCheckpoint(monitor="val/rmse_loss"),
                EarlyStopping(
                    monitor="val/rmse_loss",
                    patience=self.n_patience_early_stopping,
                ),
                LearningRateMonitor(),
            ],
        )

    def fit(self):
        dm = self.datamodule
        dm.prepare_data()
        dm.setup("")
        self.trainer.fit(self.model, datamodule=dm)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    n_epochs = 1
    batch_size = 2**10
    n_min_support = 50
    p_test = 0.10
    n_emb = 2
    l1_loss_weight = 1e-7
    l2_loss_weight = 1e-14
    lr = 1e-3

    exp_fn = lambda k: Experiment(
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        n_keys=k,
        n_min_support=n_min_support,
        p_test=p_test,
        n_emb=n_emb,
        l1_loss_weight=l1_loss_weight,
        l2_loss_weight=l2_loss_weight,
    )

    exp = exp_fn(7)
    exp.fit()

    del exp
