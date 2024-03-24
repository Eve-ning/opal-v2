from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger

from opal.data import OsuDataModule, df_k
from opal.model.delta_model import DeltaModel


@dataclass
class Experiment:
    n_keys: Sequence[int] | int | None = (7,)
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
    def df(self):
        return df_k(self.n_keys, sample=self.sample_set)

    @cached_property
    def datamodule(self):
        return OsuDataModule(
            df=self.df,
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
            qt_acc=self.datamodule.qt_acc,
            n_uid_support=self.datamodule.n_uid_support,
            n_mid_support=self.datamodule.n_mid_support,
            n_emb=self.n_emb,
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
            logger=WandbLogger(
                entity="evening",
                project="opal",
                # Gets only the dataclass fields
                # Note: __dict__ will include properties
                config={
                    k: exp.__getattribute__(k)
                    for k in exp.__dataclass_fields__.keys()
                },
                tags=["dev"],
            ),
        )

    def fit(self):
        self.trainer.fit(self.model, datamodule=self.datamodule)

    def wandb_log_weights(self):
        df_uid_emb, df_mid_emb = self.model.get_embeddings()

        wandb.log(
            {
                "model/uid_emb": wandb.Table(
                    dataframe=df_uid_emb.reset_index().assign(
                        meta=lambda x: (
                            x["username"]
                            + "/"
                            + x["year"].astype(str)
                            + "/"
                            + x["keys"].astype(str)
                        )
                    )
                ),
                "model/mid_emb": wandb.Table(
                    dataframe=df_mid_emb.reset_index().assign(
                        meta=lambda x: (
                            x["mapname"]
                            + "/"
                            + x["speed"].astype(str)
                            + "/"
                            + x["keys"].astype(str)
                        )
                    )
                ),
            }
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    n_epochs = 25
    sample_set = "full"
    batch_size = 2**10
    n_min_support = 50
    p_test = 0.10
    n_emb = 2
    l1_loss_weight = 1e-7
    l2_loss_weight = 1e-14
    lr = 1e-3

    exp_fn = lambda k: Experiment(
        n_epochs=n_epochs,
        sample_set=sample_set,
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
    exp.wandb_log_weights()
    wandb.finish()

    del exp

    exp = exp_fn(4)
    exp.fit()
    exp.wandb_log_weights()
    wandb.finish()

    del exp
