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
    sample_set: Literal["full", "1%", "10%", "1%_cached"] = "full"
    batch_size: int = 2**10
    p_test: float = 0
    n_min_support: int = 10
    n_acc_quantiles: int = 10000
    n_emb: int = 1
    n_epochs: int = 25
    n_patience_early_stopping: int = 2
    l1_loss_weight: float = 0.001
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
            le_uid=self.datamodule.le_uid,
            le_mid=self.datamodule.le_mid,
            qt_acc=self.datamodule.qt_acc,
            w_ln_ratio=self.datamodule.ln_ratio_weights,
            n_rc_emb=self.n_emb,
            n_ln_emb=self.n_emb,
            l1_loss_weight=self.l1_loss_weight,
            l2_loss_weight=self.l2_loss_weight,
        )

    @cached_property
    def trainer(self):
        return pl.Trainer(
            max_epochs=self.n_epochs,
            accelerator="gpu",
            default_root_dir="checkpoints",
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


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    exp = Experiment(
        n_epochs=25,
        sample_set="full",
        n_keys=(4, 7),
        n_min_support=50,
        p_test=0.10,
        n_emb=3,
        l1_loss_weight=1e-7,
        l2_loss_weight=1e-14,
    )
    exp.fit()

    def get_weight(w):
        return w.weight.detach().cpu().mean(dim=1).numpy()

    wandb.log(
        {
            "model/uid_emb_mean": wandb.Table(
                dataframe=pd.DataFrame(
                    {
                        "User": exp.model.uid_classes,
                        "RC": get_weight(exp.model.emb_uid_rc),
                        "LN": get_weight(exp.model.emb_uid_ln),
                    },
                )
            ),
            "model/mid_emb_mean": wandb.Table(
                dataframe=pd.DataFrame(
                    {
                        "Map": exp.model.mid_classes,
                        "RC": get_weight(exp.model.emb_mid_rc),
                        "LN": get_weight(exp.model.emb_mid_ln),
                    }
                )
            ),
            "model/uid_emb_mean_exp": wandb.Table(
                dataframe=pd.DataFrame(
                    {
                        "User": exp.model.uid_classes,
                        "RC": np.exp(get_weight(exp.model.emb_uid_rc)),
                        "LN": np.exp(get_weight(exp.model.emb_uid_ln)),
                    },
                )
            ),
            "model/mid_emb_mean_exp": wandb.Table(
                dataframe=pd.DataFrame(
                    {
                        "Map": exp.model.mid_classes,
                        "RC": np.exp(get_weight(exp.model.emb_mid_rc)),
                        "LN": np.exp(get_weight(exp.model.emb_mid_ln)),
                    }
                )
            ),
        }
    )
    wandb.finish()
