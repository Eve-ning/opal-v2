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
    p_test: float = 0
    n_min_map_support: int = 10
    n_min_user_support: int = 10
    n_acc_quantiles: int = 10000
    n_rc_emb: int = 1
    n_ln_emb: int = 1
    run_name: str = "Dev Model"
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
            n_min_map_support=self.n_min_map_support,
            n_min_user_support=self.n_min_user_support,
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
            n_rc_emb=self.n_rc_emb,
            n_ln_emb=self.n_ln_emb,
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
                ModelCheckpoint(
                    monitor="val/rmse_loss", save_top_k=1, mode="min"
                ),
                EarlyStopping(
                    monitor="val/rmse_loss",
                    patience=self.n_patience_early_stopping,
                    mode="min",
                ),
                LearningRateMonitor(),
            ],
            logger=WandbLogger(
                entity="evening", project="opal", name=self.run_name
            ),
        )

    def fit(self):
        self.trainer.fit(self.model, datamodule=self.datamodule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    exp = Experiment(
        run_name="Baseline L1 1e-7 L2 1e-14 Min Support 10",
        n_epochs=25,
        sample_set="full",
        n_keys=(4, 7),
        n_min_map_support=10,
        n_min_user_support=10,
        p_test=0.10,
        n_ln_emb=1,
        n_rc_emb=1,
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
