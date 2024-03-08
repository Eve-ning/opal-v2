from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

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
    n_keys: int | None = 7
    sample_set: Literal["full", "1%", "10%", "1%_cached"] = "full"
    p_test: float = 0
    p_remove_low_support_prob: float = 0.1
    n_acc_quantiles: int = 10000
    n_rc_emb: int = 1
    n_ln_emb: int = 1
    run_name: str = "Dev Model"
    n_epochs: int = 25
    n_patience_early_stopping: int = 2

    @cached_property
    def datamodule(self):
        return OsuDataModule(
            df=df_k(self.n_keys, sample=self.sample_set),
            p_remove_low_support_prob=self.p_remove_low_support_prob,
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
        run_name="Test run new architecture",
        n_epochs=25,
        sample_set="1%_cached",
        n_keys=7,
        p_remove_low_support_prob=0.005,
        p_test=0,
    )
    exp.fit()
    wandb.log(
        {
            "model/uid_emb_mean": wandb.Table(
                dataframe=pd.DataFrame(
                    {
                        "User": exp.model.uid_classes,
                        "RC": exp.model.emb_uid_rc.weight.detach()
                        .mean(dim=1)
                        .numpy(),
                        "LN": exp.model.emb_uid_ln.weight.detach()
                        .mean(dim=1)
                        .numpy(),
                    },
                )
            ),
            "model/mid_emb_mean": wandb.Table(
                dataframe=pd.DataFrame(
                    {
                        "Map": exp.model.mid_classes,
                        "RC": exp.model.emb_mid_rc.weight.detach()
                        .mean(dim=1)
                        .numpy(),
                        "LN": exp.model.emb_mid_rc.weight.detach()
                        .mean(dim=1)
                        .numpy(),
                    }
                )
            ),
        }
    )
    # %%
    wandb.finish()
