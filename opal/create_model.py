from __future__ import annotations

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


def train(trainer: pl.Trainer, dm: OsuDataModule, m: DeltaModel):
    trainer.fit(m, datamodule=dm)


def create_model(
    epochs: int = 20,
    keys: int | None = 7,
    sample_set: Literal["full", "1%", "10%", "1%_cached"] = "full",
    min_prob: float = 0.1,
    n_acc_quantiles: int = 10000,
    rc_emb: int = 1,
    ln_emb: int = 1,
):
    dm = OsuDataModule(
        df=df_k(keys, sample=sample_set),
        p_remove_low_support_prob=min_prob,
        n_acc_quantiles=n_acc_quantiles,
    )

    m = DeltaModel(
        le_uid=dm.uid_le,
        le_mid=dm.mid_le,
        qt_acc=dm.acc_qt,
        w_ln_ratio=dm.ln_ratio_weights,
        n_rc_emb=rc_emb,
        n_ln_emb=ln_emb,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        default_root_dir="checkpoints",
        callbacks=[
            ModelCheckpoint(monitor="val/rmse_loss", save_top_k=1, mode="min"),
            EarlyStopping(monitor="val/rmse_loss", patience=2, mode="min"),
            LearningRateMonitor(),
        ],
        logger=WandbLogger(
            entity="evening",
            project="opal",
            name="Dev Model 1D Emb VI Full",
        ),
    )
    train(trainer, dm, m)

    return dm, m


torch.set_float32_matmul_precision("medium")
dm, m = create_model(epochs=25, sample_set="full", keys=7, min_prob=0.005)
# %%
wandb.log(
    {
        "model/uid_emb_mean": wandb.Table(
            dataframe=pd.DataFrame(
                {
                    "User": dm.uid_le.classes_,
                    "RC": m.emb_uid_rc.weight.detach().mean(dim=1).numpy(),
                    "LN": m.uid_ln_mean.weight.detach().mean(dim=1).numpy(),
                },
            )
        ),
        "model/mid_emb_mean": wandb.Table(
            dataframe=pd.DataFrame(
                {
                    "Map": dm.mid_le.classes_,
                    "RC": m.mid_rc_mean.weight.detach().mean(dim=1).numpy(),
                    "LN": m.mid_ln_mean.weight.detach().mean(dim=1).numpy(),
                }
            )
        ),
    }
)
# %%
wandb.finish()
