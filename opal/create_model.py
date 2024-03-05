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
    keys: int = 7,
    sample_set: Literal["full", "1%", "10%"] = "1%",
    min_prob: float = 0.1,
    n_acc_quantiles: int = 100,
    rc_emb: int = 2,
    ln_emb: int = 2,
):
    dm = OsuDataModule(
        df=df_k(keys, sample=sample_set),
        min_prob=min_prob,
        n_acc_quantiles=n_acc_quantiles,
    )

    m = DeltaModel(
        uid_le=dm.uid_le,
        mid_le=dm.mid_le,
        acc_qt=dm.acc_qt,
        ln_ratio_weights=dm.ln_ratio_weights,
        rc_emb=rc_emb,
        ln_emb=ln_emb,
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
            name="Dev Model 3D Emb",
        ),
    )
    train(trainer, dm, m)

    return dm, m

if __name__ == "__main__":
    create_model()
