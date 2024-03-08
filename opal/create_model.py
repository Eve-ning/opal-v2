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


def create_model(
    n_epochs: int = 20,
    n_keys: int | None = 7,
    sample_set: Literal["full", "1%", "10%", "1%_cached"] = "full",
    p_remove_low_support_prob: float = 0.1,
    n_acc_quantiles: int = 10000,
    n_rc_emb: int = 1,
    n_ln_emb: int = 1,
):
    dm = OsuDataModule(
        df=df_k(n_keys, sample=sample_set),
        p_remove_low_support_prob=p_remove_low_support_prob,
        n_acc_quantiles=n_acc_quantiles,
    )

    m = DeltaModel(
        le_uid=dm.le_uid,
        le_mid=dm.le_mid,
        qt_acc=dm.qt_acc,
        w_ln_ratio=dm.ln_ratio_weights,
        n_rc_emb=n_rc_emb,
        n_ln_emb=n_ln_emb,
    )

    trainer = pl.Trainer(
        max_epochs=n_epochs,
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
    trainer.fit(m, datamodule=dm)

    return dm, m


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    dm, m = create_model(
        n_epochs=25,
        sample_set="full",
        n_keys=7,
        p_remove_low_support_prob=0.005,
    )
    wandb.log(
        {
            "model/uid_emb_mean": wandb.Table(
                dataframe=pd.DataFrame(
                    {
                        "User": m.uid_classes,
                        "RC": m.emb_uid_rc.weight.detach().mean(dim=1).numpy(),
                        "LN": m.emb_uid_ln.weight.detach().mean(dim=1).numpy(),
                    },
                )
            ),
            "model/mid_emb_mean": wandb.Table(
                dataframe=pd.DataFrame(
                    {
                        "Map": m.mid_classes,
                        "RC": m.emb_mid_rc.weight.detach().mean(dim=1).numpy(),
                        "LN": m.emb_uid_ln.weight.detach().mean(dim=1).numpy(),
                    }
                )
            ),
        }
    )
    # %%
    wandb.finish()
