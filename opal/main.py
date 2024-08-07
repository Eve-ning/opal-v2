from __future__ import annotations

from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from opal.data import OsuDataModule
from opal.model.delta_model import DeltaModel
from opal.utils import RSC_DIR


def main(
    dataset_path: str | Path,
    model_name: str,
    model_dir: str | Path = RSC_DIR / "models",
    n_keys: int = 4,
    lr: float = 1e-3,
    batch_size: int = 2**10,
    p_test: float = 0,
    n_acc_quantiles: int = 10000,
    n_emb: int = 1,
    n_epochs: int = 25,
    n_patience_early_stopping: int = 2,
    l1_loss_weight: float = 0,
    l2_loss_weight: float = 0,
):
    dm = OsuDataModule(
        n_keys=n_keys,
        dataset_path=dataset_path,
        batch_size=batch_size,
        n_acc_quantiles=n_acc_quantiles,
        p_test=p_test,
    )
    dm.prepare_data()
    dm.setup("")

    model = DeltaModel(
        one_cycle_total_steps=len(dm.ds_train) // batch_size * n_epochs,
        lr=lr,
        le_uid=dm.le_uid,
        le_mid=dm.le_mid,
        dt_uid_w=dm.dt_uid_w,
        dt_mid_w=dm.dt_mid_w,
        qt_acc=dm.qt_acc,
        n_emb_mean=n_emb,
        l1_loss_weight=l1_loss_weight,
        l2_loss_weight=l2_loss_weight,
    )

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator="cpu",
        callbacks=[
            ModelCheckpoint(
                monitor="val/rmse_loss",
                dirpath=model_dir,
                filename=model_name,
            ),
            EarlyStopping(
                monitor="val/rmse_loss",
                patience=n_patience_early_stopping,
            ),
            LearningRateMonitor(),
        ],
    )

    trainer.fit(model, datamodule=dm)

    return Path(model_dir) / model_name


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    n_epochs = 10
    batch_size = 2**10
    p_test = 0.10
    n_emb = 1
    lr = 3e-3

    main_fn = lambda k, model_name: main(
        dataset_path=RSC_DIR / "score_dataset.csv",
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        n_keys=k,
        model_name=model_name,
        p_test=p_test,
        n_emb=n_emb,
    )

    main_fn(7, "7k.ckpt")
    main_fn(4, "4k.ckpt")
