from __future__ import annotations

import argparse
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


def train(
    dataset_path: str | Path,
    model_path: Path = RSC_DIR / "models",
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
                dirpath=model_path.parent,
                filename=model_path.stem,
            ),
            EarlyStopping(
                monitor="val/rmse_loss",
                patience=n_patience_early_stopping,
            ),
            LearningRateMonitor(),
        ],
    )

    trainer.fit(model, datamodule=dm)


def entrypoint():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        "-d",
        type=str,
        required=True,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--model_path",
        "-o",
        type=str,
        required=True,
        help="Path to save the model",
    )
    parser.add_argument(
        "--n_keys",
        "-k",
        type=int,
        required=True,
        help="Number of keys",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=10,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lr",
        "-l",
        type=float,
        default=3e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=2**10,
        help="Batch size",
    )
    parser.add_argument(
        "--p_test",
        "-p",
        type=float,
        default=0,
        help="Proportion of test data. Specify 0 to train on the entire "
        "dataset",
    )
    args = parser.parse_args()
    dataset_path = Path.cwd() / args.dataset_path
    model_path = Path.cwd() / args.model_path

    train(
        dataset_path=dataset_path,
        model_path=model_path,
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        n_keys=args.n_keys,
        p_test=args.p_test,
        n_emb=1,
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    entrypoint()
