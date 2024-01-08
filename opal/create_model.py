import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from prefect import flow, task

from opal.data import OsuDataModule, df_k
from opal.model.delta_model import DeltaModel
import wandb


@task(name="Training Model")
def train(trainer: pl.Trainer, dm: OsuDataModule, m: DeltaModel):
    trainer.fit(m, datamodule=dm)


@flow(name="Create Opal Model")
def create_model():
    dm = OsuDataModule(df=df_k(7), min_map_plays=50, min_user_plays=50)

    m = DeltaModel(
        uid_le=dm.uid_le,
        mid_le=dm.mid_le,
        acc_qt=dm.acc_qt,
        ln_ratio_weights=dm.ln_ratio_weights,
        rc_emb=1,
        ln_emb=1,
        rc_delta_emb=10,
        ln_delta_emb=10,
    )

    trainer = pl.Trainer(
        max_epochs=40,
        accelerator="cpu",
        default_root_dir="checkpoints",
        callbacks=[
            ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min"),
        ],
        logger=WandbLogger(project="opal"),
    )
    train(trainer, dm, m)


if __name__ == "__main__":
    create_model()