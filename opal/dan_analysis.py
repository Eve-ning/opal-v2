import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from prefect import flow, task

from opal.data import df_dan, OsuDataModule
from opal.model.delta_model import DeltaModel


@task(name="Training Model")
def train(trainer: pl.Trainer, dm: OsuDataModule, m: DeltaModel):
    trainer.fit(m, datamodule=dm)


@flow(name="Create Opal Model")
def create_model():
    dm = OsuDataModule(df=df_dan(), min_map_plays=1, min_user_plays=1)

    m = DeltaModel(
        uid_le=dm.uid_le,
        mid_le=dm.mid_le,
        acc_qt=dm.acc_qt,
        ln_ratio_weights=dm.ln_ratio_weights,
        rc_emb=1,
        ln_emb=1,
        rc_delta_emb=3,
        ln_delta_emb=3,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        default_root_dir="dan_checkpoints",
        callbacks=[
            ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
        ],
    )
    train(trainer, dm, m)


if __name__ == "__main__":
    create_model()
