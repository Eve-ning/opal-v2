import lightning as pl
from prefect import flow, task

from data.data import df_dan, OsuDataModule
from model.model import Model


@task(name="Training Model")
def train(trainer: pl.Trainer, dm: OsuDataModule, m: Model):
    trainer.fit(m, datamodule=dm)


@flow(name="Create Opal Model")
def create_model():
    dm = OsuDataModule(df=df_dan(), min_map_plays=1, min_user_plays=1)

    m = Model(
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
        max_epochs=100, accelerator="gpu", default_root_dir="dan"
    )
    train(trainer, dm, m)


if __name__ == "__main__":
    create_model()
