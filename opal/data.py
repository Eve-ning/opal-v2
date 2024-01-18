import pandas as pd
from prefect import task

from opal.utils import db_conn


import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch.utils.data import TensorDataset, DataLoader, random_split

conn = db_conn.fn()


def df_k(keys: int = 7) -> pd.DataFrame:
    return pd.read_sql(
        rf"SELECT * FROM osu_dataset WHERE `keys` = {keys}",
        conn,
    )


def df_remove_low_support_maps(
    df: pd.DataFrame, min_map_plays: int = 50
) -> pd.DataFrame:
    return (
        df.groupby("mapname")
        .filter(lambda x: len(x) >= min_map_plays)
        .reset_index(drop=True)
    )


def df_remove_low_support_users(
    df: pd.DataFrame, min_user_plays: int = 50
) -> pd.DataFrame:
    return (
        df.groupby("username")
        .filter(lambda x: len(x) >= min_user_plays)
        .reset_index(drop=True)
    )


class OsuDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 256,
        n_train: int = 0.8,
        min_user_plays: int = 50,
        min_map_plays: int = 50,
        n_acc_quantiles: int = 1000,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.n_train = n_train
        self.df = df
        self.min_user_plays = min_user_plays
        self.min_map_plays = min_map_plays
        self.n_acc_quantiles = n_acc_quantiles

        df = df_remove_low_support_maps(df, min_map_plays)
        df = df_remove_low_support_users(df, min_user_plays)
        self.uid_le = LabelEncoder()
        self.mid_le = LabelEncoder()
        self.acc_qt = QuantileTransformer(
            n_quantiles=self.n_acc_quantiles,
            output_distribution="uniform",
        )

        self.uid_enc = self.uid_le.fit_transform(
            df["username"] + "/" + df["year"].astype(str)
        )
        self.mid_enc = self.mid_le.fit_transform(
            df["mapname"] + "/" + df["speed"].astype(str)
        )
        self.acc_tf = self.acc_qt.fit_transform(
            df["accuracy"].to_numpy().reshape(-1, 1)
        ).squeeze()

        self.ds = TensorDataset(
            torch.tensor(self.uid_enc),
            torch.tensor(self.mid_enc),
            torch.tensor(self.acc_tf).to(torch.float),
        )

        n_train = int(len(self.ds) * self.n_train)
        self.train_ds, self.test_ds = random_split(
            self.ds,
            [n_train, len(self.ds) - n_train],
            generator=torch.Generator().manual_seed(42),
        )

    @property
    def n_uid(self):
        return len(self.uid_le.classes_)

    @property
    def n_mid(self):
        return len(self.mid_le.classes_)

    @property
    def ln_ratio_weights(self):
        return (
            self.df.set_index(
                [self.df["mapname"] + "/" + self.df["speed"].astype(str)]
            )["ln_ratio"]
            .groupby(level=0)
            .first()[self.mid_le.classes_]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )
