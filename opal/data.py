from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Literal

import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader

from opal.utils import db_conn

conn = db_conn.fn()


@cache
def df_k(
    keys: int | None = 7,
    sample: Literal["full", "1%", "10%", "1%_cached"] = "full",
) -> pd.DataFrame:
    if sample == "full":
        dataset = "osu_dataset"
    elif sample == "1%":
        dataset = "osu_dataset_sample_1"
    elif sample == "10%":
        dataset = "osu_dataset_sample_10"
    elif sample == "1%_cached":
        pass
    else:
        raise ValueError(f"Invalid sample: {sample}")

    if sample == "1%_cached":
        cache_1_path = Path(__file__).parent / "osu_dataset_sample_1.csv"
        assert cache_1_path.exists(), "Cache 1% does not exist"
        df = pd.read_csv(cache_1_path)
        df = df.loc[df["keys"] == keys] if keys else df
    else:
        df = pd.read_sql(
            rf"SELECT * FROM {dataset}" rf" WHERE `keys` = {keys}"
            if keys
            else "",
            conn,
        )

    return df


def df_remove_low_support_prob(
    df: pd.DataFrame, min_prob: float = 0.1
) -> pd.DataFrame:
    return df.loc[df["prob"].rank(pct=True) > min_prob]


def df_remove_low_support_maps(
    df: pd.DataFrame, min_map_plays: int = 50
) -> pd.DataFrame:
    return (
        df.groupby("mid")
        .filter(lambda x: len(x) >= min_map_plays)
        .reset_index(drop=True)
    )


def df_remove_low_support_users(
    df: pd.DataFrame, min_user_plays: int = 50
) -> pd.DataFrame:
    return (
        df.groupby("uid")
        .filter(lambda x: len(x) >= min_user_plays)
        .reset_index(drop=True)
    )


class OsuDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 256,
        p_test: float | None = 0.2,
        p_remove_low_support_prob: float = 0.1,
        n_acc_quantiles: int = 1000,
    ):
        """DataModule for the osu! dataset

        Notes:
            The dataset is preprocessed to remove low support maps and users.
            The accuracy is transformed to a uniform distribution using a
            quantile transformer.

        Args:
            df: osu! dataset. See opal.data
            batch_size: Batch size
            p_test: The proportion of the data to use as test data.
                If None or 0, the entire dataset is used for training.
            p_remove_low_support_prob: The proportion of the data to remove
                based on the probability of the map and user.
                This is done to remove low support maps and users.
            n_acc_quantiles: The number of quantiles to use for the
                quantile transformer for the accuracy.
        """
        super().__init__()
        self.batch_size = batch_size
        self.min_prob = p_remove_low_support_prob
        self.n_acc_quantiles = n_acc_quantiles

        self.le_uid = LabelEncoder()
        self.le_mid = LabelEncoder()
        self.qt_acc = QuantileTransformer(
            n_quantiles=self.n_acc_quantiles,
            output_distribution="uniform",
        )

        # Remove low support maps and users
        #   We do this regardless of train/validation as we're simply not
        #   interested in low support maps and users
        #   This can artificially inflate the accuracy of the model, however,
        #   during deployment, we're not supporting these predictions anyway.
        df = df_remove_low_support_prob(df, p_remove_low_support_prob)

        df = df.assign(
            uid=lambda x: x["username"] + "/" + x["year"].astype(str),
            mid=lambda x: x["mapname"] + "/" + x["speed"].astype(str),
        )

        # We fit the transform on the whole dataset, doesn't cause data leakage
        df = df.assign(
            uid=lambda x: self.le_uid.fit_transform(x["uid"]),
            mid=lambda x: self.le_mid.fit_transform(x["mid"]),
        )
        df_train, df_test = (
            train_test_split(df, test_size=p_test, random_state=42)
            if p_test
            else (df, df)
        )

        # Fit the transform only on the training data to avoid data leakage
        df_train["accuracy"] = self.qt_acc.fit_transform(
            df_train["accuracy"].to_numpy().reshape(-1, 1)
        )
        df_test["accuracy"] = self.qt_acc.transform(
            df_test["accuracy"].to_numpy().reshape(-1, 1)
        )

        self.ds_train = TensorDataset(
            tensor(df_train["uid"].to_numpy()),
            tensor(df_train["mid"].to_numpy()),
            tensor(df_train["accuracy"].to_numpy()).to(float),
        )
        self.ds_test = TensorDataset(
            tensor(df_test["uid"].to_numpy()),
            tensor(df_test["mid"].to_numpy()),
            tensor(df_test["accuracy"].to_numpy()).to(float),
        )
        self.df = df

    @property
    def n_uid(self):
        return len(self.le_uid.classes_)

    @property
    def n_mid(self):
        return len(self.le_mid.classes_)

    @property
    def ln_ratio_weights(self):
        return (
            self.df.set_index(
                [self.df["mapname"] + "/" + self.df["speed"].astype(str)]
            )["ln_ratio"]
            .groupby(level=0)
            .first()[self.le_mid.classes_]
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_test, batch_size=self.batch_size, drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test, batch_size=self.batch_size, drop_last=True
        )
