from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader

from opal.utils import db_conn

conn = db_conn()


@cache
def df_k(
    keys: Sequence[int] | int | None = (7,),
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

    # Make `keys` compatible with SQL
    keys = (keys,) if isinstance(keys, int) else keys

    if sample == "1%_cached":
        cache_1_path = Path(__file__).parent / "osu_dataset_sample_1.csv"
        assert cache_1_path.exists(), "Cache 1% does not exist"
        df = pd.read_csv(cache_1_path)
        df = df.loc[df["keys"].isin(keys)] if keys else df
    else:
        if keys:
            # SQL breaks for (x,), fix with (x)
            keys = f"({keys[0]})" if len(keys) == 1 else tuple(keys)
            df = pd.read_sql(
                rf"SELECT * FROM {dataset} WHERE `keys` IN {keys}", conn
            )
        else:
            df = pd.read_sql(f"SELECT * FROM {dataset}", conn)

    return df.astype({"keys": int})


def df_remove_low_support(
    df: pd.DataFrame,
    n_min_map_support: int = 10,
    n_min_user_support: int = 10,
) -> pd.DataFrame:
    return df.loc[
        (df["n_mid"] > n_min_map_support) & (df["n_uid"] > n_min_user_support)
    ]


class OsuDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 256,
        p_test: float | None = 0.2,
        # q_remove_low_quantile_support: float = 0.1,
        n_min_user_support: int = 10,
        n_min_map_support: int = 10,
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
            # q_remove_low_quantile_support: The proportion of the data to remove
            #     based on the probability of the map and user.
            #     This is done to remove low support maps and users.
            n_min_user_support: The minimum number of plays a user must have
                to be included in the dataset.
            n_min_map_support: The minimum number of plays a map must have
                to be included in the dataset.
            n_acc_quantiles: The number of quantiles to use for the
                quantile transformer for the accuracy.
        """
        super().__init__()
        self.batch_size = batch_size
        self.n_acc_quantiles = n_acc_quantiles

        self.le_uid = LabelEncoder()
        self.le_mid = LabelEncoder()
        self.qt_acc = QuantileTransformer(
            n_quantiles=self.n_acc_quantiles,
            output_distribution="uniform",
        )

        # Note: This implicitly overwrites the uid, and mid in favour of
        # the encoded uid and mid
        df = df.assign(
            uid=lambda x: x["username"]
            + "/"
            + x["year"].astype(str)
            + "/"
            + x["keys"].astype(str),
            mid=lambda x: x["mapname"]
            + "/"
            + x["speed"].astype(str)
            + "/"
            + x["keys"].astype(str),
        )

        # Remove low support maps and users
        #   We do this regardless of train/validation as we're simply not
        #   interested in low support maps and users
        #   This can artificially inflate the accuracy of the model, however,
        #   during deployment, we're not supporting these predictions anyway.

        # We firstly calculate the quantile support for each map and user
        # Support is the number of entries for a map or user. So a popular
        # map or an active user will have a high support.
        # The quantile support is just the quantile of the support.
        df = df.merge(
            df["mid"].value_counts().rename("n_mid"),
            on="mid",
        ).merge(
            df["uid"].value_counts().rename("n_uid"),
            on="uid",
        )
        df = df_remove_low_support(
            df,
            n_min_map_support=n_min_map_support,
            n_min_user_support=n_min_user_support,
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
    def ln_ratio_weights(self) -> list[float]:
        return (
            self.df.set_index(
                [
                    self.df["mapname"]
                    + "/"
                    + self.df["speed"].astype(str)
                    + "/"
                    + self.df["keys"].astype(str)
                ]
            )["ln_ratio"]
            .groupby(level=0)
            .first()[self.le_mid.classes_]
            .to_list()
        )

    @property
    def n_uid_support(self) -> list[int]:
        return self.df["mid"].value_counts()[self.le_mid.classes_].to_list()

    @property
    def n_mid_support(self) -> list[int]:
        return self.df["uid"].value_counts()[self.le_uid.classes_].to_list()

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
