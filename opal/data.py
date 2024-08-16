from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    QuantileTransformer,
    minmax_scale,
)
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader

from opal.dict_transformer import DictionaryTransformer


class OsuDataModule(LightningDataModule):
    def __init__(
        self,
        n_keys: int,
        dataset_path: str | Path,
        batch_size: int = 256,
        p_test: float | None = 0.2,
        n_acc_quantiles: int = 1000,
    ):
        """DataModule for the osu! dataset

        Args:
            n_keys: The number of keys of the dataset to use.
            batch_size: Batch size
            p_test: The proportion of the data to use as test data.
                If None or 0, the entire dataset is used for training.
            n_acc_quantiles: The number of quantiles to use for the
                quantile transformer for the accuracy.
        """
        super().__init__()
        self.n_keys = n_keys
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.n_acc_quantiles = n_acc_quantiles

        self.p_test = p_test
        self.le_uid = LabelEncoder()
        self.le_mid = LabelEncoder()
        self.dt_uid_w = DictionaryTransformer()
        self.dt_mid_w = DictionaryTransformer()
        self.qt_acc = QuantileTransformer(
            n_quantiles=self.n_acc_quantiles,
            output_distribution="normal",
        )
        self.df = None

    def prepare_data(self) -> None:
        df = pd.read_csv(self.dataset_path)
        self.df = df[df["keys"] == self.n_keys]

    def setup(self, stage: str) -> None:
        self.df["year"] = pd.to_datetime(
            self.df["days_since_epoch"], unit="D"
        ).dt.year
        self.df = self.df.assign(
            uid=lambda x: x["uid"].astype(str) + "/" + x["year"].astype(str),
            mid=lambda x: x["mid"].astype(str) + "/" + x["speed"].astype(str),
        )[["uid", "mid", "accuracy"]].assign(
            uid=lambda x: self.le_uid.fit_transform(x["uid"]),
            mid=lambda x: self.le_mid.fit_transform(x["mid"]),
        )

        # Evaluate weight from PageRank
        # We construct our IDs as tuples because NetworkX treats all of them
        # under the same class, nodes. This makes it difficult to retrieve the
        # PageRank later. So we "tag" them with their respective classes.
        g = nx.from_edgelist(
            [
                (("uid", u), ("mid", m))
                for u, m in self.df[["uid", "mid"]].values
            ]
        )
        pr = nx.pagerank(g)

        # A neat trick, if we do (X, Y), it'll make a MultiIndex, which we can
        # retrieve with df.loc["X"]
        df_pr = pd.DataFrame({"w": pr.values()}, index=pr.keys())

        df_pr_uid = df_pr.loc["uid"]
        df_pr_uid[:] = minmax_scale(df_pr_uid)
        self.dt_uid_w.fit(df_pr_uid["w"])

        df_pr_mid = df_pr.loc["mid"]
        df_pr_mid[:] = minmax_scale(df_pr_mid)
        self.dt_mid_w.fit(df_pr_mid["w"])

        if self.p_test:
            df_train, df_test = train_test_split(
                self.df, test_size=self.p_test, random_state=42
            )

            # Fit the transform only on the training data to avoid data leakage
            df_train["accuracy"] = self.qt_acc.fit_transform(
                df_train[["accuracy"]].values
            )
            df_test["accuracy"] = self.qt_acc.transform(
                df_test[["accuracy"]].values
            )
        else:
            df_train = self.df[:]
            df_test = self.df[0 : self.batch_size]
            df_train["accuracy"] = self.qt_acc.fit_transform(
                df_train[["accuracy"]].values
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

    @property
    def n_uid(self):
        return len(self.le_uid.classes_)

    @property
    def n_mid(self):
        return len(self.le_mid.classes_)

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
