from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch import nn, Tensor, tensor
from torch.nn.functional import softplus, hardsigmoid

from opal.model.positive_linear import PositiveLinear


class DeltaModel(pl.LightningModule):
    def __init__(
        self,
        le_uid: LabelEncoder,
        le_mid: LabelEncoder,
        qt_acc: QuantileTransformer,
        n_uid_support: list,
        n_mid_support: list,
        n_emb: int = 2,
        n_delta_neurons: int = 4,
        lr: float = 0.003,
        l1_loss_weight: float = 0.001,
        l2_loss_weight: float = 0,
    ):
        """Delta Model for Osu!Mania Score Prediction

        Args:
            le_uid: Label Encoder for User IDs
            le_mid: Label Encoder for Beatmap IDs
            qt_acc: Quantile Transformer for Accuracy
            n_uid_support: Number of Beatmaps each User has played
            n_mid_support: Number of Users each Beatmap has scores on
            n_emb: Number of Embedding Dimensions
            n_delta_neurons: Number of Neurons in the Delta to Accuracy Layers
            lr: Learning Rate
            l1_loss_weight: L1 Loss Weight
            l2_loss_weight: L2 Loss Weight

        """
        super().__init__()
        self.le_uid = le_uid
        self.le_mid = le_mid
        self.qt_acc = qt_acc
        self.n_uid_support = nn.Parameter(
            tensor(n_uid_support, dtype=torch.int),
            requires_grad=False,
        )
        self.n_mid_support = nn.Parameter(
            tensor(n_mid_support, dtype=torch.int),
            requires_grad=False,
        )
        self.lr = lr
        self.l1_loss_weight = l1_loss_weight
        self.l2_loss_weight = l2_loss_weight

        n_uid = len(le_uid.classes_)
        n_mid = len(le_mid.classes_)

        # Embeddings for User and Beatmap IDs
        self.emb_uid = nn.Embedding(n_uid, n_emb)
        self.emb_mid = nn.Embedding(n_mid, n_emb)

        # Batch Normalization for Embedding Differences
        self.bn = nn.BatchNorm1d(n_emb, affine=False)

        # Linear Layers for Embedding Differences to Accuracy Mean and Variance
        # The positive linear layer ensures the estimated function is
        # monotonic increasing
        self.delta_to_acc = nn.Sequential(
            PositiveLinear(n_emb, n_delta_neurons),
            nn.Tanh(),
            PositiveLinear(n_delta_neurons, 2),
        )

        self.save_hyperparameters()

    def forward(self, x_uid, x_mid):
        # Convert the One-Hot Encoded User ID and Beatmap ID to Embeddings
        x_uid = self.emb_uid(x_uid)
        x_mid = self.emb_mid(x_mid)

        # Calculate the difference between the User and Beatmap ID Embeddings
        x_delta = x_uid - x_mid

        # Normalize the Embedding Differences
        x_delta = self.bn(x_delta)

        # Convert the Embedding Differences to Accuracy Mean and Variance
        y = self.delta_to_acc(x_delta)

        # Squash the predictions to the range [0, 1]
        y_mean = hardsigmoid(y[:, 0])

        # Make sure the variance is positive
        y_var = softplus(y[:, 1])

        return y_mean, y_var

    @staticmethod
    def loss_nll(
        mean: Tensor,
        var: Tensor,
        target: Tensor,
        eps: float = 1e-10,
    ):
        """Negative Log Likelihood Loss for Gaussian Distribution

        Args:
            mean: Mean of the Gaussian Distribution
            var: Variance of the Gaussian Distribution
            target: Target Value
            eps: Epsilon to prevent NaNs. Defaults to 1e-10.
        """
        return (
            (torch.log(var + eps) / 2)
            + ((target - mean) ** 2 / (2 * var + eps))
        ).mean()

    @staticmethod
    def loss_nll_laplace(
        mean: Tensor,
        var: Tensor,
        target: Tensor,
        eps: float = 1e-10,
    ):
        """Negative Log Likelihood Loss for Laplace Distribution

        Args:
            mean: Mean of the Gaussian Distribution
            var: Variance of the Gaussian Distribution
            target: Target Value
            eps: Epsilon to prevent NaNs. Defaults to 1e-10.
        """
        std = torch.sqrt(var + eps)
        return (torch.log(2 * std) + torch.abs(target - mean) / std).mean()

    def decoded_rmse(self, mean: Tensor, var: Tensor, target: Tensor):  # noqa
        return (
            nn.MSELoss()(
                tensor(self.decode_acc(mean.cpu().reshape(-1, 1))),
                tensor(self.decode_acc(target.cpu().reshape(-1, 1))),
            )
            ** 0.5
        )

    def step(self, batch: tuple[Tensor, Tensor, Tensor], loss_fn: Any):
        x_uid, x_mid, y = batch
        y_pred_mean, y_pred_var = self(x_uid, x_mid)
        return loss_fn(y_pred_mean, y_pred_var, y)

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ):
        self.log(
            "train/nll_loss",
            (loss := self.step(batch, self.loss_nll_laplace)),
            prog_bar=True,
        )
        # Add l1 regularization to the embeddings
        l1 = (
            torch.exp(self.emb_uid.weight).sum()
            + torch.exp(self.emb_mid.weight).sum()
        )
        self.log("train/l1_loss", l1, prog_bar=True)
        l2 = l1**2
        self.log("train/l2_loss", l2)
        return loss + self.l1_loss_weight * l1 + self.l2_loss_weight * l2

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ):
        self.log(
            "val/rmse_loss",
            (loss := self.step(batch, self.decoded_rmse)),
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        self.log(
            "test/rmse_loss",
            (loss := self.step(batch, self.decoded_rmse)),
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=1,
            factor=0.5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/rmse_loss",
        }

    @property
    def uid_classes(self) -> np.ndarray:
        return np.array(self.le_uid.classes_)

    @property
    def mid_classes(self) -> np.ndarray:
        return np.array(self.le_mid.classes_)

    def get_embeddings(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns the Embeddings for Users and Beatmaps

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: User and Beatmap Embeddings
            The DataFrames contain the Embeddings and the Support for each
            User and Beatmap.
            The Support is the number of scores each User and Beatmap is
            associated with.
        """
        uid = self.emb_uid.weight.detach().numpy()
        mid = self.emb_mid.weight.detach().numpy()

        df_emb_uid = pd.DataFrame(
            {
                **{f"D{k}": emb for k, emb in enumerate(uid.T)},
                **{"support": self.n_uid_support},
            },
            index=self.multi_index_uid(),
        )

        df_emb_mid = pd.DataFrame(
            {
                **{f"D{k}": emb for k, emb in enumerate(mid.T)},
                **{"support": self.n_mid_support},
            },
            index=self.multi_index_mid(),
        )

        return df_emb_uid, df_emb_mid

    def predict_user(self, username: str | list[str]) -> pd.DataFrame:
        """Predicts the accuracy for all beatmaps for a given user

        Args:
            username: User Name

        Returns:
            DataFrame: Mean, Lower Bound, and Upper Bound.
            The lower and upper bounds are one standard deviation away from
             the mean.

        """
        username = [username] if isinstance(username, str) else username
        uid = self.encode_uid(username)
        n_mid = len(self.mid_classes)
        x_uid = tensor(uid).repeat(n_mid)
        x_mid = tensor(range(n_mid))

        y_mean, y_var = self(x_uid, x_mid)
        y_std = torch.sqrt(y_var)

        y_mean_decode = self.decode_acc(
            y_mean.detach().numpy().reshape(-1, 1)
        )[:, 0]
        y_mean_upper_decode = self.decode_acc(
            (y_mean + y_std).detach().numpy().reshape(-1, 1)
        )[:, 0]
        y_mean_lower_decode = self.decode_acc(
            (y_mean - y_std).detach().numpy().reshape(-1, 1)
        )[:, 0]

        df_mean = pd.DataFrame(
            {
                "Mean": y_mean_decode,
                "Lower Bound": y_mean_lower_decode,
                "Upper Bound": y_mean_upper_decode,
            },
            index=self.multi_index_mid(),
        )
        return df_mean

    def predict_map(self, mapname: str | list[str]) -> pd.DataFrame:
        """Predicts the accuracy for all users for a given beatmap

        Args:
            mapname: Map Name
            speed: Map Speed

        Returns:
            DataFrame: Mean, Lower Bound, and Upper Bound.
            The lower and upper bounds are one standard deviation away from
             the mean.

        """

        mapname = [mapname] if isinstance(mapname, str) else mapname
        mid = self.encode_mid(mapname)
        n_uid = len(self.uid_classes)
        x_uid = tensor(range(n_uid))
        x_mid = tensor(mid).repeat(n_uid)

        y_mean, y_var = self(x_uid, x_mid)
        y_std = torch.sqrt(y_var)

        y_mean_decode = self.decode_acc(
            y_mean.detach().numpy().reshape(-1, 1)
        )[:, 0]
        y_mean_upper_decode = self.decode_acc(
            (y_mean + y_std).detach().numpy().reshape(-1, 1)
        )[:, 0]
        y_mean_lower_decode = self.decode_acc(
            (y_mean - y_std).detach().numpy().reshape(-1, 1)
        )[:, 0]

        df_mean = pd.DataFrame(
            {
                "Mean": y_mean_decode,
                "Lower Bound": y_mean_lower_decode,
                "Upper Bound": y_mean_upper_decode,
            },
            index=self.multi_index_uid(),
        )
        return df_mean

    def multi_index_uid(self):
        return pd.MultiIndex.from_frame(
            pd.Series(self.uid_classes)
            .str.split("/", expand=True)
            .rename(columns={0: "User Name", 1: "Year", 2: "Keys"})
            .astype({"Year": int, "Keys": float})
            # We cast to float, then int to handle strings with decimals
            .astype({"Keys": int})
        )

    def multi_index_mid(self):
        return pd.MultiIndex.from_frame(
            pd.Series(self.mid_classes)
            .str.split("/", expand=True)
            .rename(columns={0: "Map Name", 1: "Speed", 2: "Keys"})
            .astype({"Speed": int, "Keys": float})
            # We cast to float, then int to handle strings with decimals
            .astype({"Keys": int})
        )

    def decode_acc(self, x):
        x = [x] if isinstance(x, (int, float)) else x
        return self.qt_acc.inverse_transform(x)

    def encode_acc(self, x):
        x = [x] if isinstance(x, (int, float)) else x
        return self.qt_acc.transform(x)

    def decode_uid(self, x):
        x = [x] if isinstance(x, (int, float)) else x
        return self.le_uid.inverse_transform(x)

    def decode_mid(self, x):
        x = [x] if isinstance(x, (int, float)) else x
        return self.le_mid.inverse_transform(x)

    def encode_uid(self, x):
        x = [x] if isinstance(x, str) else x
        return self.le_uid.transform(x)

    def encode_mid(self, x):
        x = [x] if isinstance(x, str) else x
        return self.le_mid.transform(x)
