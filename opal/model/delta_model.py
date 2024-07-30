from __future__ import annotations

from typing import Any, TypeAlias

import lightning as pl
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch import nn, Tensor, tensor
from torch.nn.functional import softplus

from opal.dict_transformer import DictionaryTransformer
from opal.model.positive_linear import PositiveLinear

Batch: TypeAlias = tuple[Tensor, Tensor, Tensor]


class DeltaModel(pl.LightningModule):
    def __init__(
        self,
        one_cycle_total_steps: int,
        le_uid: LabelEncoder,
        le_mid: LabelEncoder,
        dt_uid_w: DictionaryTransformer,
        dt_mid_w: DictionaryTransformer,
        qt_acc: QuantileTransformer,
        n_emb_mean: int = 2,
        n_emb_var: int = 1,
        n_delta_mean_neurons: int = 4,
        n_delta_var_neurons: int = 20,
        lr: float = 0.003,
        l1_loss_weight: float = 0.001,
        l2_loss_weight: float = 0,
    ):
        """Delta Model for Osu!Mania Score Prediction

        Args:
            le_uid: Label Encoder for User IDs
            le_mid: Label Encoder for Beatmap IDs
            qt_acc: Quantile Transformer for Accuracy
            n_emb_mean: Number of Embedding Dimensions
            lr: Learning Rate
            l1_loss_weight: L1 Loss Weight
            l2_loss_weight: L2 Loss Weight

        """
        super().__init__()
        self.total_steps = one_cycle_total_steps
        self.le_uid = le_uid
        self.le_mid = le_mid
        self.dt_uid_w = dt_uid_w
        self.dt_mid_w = dt_mid_w
        self.qt_acc = qt_acc
        self.lr = lr
        self.l1_loss_weight = l1_loss_weight
        self.l2_loss_weight = l2_loss_weight

        n_uid = len(le_uid.classes_)
        n_mid = len(le_mid.classes_)

        # Embeddings for User and Beatmap IDs
        self.emb_uid_mean = nn.Embedding(n_uid, n_emb_mean)
        self.emb_mid = nn.Embedding(n_mid, n_emb_mean)
        self.emb_uid_var = nn.Embedding(n_uid, n_emb_var)

        # Batch Normalization for Embedding Differences
        self.bn_mean = nn.BatchNorm1d(n_emb_mean, affine=False)
        self.bn_var = nn.BatchNorm1d(n_emb_var, affine=False)

        # Linear Layers for Embedding Differences to Accuracy Mean and Variance
        # The positive linear layer ensures the estimated function is
        # monotonic increasing
        self.delta_to_acc_mean = nn.Sequential(
            PositiveLinear(n_emb_mean, n_delta_mean_neurons),
            nn.Tanh(),
            PositiveLinear(n_delta_mean_neurons, 1),
        )
        self.delta_to_acc_var = nn.Sequential(
            PositiveLinear(n_emb_var, n_delta_var_neurons),
            nn.Softplus(),
            PositiveLinear(n_delta_var_neurons, n_delta_var_neurons),
            nn.Softplus(),
            PositiveLinear(n_delta_var_neurons, 1),
        )

    def forward(self, x_uid, x_mid):
        # Convert the One-Hot Encoded User ID and Beatmap ID to Embeddings
        x_uid_mean = self.emb_uid_mean(x_uid)
        x_uid_var = self.emb_uid_var(x_uid)

        x_mid = self.emb_mid(x_mid)

        # Calculate the difference between the User and Beatmap ID Embeddings
        x_delta = x_uid_mean - x_mid

        # Normalize the Embedding Differences
        x_delta = self.bn_mean(x_delta)
        x_uid_var = self.bn_var(x_uid_var)

        # Convert the Embedding Differences to Accuracy Mean and Variance
        y_mean = self.delta_to_acc_mean(x_delta).squeeze()
        y_var = softplus(self.delta_to_acc_var(x_uid_var)).squeeze()

        return y_mean, y_var

    @staticmethod
    def loss_nll_laplace(
        mean: Tensor,
        var: Tensor,
        target: Tensor,
        weight: Tensor,
        eps: float = 1e-10,
    ):
        """Negative Log Likelihood Loss for Laplace Distribution

        Args:
            mean: Mean of the Gaussian Distribution
            var: Variance of the Gaussian Distribution
            target: Target Value
            eps: Epsilon to prevent NaNs. Defaults to 1e-10.
        """
        scale = torch.sqrt(var + eps) / 2
        loss = (
            torch.log(2 * scale) + torch.abs(target - mean) / scale
        ) * weight
        return loss.mean()

    def loss_decoded_rmse(
        self, mean: Tensor, var: Tensor, target: Tensor, weight: Tensor
    ):
        return (
            nn.MSELoss()(
                tensor(self.decode_acc(mean.cpu().reshape(-1, 1))),
                tensor(self.decode_acc(target.cpu().reshape(-1, 1))),
            )
            ** 0.5
        )

    def step(self, batch: Batch, loss_fn: Any):
        x_uid, x_mid, y = batch
        y_pred_mean, y_pred_var = self(x_uid, x_mid)
        x_uid_w = tensor(self.dt_uid_w.transform(x_uid.tolist()))
        x_mid_w = tensor(self.dt_mid_w.transform(x_mid.tolist()))
        return loss_fn(y_pred_mean, y_pred_var, y, x_uid_w * x_mid_w)

    def training_step(self, batch: Batch, batch_idx: int):
        self.log(
            "train/nll_loss",
            (loss := self.step(batch, self.loss_nll_laplace)),
            prog_bar=True,
        )
        # # Add l1 regularization to the embeddings
        # l1 = (
        #     torch.exp(self.emb_uid_mean.weight).sum()
        #     + torch.exp(self.emb_mid_mean.weight).sum()
        # )
        # self.log("train/l1_loss", l1, prog_bar=True)
        # l2 = l1**2
        # self.log("train/l2_loss", l2)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        self.log(
            "val/rmse_loss",
            (loss := self.step(batch, self.loss_decoded_rmse)),
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Batch, batch_idx: int):
        self.log(
            "test/rmse_loss",
            (loss := self.step(batch, self.loss_decoded_rmse)),
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        one_cycle = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr * 10,
            pct_start=0.1,
            three_phase=True,
            total_steps=self.total_steps,
            anneal_strategy="linear",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": one_cycle,
                "interval": "step",
            },
        }

    @property
    def uid_classes(self) -> np.ndarray:
        return np.array(self.le_uid.classes_)

    @property
    def mid_classes(self) -> np.ndarray:
        return np.array(self.le_mid.classes_)

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
                "mean": y_mean_decode,
                "lower_bound": y_mean_lower_decode,
                "upper_bound": y_mean_upper_decode,
            },
            index=self.multi_index_mid(),
        )
        return df_mean

    def predict_map(self, mid: str | list[str]) -> pd.DataFrame:
        """Predicts the accuracy for all users for a given beatmap

        Args:
            mid: Map Name
            speed: Map Speed

        Returns:
            A list of dict:
                [
                    {"map
                ]
            The lower and upper bounds are one standard deviation away from
             the mean.

        """

        mid = [mid] if isinstance(mid, str) else mid
        mid = self.encode_mid(mid)
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
                "mean": y_mean_decode,
                "lower_bound": y_mean_lower_decode,
                "upper_bound": y_mean_upper_decode,
            },
            index=self.multi_index_uid(),
        )
        return df_mean

    def multi_index_uid(self):
        return pd.MultiIndex.from_frame(
            pd.Series(self.uid_classes)
            .str.split("/", expand=True)
            .rename(columns={0: "uid", 1: "year"})
            .astype(int)
        )

    def multi_index_mid(self):
        return pd.MultiIndex.from_frame(
            pd.Series(self.mid_classes)
            .str.split("/", expand=True)
            .rename(columns={0: "mid", 1: "speed"})
            .astype(int)
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
