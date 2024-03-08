from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch import nn
from torch.nn.functional import softplus, hardsigmoid

from opal.model.positive_linear import PositiveLinear


class DeltaModel(pl.LightningModule):
    def __init__(
        self,
        le_uid: LabelEncoder,
        le_mid: LabelEncoder,
        qt_acc: QuantileTransformer,
        w_ln_ratio: list,
        n_rc_emb: int = 1,
        n_ln_emb: int = 1,
        n_delta_neurons: int = 4,
        lr: float = 0.003,
    ):
        """Delta Model for Osu!Mania Score Prediction

        Args:
            le_uid: Label Encoder for User IDs
            le_mid: Label Encoder for Beatmap IDs
            qt_acc: Quantile Transformer for Accuracy
            w_ln_ratio: Weights for LN Ratio
            n_rc_emb: Number of Embeddings for Relative Coordinates
            n_ln_emb: Number of Embeddings for Local Normals
        """
        super().__init__()
        self.le_uid = le_uid
        self.le_mid = le_mid
        self.qt_acc = qt_acc
        self.w_ln_ratio = nn.Parameter(
            torch.tensor(w_ln_ratio, dtype=torch.float),
            requires_grad=False,
        )
        self.lr = lr

        n_uid = len(le_uid.classes_)
        n_mid = len(le_mid.classes_)

        # Embeddings for User and Beatmap IDs
        self.emb_uid_rc = nn.Embedding(n_uid, n_rc_emb)
        self.emb_mid_rc = nn.Embedding(n_mid, n_rc_emb)
        self.emb_uid_ln = nn.Embedding(n_uid, n_ln_emb)
        self.emb_mid_ln = nn.Embedding(n_mid, n_ln_emb)

        # Batch Normalization for Embedding Differences
        self.bn_rc = nn.BatchNorm1d(n_rc_emb, affine=False)
        self.bn_ln = nn.BatchNorm1d(n_ln_emb, affine=False)

        # Linear Layers for Embedding Differences to Accuracy Mean and Variance
        # The positive linear layer ensures the estimated function is
        # monotonic increasing
        self.delta_rc_to_acc = nn.Sequential(
            PositiveLinear(n_rc_emb, n_delta_neurons),
            nn.Tanh(),
            PositiveLinear(n_delta_neurons, 2),
        )
        self.delta_ln_to_acc = nn.Sequential(
            PositiveLinear(n_ln_emb, n_delta_neurons),
            nn.Tanh(),
            PositiveLinear(n_delta_neurons, 2),
        )

        self.save_hyperparameters()

    def forward(self, x_uid, x_mid):
        # Get the LN Ratio for the current beatmap
        w_ln_ratio = self.w_ln_ratio[x_mid]

        # Convert the One-Hot Encoded User ID and Beatmap ID to Embeddings
        x_uid_rc = self.emb_uid_rc(x_uid)
        x_mid_rc = self.emb_mid_rc(x_mid)
        x_uid_ln = self.emb_uid_ln(x_uid)
        x_mid_ln = self.emb_mid_ln(x_mid)

        # Calculate the difference between the User and Beatmap ID Embeddings
        x_rc_delta = x_uid_rc - x_mid_rc
        x_ln_delta = x_uid_ln - x_mid_ln

        # Normalize the Embedding Differences
        x_rc_delta = self.bn_rc(x_rc_delta)
        x_ln_delta = self.bn_ln(x_ln_delta)

        # Convert the Embedding Differences to Accuracy Mean and Variance
        y_rc = self.delta_rc_to_acc(x_rc_delta)
        y_ln = self.delta_ln_to_acc(x_ln_delta)

        # Squash the predictions to the range [0, 1]
        y_rc_mean = hardsigmoid(y_rc[:, 0])
        y_ln_mean = hardsigmoid(y_ln[:, 0])

        # Make sure the variance is positive
        y_rc_var = softplus(y_rc[:, 1])
        y_ln_var = softplus(y_ln[:, 1])

        # Combine the predictions using the LN Ratio
        y_mean = y_rc_mean * (1 - w_ln_ratio) + y_ln_mean * w_ln_ratio
        y_var = y_rc_var * (1 - w_ln_ratio) + y_ln_var * w_ln_ratio

        return y_mean, y_var

    @staticmethod
    def loss_nll(
        mean: torch.Tensor,
        var: torch.Tensor,
        target: torch.Tensor,
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

    def decoded_rmse(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,  # noqa
        target: torch.Tensor,
    ):
        return (
            nn.MSELoss()(
                torch.tensor(self.decode_acc(mean)),
                torch.tensor(self.decode_acc(target)),
            )
            ** 0.5
        )

    def step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        loss_fn: Any,
    ):
        x_uid, x_mid, y = batch
        y_pred_mean, y_pred_var = self(x_uid, x_mid)
        return loss_fn(y_pred_mean, y_pred_var, y)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        self.log(
            "train/nll_loss",
            (loss := self.step(batch, self.loss_nll)),
            prog_bar=True,
        )
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        self.log(
            "val/rmse_loss",
            (loss := self.step(batch, self.decoded_rmse)),
            prog_bar=True,
        )
        return loss

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
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
            patience=2,
            factor=0.5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/rmse_loss",
        }

    def decode_acc(self, x):
        return self.qt_acc.inverse_transform(x.cpu().reshape(-1, 1))

    def encode_acc(self, x):
        return self.qt_acc.transform(x.cpu().reshape(-1, 1)).squeeze()

    def decode_uid(self, x):
        return self.le_uid.inverse_transform(x)

    def decode_mid(self, x):
        return self.le_mid.inverse_transform(x)

    def encode_uid(self, x):
        return self.le_uid.transform(x)

    def encode_mid(self, x):
        return self.le_mid.transform(x)

    @property
    def uid_classes(self) -> np.ndarray:
        return np.array(self.le_uid.classes_)

    @property
    def mid_classes(self) -> np.ndarray:
        return np.array(self.le_mid.classes_)

    def predict_all(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Predicts the accuracy for all user-beatmap pairs

        Returns:
            DataFrames of the mean and variance of the accuracy respectively.
        """
        n_uid = len(self.uid_classes)
        n_mid = len(self.mid_classes)

        # Construct the Cartesian Product of User and Beatmap IDs
        cart = torch.cartesian_prod(torch.arange(n_uid), torch.arange(n_mid))
        x_uid = cart[:, 0]
        x_mid = cart[:, 1]

        y_mean = self.qt_acc.inverse_transform(
            self(x_uid, x_mid)[0].detach().numpy().reshape(-1, 1)
        ).reshape(n_uid, n_mid)
        y_var = self.qt_acc.inverse_transform(
            self(x_uid, x_mid)[1].detach().numpy().reshape(-1, 1)
        ).reshape(n_uid, n_mid)

        return pd.DataFrame(
            y_mean, columns=self.mid_classes, index=self.uid_classes
        ), pd.DataFrame(
            y_var, columns=self.mid_classes, index=self.uid_classes
        )
