from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch import nn

from opal.model.positive_linear import PositiveLinear


class DeltaModel(pl.LightningModule):
    def __init__(
        self,
        uid_le: LabelEncoder,
        mid_le: LabelEncoder,
        acc_qt: QuantileTransformer,
        ln_ratio_weights: list,
        rc_emb: int = 1,
        ln_emb: int = 1,
    ):
        super().__init__()
        self.uid_le = uid_le
        self.mid_le = mid_le
        self.acc_qt = acc_qt
        self.ln_ratio_weights = nn.Parameter(
            torch.tensor(ln_ratio_weights, dtype=torch.float),
            requires_grad=False,
        )
        n_uid = len(uid_le.classes_)
        n_mid = len(mid_le.classes_)

        self.uid_rc_emb_mean = nn.Embedding(n_uid, rc_emb)
        self.uid_rc_emb_var = nn.Embedding(n_uid, rc_emb)
        self.mid_rc_emb_mean = nn.Embedding(n_mid, rc_emb)
        self.mid_rc_emb_var = nn.Embedding(n_mid, rc_emb)
        self.uid_ln_emb_mean = nn.Embedding(n_uid, ln_emb)
        self.uid_ln_emb_var = nn.Embedding(n_uid, ln_emb)
        self.mid_ln_emb_mean = nn.Embedding(n_mid, ln_emb)
        self.mid_ln_emb_var = nn.Embedding(n_mid, ln_emb)

        self.rc_emb_bn = nn.BatchNorm1d(rc_emb, affine=False)
        self.ln_emb_bn = nn.BatchNorm1d(ln_emb, affine=False)

        self.delta_rc_to_acc_mean = nn.Sequential(
            PositiveLinear(rc_emb, 4),
            nn.ReLU(),
            PositiveLinear(4, 1),
            nn.Hardsigmoid(),
        )
        self.delta_ln_to_acc_mean = nn.Sequential(
            PositiveLinear(ln_emb, 4),
            nn.ReLU(),
            PositiveLinear(4, 1),
            nn.Hardsigmoid(),
        )
        self.var_rc_to_acc_var = nn.Sequential(
            nn.Linear(rc_emb * 2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.ReLU(),
        )
        self.var_ln_to_acc_var = nn.Sequential(
            nn.Linear(ln_emb * 2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.ReLU(),
        )
        self.save_hyperparameters()

    def forward(self, uid, mid, freeze_random=False):
        w_ln_ratio = self.ln_ratio_weights[mid]

        uid_rc_emb_mean = self.uid_rc_emb_mean(uid)
        uid_rc_emb_var = self.uid_rc_emb_var(uid)
        mid_rc_emb_mean = self.mid_rc_emb_mean(mid)
        mid_rc_emb_var = self.mid_rc_emb_var(mid)
        uid_ln_emb_mean = self.uid_ln_emb_mean(uid)
        uid_ln_emb_var = self.uid_ln_emb_var(uid)
        mid_ln_emb_mean = self.mid_ln_emb_mean(mid)
        mid_ln_emb_var = self.mid_ln_emb_var(mid)

        # Perform Variational Inference
        if freeze_random:
            uid_rc_emb = uid_rc_emb_mean
            mid_rc_emb = mid_rc_emb_mean
            uid_ln_emb = uid_ln_emb_mean
            mid_ln_emb = mid_ln_emb_mean
        else:
            uid_rc_emb = uid_rc_emb_mean + torch.randn_like(
                uid_rc_emb_var
            ) * torch.exp(uid_rc_emb_var / 2)
            mid_rc_emb = mid_rc_emb_mean + torch.randn_like(
                mid_rc_emb_var
            ) * torch.exp(mid_rc_emb_var / 2)
            uid_ln_emb = uid_ln_emb_mean + torch.randn_like(
                uid_ln_emb_var
            ) * torch.exp(uid_ln_emb_var / 2)
            mid_ln_emb = mid_ln_emb_mean + torch.randn_like(
                mid_ln_emb_var
            ) * torch.exp(mid_ln_emb_var / 2)

        rc_emb_delta = uid_rc_emb - mid_rc_emb
        ln_emb_delta = uid_ln_emb - mid_ln_emb

        rc_emb_delta = self.rc_emb_bn(rc_emb_delta)
        ln_emb_delta = self.ln_emb_bn(ln_emb_delta)

        rc_acc_mean = self.delta_rc_to_acc_mean(rc_emb_delta)
        ln_acc_mean = self.delta_ln_to_acc_mean(ln_emb_delta)

        rc_acc_mean = rc_acc_mean[:, 0]
        ln_acc_mean = ln_acc_mean[:, 0]

        rc_acc_var = self.var_rc_to_acc_var(
            torch.cat((uid_rc_emb_var, mid_rc_emb_var), dim=1)
        )
        ln_acc_var = self.var_ln_to_acc_var(
            torch.cat((uid_ln_emb_var, mid_ln_emb_var), dim=1)
        )

        rc_acc_var = rc_acc_var[:, 0]
        ln_acc_var = ln_acc_var[:, 0]

        y_mean = rc_acc_mean * (1 - w_ln_ratio) + ln_acc_mean * w_ln_ratio
        y_var = rc_acc_var * (1 - w_ln_ratio) + ln_acc_var * w_ln_ratio
        return y_mean, y_var

    @staticmethod
    def loss_nll(y_pred_mean, y_pred_var, y, eps=1e-3):
        return (torch.log(y_pred_var + eps) / 2) + (
            (y - y_pred_mean) ** 2 / (2 * y_pred_var + eps)
        )

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x_uid, x_mid, y = batch
        y_pred_mean, y_pred_var = self(x_uid, x_mid)
        loss = self.loss_nll(y_pred_mean, y_pred_var, y).mean()
        # loss = nn.MSELoss()(y_pred_mean, y)
        self.log("train/nll_loss", loss, prog_bar=True)
        # self.log("train/mse_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x_uid, x_mid, y = batch
        y_pred, _ = self(x_uid, x_mid, freeze_random=True)
        # y_pred = self(x_uid, x_mid, freeze_random=True)
        loss = nn.MSELoss()(
            torch.tensor(self.decode_acc(y_pred)),
            torch.tensor(self.decode_acc(y)),
        )
        self.log("val/rmse_loss", loss**0.5, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x_uid, x_mid, y = batch
        y_pred, _ = self(x_uid, x_mid, freeze_random=True)
        # y_pred = self(x_uid, x_mid, freeze_random=True)
        loss = nn.MSELoss()(
            torch.tensor(self.decode_acc(y_pred)),
            torch.tensor(self.decode_acc(y)),
        )
        self.log("test/rmse_loss", loss**0.5, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.003)
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
        return self.acc_qt.inverse_transform(x.cpu().reshape(-1, 1))

    def encode_acc(self, x):
        return self.acc_qt.transform(x.cpu().reshape(-1, 1)).squeeze()

    def decode_uid(self, x):
        return self.uid_le.inverse_transform(x)

    def decode_mid(self, x):
        return self.mid_le.inverse_transform(x)

    def encode_uid(self, x):
        return self.uid_le.transform(x)

    def encode_mid(self, x):
        return self.mid_le.transform(x)

    @property
    def uid_classes(self) -> np.ndarray:
        return np.array(self.uid_le.classes_)

    @property
    def mid_classes(self) -> np.ndarray:
        return np.array(self.mid_le.classes_)

    def predict_all(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        n_uid = len(self.uid_classes)
        n_mid = len(self.mid_classes)
        a = torch.cartesian_prod(torch.arange(n_uid), torch.arange(n_mid))
        uids = a[:, 0]
        mids = a[:, 1]
        accs = self.acc_qt.inverse_transform(
            self(uids, mids)[0].detach().numpy().reshape(-1, 1)
        ).reshape(n_uid, n_mid)
        accs_var = self.acc_qt.inverse_transform(
            self(uids, mids)[1].detach().numpy().reshape(-1, 1)
        ).reshape(n_uid, n_mid)

        return pd.DataFrame(
            accs, columns=self.mid_classes, index=self.uid_classes
        ), pd.DataFrame(
            accs_var, columns=self.mid_classes, index=self.uid_classes
        )
