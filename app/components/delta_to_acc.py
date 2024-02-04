from typing import TYPE_CHECKING

import torch


import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import interp1d

if TYPE_CHECKING:
    from opal.model.delta_model import DeltaModel


def st_delta_to_acc(m: "DeltaModel"):
    x_delta = torch.linspace(-4, 4, 100)
    rc_dim = m.uid_rc_emb.weight.shape[1]
    ln_dim = m.uid_ln_emb.weight.shape[1]
    y_rc = (
        m.delta_rc_to_acc(x_delta.repeat((rc_dim, 1)).T)
        .squeeze()
        .detach()
        .numpy()
    )
    y_ln = (
        m.delta_ln_to_acc(x_delta.repeat((ln_dim, 1)).T)
        .squeeze()
        .detach()
        .numpy()
    )

    y_rc_inverse_func = interp1d(
        y_rc,
        x_delta,
        fill_value="extrapolate",
    )
    y_ln_inverse_func = interp1d(
        y_ln,
        x_delta,
        fill_value="extrapolate",
    )

    x_inv = torch.linspace(0, 1, 100)
    y_rc_inv = y_rc_inverse_func(x_inv)
    y_ln_inv = y_ln_inverse_func(x_inv)

    left, right = st.columns(2)
    left.plotly_chart(
        go.Figure(
            data=[
                go.Scatter(x=x_delta, y=y_rc, name="RC"),
                go.Scatter(x=x_delta, y=y_ln, name="LN"),
            ],
            layout=go.Layout(
                dict(
                    xaxis_title="Delta",
                    yaxis_title="Accuracy",
                    legend_title="Type",
                    # limits
                    yaxis_range=[0, 1],
                )
            ),
        ),
        use_container_width=True,
    )

    right.plotly_chart(
        go.Figure(
            data=[
                go.Scatter(x=x_inv, y=y_rc_inv, name="RC"),
                go.Scatter(x=x_inv, y=y_ln_inv, name="LN"),
            ],
            layout=go.Layout(
                dict(
                    xaxis_title="Accuracy",
                    yaxis_title="Delta",
                    legend_title="Type",
                )
            ),
        ),
        use_container_width=True,
    )
