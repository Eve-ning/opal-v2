from typing import TYPE_CHECKING

import torch


import plotly.graph_objects as go
import streamlit as st

if TYPE_CHECKING:
    from opal.model.delta_model import DeltaModel


def delta_to_acc(m: "DeltaModel"):
    x_delta = torch.linspace(-7, 7, 100)
    y_rc = m.delta_rc_to_acc(x_delta.reshape(-1, 1)).squeeze().detach().numpy()
    y_ln = m.delta_ln_to_acc(x_delta.reshape(-1, 1)).squeeze().detach().numpy()

    st.plotly_chart(
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
                )
            ),
        )
    )
