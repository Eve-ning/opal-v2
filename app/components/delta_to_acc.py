from typing import TYPE_CHECKING

import plotly.graph_objects as go
import streamlit as st
import torch
from torch.nn.functional import softplus

if TYPE_CHECKING:
    from opal.model.delta_model import DeltaModel


def st_delta_to_acc(m: "DeltaModel", xlim: tuple[float, float] = (-7, 7)):
    x_delta = torch.linspace(*xlim, 100)
    dims = m.emb_uid_mean.weight.shape[1]
    y_means = []
    for d in range(dims):
        # Create a zeroed tensor
        # Then replace the dth element with x_delta
        # This simulates the effect of x_delta solely on the dth dimension
        x_delta_d = torch.zeros(len(x_delta), dims)
        x_delta_d[:, d] = x_delta
        y_mean = m.delta_to_acc_mean(x_delta_d).squeeze()
        y_means.append(y_mean)

    st.header("Delta to Accuracy Mapping")
    st.write(
        """
        The following plots shows the effect of the **Delta**, which is the 
        $E_{u}- E_{m}$, on **Accuracy**. We predict 2 metrics, the mean and
        variance of the accuracy, assuming that the accuracy is a Laplace
        distribution.
        
        The Laplace distribution is given by:
        $$
        f(x | \\mu, s) = \\frac{1}{2s} \\exp\\left(-\\frac{|x - \\mu|}{s}\\right)
        $$
        where $\\mu$ is the mean and $s$ is the scale parameter. The scale
        parameter is related to the variance by $\\sigma^2 = 2s^2$.
        """
    )
    st.plotly_chart(
        go.Figure(
            data=[
                go.Scatter(
                    x=x_delta,
                    y=m.qt_acc.inverse_transform(
                        y.detach().numpy().reshape(-1, 1)
                    ).squeeze(),
                    name=f"D{e}",
                )
                for e, y in enumerate(y_means)
            ],
            layout=go.Layout(
                dict(
                    xaxis_title="Delta",
                    yaxis_title="Accuracy Mean",
                    legend_title="Type",
                )
            ),
        ),
        use_container_width=True,
    )
