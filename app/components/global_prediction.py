import plotly.graph_objects as go
import streamlit as st


def global_prediction(map_pred, user_pred, mean, lower_bound, upper_bound):
    st.markdown(
        "## Global Statistics",
        help="The following statistics are based on global predictions of the "
        "selected user and map.",
    )
    st.plotly_chart(
        # Add a y vertical line at the mean
        go.Figure(
            [
                go.Histogram(
                    name="Accuracy",
                    x=map_pred["mean"],
                ),
            ]
        )
        .update_layout(
            title=f"Map Global Accuracy Distribution",
            xaxis_title="Accuracy",
            yaxis_title="Count",
            xaxis=dict(range=[0.85, 1]),
        )
        .add_vline(x=mean, line=dict(color="red", width=2))
        .add_vrect(
            lower_bound,
            upper_bound,
            fillcolor="yellow",
            opacity=0.2,
            line=dict(width=0),
        )
    )
    st.plotly_chart(
        go.Figure(
            [
                go.Histogram(
                    name="Accuracy",
                    x=user_pred["mean"],
                    histnorm="probability",
                ),
            ]
        )
        .update_layout(
            title=f"Player Global Accuracy Distribution",
            xaxis_title="Accuracy",
            yaxis_title="Count",
            xaxis=dict(range=[0.85, 1]),
            # format y axis as percent
            yaxis=dict(tickformat=".0%"),
        )
        .add_vline(x=mean, line=dict(color="red", width=2))
        .add_vrect(
            lower_bound,
            upper_bound,
            fillcolor="yellow",
            opacity=0.2,
            line=dict(width=0),
        )
    )
