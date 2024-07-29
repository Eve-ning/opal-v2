import plotly.graph_objects as go
import streamlit as st


def st_leaderboard(map_pred, user_pred, mean, lower_bound, upper_bound):
    with st.expander("Map Leaderboard"):
        st.markdown("This graph shows the accuracy distribution across")
        st.plotly_chart(
            go.Figure(
                [
                    go.Histogram(
                        name="Accuracy",
                        x=map_pred["mean"],
                        histnorm="probability",
                    ),
                ]
            )
            .update_layout(
                title="Map Accuracy Distribution",
                xaxis_title="Accuracy",
                yaxis_title="Frequency",
                xaxis=dict(range=[0.85, 1], tickformat=".0%"),
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
    with st.expander("Player Leaderboard"):
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
                title="Player Accuracy Distribution",
                xaxis_title="Accuracy",
                yaxis_title="Frequency",
                xaxis=dict(range=[0.85, 1], tickformat=".0%"),
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
