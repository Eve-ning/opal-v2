import plotly.graph_objects as go
import streamlit as st


def st_leaderboard(map_pred, user_pred, mean, lower_bound, upper_bound):
    with st.expander("Map Leaderboard"):
        st.markdown(
            "The red line shows how the player would perform, "
            "compared to other players on this map. "
            "The yellow rectangle shows the 75% confidence interval lower and "
            "upper bound. This means that if the player plays the map 100 "
            "times, 75 of them are likely to fall within this range."
        )
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
        st.markdown(
            "The red line shows how the player would perform, "
            "compared to their own plays on other maps. "
            "The yellow rectangle shows the 75% confidence interval lower and "
            "upper bound. This means that if the player plays the map 100 "
            "times, 75 of them are likely to fall within this range."
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
