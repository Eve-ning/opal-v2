import plotly.graph_objects as go
import streamlit as st


def st_map_leaderboard(df_acc):
    acc_mean, acc_lower, acc_upper = (
        st.session_state["acc_mean"],
        st.session_state["acc_lower"],
        st.session_state["acc_upper"],
    )
    username = st.session_state["username"]
    year = st.session_state["year"]
    with st.expander("Map Leaderboard"):
        st.markdown(
            f"The red line shows how :orange[{username} @ {year}] would "
            f"perform, compared to :orange[other players on this map]. "
            "The yellow rectangle shows the :blue[75% confidence interval]. "
            f"This means that if :orange[{username} @ {year}] plays the map "
            f":blue[100] times, :blue[75] of them are likely to fall within "
            f"this range."
        )
        st.plotly_chart(
            go.Figure(
                [
                    go.Histogram(
                        name="Accuracy",
                        x=df_acc["mean"],
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
            .add_vline(x=acc_mean, line=dict(color="red", width=2))
            .add_vrect(
                acc_lower,
                acc_upper,
                fillcolor="yellow",
                opacity=0.2,
                line=dict(width=0),
            )
        )


def st_player_leaderboard(df_acc):
    acc_mean, acc_lower, acc_upper = (
        st.session_state["acc_mean"],
        st.session_state["acc_lower"],
        st.session_state["acc_upper"],
    )
    username = st.session_state["username"]
    year = st.session_state["year"]
    with st.expander("Player Leaderboard"):
        st.markdown(
            f"The red line shows how :orange[{username} @ {year}] would "
            "compared to their :orange[other plays on other maps]. "
            "The yellow rectangle shows the :blue[75% confidence interval]. "
            f"This means that if :orange[{username} @ {year}] plays the map "
            f":blue[100] times, :blue[75] of them are likely to fall within "
            f"this range."
        )
        st.plotly_chart(
            go.Figure(
                [
                    go.Histogram(
                        name="Accuracy",
                        x=df_acc["mean"],
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
            .add_vline(x=acc_mean, line=dict(color="red", width=2))
            .add_vrect(
                acc_lower,
                acc_upper,
                fillcolor="yellow",
                opacity=0.2,
                line=dict(width=0),
            )
        )


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
