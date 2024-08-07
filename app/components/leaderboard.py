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
    mapname = st.session_state["mapname"]
    speed_str = st.session_state["speed_str"]
    with st.expander("Map Leaderboard"):
        st.markdown(
            f"The red line shows how :orange[{username} @ {year}] would "
            f"perform, compared to :orange[other players on this map]. "
            "The yellow rectangle shows the :blue[75% confidence interval]. "
        )
        st.caption(
            "Confidence Interval?",
            help=f"This means that if :orange[{username} @ {year}] plays "
            f":orange[{mapname} @ {speed_str}] "
            f":blue[100] times, :blue[75] of them are likely to fall within "
            f"this range.",
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
    mapname = st.session_state["mapname"]
    speed_str = st.session_state["speed_str"]
    with st.expander("Player Leaderboard"):
        st.markdown(
            f"The red line shows how :orange[{username} @ {year}] would "
            "compared to their :orange[other plays on other maps]. "
            "The yellow rectangle shows the :blue[75% confidence interval]. "
        )
        st.caption(
            "Confidence Interval?",
            help=f"This means that if :orange[{username} @ {year}] plays "
            f":orange[{mapname} @ {speed_str}] "
            f":blue[100] times, :blue[75] of them are likely to fall within "
            f"this range.",
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
