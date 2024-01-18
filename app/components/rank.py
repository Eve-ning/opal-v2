import sys
from pathlib import Path

import numpy as np
import plotly.express as px

PROJECT_DIR = Path(__file__).parents[1]
sys.path.append(PROJECT_DIR.as_posix())
import pandas as pd
import streamlit as st

THIS_DIR = Path(__file__).parent


@st.cache_data()
def player_rank(df: pd.DataFrame, n_uid: int):
    left, right = st.columns(2)
    left.metric(
        "Player Median Rank",
        f"{np.median(df).astype(int)} / {n_uid}",
    )
    right.markdown(
        "The value on the left is a rough estimate on your rank "
        "based on all the maps in the dataset"
    )


@st.cache_data()
def player_rank_hist(df: pd.DataFrame, username: str, useryear: str):
    st.plotly_chart(
        px.histogram(x=df).update_layout(
            title=f"Rank Distribution of {username} in {useryear}",
            xaxis_title="Rank",
            yaxis_title="Count",
        )
    )


@st.cache_data()
def map_ranks(df: pd.DataFrame):
    st.markdown("Players with at least ...")
    r_b, r_a, r_s = st.columns(3)

    r_b.metric("B Rank", f"{(df >= 0.85).mean():.2%}")
    r_a.metric("A Rank", f"{(df >= 0.90).mean():.2%}")
    r_s.metric("S Rank", f"{(df >= 0.95).mean():.2%}")


@st.cache_data()
def map_rank_hist(df: pd.DataFrame, mapname: str, mapspeed_str: str):
    st.plotly_chart(
        px.histogram(x=df)
        .update_layout(
            title=f"Accuracy Distribution of {mapname} {mapspeed_str}",
            xaxis_title="Accuracy",
            yaxis_title="Count",
            xaxis_range=[None, 1],
        )
        .add_vline(0.95, line_color="red", line_width=2, opacity=0.65)
        .add_vline(0.90, line_color="yellow", line_width=2, opacity=0.65)
        .add_vline(0.85, line_color="green", line_width=2, opacity=0.65)
    )