import streamlit as st
import pandas as pd


def select_user(
    df_uid_name: pd.DataFrame,
    df_uid_year: pd.DataFrame,
):
    left, right = st.columns(2)
    username = left.selectbox("User", df_uid_name.unique())
    useryear = right.select_slider(
        "Year", df_uid_year[df_uid_name == username].unique()
    )
    return username, useryear


def select_map(
    df_mid_name: pd.DataFrame,
    df_mid_speed: pd.DataFrame,
):
    mapname = st.selectbox("Map", df_mid_name.unique())
    speed_options = df_mid_speed[df_mid_name == mapname].unique()
    mapspeed = st.radio(
        "Speed",
        speed_options,
        format_func={"-1": "HT", "0": "NM", "1": "DT"}.get,
        horizontal=True,
    )
    return mapname, mapspeed
