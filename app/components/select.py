import streamlit as st
import pandas as pd


def select(
    df_uid_name: pd.DataFrame,
    df_uid_year: pd.DataFrame,
    df_mid_name: pd.DataFrame,
    df_mid_speed: pd.DataFrame,
):
    left, right = st.columns(2)
    username = left.selectbox("User", df_uid_name.unique())
    useryear = right.select_slider(
        "Year", df_uid_year[df_uid_name == username].unique()
    )

    left, right = st.columns(2)
    mapname = left.selectbox("Map", df_mid_name.unique())
    speed_options = df_mid_speed[df_mid_name == mapname].unique()
    mapspeed = right.radio("Speed", speed_options, horizontal=True)
    return username, useryear, mapname, mapspeed
