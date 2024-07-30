import random
from pathlib import Path

import streamlit as st
import pandas as pd
import sys

PROJECT_DIR = Path(__file__).parents[1]
sys.path.append(PROJECT_DIR.as_posix())
from opal.model.delta_model import DeltaModel


def st_select_user(username_opts: pd.Series, year_opts: pd.Series):
    username = st.selectbox("User", username_opts.unique())
    years = year_opts[username_opts == username].unique()
    if len(years) == 1:
        st.metric("Year", (year := years[0]))
    else:
        year = st.select_slider("Year", years)
    return username, int(year)


def st_select_map(mapname_opts: pd.Series, speed_opts: pd.Series):
    mapname = st.selectbox("Map", mapname_opts.unique())
    speed_options = speed_opts[mapname_opts == mapname].unique()
    speed = st.radio(
        "Speed",
        speed_options,
        format_func={-1: "Half Time", 0: "Normal Time", 1: "Double Time"}.get,
        horizontal=True,
    )
    return mapname, int(speed)


@st.cache_resource()
def load_model(path: Path) -> DeltaModel:
    return DeltaModel.load_from_checkpoint(Path(path).as_posix()).eval().cpu()


def st_select_model(model_search_pth: Path) -> tuple[DeltaModel, str]:
    model_path = st.selectbox(
        "Model Path",
        options=list(p for p in model_search_pth.glob("**/*.ckpt")),
        placeholder="Select a model",
    )
    return load_model(model_search_pth / model_path), Path(model_path).name
