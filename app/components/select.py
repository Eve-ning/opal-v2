from pathlib import Path

import streamlit as st
import pandas as pd
import sys

PROJECT_DIR = Path(__file__).parents[1]
sys.path.append(PROJECT_DIR.as_posix())
from opal.model.delta_model import DeltaModel


def st_select_user(
    name_opts: pd.DataFrame,
    year_opts: pd.DataFrame,
):
    left, right = st.columns(2)
    username = left.selectbox("User", name_opts.unique())
    useryear = right.select_slider(
        "Year", year_opts[name_opts == username].unique()
    )
    return username, useryear


def st_select_map(name_opts: pd.Series, speed_opts: pd.Series):
    mapname = st.selectbox("Map", name_opts.unique())
    speed_options = speed_opts[name_opts == mapname].unique()
    mapspeed = st.radio(
        "Speed",
        speed_options,
        format_func={"-1": "HT", "0": "NM", "1": "DT"}.get,
        horizontal=True,
    )
    return mapname, mapspeed


@st.cache_resource()
def load_model(path: Path) -> DeltaModel:
    return DeltaModel.load_from_checkpoint(Path(path).as_posix()).eval().cpu()


def st_select_model(model_search_pth: Path) -> tuple[DeltaModel, str]:
    model_path = st.selectbox(
        "Model Path",
        # format_func=lambda x: x.parts[-3],
        options=list(p for p in model_search_pth.glob("**/*.ckpt")),
        placeholder="Select a model",
    )
    return load_model(model_search_pth / model_path), model_path
