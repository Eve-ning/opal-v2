import sys

from pathlib import Path

PROJECT_DIR = Path(__file__).parents[1]
sys.path.append(PROJECT_DIR.as_posix())

import pandas as pd
import streamlit as st

from opal.model.delta_model import DeltaModel


@st.cache_data()
def predict_all(
    _m: DeltaModel,
    model_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _m.predict_all()


def mapspeed_to_str(mapspeed):
    return {"-1": "HT", "0": "NM", "1": "DT", -1: "HT", 0: "NM", 1: "DT"}.get(
        mapspeed
    )


def mapspeed_to_int(mapspeed):
    return {"HT": -1, "NM": 0, "DT": 1}.get(mapspeed)
