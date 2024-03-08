import sys

from pathlib import Path

PROJECT_DIR = Path(__file__).parents[1]
sys.path.append(PROJECT_DIR.as_posix())

import pandas as pd
import streamlit as st

from opal.model.delta_model import DeltaModel


@st.cache_data()
def model_emb(
    _m: DeltaModel,
    model_id: str,
):
    w_uid_rc = _m.emb_uid_rc.weight.detach().numpy().squeeze()
    w_uid_ln = _m.emb_uid_ln.weight.detach().numpy().squeeze()
    w_mid_rc = _m.emb_mid_rc.weight.detach().numpy().squeeze()
    w_mid_ln = _m.emb_mid_ln.weight.detach().numpy().squeeze()
    ln_ratio = _m.w_ln_ratio.detach().numpy().squeeze()
    df_mid = pd.DataFrame(
        [_m.le_mid.classes_, w_mid_rc, w_mid_ln, ln_ratio]
    ).T.rename(columns={0: "mid", 1: "RC", 2: "LN", 3: "ln_ratio"})
    df_mid[["name", "speed"]] = df_mid["mid"].str.split("/", expand=True)
    df_uid = pd.DataFrame([_m.le_uid.classes_, w_uid_rc, w_uid_ln]).T.rename(
        columns={0: "uid", 1: "RC", 2: "LN"}
    )
    df_uid[["name", "year"]] = df_uid["uid"].str.split("/", expand=True)
    return (
        df_mid.drop(columns=["mid"]),
        df_uid.drop(columns=["uid"]),
    )


@st.cache_data()
def predict_all(
    _m: DeltaModel,
    model_id: str,
) -> pd.DataFrame:
    accs = _m.predict_all()
    return pd.DataFrame(accs, columns=_m.mid_classes, index=_m.uid_classes)


def mapspeed_to_str(mapspeed):
    return {"-1": "HT", "0": "NM", "1": "DT", -1: "HT", 0: "NM", 1: "DT"}.get(
        mapspeed
    )


def mapspeed_to_int(mapspeed):
    return {"HT": -1, "NM": 0, "DT": 1}.get(mapspeed)
