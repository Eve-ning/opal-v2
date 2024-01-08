import sys
from pathlib import Path

from components.delta_to_acc import delta_to_acc
from components.map_embeddings import map_embeddings
from components.player_embeddings import player_embeddings

PROJECT_DIR = Path(__file__).parents[1]
sys.path.append(PROJECT_DIR.as_posix())
import pandas as pd
import streamlit as st

from opal.model.delta_model import DeltaModel

THIS_DIR = Path(__file__).parent

st.title("Dan Analysis")


@st.cache_resource()
def load_model(path: str) -> DeltaModel:
    return (
        DeltaModel.load_from_checkpoint(PROJECT_DIR / Path(path).as_posix())
        .eval()
        .cpu()
    )


@st.cache_data()
def get_model_embeddings(
    _m: DeltaModel,
    path: str,
):
    w_uid_rc = _m.uid_rc_emb.weight.detach().numpy().squeeze()
    w_uid_ln = _m.uid_ln_emb.weight.detach().numpy().squeeze()
    w_mid_rc = _m.mid_rc_emb.weight.detach().numpy().squeeze()
    w_mid_ln = _m.mid_ln_emb.weight.detach().numpy().squeeze()
    ln_ratio = _m.ln_ratio_weights.detach().numpy().squeeze()
    df_mid = pd.DataFrame(
        [_m.mid_le.classes_, w_mid_rc, w_mid_ln, ln_ratio]
    ).T.rename(columns={0: "mid", 1: "RC", 2: "LN", 3: "ln_ratio"})

    df_uid = pd.DataFrame([_m.uid_le.classes_, w_uid_rc, w_uid_ln]).T.rename(
        columns={0: "uid", 1: "RC", 2: "LN"}
    )

    return df_mid, df_uid


model_path = st.selectbox(
    "Model Path",
    format_func=lambda x: x.parts[-3],
    options=list(PROJECT_DIR.glob("**/*.ckpt")),
)

m = load_model(model_path)
#
# df_uid = (
#     pd.Series(m.uid_le.classes_)
#     .str.split("/", expand=True)
#     .rename(columns={0: "name", 1: "year"})
# )
# st.dataframe(df_uid)
# left, right = st.columns(2)
# uid_name = left.selectbox("User", df_uid["name"].unique())
# uid_year = right.selectbox(
#     "Year", df_uid[df_uid["name"] == uid_name]["year"].unique()
# )
#
# w_uid_rc, w_uid_ln, w_mid_rc, w_mid_ln, ln_ratio = get_model_embeddings(
#     m, model_path
# )
#
# w_uid_rc.shape
# w_uid_ln.shape
# w_mid_rc.shape
# w_mid_ln.shape
# ln_ratio.shape

# %%

if True:
    df_mid, df_uid = get_model_embeddings(m, model_path)

    st.header("Map Embeddings")
    is_dans_only = st.checkbox("Dans Only", value=True)
    map_embeddings(df_mid, dans_only=is_dans_only)

    st.header("Player Embeddings")
    player_embeddings(df_uid)  # [df_uid["uid"].str.startswith(uid)])

    st.header("Delta to Accuracy Mapping")
    st.write(
        "This function maps the difference between player and map "
        "embeddings to accuracy."
    )
    delta_to_acc(m)
