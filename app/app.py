import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parents[1]
sys.path.append(PROJECT_DIR.as_posix())

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from model.model import Model

THIS_DIR = Path(__file__).parent
# %%

st.title("Dan Analysis")


@st.cache_resource()
def load_model(path: str) -> Model:
    return Model.load_from_checkpoint(PROJECT_DIR / Path(path).as_posix()).eval().cpu()


@st.cache_data()
def get_model_embeddings(
    _m: Model,
    path: str,
    n_mid: int = 1000,
    n_uid: int = 100,
):
    ln_ratio = pd.read_csv("ln_ratio_all.csv", index_col=0)
    w_uid_rc = _m.uid_emb_rc.weight.detach().numpy().squeeze()
    w_uid_ln = _m.uid_emb_ln.weight.detach().numpy().squeeze()
    w_mid_rc = _m.mid_emb_rc.weight.detach().numpy().squeeze()
    w_mid_ln = _m.mid_emb_ln.weight.detach().numpy().squeeze()
    df_mid = (
        (
            pd.DataFrame([_m.mid_le.classes_, w_mid_rc, w_mid_ln, ln_ratio]).T.rename(
                columns={0: "mid", 1: "RC", 2: "LN", 3: "ln_ratio"}
            )
        )
        .merge(ln_ratio, left_on="mid", right_index=True)
        .drop("ln_ratio_x", axis=1)
        .rename(columns={"ln_ratio_y": "ln_ratio"})
        .assign(mid=lambda x: x.mid.str[:-2])
    )

    df_uid = pd.DataFrame([_m.uid_le.classes_, w_uid_rc, w_uid_ln]).T.rename(
        columns={0: "uid", 1: "RC", 2: "LN"}
    )

    return (
        (df_mid.sample(n_mid, random_state=0) if n_mid else df_mid),
        (df_uid.sample(n_uid, random_state=0) if n_uid else df_uid),
    )


model_path = st.selectbox(
    "Model Path",
    format_func=lambda x: x.parts[-3],
    options=list(PROJECT_DIR.glob("**/*.ckpt")),
)


m = load_model(model_path)

sample_mid = st.checkbox("Sample Maps", value=True)
n_mid = (
    st.slider(
        "No. of Maps",
        min_value=10,
        max_value=(max_n_mid := len(m.mid_le.classes_)),
        step=10,
        disabled=not sample_mid,
    )
    if sample_mid
    else None
)
sample_uid = st.checkbox("Sample Players", value=True)
n_uid = (
    st.slider(
        "No. of Players",
        min_value=100,
        max_value=(max_n_uid := len(m.uid_le.classes_)),
        step=100,
        disabled=not sample_uid,
    )
    if sample_uid
    else None
)

df_mid, df_uid = get_model_embeddings(
    m,
    model_path,
    n_mid=n_mid,
    n_uid=n_uid,
)

st.header("Map Embeddings")
st.write(
    "These embeddings represent the difficulty of maps. "
    "The larger the values (top right), the harder the map. "
    "Each dimension (x, y) represents a different difficulty element, "
    "in this case, we have RC and LN."
)
m_scaled_tab, m_unscaled_tab = st.tabs(["Scaled", "Unscaled"])

with m_scaled_tab:
    st.info(
        "**Embeddings are scaled** by the ratio of LN/RC notes. "
        "Therefore, if a map is LN-hard, but has only a few LNs, "
        "the LN embedding will be small."
    )
    st.plotly_chart(
        px.scatter(
            df_mid,
            x=df_mid["RC"] * (1 - df_mid["ln_ratio"]),
            y=df_mid["LN"] * df_mid["ln_ratio"],
            # text=df_mid["mid"].str[:25] + "...",
            hover_name="mid",
            labels={"x": "RC", "y": "LN"},
            # size=np.ones_like(df_mid["RC"], dtype=int),
        ),
        use_container_width=True,
    )

with m_unscaled_tab:
    st.info(
        "**Embeddings are NOT scaled** by the ratio of LN/RC notes. "
        "Therefore, if a map is LN-hard, but has only a few LNs, "
        "the LN embedding will remain large. "
        "Which isn't representative of the map."
    )
    st.plotly_chart(
        px.scatter(
            df_mid,
            x="RC",
            y="LN",
            # text=df_mid["mid"],
            labels={"x": "RC", "y": "LN"},
            hover_name="mid",
            # size=np.ones_like(df_mid["RC"], dtype=int),
            # color=df_mid["mid"].str.startswith("RC").replace({True: "RC", False: "LN"}),
            # color_discrete_map={"RC": "#0099FF", "LN": "#99AA33"},
        ),
        use_container_width=True,
    )

st.header("Player Embeddings")
st.plotly_chart(
    px.scatter(
        df_uid,
        x=df_uid["RC"],
        y=df_uid["LN"],
        # text="uid",
        hover_name="uid",
    ),
    use_container_width=True,
)

# %%
import torch
import plotly.graph_objects as go

x_delta = torch.linspace(-7, 7, 100)
y_rc = m.delta_rc_to_acc(x_delta.reshape(-1, 1)).squeeze().detach().numpy()
y_ln = m.delta_ln_to_acc(x_delta.reshape(-1, 1)).squeeze().detach().numpy()

st.header("Delta to Accuracy Mapping")
st.write(
    "This function maps the difference between player and map "
    "embeddings to accuracy."
)
st.plotly_chart(
    go.Figure(
        data=[
            go.Scatter(x=x_delta, y=y_rc, name="RC"),
            go.Scatter(x=x_delta, y=y_ln, name="LN"),
        ],
        layout=go.Layout(
            dict(
                xaxis_title="Delta",
                yaxis_title="Accuracy",
                legend_title="Type",
            )
        ),
    )
)
