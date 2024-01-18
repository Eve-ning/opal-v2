import sys
from pathlib import Path

import numpy as np
import plotly.express as px

from app.utils import mapspeed_to_str
from components.delta_to_acc import delta_to_acc
from components.rank import (
    player_rank,
    player_rank_hist,
    map_rank_hist,
    map_ranks,
)
from components.select import select_user, select_map, select_model

PROJECT_DIR = Path(__file__).parents[1]
sys.path.append(PROJECT_DIR.as_posix())
import pandas as pd
import streamlit as st

from opal.data import OsuDataModule, df_k
from opal.model.delta_model import DeltaModel

THIS_DIR = Path(__file__).parent

st.title("Dan Analysis")


@st.cache_data()
def get_model_embeddings(
    _m: DeltaModel,
    model_id: str,
):
    w_uid_rc = _m.uid_rc_emb.weight.detach().numpy().squeeze()
    w_uid_ln = _m.uid_ln_emb.weight.detach().numpy().squeeze()
    w_mid_rc = _m.mid_rc_emb.weight.detach().numpy().squeeze()
    w_mid_ln = _m.mid_ln_emb.weight.detach().numpy().squeeze()
    ln_ratio = _m.ln_ratio_weights.detach().numpy().squeeze()
    df_mid = pd.DataFrame(
        [_m.mid_le.classes_, w_mid_rc, w_mid_ln, ln_ratio]
    ).T.rename(columns={0: "mid", 1: "RC", 2: "LN", 3: "ln_ratio"})
    df_mid[["name", "speed"]] = df_mid["mid"].str.split("/", expand=True)
    df_uid = pd.DataFrame([_m.uid_le.classes_, w_uid_rc, w_uid_ln]).T.rename(
        columns={0: "uid", 1: "RC", 2: "LN"}
    )
    df_uid[["name", "year"]] = df_uid["uid"].str.split("/", expand=True)
    return (
        df_mid.drop(columns=["mid"]).set_index(["name", "speed"]),
        df_uid.drop(columns=["uid"]).set_index(["name", "year"]),
    )


@st.cache_data()
def predict_all(
    _m: DeltaModel,
    model_id: str,
) -> pd.DataFrame:
    accs = _m.predict_all()
    return pd.DataFrame(accs, columns=_m.mid_classes, index=_m.uid_classes)


m, model_id = select_model(PROJECT_DIR)

df_mid_all, df_uid_all = get_model_embeddings(m, model_id)
df_accs = predict_all(m, model_id)

n_uid, n_mid = len(m.uid_classes), len(m.mid_classes)

with st.expander("Model Analysis"):
    st.markdown("# Delta to Accuracy Mapping")
    delta_to_acc(m)

    if False:
        with st.expander("Global Analysis"):
            df_mid, df_uid = get_model_embeddings(m, model_path)
            st.header("Map Embeddings")
            map_embeddings(df_mid)
            st.header("Player Embeddings")
            player_embeddings(df_uid)


with st.expander("User Analysis"):
    # TODO: reduce verbosity of this index selection
    username, useryear = select_user(
        df_uid_name=df_uid_all.index.get_level_values("name"),
        df_uid_year=df_uid_all.index.get_level_values("year"),
    )

    df_rank_uid = df_accs.rank(axis=0, ascending=False)
    df_rank_uid_i = df_rank_uid.loc[f"{username}/{useryear}"]

    player_rank(df_rank_uid_i, n_uid)
    player_rank_hist(df_rank_uid_i, username, useryear)

with st.expander("Map Analysis"):
    # TODO: reduce verbosity of this index selection
    mapname, mapspeed = select_map(
        df_mid_name=df_mid_all.index.get_level_values("name"),
        df_mid_speed=df_mid_all.index.get_level_values("speed"),
    )
    mapspeed_str = mapspeed_to_str(mapspeed)
    df_accs_mid = df_accs.loc[:, f"{mapname}/{mapspeed}"]

    map_ranks(df_accs_mid)
    map_rank_hist(df_accs_mid, mapname, mapspeed_str)

# %%
with st.expander("Star Rating Analysis"):
    threshold = 0.9
    df_accs_thres: pd.DataFrame = (
        (df_accs >= threshold)
        .mean(axis=0)
        .rename("thres")
        .rename_axis("mapname")
        .to_frame()
    )
    st.plotly_chart(
        px.histogram(df_accs_thres, nbins=100)
        .update_layout(
            title=f"{threshold:.2%} Threshold Distribution",
            xaxis_title=f"% of Population above {threshold:.2%}",
            yaxis_title="Count of Maps",
            showlegend=False,
        )
        .update_xaxes(autorange="reversed"),
        use_container_width=True,
    )
    st.markdown(
        f"""
        The distribution of how many players can achieve {threshold:.2%}. 
        - The further to the right, the rarer the {threshold:.2%} of a map, 
        which corresponds higher difficulty. 
        - The higher the peak, the more maps are in that difficulty range.
        """
    )
    st.info(
        """
        Setting a low threshold has a unique property of reducing the effects
        of gimmicked maps. Most of them are designed to be **extremely**
        difficult to achieve a high accuracy, however, are easy to achieve
        a decent one.        
        """
    )
# %%
min_sr, max_sr = 2.5, 10

# TODO: Find a better formula to convert the threshold to sr
df_accs_thres["new_sr"] = (max_sr - min_sr) * (
    np.sin((1 - df_accs_thres["thres"]) * np.pi / 2)
) + min_sr


# %%
dm = OsuDataModule(df_k(7))
df = dm.df.groupby(["mid", "uid"]).agg({"accuracy": "mean"}).reset_index()
df_p = df.pivot(index="uid", columns="mid", values="accuracy")

# %%
df_err = accs - df_p
df_m_err = (df_err.abs()).quantile(0.85, axis=0).rename("uncertainty")

# We do this as we only have srs of NT
df_m_err = df_m_err[df_m_err.index.str.endswith("/0")].to_frame()
df_m_err["uncertainty_pct"] = df_m_err["uncertainty"].rank(pct=True)

# %%
df_sr = (
    df[df["mid"].str.endswith("/0")]
    .groupby("mid")
    .agg({"sr": "first"})
    .rename(columns={"sr": "old_sr"})
)
df_sr = df_accs_thres.merge(df_sr, right_index=True, left_index=True).merge(
    df_m_err, right_index=True, left_index=True
)
# %%
df_sr["delta_sr"] = df_sr["new_sr"] - df_sr["old_sr"]
# %%
st.plotly_chart(px.histogram(df_sr, x="new_sr", nbins=100))
st.plotly_chart(px.histogram(df_sr, x="old_sr", nbins=100))
st.dataframe(df_sr.sort_values("delta_sr", ascending=False))

#
# @st.cache_data()
# def get_df(k=7):
#     return df_k.fn(k)
#
#
# # %%
# df_sr = (
#     get_df()
#     .groupby("mapname")
#     .agg({"sr": "first"})
#     .rename(columns={"sr": "old_sr"})
# )
# df_sr.index += "/0"
# df_sr = df_accs_thres.merge(df_sr, right_index=True, left_index=True)
# df_sr["delta_sr"] = df_sr["new_sr"] - df_sr["old_sr"]
# # %%
# st.plotly_chart(px.histogram(df_sr, x="new_sr", nbins=100))
# st.plotly_chart(px.histogram(df_sr, x="old_sr", nbins=100))
# st.dataframe(df_sr.sort_values("delta_sr", ascending=False))
#
# # %%
# df: pd.DataFrame = get_df()
# # # %%
# #
# # a = (
#     (
#         df.assign(
#             mid=lambda x: x["mapname"] + "/" + x["speed"].astype(str),
#             uid=lambda x: x["username"] + "/" + x["year"].astype(str),
#         )
#         .groupby(["mid", "uid"])
#         .agg({"accuracy": "mean"})
#         .reset_index()
#         .pivot(index="mid", columns="uid", values="accuracy")
#     )
#     .dropna(how="all", axis=0)
#     .dropna(how="all", axis=1)
# )
