import sys

from pathlib import Path

PROJECT_DIR = Path(__file__).parents[1]
sys.path.append(PROJECT_DIR.as_posix())

from components.delta_to_acc import st_delta_to_acc
from components.select import st_select_model, st_select_user, st_select_map
from components.map_embeddings import st_map_emb
from components.rank import (
    st_map_rank_hist,
    st_map_ranks,
    st_player_rank,
    st_player_rank_hist,
)

from components.player_embeddings import st_player_emb
from utils import model_emb, predict_all, mapspeed_to_str
import streamlit as st

st.title("Dan Analysis")


m, model_id = st_select_model(PROJECT_DIR)
df_mid, df_uid = model_emb(m, model_id)
n_uid, n_mid = len(m.uid_classes), len(m.mid_classes)

with st.expander("Model Analysis"):
    st.header("Delta to Accuracy Mapping")
    st_delta_to_acc(m)

with st.expander("Global Analysis"):
    st.header("Map Embeddings")
    st_map_emb(df_mid)
    st.header("Player Embeddings")
    st_player_emb(df_uid)

df_accs = predict_all(m, model_id)
with st.expander("User Analysis"):
    username, useryear = st_select_user(
        name_opts=df_uid["name"],
        year_opts=df_uid["year"],
    )

    df_rank_uid_i = df_accs.rank(
        axis=0,
        ascending=False,
    ).loc[f"{username}/{useryear}"]

    st_player_rank(df_rank_uid_i, n_uid)
    st_player_rank_hist(df_rank_uid_i, username, useryear)

with st.expander("Map Analysis"):
    mapname, mapspeed = st_select_map(
        name_opts=df_mid["name"],
        speed_opts=df_mid["speed"],
    )
    mapspeed_str = mapspeed_to_str(mapspeed)
    df_accs_mid = df_accs.loc[:, f"{mapname}/{mapspeed}"]

    st_map_ranks(df_accs_mid)
    st_map_rank_hist(df_accs_mid, mapname, mapspeed_str)
# %%
# from opal.data import OsuDataModule, df_k
#
# df_raw = OsuDataModule(df_k(7)).df
# # %%
# df_piv = (
#     df_raw.groupby(["mid", "uid"])
#     .agg({"accuracy": "mean"})
#     .reset_index()
#     .pivot(index="uid", columns="mid", values="accuracy")
# ).to_csv("df_piv.csv.gz", index=False, compression="gzip")
# # %%
# df_raw[["mid", "uid", "accuracy"]].to_csv(
#     "df_raw.csv.gz", index=False, compression="gzip"
# )
# # %%
# df_unc = (
#     (df_accs - df_piv)
#     .abs()
#     .quantile(0.85, axis=0)
#     .rename("uncertainty")
#     .to_frame()
#     .query("index.str.endswith('/0')")
#     .assign(uncertainty_pct=lambda x: x.rank(pct=True))
# )
#
# # %%
# threshold = st.slider("Threshold", 0.8, 1.0, 0.925, 0.01)
# df_accs_thres: pd.DataFrame = (
#     (df_accs >= threshold)
#     .mean(axis=0)
#     .rename("thres")
#     .rename_axis("mapname")
#     .to_frame()
#     .query("mapname.str.endswith('/0')")
# )
#
# df_sr = (
#     df_raw.query("mid.str.endswith('/0')")
#     .groupby("mid")
#     .agg({"sr": "first"})
#     .merge(df_accs_thres, right_index=True, left_index=True)
#     .merge(df_unc, right_index=True, left_index=True)
#     .assign(
#         sr_rank=lambda x: x["sr"].rank(ascending=False),
#         thres_rank=lambda x: x["thres"].rank(ascending=True),
#         delta_rank=lambda x: x["sr_rank"] - x["thres_rank"],
#     )
# )
# st.dataframe(df_sr)
# # %%
