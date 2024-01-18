import sys
from pathlib import Path

import numpy as np
import plotly.express as px

from components.delta_to_acc import delta_to_acc
from components.select import select_user, select_map

PROJECT_DIR = Path(__file__).parents[1]
sys.path.append(PROJECT_DIR.as_posix())
import pandas as pd
import streamlit as st

from opal.data import df_k
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
    df_mid[["name", "speed"]] = df_mid["mid"].str.split("/", expand=True)
    # df_mid["speed"] = df_mid["speed"].replace(
    #     {"-1": "HT", "0": "NT", "1": "DT"}
    # )

    df_uid = pd.DataFrame([_m.uid_le.classes_, w_uid_rc, w_uid_ln]).T.rename(
        columns={0: "uid", 1: "RC", 2: "LN"}
    )
    df_uid[["name", "year"]] = df_uid["uid"].str.split("/", expand=True)
    # df_uid["RC"] = df_uid["RC"].apply(
    #     lambda x: x if isinstance(x, Seqe) else [x]
    # )
    # df_uid["LN"] = df_uid["LN"].apply(
    #     lambda x: x if isinstance(x, Seqe) else [x]
    # )
    # df_mid["RC"] = df_mid["RC"].apply(
    #     lambda x: x if isinstance(x, Seqe) else [x]
    # )
    # df_mid["LN"] = df_mid["LN"].apply(
    #     lambda x: x if isinstance(x, Seqe) else [x]
    # )
    return (
        df_mid.drop(columns=["mid"]).set_index(["name", "speed"]),
        df_uid.drop(columns=["uid"]).set_index(["name", "year"]),
    )


@st.cache_data()
def predict_all(
    _m: DeltaModel,
    path: str,
):
    return m.predict_all()


model_path = st.selectbox(
    "Model Path",
    format_func=lambda x: x.parts[-3],
    options=list(p for p in PROJECT_DIR.glob("**/*.ckpt")),
    placeholder="Select a model",
)

m = load_model(model_path)
df_mid_all, df_uid_all = get_model_embeddings(m, model_path)
# %%
accs = predict_all(m, model_path)
df_accs = pd.DataFrame(accs, columns=m.mid_classes, index=m.uid_classes)

n_uid = len(m.uid_classes)
n_mid = len(m.mid_classes)
# %%
ar_mid_rc = np.array(df_mid_all["RC"].tolist())
ar_mid_ln = np.array(df_mid_all["LN"].tolist())
ar_uid_rc = np.array(df_uid_all["RC"].tolist())
ar_uid_ln = np.array(df_uid_all["LN"].tolist())
# %%

with st.expander("Model Analysis"):
    st.markdown("# Delta to Accuracy Mapping")
    delta_to_acc(m)

if False:
    with st.expander("Global Analysis"):
        cols = st.columns(4)
        cols[0].metric("Map RC Shape", str(ar_mid_rc.shape))
        cols[1].metric("Map LN Shape", str(ar_mid_ln.shape))
        cols[2].metric("Player RC Shape", str(ar_uid_rc.shape))
        cols[3].metric("Player LN Shape", str(ar_uid_ln.shape))

        df_mid, df_uid = get_model_embeddings(m, model_path)
        st.header("Map Embeddings")
        map_embeddings(df_mid)
        st.header("Player Embeddings")
        player_embeddings(df_uid)
        st.header("Delta to Accuracy Mapping")
        st.write(
            "This function maps the difference between player and map "
            "embeddings to accuracy."
        )
        delta_to_acc(m)


# %%
@st.cache_data()
def player_rank(df: pd.DataFrame, n_uid: int):
    left, right = st.columns(2)
    left.metric(
        "Player Median Rank",
        f"{np.median(df).astype(int)} / {n_uid}",
    )
    right.markdown(
        "The value on the left is a rough estimate on your rank "
        "based on all the maps in the dataset"
    )


@st.cache_data()
def player_rank_hist(df: pd.DataFrame, username: str, useryear: str):
    st.plotly_chart(
        px.histogram(x=df).update_layout(
            title=f"Rank Distribution of {username} in {useryear}",
            xaxis_title="Rank",
            yaxis_title="Count",
        )
    )


@st.cache_data()
def map_ranks(df: pd.DataFrame):
    st.markdown("Players with at least ...")
    r_b, r_a, r_s = st.columns(3)

    r_b.metric("B Rank", f"{(df >= 0.85).mean():.2%}")
    r_a.metric("A Rank", f"{(df >= 0.90).mean():.2%}")
    r_s.metric("S Rank", f"{(df >= 0.95).mean():.2%}")


@st.cache_data()
def map_rank_hist(df: pd.DataFrame, mapname: str, mapspeed_str: str):
    st.plotly_chart(
        px.histogram(x=df)
        .update_layout(
            title=f"Accuracy Distribution of {mapname} {mapspeed_str}",
            xaxis_title="Accuracy",
            yaxis_title="Count",
            xaxis_range=[None, 1],
        )
        .add_vline(0.95, line_color="red", line_width=2, opacity=0.65)
        .add_vline(0.90, line_color="yellow", line_width=2, opacity=0.65)
        .add_vline(0.85, line_color="green", line_width=2, opacity=0.65)
    )


with st.expander("User Analysis"):
    username, useryear = select_user(
        df_uid_name=df_uid_all.index.get_level_values("name"),
        df_uid_year=df_uid_all.index.get_level_values("year"),
    )

    df_rank_uid = df_accs.rank(axis=0, ascending=False)
    df_rank_uid_i = df_rank_uid.loc[f"{username}/{useryear}"]

    player_rank(df_rank_uid_i, n_uid)
    player_rank_hist(df_rank_uid_i, username, useryear)

with st.expander("Map Analysis"):
    mapname, mapspeed = select_map(
        df_mid_name=df_mid_all.index.get_level_values("name"),
        df_mid_speed=df_mid_all.index.get_level_values("speed"),
    )
    mapspeed_str = {"-1": "HT", "0": "NM", "1": "DT"}.get(mapspeed)

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
        .update_xaxes(autorange="reversed")
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

df_accs_thres["new_sr"] = (max_sr - min_sr) * (
    np.sin((1 - df_accs_thres["thres"]) * np.pi / 2)
) + min_sr

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

# %%
from opal.data import OsuDataModule, df_k

dm = OsuDataModule(df_k(7))
df = dm.df.groupby(["mid", "uid"]).agg({"accuracy": "mean"}).reset_index()
# %%
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
