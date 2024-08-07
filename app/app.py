import re
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).parents[1]
sys.path.append(PROJECT_DIR.as_posix())


import pandas as pd
import streamlit as st

from components.confidence import st_confidence
from opal.model.delta_model import DeltaModel
from opal.utils import RSC_DIR

from components.bound_metrics import st_boundary_metrics
from components.rating import st_map_rating, st_player_rating

from components.leaderboard import st_player_leaderboard, st_map_leaderboard
from components.select import st_select_model, st_select_user, st_select_map

st.set_page_config(
    page_title="Opal",
    page_icon="☄️",
    initial_sidebar_state="expanded",
)
st.title("Opal: Delta Model")

THIS_DIR = Path(__file__).parent


@st.cache_data
def read_map_metadata():
    return (
        pd.read_csv(THIS_DIR / "map_metadata.csv")
        .set_index("mid")["mapname"]
        .to_dict()
    )


@st.cache_data
def read_player_metadata():
    return (
        pd.read_csv(THIS_DIR / "player_metadata.csv")
        .set_index("uid")["username"]
        .to_dict()
    )


map_metadata = read_map_metadata()
player_metadata = read_player_metadata()

with st.sidebar:
    m, model_name = st_select_model(PROJECT_DIR / "app")
    m: DeltaModel
    # Extract all numerical values from the model_id as keys
    KEYS = int(re.findall(r"\d+", model_name)[0])
    st.caption(f"Model ID: {model_name}, Keys: {KEYS}")

    df_uid = pd.DataFrame(
        m.emb_uid_mean.weight.detach(),
        columns=["rating"],
        index=m.le_uid.classes_,
    ).reset_index(names=["user"])
    df_uid.columns = df_uid.columns.astype(str)
    df_uid[["uid", "year"]] = (
        df_uid["user"].str.split("/", expand=True).astype(int)
    )
    df_uid = (
        df_uid.drop("user", axis=1)
        .assign(
            username=lambda x: x["uid"].apply(lambda i: player_metadata[i]),
            uid_le=lambda x: m.le_uid.transform(
                x["uid"].astype(str) + "/" + x["year"].astype(str)
            ),
            pagerank=lambda x: m.dt_uid_w.transform(x["uid_le"]),
        )
        .assign(
            pagerank_qt=lambda x: x["pagerank"].rank(pct=True),
        )
    )
    df_mid = (
        pd.DataFrame(
            m.emb_mid.weight.detach(),
            columns=["rating"],
            index=m.le_mid.classes_,
        )
        .rename(lambda x: str(x))
        .reset_index(names=["map"])
    )
    df_mid.columns = df_mid.columns.astype(str)
    df_mid[["mid", "speed"]] = (
        df_mid["map"].str.split("/", expand=True).astype(int)
    )
    df_mid = (
        df_mid.drop("map", axis=1)
        .assign(
            mapname=lambda x: x["mid"].apply(lambda i: map_metadata[i]),
            mid_le=lambda x: m.le_mid.transform(
                x["mid"].astype(str) + "/" + x["speed"].astype(str)
            ),
            pagerank=lambda x: m.dt_mid_w.transform(x["mid_le"]),
        )
        .assign(
            pagerank_qt=lambda x: x["pagerank"].rank(pct=True),
        )
    )

    n_uid, n_mid = len(m.uid_classes), len(m.mid_classes)
    username, year = st_select_user(
        username_opts=df_uid["username"],
        year_opts=df_uid["year"],
    )

    mapname, speed = st_select_map(
        mapname_opts=df_mid["mapname"],
        speed_opts=df_mid["speed"],
    )
    speed_str = {-1: "HT", 0: "NT", 1: "DT"}[speed]

    u_rating, uid, upr = df_uid.loc[
        (df_uid["username"] == username) & (df_uid["year"] == year),
        ["rating", "uid", "pagerank"],
    ].iloc[0]

    m_rating, mid, mpr = df_mid.loc[
        (df_mid["mapname"] == mapname) & (df_mid["speed"] == speed),
        ["rating", "mid", "pagerank"],
    ].iloc[0]
    uid, mid = int(uid), int(mid)

    df_map_pred = m.predict_map(f"{mid}/{speed}")
    df_user_pred = m.predict_user(f"{uid}/{year}")

    acc_pred = df_map_pred.loc[uid, year]

    acc_mean, acc_lower, acc_upper = (
        acc_pred["mean"],
        acc_pred["lower_bound"],
        acc_pred["upper_bound"],
    )

    st.header(":wave: Hey! [Try AlphaOsu!](https://alphaosu.keytoix.vip/)")
    st.caption("AlphaOsu is a pp recommender system with a website UI. ")
    st.caption(
        "Opal doesn't require monetary support, but they do. "
        "If you do enjoy using their services, "
        "you can [support them](https://alphaosu.keytoix.vip/support)"
    )
    st.session_state.update(
        {
            "uid": uid,
            "mid": mid,
            "year": year,
            "speed": speed,
            "username": username,
            "mapname": mapname,
            "speed_str": speed_str,
            "u_rating": u_rating,
            "m_rating": m_rating,
            "upr": upr,
            "mpr": mpr,
            "keys": KEYS,
            "acc_mean": acc_mean,
            "acc_lower": acc_lower,
            "acc_upper": acc_upper,
        }
    )

st.success(
    ":wave: Thanks for using Opal! "
    "This is still in early access, we're improving things as we go along, "
    "and we appreciate feedback! "
    "Let us (@dev_evening on Twitter) know how it can be better."
)


st_boundary_metrics()
st_confidence()
st_map_rating(df_mid)
st_player_rating(df_uid)
st_map_leaderboard(df_map_pred)
st_player_leaderboard(df_user_pred)

st.caption("Developed by [Evening](https://twitter.com/dev_evening).")
