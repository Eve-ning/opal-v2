import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

from components.confidence import st_confidence
from opal.model.delta_model import DeltaModel

PROJECT_DIR = Path(__file__).parents[1]
sys.path.append(PROJECT_DIR.as_posix())

from components.bound_metrics import st_boundary_metrics
from components.delta_to_acc import st_delta_to_acc
from components.rating import st_map_rating, st_player_rating
from components.leaderboard import st_leaderboard
from components.select import st_select_model, st_select_user, st_select_map

st.set_page_config(
    page_title="Opal",
    page_icon="☄️",
    initial_sidebar_state="expanded",
)
st.title("Opal")

THIS_DIR = Path(__file__).parent
map_metadata = (
    pd.read_csv(THIS_DIR / "map_metadata.csv")
    .set_index("mid")["mapname"]
    .to_dict()
)
player_metadata = (
    pd.read_csv(THIS_DIR / "player_metadata.csv")
    .set_index("uid")["username"]
    .to_dict()
)


with st.sidebar:
    m, model_name = st_select_model(PROJECT_DIR / "app")
    m: DeltaModel
    # Extract all numerical values from the model_id as keys
    KEYS = int(re.findall(r"\d+", model_name)[0])
    st.caption(f"Model ID: {model_name}, Keys: {KEYS}")

    df_uid = pd.DataFrame(
        m.emb_uid_mean.weight.detach(),
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

    pca = PCA(n_components=2, whiten=True, random_state=42)
    pca.fit(pd.concat([df_uid[["0", "1"]], df_mid[["0", "1"]]]))
    df_uid[["0", "1"]] = pca.transform(df_uid[["0", "1"]])
    df_mid[["0", "1"]] = pca.transform(df_mid[["0", "1"]])

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

    ud0, ud1, uid, upr = df_uid.loc[
        (df_uid["username"] == username) & (df_uid["year"] == year),
        ["0", "1", "uid", "pagerank"],
    ].iloc[0]

    md0, md1, mid, mpr = df_mid.loc[
        (df_mid["mapname"] == mapname) & (df_mid["speed"] == speed),
        ["0", "1", "mid", "pagerank"],
    ].iloc[0]
    uid, mid = int(uid), int(mid)

    st.header(":wave: Hey! [Try AlphaOsu!](https://alphaosu.keytoix.vip/)")
    st.caption("AlphaOsu is a pp recommender system with a website UI. ")
    st.caption(
        "Opal doesn't require monetary support, but they do. "
        "If you do enjoy using their services, "
        "you can [support them](https://alphaosu.keytoix.vip/support)"
    )

st.success(
    ":wave: Thanks for using Opal! "
    "This is still in early access, we're improving things as we go along, "
    "and we appreciate feedback! "
    "Let us (@dev_evening on Twitter) know how it can be better."
)
map_pred = m.predict_map(f"{mid}/{speed}")
user_pred = m.predict_user(f"{uid}/{year}")

map_play_pred = map_pred.loc[uid, year]

mean, lower_bound, upper_bound = (
    map_play_pred["mean"],
    map_play_pred["lower_bound"],
    map_play_pred["upper_bound"],
)
st_boundary_metrics(mean, lower_bound, upper_bound)
st_confidence(upr, mpr)

st_map_rating(
    df_mid,
    md=md0,
    maplabel=f"{mapname} {speed_str}",
    enable_dans=KEYS == 7,
)
st_player_rating(df_uid, ud=ud0, userlabel=f"{username} @ {year}")

st_leaderboard(map_pred, user_pred, mean, lower_bound, upper_bound)

with st.expander("Debugging Tools"):
    xlim = st.slider(
        "Delta Range (Debug)",
        min_value=5,
        max_value=10,
        value=7,
    )
    st_delta_to_acc(m, xlim=(-xlim, xlim))

st.caption("Developed by [Evening](https://twitter.com/dev_evening).")
