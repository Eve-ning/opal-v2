import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

PROJECT_DIR = Path(__file__).parents[1]
sys.path.append(PROJECT_DIR.as_posix())

from components.bound_metrics import st_boundary_metrics
from components.delta_to_acc import st_delta_to_acc
from components.embeddings import st_map_emb, st_player_emb
from components.global_prediction import st_global_preds
from components.select import st_select_model, st_select_user, st_select_map

st.set_page_config(
    page_title="Opal: Delta Embedding Approach",
    page_icon="☄️",
    initial_sidebar_state="expanded",
)
st.title("Opal: Delta Embedding Approach")

with st.sidebar:
    m, model_id = st_select_model(PROJECT_DIR / "app")

    # Extract all numerical values from the model_id as keys
    KEYS = int(re.search(r"\d+", model_id).group())
    st.caption(f"Model ID: {model_id}, Keys: {KEYS}")

    df_uid, df_mid = m.get_embeddings()
    df_uid, df_mid = df_uid.reset_index(), df_mid.reset_index()

    pca = PCA(n_components=2, whiten=True, random_state=42)
    if st.checkbox(
        "Align Embeddings (PCA)",
        help="Applying PCA extracts the most important dimensions "
        "from the embeddings, where d0 will be aligned to the most important "
        "feature and d1 the second most important. This alignment "
        "improves interpretation, but will not change the model's prediction. "
        "PCA may flip the axes, so the interpretation may be different.",
    ):
        pca.fit(pd.concat([df_uid[["d0", "d1"]], df_mid[["d0", "d1"]]]))
        df_uid[["d0", "d1"]] = pca.transform(df_uid[["d0", "d1"]])
        df_mid[["d0", "d1"]] = pca.transform(df_mid[["d0", "d1"]])

    n_uid, n_mid = len(m.uid_classes), len(m.mid_classes)
    username, useryear = st_select_user(
        name_opts=df_uid["username"],
        year_opts=df_uid["year"],
    )
    usersupp = df_uid[
        (df_uid["username"] == username) & (df_uid["year"] == useryear)
    ]["support"]

    mapname, mapspeed = st_select_map(
        name_opts=df_mid["mapname"],
        speed_opts=df_mid["speed"],
    )

    mapsupp = df_mid[
        (df_mid["mapname"] == mapname) & (df_mid["speed"] == mapspeed)
    ]["support"]
    st.markdown(
        "## User and Map Support",
        help="The support is the number of plays associated to this user or map. "
        "Therefore, if a user or map has a low support, the model's "
        "prediction will be less accurate. Keep this in mind.",
    )
    st.metric("User Support", usersupp)
    st.metric("Map Support", mapsupp)

    st.header(":wave: Hey! [Try AlphaOsu!](https://alphaosu.keytoix.vip/)")
    st.caption("AlphaOsu is a pp recommender system with a website UI. ")
    st.caption(
        "Opal doesn't require monetary support, but they do. "
        "If you do enjoy using their services, "
        "you can [support them](https://alphaosu.keytoix.vip/support)"
    )

st.success(
    ":wave: Thanks for using Opal! "
    "This is still in early access, we're improving things as we go along, and we appreciate feedback! "
    "Let us (@dev_evening on Twitter) know how it can be better." 
)
map_pred = m.predict_map(f"{mapname}/{mapspeed}/{KEYS}")
user_pred = m.predict_user(f"{username}/{useryear}/{KEYS}")
map_play_pred = map_pred.loc[username, useryear, KEYS]

mean, lower_bound, upper_bound = (
    map_play_pred["mean"],
    map_play_pred["lower_bound"],
    map_play_pred["upper_bound"],
)
st_boundary_metrics(mean, lower_bound, upper_bound)

with st.expander("Embedding Analysis"):
    st_map_emb(
        df_mid,
        highlight_map=df_mid[(df_mid["mapname"] == mapname)],
        # We're not supporting 4K dans yet since they don't have leaderboards
        enable_dans=KEYS == 7,
    )
    st_player_emb(
        df_uid,
        highlight_user=df_uid[(df_uid["username"] == username)],
    )

st_global_preds(map_pred, user_pred, mean, lower_bound, upper_bound)

with st.expander("Model Analysis"):
    xlim = st.slider(
        "Delta Range (Debug)",
        min_value=5,
        max_value=10,
        value=7,
    )
    st_delta_to_acc(m, xlim=(-xlim, xlim))

st.caption("Developed by [Evening](https://twitter.com/dev_evening).")
