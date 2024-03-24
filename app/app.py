import sys
from pathlib import Path

import streamlit as st
from sklearn.decomposition import PCA

PROJECT_DIR = Path(__file__).parents[1]
sys.path.append(PROJECT_DIR.as_posix())

from components.bound_metrics import bound_metrics
from components.delta_to_acc import st_delta_to_acc
from components.embeddings import st_map_emb, st_player_emb
from components.global_prediction import global_prediction
from components.select import st_select_model, st_select_user, st_select_map


st.title("Dan Analysis")


with st.sidebar:
    m, model_id = st_select_model(PROJECT_DIR / "app")
    df_uid, df_mid = m.get_embeddings()
    pca = PCA(
        n_components=2,
        whiten=True,
        random_state=42,
    )
    do_pca = st.checkbox(
        "Align Embeddings (PCA)",
        help="Applying PCA extracts the most important dimensions "
        "from the embeddings, where d0 will be aligned to the most important "
        "feature and d1 the second most important. This alignment "
        "improves interpretation, but will not change the model's prediction.",
    )
    if do_pca:
        df_uid[["d0", "d1"]] = pca.fit_transform(df_uid[["d0", "d1"]])
        df_mid[["d0", "d1"]] = pca.transform(df_mid[["d0", "d1"]])
    df_uid, df_mid = df_uid.reset_index(), df_mid.reset_index()
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


map_pred = m.predict_map(f"{mapname}/{mapspeed}/7")
user_pred = m.predict_user(f"{username}/{useryear}/7")
map_play_pred = map_pred.loc[username, useryear, 7]

mean, lower_bound, upper_bound = (
    map_play_pred["mean"],
    map_play_pred["lower_bound"],
    map_play_pred["upper_bound"],
)
bound_metrics(mean, lower_bound, upper_bound)
with st.expander("Embedding Analysis"):
    st_map_emb(df_mid, df_mid[(df_mid["mapname"] == mapname)])
    st_player_emb(df_uid, df_uid[(df_uid["username"] == username)])

global_prediction(map_pred, user_pred, mean, lower_bound, upper_bound)

with st.expander("Model Analysis"):
    st_delta_to_acc(m)
