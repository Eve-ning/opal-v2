import sys

from pathlib import Path

PROJECT_DIR = Path(__file__).parents[1]
sys.path.append(PROJECT_DIR.as_posix())

from components.delta_to_acc import st_delta_to_acc
from components.select import st_select_model, st_select_user, st_select_map
from components.embeddings import st_map_emb, st_player_emb

from utils import mapspeed_to_str
import streamlit as st

st.title("Dan Analysis")


m, model_id = st_select_model(PROJECT_DIR / "app")
df_uid, df_mid = m.get_embeddings()
df_uid, df_mid = df_uid.reset_index(), df_mid.reset_index()
n_uid, n_mid = len(m.uid_classes), len(m.mid_classes)

with st.expander("Model Analysis"):
    st_delta_to_acc(m)

with st.expander("Global Analysis"):
    st.header("Map Embeddings")
    st_map_emb(df_mid)
    st.header("Player Embeddings")
    st_player_emb(df_uid)

with st.sidebar:
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
left, right = st.columns(2)
left.metric("User Support", usersupp)
right.metric("Map Support", mapsupp)

map_pred = m.predict_map(f"{mapname}/{mapspeed}/7")
user_pred = m.predict_user(f"{username}/{useryear}/7")
map_play_pred = map_pred.loc[username, useryear, 7]

mean, lower_bound, upper_bound = (
    map_play_pred["mean"],
    map_play_pred["lower_bound"],
    map_play_pred["upper_bound"],
)

st.markdown(
    "## Bounds",
    help="The bounds are the 75% confidence interval for the prediction. "
    "Which means, if you played the map 100 times, "
    "it's likely that 75 of those scores fall within the bounds.",
)
cols = st.columns(3)
cols[0].metric(
    "Lower Bound",
    f"{lower_bound:.2%}",
    delta=f"-{(mean - lower_bound):.2%}",
)
cols[1].metric("Accuracy", f"{mean:.2%}")
cols[2].metric(
    "Upper Bound",
    f"{upper_bound:.2%}",
    delta=f"{(upper_bound - mean):.2%}",
)

st.markdown(
    "## Global Statistics",
    help="The following statistics are based on global predictions of the "
    "selected user and map.",
)

import plotly.graph_objects as go

st.markdown(f"### Accuracy Distribution\n")
st.plotly_chart(
    # Add a y vertical line at the mean
    go.Figure(
        [
            go.Histogram(
                name="Accuracy",
                x=map_pred["mean"],
            ),
        ]
    )
    .update_layout(
        title=f"Global Distribution for {mapname} @ {mapspeed_to_str(mapspeed)}",
        xaxis_title="Accuracy",
        yaxis_title="Count",
        xaxis=dict(range=[0.85, 1]),
    )
    .add_vline(x=mean, line=dict(color="red", width=2)),
)
st.plotly_chart(
    go.Figure(
        [
            go.Histogram(
                name="Accuracy", x=user_pred["mean"], histnorm="probability"
            ),
        ]
    )
    .update_layout(
        title=f"Accuracy Distribution for {username} @ {useryear}",
        xaxis_title="Accuracy",
        yaxis_title="Count",
        xaxis=dict(range=[0.85, 1]),
        # format y axis as percent
        yaxis=dict(tickformat=".0%"),
    )
    .add_vline(x=mean, line=dict(color="red", width=2)),
)
