import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DAN_MAPPING = {
    "0th": 0,
    "1st": 1,
    "2nd": 2,
    "3rd": 3,
    "4th": 4,
    "5th": 5,
    "6th": 6,
    "7th": 7,
    "8th": 8,
    "9th": 9,
    "10th": 10,
    "Gamma": 11,
    "Azimuth": 12,
    "Zenith": 13,
}


def st_map_emb(df):
    st.error(
        "Note that the embeddings dimensions do not have a fixed meaning. "
        "However, it's possible to interpret them as a measure of difficulty. "
        "Where the larger the value, the harder the map."
    )
    dans_only = st.checkbox(
        "Dans Only",
        value=True,
        help="Dans are the Dan Courses in the game."
        "Each level is a full course of different types of "
        "patterns. It's a way to measure a player's skill. "
        "We use this as a measure to sanity check the embeddings.",
    )
    if dans_only:
        df = df[
            (
                df["mapname"].str.contains("Regular Dan Phase")
                | df["mapname"].str.contains("LN Dan Phase")
            )
            & (df["speed"] == 0)
        ]
    name = df["mapname"] + " " + df["speed"].astype(str)

    # Extract the dan number and color
    name_text = (
        (
            name.str.extract(r"\[\b(\w+)\b")
            .replace(DAN_MAPPING)
            .values.flatten()
        )
        if dans_only
        else None
    )
    name_color = (
        name.str.extract(r"\-\s\b(\w+)\b").values.flatten()
        if dans_only
        else None
    )

    st.plotly_chart(
        px.scatter(
            data_frame=df,
            x="d0",
            y="d1",
            hover_name=name,
            size=[1] * len(df) if dans_only else None,
            text=name_text,
            color=name_color,
        ).update_layout(
            font=dict(size=16)
        ),  # Set the font size here
        use_container_width=True,
    )


def st_player_emb(df):
    # Concatenate name and year for display on plotly
    st.plotly_chart(
        px.scatter(
            df,
            x="d0",
            y="d1",
            hover_name=df["username"] + " " + df["year"].astype(str),
        ),
        use_container_width=True,
    )
