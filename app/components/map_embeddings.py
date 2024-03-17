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
        "Currently, this is rather experimental. "
        "We're working on making this more intuitive."
    )
    st.write(
        "These embeddings represent the difficulty of maps. "
        "The larger the values (top right), the harder the map. "
        "Each dimension (x, y) represents a different difficulty element, "
        "in this case, we have RC and LN."
    )
    dans_only = st.checkbox("Dans Only", value=True)
    df = df.copy().reset_index()
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
    st.text(df.columns)

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
        )  # Set the font size here
    )
