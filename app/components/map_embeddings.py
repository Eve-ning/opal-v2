import numpy as np
import plotly.express as px
import streamlit as st


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
    if dans_only:
        df = df[
            (
                df["name"].str.contains("Regular Dan Phase")
                | df["name"].str.contains("LN Dan Phase")
            )
            & (df["speed"] == "0")
        ]
    rc, ln, ln_ratio = df["RC"], df["LN"], df["ln_ratio"]
    name = df["name"] + " " + df["speed"]

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

    def plot(rc, ln, name):
        return px.scatter(
            x=rc,
            y=ln,
            hover_name=name,
            labels={"x": "RC", "y": "LN"},
            size=[1] * len(df) if dans_only else None,
            text=name_text,
            color=name_color,
        ).update_layout(
            font=dict(size=16)
        )  # Set the font size here

    m_scaled_tab, m_unscaled_tab = st.tabs(["Scaled", "Unscaled"])
    with m_scaled_tab:
        st.info(
            "**Embeddings are scaled** by the ratio of LN/RC notes. "
            "Therefore, if a map is LN-hard, but has only a few LNs, "
            "the LN embedding will be small."
        )
        st.plotly_chart(
            plot(rc * (1 - ln_ratio), ln * ln_ratio, name),
            use_container_width=True,
        )

    with m_unscaled_tab:
        st.info(
            "**Embeddings are NOT scaled** by the ratio of LN/RC notes. "
            "Therefore, if a map is LN-hard, but has only a few LNs, "
            "the LN embedding will remain large. "
            "Which isn't representative of the map."
        )
        st.plotly_chart(
            plot(rc, ln, name),
            use_container_width=True,
        )
