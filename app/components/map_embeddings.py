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


def map_embeddings(df, p_mid=1, dans_only=True):
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

    m_scaled_tab, m_unscaled_tab = st.tabs(["Scaled", "Unscaled"])

    def plot(df, scaled):
        if dans_only:
            df = df[
                (
                    df["mid"].str.contains("Regular Dan Phase")
                    | df["mid"].str.contains("LN Dan Phase")
                )
                & df["mid"].str.endswith("/0")
            ]
        df = df.sample(frac=p_mid, random_state=0)
        x = df["RC"] * (1 - df["ln_ratio"]) if scaled else df["RC"]
        y = df["LN"] * df["ln_ratio"] if scaled else df["LN"]
        return px.scatter(
            df,
            x=x,
            y=y,
            hover_name="mid",
            labels={"x": "RC", "y": "LN"},
            size=[1] * len(df) if dans_only else None,
            text=df["mid"]
            .str.extract(r"\[\b(\w+)\b")
            .replace(
                {
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
            )
            .values.flatten()
            if dans_only
            else None,
            color=df["mid"].str.extract(r"\-\s\b(\w+)\b").values.flatten()
            if dans_only
            else None,
        ).update_layout(
            font=dict(
                size=16,  # Set the font size here
            ),
        )

    with m_scaled_tab:
        st.info(
            "**Embeddings are scaled** by the ratio of LN/RC notes. "
            "Therefore, if a map is LN-hard, but has only a few LNs, "
            "the LN embedding will be small."
        )

        st.plotly_chart(
            plot(df, scaled=True),
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
            plot(df, scaled=False),
            use_container_width=True,
        )
