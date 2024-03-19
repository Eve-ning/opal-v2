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


def st_map_emb(df, highlight_map):
    st.error(
        "Note that the embeddings dimensions do not have a fixed meaning. "
        "However, it's possible to interpret them as a measure of difficulty, "
        "where the larger the value, the harder the map."
    )
    dans_only = st.checkbox(
        "Dans Only",
        value=False,
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
        # Extract the dan number and color
        name = df["mapname"] + " " + df["speed"].astype(str)
        name_text = (
            name.str.extract(r"\[\b(\w+)\b")
            .replace(DAN_MAPPING)
            .values.flatten()
        )
        name_color = name.str.extract(r"\-\s\b(\w+)\b").values.flatten()
        fig = px.scatter(
            data_frame=df,
            x="d0",
            y="d1",
            hover_name=name,
            size=[1] * len(df),
            text=name_text,
            color=name_color,
        )
    else:
        fig = px.density_contour(data_frame=df, x="d0", y="d1")
    st.plotly_chart(
        fig.update_layout(font=dict(size=16)).add_scatter(
            x=highlight_map["d0"],
            y=highlight_map["d1"],
            mode="markers",
            marker=dict(size=10, color="red"),
            hoverinfo="text",
            hovertext=highlight_map["mapname"]
            + " "
            + highlight_map["speed"].apply({-1: "HT", 0: "NM", 1: "DT"}.get),
            showlegend=False,
        ),
        use_container_width=True,
    )


def st_player_emb(df, highlight_user):
    import plotly.graph_objects as go

    # Concatenate name and year for display on plotly
    year = highlight_user["year"]
    year_scaled = year - year.min()
    year_scaled /= year_scaled.max() if year_scaled.max() != 0 else 1
    year_scaled = year_scaled * 5 + 10
    st.plotly_chart(
        go.Figure(
            px.density_contour(
                df,
                x="d0",
                y="d1",
            ),
        ).add_scatter(
            x=highlight_user["d0"],
            y=highlight_user["d1"],
            mode="markers+lines",
            marker=dict(
                size=year_scaled,
                color="red",
            ),
            hoverinfo="text",
            hovertext=highlight_user["username"]
            + " "
            + highlight_user["year"].astype(str),
            showlegend=False,
        ),
        use_container_width=True,
    )
