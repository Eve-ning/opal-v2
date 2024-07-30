import plotly.express as px
import streamlit as st

import plotly.graph_objects as go


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


def st_map_rating(df, md, maplabel, enable_dans: bool = False):
    with st.expander("Map Rating"):
        st.markdown(
            "The red line shows the difficulty of the map, "
            "compared to all other maps"
        )
        dans_only = st.checkbox(
            "Dans Only (Only for 7K)",
            value=False,
            disabled=not enable_dans,
            help="Dans are the Dan Courses in the game. "
            "Each level is a full course of different types of "
            "patterns. It's a way to measure a player's skill. "
            "We use this as a measure to sanity check the embeddings.",
        )

        if dans_only and enable_dans:
            df = df.query(
                "(mapname.str.contains('Regular Dan Phase') | "
                "mapname.str.contains('LN Dan Phase')) "
            )

            # Extract the dan number and color
            df["danname"] = df["mapname"].str.extract(r"\[(\w+)\b")
            df["dannum"] = df["danname"].replace(DAN_MAPPING)
            df["dantype"] = df["mapname"].str.extract(r"\-\s\b(\w+)\b")
            df["speedtxt"] = df["speed"].apply({-1: "HT", 0: "", 1: "DT"}.get)

            fig = px.scatter(
                data_frame=df.rename({"pagerank_qt": "Confidence"}, axis=1),
                x=df["0"],
                y=df["dantype"] + " " + df["speedtxt"],
                hover_name=df["danname"] + " " + df["dantype"],
                size=[1] * len(df),
                text="dannum",
                color="Confidence",
                color_continuous_scale="RdYlGn",
            ).update_layout(
                yaxis_title="Dan Category",
            )
        else:
            fig = px.histogram(
                data_frame=df,
                x="0",
                histnorm="probability",
            ).update_layout(
                yaxis=dict(tickformat=".0%"),
                yaxis_title="Frequency",
            )

        st.plotly_chart(
            fig.update_layout(
                font=dict(size=16),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                xaxis_title="Rating",
            )
            .add_vline(x=md, line_color="#F00")
            .add_scatter(
                x=[md], line_color="#F00", name=maplabel, visible="legendonly"
            ),
            use_container_width=True,
        )


def st_player_rating(df, ud, userlabel):
    with st.expander("Player Rating"):
        st.markdown(
            "The red line shows the rating of the player, "
            "compared to all other players"
        )
        st.plotly_chart(
            go.Figure(
                px.histogram(
                    df,
                    x="0",
                    labels={"0": "Rating"},
                    histnorm="probability",
                ),
            )
            .add_vline(
                x=ud,
                line_color="#F00",
            )
            .add_scatter(
                x=[ud],
                line_color="#F00",
                name=userlabel,
            )
            .update_layout(
                font=dict(size=16),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                yaxis=dict(tickformat=".0%"),
                yaxis_title="Frequency",
            ),
            use_container_width=True,
        )
