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


def st_map_rating(df):
    enable_dans = st.session_state["keys"] == 7
    m_rating = st.session_state["m_rating"]
    mapname = st.session_state["mapname"]
    speed_str = st.session_state["speed_str"]

    with st.expander("Map Rating"):
        st.markdown(
            f"The red line shows the difficulty of :orange[{mapname} on {speed_str}], "
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
                x=df["rating"],
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
                x="rating",
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
            ).add_vline(x=m_rating, line_color="#F00"),
            use_container_width=True,
        )


def st_player_rating(df):
    u_rating = st.session_state["u_rating"]
    username = st.session_state["username"]
    year = st.session_state["year"]
    with st.expander("Player Rating"):
        st.markdown(
            f"The red line indicates the rating of :orange[{username} @ {year}], "
            "compared to all players"
        )
        st.plotly_chart(
            go.Figure(
                px.histogram(
                    x=df["rating"],
                    histnorm="probability",
                ),
            )
            .add_vline(
                x=u_rating,
                line_color="#F00",
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
        st.subheader("Rating History")
        st.plotly_chart(
            px.line(
                df.loc[df["username"] == username, ["year", "rating"]]
                .sort_values("year", ascending=False)
                .reset_index(drop=True),
                x="year",
                y="rating",
            ).add_vline(
                x=year,
                line_color="#F00",
            ),
            use_container_width=True,
        )
