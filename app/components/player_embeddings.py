import numpy as np
import streamlit as st
import plotly.express as px


def st_player_emb(df):
    # Concatenate name and year for display on plotly
    st.plotly_chart(
        px.scatter(
            df,
            x=df["RC"],
            y=df["LN"],
            hover_name=df["name"] + " " + df["year"],
        ),
        use_container_width=True,
    )
