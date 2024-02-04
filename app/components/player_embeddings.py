import numpy as np
import streamlit as st
import plotly.express as px


def st_player_emb(df):
    df = df.reset_index()
    df["RC"] = np.mean(df["RC"].tolist(), axis=1)
    df["LN"] = np.mean(df["LN"].tolist(), axis=1)

    # Concatenate name and year for display on plotly
    df["name"] = df["name"] + " " + df["year"]

    st.plotly_chart(
        px.scatter(df, x=df["RC"], y=df["LN"], hover_name="name"),
        use_container_width=True,
    )
