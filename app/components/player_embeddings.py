import streamlit as st
import plotly.express as px


def player_embeddings(df):
    st.plotly_chart(
        px.scatter(
            df,
            x=df["RC"],
            y=df["LN"],
            hover_name="uid",
        ),
        use_container_width=True,
    )
