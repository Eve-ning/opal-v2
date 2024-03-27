import streamlit as st


def st_support(user, map):
    st.markdown(
        "## Supports",
        help="The support is the number of plays associated to this user or map. "
        "Therefore, if a user or map has a low support, the model's "
        "prediction will be less accurate. Keep this in mind.",
    )
    left, right = st.columns(2)
    left.metric("User", int(user["support"].iloc[0]))
    right.metric("Map", int(map["support"].iloc[0]))
