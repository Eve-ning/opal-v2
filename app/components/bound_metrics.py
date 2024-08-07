import streamlit as st


def st_boundary_metrics():
    acc_mean, acc_lower, acc_upper = (
        st.session_state["acc_mean"],
        st.session_state["acc_lower"],
        st.session_state["acc_upper"],
    )
    username, year, mapname, speed_str = (
        st.session_state["username"],
        st.session_state["year"],
        st.session_state["mapname"],
        st.session_state["speed_str"],
    )

    st.markdown(
        "## Accuracy Prediction",
        help=f"This means that if :orange[{username} @ {year}] plays "
        f":orange[{mapname} @ {speed_str}] "
        f":blue[100] times, :blue[75] of them are likely to fall between "
        f":blue[{acc_lower:.2%}] and :blue[{acc_upper:.2%}].",
    )
    cols = st.columns(3)
    cols[0].metric(
        "Lower Bound",
        f"{acc_lower:.2%}",
        delta=f"-{(acc_mean - acc_lower):.2%}",
    )
    cols[1].metric("Accuracy", f"{acc_mean:.2%}")
    cols[2].metric(
        "Upper Bound",
        f"{acc_upper:.2%}",
        delta=f"{(acc_upper - acc_mean):.2%}",
    )
