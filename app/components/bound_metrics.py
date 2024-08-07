import streamlit as st


def st_boundary_metrics():
    acc_mean, acc_lower, acc_upper = (
        st.session_state["acc_mean"],
        st.session_state["acc_lower"],
        st.session_state["acc_upper"],
    )

    st.markdown(
        "## Accuracy Prediction",
        help="The bounds are the 75% confidence interval for the prediction. "
        "Which means, if you played the map 100 times, "
        "it's likely that 75 of those scores fall within the bounds.",
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
