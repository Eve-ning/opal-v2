import streamlit as st


def st_boundary_metrics(mean, lower_bound, upper_bound):
    st.markdown(
        "## 75% Confidence Prediction Bounds",
        help="The bounds are the 75% confidence interval for the prediction. "
        "Which means, if you played the map 100 times, "
        "it's likely that 75 of those scores fall within the bounds.",
    )
    cols = st.columns(3)
    cols[0].metric(
        "Lower Bound",
        f"{lower_bound:.2%}",
        delta=f"-{(mean - lower_bound):.2%}",
    )
    cols[1].metric("Accuracy", f"{mean:.2%}")
    cols[2].metric(
        "Upper Bound",
        f"{upper_bound:.2%}",
        delta=f"{(upper_bound - mean):.2%}",
    )
