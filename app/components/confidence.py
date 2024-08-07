import streamlit as st

mappings = {
    (0.0, 0.01): "Very Low",
    (0.01, 0.05): "Low",
    (0.05, 0.1): "Medium",
    (0.1, 1.0): "High",
}


def float_to_str_mapping(value):
    """Converts a float to a string based on a predefined mapping.

    Args:
        value: The float value to convert.

    Returns:
        The corresponding string label ("High", "Medium", "Low", or "Very Low")
        based on the mapping. If the value falls outside the defined ranges,
        returns "Unknown".
    """
    for (low, high), label in mappings.items():
        if low <= value < high:
            return f"{label}"
    return "Unknown"


def st_confidence():
    upr, mpr = st.session_state["upr"], st.session_state["mpr"]
    st.subheader(
        "Prediction Confidence",
        help="This measures the trustworthiness of the predictions. ",
    )

    left, right = st.columns(2)
    right.metric("User", float_to_str_mapping(upr))
    left.metric("Map", float_to_str_mapping(mpr))
