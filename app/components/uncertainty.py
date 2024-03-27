import streamlit as st

mappings = {
    (0.0, 0.5): "High",
    (0.5, 0.8): "Medium",
    (0.8, 0.95): "Low",
    (0.95, 1.0): "Very Low",
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
            return f"{label} : {value:.2f}"
    return "Unknown"


def st_uncertainty(user, map):
    st.subheader(
        "Prediction Confidence",
        help="The prediction confidence measures how confident the model is "
        "in its prediction. The higher the confidence, the more accurate "
        "the prediction is.",
    )

    userq = float(user["dv0_q"])
    mapq = float(map["dv0_q"])

    left, right = st.columns(2)
    right.metric("User", float_to_str_mapping(userq))
    left.metric("Map", float_to_str_mapping(mapq))
