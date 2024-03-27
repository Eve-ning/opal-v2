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
            return f"{label} ({value:.2f})"
    return "Unknown"


def st_confidence(user, map):
    st.subheader(
        "Prediction Confidence",
        help="The prediction confidence measures how confident the model is. "
        "It is determined by both the user's and map's confidence.",
    )

    userq = float(user["confidence_q"])
    mapq = float(map["confidence_q"])

    left, right = st.columns(2)
    right.metric("User", float_to_str_mapping(userq))
    left.metric("Map", float_to_str_mapping(mapq))
