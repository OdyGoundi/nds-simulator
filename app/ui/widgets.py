import streamlit as st


def slider_with_input(label: str, min_value: float, max_value: float,
                      value: float, step: float, key: str, fmt: str = "%.6f") -> float:
    if key not in st.session_state:
        st.session_state[key] = float(value)

    slider_key = f"{key}_slider"
    input_key = f"{key}_input"
    if slider_key not in st.session_state:
        slider_val = max(min_value, min(max_value, float(value)))
        st.session_state[slider_key] = slider_val
    if input_key not in st.session_state:
        st.session_state[input_key] = float(value)

    def sync_from_slider():
        val = float(st.session_state[slider_key])
        st.session_state[key] = val
        st.session_state[input_key] = val

    def sync_from_input():
        st.session_state[key] = float(st.session_state[input_key])

    c1, c2 = st.columns([2, 1], gap="small")

    with c1:
        st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            value=float(st.session_state[slider_key]),
            step=step,
            key=slider_key,
            on_change=sync_from_slider,
        )

    with c2:
        st.number_input(
            " ",
            min_value=min_value,
            value=float(st.session_state[input_key]),
            step=step,
            format=fmt,
            key=input_key,
            on_change=sync_from_input,
        )

    return float(st.session_state[key])
