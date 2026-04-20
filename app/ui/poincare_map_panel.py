from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app.logic.poincare_map import (
    DIRECTION_LABEL_BY_VALUE,
    DIRECTION_VALUE_BY_LABEL,
    PoincareMapConfig,
    axis_pair_options,
    compute_poincare_map,
)


SHOW_KEY = "show_poincare_map_tab1"
SECTION_INDEX_KEY = "poincare_section_idx_tab1"
SECTION_VALUE_KEY = "poincare_section_value_tab1"
DIRECTION_KEY = "poincare_direction_label_tab1"
AXIS_PAIR_KEY = "poincare_axis_pair_tab1"
AXIS_PAIR_SIG_KEY = "poincare_axis_pair_sig_tab1"
MAX_POINTS_KEY = "poincare_max_points_tab1"


def _axis_name(var_names: Sequence[str], index: int) -> str:
    if 0 <= int(index) < len(var_names):
        return str(var_names[int(index)])
    return f"y{int(index) + 1}"


def _pair_label(var_names: Sequence[str], pair: tuple[int, int]) -> str:
    return f"{_axis_name(var_names, pair[0])} vs {_axis_name(var_names, pair[1])}"


def render_poincare_map_panel(
    *,
    t: np.ndarray,
    y: np.ndarray,
    var_names: Sequence[str],
    preferred_axes: Sequence[int],
    title_prefix: str = "",
) -> None:
    st.markdown("**Poincaré map (from current trajectory)**")
    show_map = st.checkbox(
        "Show Poincaré map",
        value=bool(st.session_state.get(SHOW_KEY, False)),
        key=SHOW_KEY,
        help=(
            "Extract section crossings from the trajectory already shown above and plot "
            "their projection on a selected pair of axes."
        ),
    )
    if not show_map:
        return

    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim != 2 or int(y_arr.shape[0]) < 3:
        st.info("Poincaré map display requires at least 3 state variables.")
        return

    n_dim = int(y_arr.shape[0])
    default_section = int(preferred_axes[0]) if preferred_axes else 0
    if SECTION_INDEX_KEY not in st.session_state:
        st.session_state[SECTION_INDEX_KEY] = min(max(default_section, 0), n_dim - 1)
    if MAX_POINTS_KEY not in st.session_state:
        st.session_state[MAX_POINTS_KEY] = 5000
    if DIRECTION_KEY not in st.session_state:
        st.session_state[DIRECTION_KEY] = DIRECTION_LABEL_BY_VALUE[+1]

    controls_top = st.columns([1.35, 1.0, 1.0, 1.0], gap="small")
    with controls_top[0]:
        section_index = int(
            st.selectbox(
                "Section plane",
                options=list(range(n_dim)),
                format_func=lambda i: f"{_axis_name(var_names, i)} = const",
                key=SECTION_INDEX_KEY,
            )
        )
    with controls_top[1]:
        section_value = float(
            st.number_input(
                "Plane value",
                value=float(st.session_state.get(SECTION_VALUE_KEY, 0.0)),
                step=0.1,
                format="%.6f",
                key=SECTION_VALUE_KEY,
            )
        )
    with controls_top[2]:
        direction_label = str(
            st.selectbox(
                "Direction",
                options=[
                    DIRECTION_LABEL_BY_VALUE[+1],
                    DIRECTION_LABEL_BY_VALUE[0],
                    DIRECTION_LABEL_BY_VALUE[-1],
                ],
                key=DIRECTION_KEY,
            )
        )
    with controls_top[3]:
        max_points = int(
            st.number_input(
                "Max points",
                min_value=100,
                max_value=20000,
                value=int(st.session_state.get(MAX_POINTS_KEY, 5000)),
                step=100,
                key=MAX_POINTS_KEY,
            )
        )

    pair_options = axis_pair_options(n_dim, section_index, preferred_axes=preferred_axes)
    if not pair_options:
        st.info("Not enough remaining dimensions to render a 2D Poincaré map.")
        return

    pair_sig = (n_dim, section_index, tuple(int(v) for v in preferred_axes))
    if st.session_state.get(AXIS_PAIR_SIG_KEY) != pair_sig or st.session_state.get(AXIS_PAIR_KEY) not in pair_options:
        st.session_state[AXIS_PAIR_KEY] = pair_options[0]
        st.session_state[AXIS_PAIR_SIG_KEY] = pair_sig

    axis_pair = tuple(
        int(v)
        for v in st.selectbox(
            "Map axes",
            options=list(pair_options),
            format_func=lambda pair: _pair_label(var_names, pair),
            key=AXIS_PAIR_KEY,
        )
    )

    cfg = PoincareMapConfig(
        section_index=section_index,
        section_value=section_value,
        direction=int(DIRECTION_VALUE_BY_LABEL[direction_label]),
        axis_pair=(axis_pair[0], axis_pair[1]),
        max_points=max_points,
    )

    try:
        result = compute_poincare_map(t, y_arr, cfg, preferred_axes=preferred_axes)
    except Exception as exc:
        st.warning(f"Poincaré map computation failed: {exc}")
        return

    if result.hit_count == 0:
        st.warning("No section crossings found with the current plane settings.")
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    fig.set_dpi(150)
    ax.scatter(
        result.x_values,
        result.y_values,
        s=7,
        color="crimson",
        alpha=0.82,
        edgecolors="none",
    )
    section_axis_name = _axis_name(var_names, result.section_index)
    axis_x_name = _axis_name(var_names, result.axis_pair[0])
    axis_y_name = _axis_name(var_names, result.axis_pair[1])
    plot_title = f"Poincaré map on {section_axis_name}={result.section_value:g}"
    if title_prefix:
        plot_title = f"{title_prefix} – {plot_title}"
    ax.set_title(plot_title)
    ax.set_xlabel(axis_x_name)
    ax.set_ylabel(axis_y_name)
    ax.grid(True, linewidth=0.3)
    st.pyplot(fig, clear_figure=True)
    st.caption(
        f"Hits found: {result.hit_count} | displayed: {result.display_count} | direction: {result.direction:+d}"
    )
