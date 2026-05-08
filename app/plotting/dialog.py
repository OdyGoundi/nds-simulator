"""Per-plot settings popup. ``render_plot_settings_button`` wires a button +
``st.dialog`` (with an inline ``st.expander`` fallback for older Streamlit) and
persists a ``PlotSettings`` value under one session-state key."""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import streamlit as st

from app.plotting.settings import PlotSettings


def get_plot_settings(state_key: str, default: PlotSettings) -> PlotSettings:
    val = st.session_state.get(state_key)
    return val if isinstance(val, PlotSettings) else default


def set_plot_settings(state_key: str, settings: PlotSettings) -> None:
    st.session_state[state_key] = settings


def render_plot_settings_button(
    state_key: str,
    default: PlotSettings,
    *,
    label: str = "Open plot settings",
    has_color: bool = True,
    has_square: bool = True,
    button_key: Optional[str] = None,
) -> PlotSettings:
    """Render the trigger button + popup. Returns the current persisted settings.

    ``has_color`` / ``has_square`` hide the corresponding widget when the plot
    can't honor that setting (e.g. multi-line Lyapunov, 3D phase, time series)."""
    open_flag_key = f"{state_key}__open"
    if open_flag_key not in st.session_state:
        st.session_state[open_flag_key] = False

    if st.button(label, key=button_key or f"{state_key}__btn"):
        st.session_state[open_flag_key] = True

    if st.session_state[open_flag_key]:
        dialog_decorator = getattr(st, "dialog", None)
        if dialog_decorator is None:
            with st.expander("Plot settings", expanded=True):
                _render_settings_widgets(
                    state_key, default, has_color=has_color, has_square=has_square
                )
                if st.button("Close", key=f"{state_key}__close_inline"):
                    st.session_state[open_flag_key] = False
        else:

            @dialog_decorator("Plot settings")
            def _open_dialog() -> None:
                _render_settings_widgets(
                    state_key, default, has_color=has_color, has_square=has_square
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Reset to defaults", key=f"{state_key}__reset"):
                        set_plot_settings(state_key, default)
                        st.rerun()
                with col2:
                    if st.button("Close", key=f"{state_key}__close"):
                        st.session_state[open_flag_key] = False
                        st.rerun()

            _open_dialog()
            st.session_state[open_flag_key] = False

    return get_plot_settings(state_key, default)


def _render_settings_widgets(
    state_key: str,
    default: PlotSettings,
    *,
    has_color: bool,
    has_square: bool,
) -> None:
    current = get_plot_settings(state_key, default)

    new_color = current.color
    if has_color:
        new_color = st.color_picker(
            "Color",
            value=current.color,
            key=f"{state_key}__color",
        )

    new_linewidth = st.slider(
        "Line width",
        min_value=0.05,
        max_value=3.0,
        value=float(min(max(current.linewidth, 0.05), 3.0)),
        step=0.05,
        key=f"{state_key}__lw",
    )

    new_grid = st.checkbox(
        "Show grid",
        value=bool(current.grid),
        key=f"{state_key}__grid",
    )

    new_tick_density = st.slider(
        "Tick density (max ticks per axis)",
        min_value=2,
        max_value=20,
        value=int(min(max(current.tick_density, 2), 20)),
        step=1,
        key=f"{state_key}__ticks",
    )

    new_decimals = st.slider(
        "Tick decimals",
        min_value=0,
        max_value=6,
        value=int(min(max(current.decimals, 0), 6)),
        step=1,
        key=f"{state_key}__dec",
    )

    new_square = current.square_axis
    if has_square:
        new_square = st.checkbox(
            "Square axis (1:1 aspect)",
            value=bool(current.square_axis),
            key=f"{state_key}__sq",
        )

    set_plot_settings(
        state_key,
        replace(
            current,
            color=str(new_color),
            linewidth=float(new_linewidth),
            grid=bool(new_grid),
            tick_density=int(new_tick_density),
            decimals=int(new_decimals),
            square_axis=bool(new_square),
        ),
    )
