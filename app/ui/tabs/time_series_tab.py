from __future__ import annotations

import numpy as np
import streamlit as st

from app.helpers import downsample_trajectory
from app.plotting import LINE_COLORS, plot_single_variable, plot_time_series
from app.state import TimeSeriesKeys
from app.ui.tabs.phase_tab import PhaseTabResult


def render_time_series_tab(tab, *, phase_result: PhaseTabResult) -> None:
    t_plot = phase_result.t_plot
    y_plot = phase_result.y_plot
    var_names = phase_result.var_names
    max_plot_points_i = phase_result.max_plot_points
    integration = phase_result.integration
    system_label = phase_result.system_label

    with tab:
        st.markdown("**Time series (post-transient)**")

        t_min = float(t_plot[0])
        t_max = float(t_plot[-1])
        time_step_ui = max(float(integration.dt), 1e-6)
        current_ts_range = (float(t_min), float(t_max))

        if TimeSeriesKeys.WINDOW_START not in st.session_state:
            st.session_state[TimeSeriesKeys.WINDOW_START] = t_min
        if TimeSeriesKeys.WINDOW_END not in st.session_state:
            st.session_state[TimeSeriesKeys.WINDOW_END] = t_max
        if TimeSeriesKeys.WINDOW_RANGE not in st.session_state:
            st.session_state[TimeSeriesKeys.WINDOW_RANGE] = current_ts_range

        # Reset to full range whenever the available integration window changes.
        if tuple(st.session_state.get(TimeSeriesKeys.WINDOW_RANGE, ())) != current_ts_range:
            st.session_state[TimeSeriesKeys.WINDOW_START] = t_min
            st.session_state[TimeSeriesKeys.WINDOW_END] = t_max
            st.session_state[TimeSeriesKeys.WINDOW_RANGE] = current_ts_range

        # Keep persisted values inside current bounds when t0/tf/transient changes.
        st.session_state[TimeSeriesKeys.WINDOW_START] = min(
            max(float(st.session_state[TimeSeriesKeys.WINDOW_START]), t_min),
            t_max,
        )
        st.session_state[TimeSeriesKeys.WINDOW_END] = min(
            max(float(st.session_state[TimeSeriesKeys.WINDOW_END]), t_min),
            t_max,
        )

        tw_header_col, twc1, twc2 = st.columns([1.2, 1, 1], gap="small")
        with tw_header_col:
            st.markdown("**Time window**")
            st.caption(
                f"Available range from Tab 1 integration: [{t_min:.3f}, {t_max:.3f}]"
            )
        with twc1:
            t_view_start = st.number_input(
                "start time",
                min_value=t_min,
                max_value=t_max,
                value=float(st.session_state[TimeSeriesKeys.WINDOW_START]),
                step=time_step_ui,
                format="%.6f",
                key=TimeSeriesKeys.WINDOW_START,
            )
        with twc2:
            t_view_end = st.number_input(
                "end time",
                min_value=t_min,
                max_value=t_max,
                value=float(st.session_state[TimeSeriesKeys.WINDOW_END]),
                step=time_step_ui,
                format="%.6f",
                key=TimeSeriesKeys.WINDOW_END,
            )

        if float(t_view_start) >= float(t_view_end):
            st.warning("Invalid time window: 'start time' must be smaller than 'end time'. Showing full range.")
            t_view_start = t_min
            t_view_end = t_max

        ts_mask = (t_plot >= float(t_view_start)) & (t_plot <= float(t_view_end))
        if int(np.count_nonzero(ts_mask)) < 2:
            st.warning("Time window contains fewer than 2 samples. Showing full range.")
            ts_mask = np.ones_like(t_plot, dtype=bool)

        t_ts = t_plot[ts_mask]
        y_ts = y_plot[:, ts_mask]
        t_ts_plot, y_ts_plot = downsample_trajectory(t_ts, y_ts, max_plot_points_i)
        st.caption(
            f"Showing t in [{float(t_view_start):.3f}, {float(t_view_end):.3f}] | "
            f"samples: {len(t_ts_plot)}/{len(t_ts)} (stored window) | stored total: {len(t_plot)}"
        )

        default_sel = [0] if len(var_names) > 0 else []
        selected_names = st.multiselect(
            "Select variable(s)",
            options=var_names,
            default=[var_names[i] for i in default_sel] if default_sel else [],
        )

        if not selected_names:
            st.info("Select at least one variable to plot.")
        else:
            selected_indices = [var_names.index(name) for name in selected_names]

            fig_ts = plot_time_series(
                t=t_ts_plot,
                y=y_ts_plot,
                indices=selected_indices,
                var_names=var_names,
                title=f"{system_label} – time series",
            )
            st.pyplot(fig_ts, clear_figure=True)

        selected_names = st.multiselect(
            "Select variable(s) to display (one plot per variable)",
            options=var_names,
            default=[],
        )

        if selected_names:
            plot_indices = [var_names.index(name) for name in selected_names]
        else:
            plot_indices = list(range(len(var_names)))

        for plot_pos, var_idx in enumerate(plot_indices):
            fig = plot_single_variable(
                t=t_ts_plot,
                y_var=y_ts_plot[var_idx, :],
                var_name=var_names[var_idx],
                title=f"{system_label} – {var_names[var_idx]} vs time",
                color=LINE_COLORS[plot_pos % len(LINE_COLORS)],
            )
            st.pyplot(fig, clear_figure=True)
