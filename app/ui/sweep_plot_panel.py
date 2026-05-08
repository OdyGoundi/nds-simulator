import json

import numpy as np
import pandas as pd
import streamlit as st

from app.export_utils import build_sweep_config
from app.helpers import decimate_indices, downsample_xy
from app.logic.reservoir_sampling import ensure_xy_reservoir, get_xy_reservoir_points
from app.plotting import (
    BIFURCATION_DEFAULTS,
    LYAPUNOV_DEFAULTS,
    axis_bounds as _axis_bounds,
    plot_bifurcation,
    plot_lyapunov_sweep,
    render_plot_settings_button,
)
from app.state import (
    BifPlotKeys,
    LyaPlotKeys,
    LyapunovDataKeys,
    PlotSettingsKeys,
    SweepDataKeys,
)
from app.services.sweep_state_service import (
    MAX_BIF_RESERVOIR_POINTS,
    MAX_SWEEP_ROWS_IN_MEMORY,
    _apply_sweep_config_to_state,
)
from app.ui.sweep_controls import SweepControlsResult


MAX_BIF_PLOT_POINTS = MAX_SWEEP_ROWS_IN_MEMORY + MAX_BIF_RESERVOIR_POINTS
MAX_LYA_PLOT_POINTS = 200_000


def _save_sweep_config(ctrl: SweepControlsResult) -> None:
    sweep_config = build_sweep_config(
        app_name=ctrl.app_name,
        repo_root=ctrl.repo_root,
        system=ctrl.system,
        integration=ctrl.integration,
        initial=ctrl.initial,
        solve_tols=ctrl.solve_tols,
        sweep_param=ctrl.sweep_param,
        sweep_start=ctrl.sweep_start,
        sweep_stop=ctrl.sweep_stop,
        sweep_step=ctrl.sweep_step,
        dt_sweep=ctrl.dt_sweep,
        tf_sweep=ctrl.tf_sweep,
        section_var=ctrl.section_var,
        section_index=ctrl.section_index,
        section_value=ctrl.section_value,
        section_expr=str(ctrl.section_expr or ""),
        direction=ctrl.direction,
        method=ctrl.method,
        tol=ctrl.tol,
        output_var=ctrl.out_var,
        output_index=ctrl.output_index,
        observable="extrema" if ctrl.use_extrema else "poincare",
        extrema_kind=ctrl.extrema_kind,
        transient_frac=ctrl.transient_frac,
        transient_steps_est=ctrl.transient_steps_sweep,
        warm_start=ctrl.warm_start,
        early_stop=ctrl.early_stop,
        max_hits=ctrl.max_hits,
        chunk_time=ctrl.chunk_time,
        parallel_bif=ctrl.parallel_bif,
        workers_bif=ctrl.workers_bif,
        rtol_sweep=ctrl.rtol_sweep,
        atol_sweep=ctrl.atol_sweep,
        qr_interval_lya=ctrl.qr_interval_lya,
        lyapunov_transient_frac=ctrl.transient_frac_lya,
        lyapunov_transient_steps=ctrl.transient_steps_lya,
        parallel_lya=ctrl.parallel_enabled,
        workers_lya=ctrl.workers,
        clip_lyapunov=ctrl.clip_lyapunov,
        clip_min=ctrl.clip_min,
        sweep_fingerprint=ctrl.sweep_meta,
        lya_fingerprint=ctrl.lya_meta,
    )
    st.session_state[SweepDataKeys.CONFIG] = sweep_config
    st.success("Sweep configuration saved. Download from the Export tab.")


def render_sweep_plots(ctrl: SweepControlsResult, df_plot) -> None:
    # Caller is responsible for the `with tab:` context.

    with ctrl.left_col:
        st.divider()
        if df_plot is None or len(df_plot) == 0:
            st.info("No sweep data yet. Click 'Generate Bifurcation Diagram' to start.")
        else:
            if not isinstance(df_plot, pd.DataFrame):
                df_plot = pd.DataFrame(df_plot)

            x_vals = np.asarray(df_plot[ctrl.sweep_param].to_numpy(), dtype=float)
            y_vals = np.asarray(df_plot[ctrl.ycol].to_numpy(), dtype=float)
            reservoir_state = ensure_xy_reservoir(
                st.session_state.get(SweepDataKeys.RESERVOIR),
                capacity=MAX_BIF_RESERVOIR_POINTS,
            )
            st.session_state[SweepDataKeys.RESERVOIR] = reservoir_state
            x_hist_raw, y_hist_raw = get_xy_reservoir_points(reservoir_state)
            history_budget = max(0, int(MAX_BIF_PLOT_POINTS) - int(x_vals.size))
            if history_budget > 0 and x_hist_raw.size > 0:
                x_hist_plot, y_hist_plot = downsample_xy(x_hist_raw, y_hist_raw, history_budget)
            else:
                x_hist_plot = np.empty(0, dtype=float)
                y_hist_plot = np.empty(0, dtype=float)

            if x_hist_plot.size > 0:
                y_bounds_data = np.concatenate((y_hist_plot, y_vals), axis=0)
            else:
                y_bounds_data = y_vals
            x_start = float(ctrl.sweep_start)
            x_stop_data = float(st.session_state.get(SweepDataKeys.LAST_PV, ctrl.sweep_stop))
            if not np.isfinite(x_stop_data):
                x_stop_data = float(ctrl.sweep_stop)
            if x_stop_data <= x_start:
                x_stop_data = x_start + max(1e-12, float(abs(ctrl.sweep_step)))
            x_auto = (float(x_start), float(x_stop_data))
            y_auto = _axis_bounds(y_bounds_data)

            bif_bounds_sig = (
                str(ctrl.sweep_param),
                str(ctrl.ycol),
                int(x_vals.size),
                int(x_hist_plot.size),
                int(st.session_state[SweepDataKeys.RESERVOIR].get("seen", 0)),
                float(x_auto[0]),
                float(x_auto[1]),
                float(y_auto[0]),
                float(y_auto[1]),
            )
            if st.session_state.get(BifPlotKeys.BOUNDS_SIG) != bif_bounds_sig:
                st.session_state[BifPlotKeys.XLIM_MIN] = float(x_auto[0])
                st.session_state[BifPlotKeys.XLIM_MAX] = float(x_auto[1])
                st.session_state[BifPlotKeys.YLIM_MIN] = float(y_auto[0])
                st.session_state[BifPlotKeys.YLIM_MAX] = float(y_auto[1])
                st.session_state[BifPlotKeys.BOUNDS_SIG] = bif_bounds_sig

            x_view = (
                float(st.session_state.get(BifPlotKeys.XLIM_MIN, x_auto[0])),
                float(st.session_state.get(BifPlotKeys.XLIM_MAX, x_auto[1])),
            )
            y_view = (
                float(st.session_state.get(BifPlotKeys.YLIM_MIN, y_auto[0])),
                float(st.session_state.get(BifPlotKeys.YLIM_MAX, y_auto[1])),
            )
            if not (x_view[0] < x_view[1] and y_view[0] < y_view[1]):
                st.warning("Invalid bifurcation axis limits detected. Reverting to data bounds.")
                x_view = x_auto
                y_view = y_auto
                st.session_state[BifPlotKeys.XLIM_MIN] = float(x_auto[0])
                st.session_state[BifPlotKeys.XLIM_MAX] = float(x_auto[1])
                st.session_state[BifPlotKeys.YLIM_MIN] = float(y_auto[0])
                st.session_state[BifPlotKeys.YLIM_MAX] = float(y_auto[1])

            st.markdown("**Axis limits (view window)**")
            lim_c1, lim_c2, lim_c3, lim_c4 = st.columns([1, 1, 1, 1], gap="small")
            with lim_c1:
                st.number_input(
                    f"{ctrl.sweep_param} min",
                    format="%.6f",
                    key=BifPlotKeys.XLIM_MIN,
                )
            with lim_c2:
                st.number_input(
                    f"{ctrl.sweep_param} max",
                    format="%.6f",
                    key=BifPlotKeys.XLIM_MAX,
                )
            with lim_c3:
                st.number_input(
                    f"{ctrl.out_var} min",
                    format="%.6f",
                    key=BifPlotKeys.YLIM_MIN,
                )
            with lim_c4:
                st.number_input(
                    f"{ctrl.out_var} max",
                    format="%.6f",
                    key=BifPlotKeys.YLIM_MAX,
                )
            st.caption(
                f"Default bounds: {ctrl.sweep_param} [{x_auto[0]:.4g}, {x_auto[1]:.4g}], "
                f"{ctrl.out_var} [{y_auto[0]:.4g}, {y_auto[1]:.4g}]"
            )
            expected_params = 0
            if float(ctrl.sweep_step) > 0:
                expected_params = int(
                    np.floor((float(x_auto[1]) - float(x_auto[0])) / float(ctrl.sweep_step) + 1e-12)
                ) + 1
            observed_params = int(np.unique(np.round(x_vals, 12)).size)
            if expected_params > 0 and observed_params < expected_params:
                missing_params = int(expected_params - observed_params)
                st.caption(
                    f"Recent-buffer coverage: {observed_params}/{expected_params} parameter values have hits "
                    f"(missing {missing_params})."
                )

            if ctrl.use_extrema:
                ylabel = f"{ctrl.out_var} local extrema ({ctrl.extrema_kind})"
            else:
                section_label = str(ctrl.section_expr).strip()
                if section_label:
                    ylabel = f"{ctrl.out_var} on section ({section_label})"
                else:
                    ylabel = f"{ctrl.out_var} on section ({ctrl.section_var}={ctrl.section_value})"

            bif_settings = render_plot_settings_button(
                PlotSettingsKeys.BIFURCATION_TAB3,
                default=BIFURCATION_DEFAULTS,
                has_square=False,
            )
            fig = plot_bifurcation(
                x_vals=x_vals,
                y_vals=y_vals,
                x_history=x_hist_plot if x_hist_plot.size > 0 else None,
                y_history=y_hist_plot if x_hist_plot.size > 0 else None,
                boundaries=st.session_state.get(SweepDataKeys.BOUNDARIES, []),
                xlabel=ctrl.sweep_param,
                ylabel=ylabel,
                x_view=x_view,
                y_view=y_view,
                settings=bif_settings,
            )
            st.pyplot(fig, clear_figure=True)
            total_plotted = int(x_hist_plot.size + x_vals.size)
            st.caption(
                f"Plotted points: recent {len(x_vals):,} + reservoir {len(x_hist_plot):,} = {total_plotted:,}"
            )

            last_pv = st.session_state.get(SweepDataKeys.LAST_PV, None)
            if last_pv is not None:
                st.caption(f"Accumulated sweep up to {ctrl.sweep_param} = {float(last_pv):g} | Rows: {len(df_plot)}")
            else:
                try:
                    st.caption(f"Accumulated sweep | Rows: {len(df_plot)}")
                except Exception:
                    pass
            if bool(st.session_state.get(SweepDataKeys.ROWS_CLIPPED, False)):
                reservoir_seen = int(st.session_state[SweepDataKeys.RESERVOIR].get("seen", 0))
                st.caption(
                    f"Stored sweep rows are capped at {MAX_SWEEP_ROWS_IN_MEMORY:,} (recent full-resolution). "
                    f"Dropped history is kept as a reservoir sample up to {MAX_BIF_RESERVOIR_POINTS:,} "
                    f"points from {reservoir_seen:,} dropped rows."
                )

    with ctrl.right_col:
        st.divider()
        lya_data = st.session_state.get(LyapunovDataKeys.ACC_DATA, None)
        if lya_data is None:
            st.info("No Lyapunov sweep data yet. Click 'Generate Lyapunov Diagram'.")
        else:
            param_vals = lya_data.get("param_vals", np.array([], dtype=float))
            lambdas_arr = lya_data.get("lambdas", np.zeros((0, len(ctrl.var_names))))
            errors = lya_data.get("errors", [])

            if param_vals.size == 0 or lambdas_arr.size == 0:
                st.info("No Lyapunov sweep data yet. Click 'Generate Lyapunov Diagram'.")
            else:
                plot_lambdas = np.array(lambdas_arr, dtype=float)
                if ctrl.clip_lyapunov:
                    plot_lambdas = np.maximum(plot_lambdas, float(ctrl.clip_min))
                lya_idx = decimate_indices(int(np.asarray(param_vals).size), MAX_LYA_PLOT_POINTS)
                param_vals_plot = np.asarray(param_vals, dtype=float)[lya_idx]
                if plot_lambdas.ndim == 1:
                    plot_lambdas_plot = np.asarray(plot_lambdas, dtype=float)[lya_idx]
                else:
                    plot_lambdas_plot = np.asarray(plot_lambdas, dtype=float)[lya_idx, :]
                if np.asarray(plot_lambdas_plot).ndim == 1:
                    plot_lambdas_plot = np.asarray(plot_lambdas_plot, dtype=float)[:, None]

                st.markdown("**Lyapunov exponents**")
                x_auto = _axis_bounds(np.asarray(param_vals, dtype=float))
                y_auto = _axis_bounds(np.asarray(plot_lambdas, dtype=float))
                lya_bounds_sig = (
                    str(ctrl.sweep_param),
                    bool(ctrl.clip_lyapunov),
                    float(ctrl.clip_min),
                    int(plot_lambdas.shape[0]),
                    int(plot_lambdas.shape[1]) if plot_lambdas.ndim == 2 else 0,
                    float(x_auto[0]),
                    float(x_auto[1]),
                    float(y_auto[0]),
                    float(y_auto[1]),
                )
                if st.session_state.get(LyaPlotKeys.BOUNDS_SIG) != lya_bounds_sig:
                    st.session_state[LyaPlotKeys.XLIM_MIN] = float(x_auto[0])
                    st.session_state[LyaPlotKeys.XLIM_MAX] = float(x_auto[1])
                    st.session_state[LyaPlotKeys.YLIM_MIN] = float(y_auto[0])
                    st.session_state[LyaPlotKeys.YLIM_MAX] = float(y_auto[1])
                    st.session_state[LyaPlotKeys.BOUNDS_SIG] = lya_bounds_sig

                x_view = (
                    float(st.session_state.get(LyaPlotKeys.XLIM_MIN, x_auto[0])),
                    float(st.session_state.get(LyaPlotKeys.XLIM_MAX, x_auto[1])),
                )
                y_view = (
                    float(st.session_state.get(LyaPlotKeys.YLIM_MIN, y_auto[0])),
                    float(st.session_state.get(LyaPlotKeys.YLIM_MAX, y_auto[1])),
                )
                if not (x_view[0] < x_view[1] and y_view[0] < y_view[1]):
                    st.warning("Invalid Lyapunov axis limits detected. Reverting to data bounds.")
                    x_view = x_auto
                    y_view = y_auto
                    st.session_state[LyaPlotKeys.XLIM_MIN] = float(x_auto[0])
                    st.session_state[LyaPlotKeys.XLIM_MAX] = float(x_auto[1])
                    st.session_state[LyaPlotKeys.YLIM_MIN] = float(y_auto[0])
                    st.session_state[LyaPlotKeys.YLIM_MAX] = float(y_auto[1])

                lya_settings = render_plot_settings_button(
                    PlotSettingsKeys.LYAPUNOV_TAB3,
                    default=LYAPUNOV_DEFAULTS,
                    has_color=False,
                    has_square=False,
                )
                fig_lya = plot_lyapunov_sweep(
                    param_vals=param_vals_plot,
                    lambdas=plot_lambdas_plot,
                    boundaries=st.session_state.get(LyapunovDataKeys.BOUNDARIES, []),
                    xlabel=ctrl.sweep_param,
                    x_view=x_view,
                    y_view=y_view,
                    settings=lya_settings,
                )
                st.pyplot(fig_lya, clear_figure=True)
                st.caption(f"Plotted points: {len(param_vals_plot)}/{len(param_vals)}")

                st.markdown("**Axis limits (view window)**")
                lim_c1, lim_c2, lim_c3, lim_c4 = st.columns([1, 1, 1, 1], gap="small")
                with lim_c1:
                    st.number_input(
                        f"{ctrl.sweep_param} min",
                        format="%.6f",
                        key=LyaPlotKeys.XLIM_MIN,
                    )
                with lim_c2:
                    st.number_input(
                        f"{ctrl.sweep_param} max",
                        format="%.6f",
                        key=LyaPlotKeys.XLIM_MAX,
                    )
                with lim_c3:
                    st.number_input(
                        "lambda min",
                        format="%.6f",
                        key=LyaPlotKeys.YLIM_MIN,
                    )
                with lim_c4:
                    st.number_input(
                        "lambda max",
                        format="%.6f",
                        key=LyaPlotKeys.YLIM_MAX,
                    )
                st.caption(
                    f"Default bounds: {ctrl.sweep_param} [{x_auto[0]:.4g}, {x_auto[1]:.4g}], "
                    f"lambda [{y_auto[0]:.4g}, {y_auto[1]:.4g}]"
                )

                if ctrl.clip_lyapunov:
                    st.caption(f"Clipped exponents below {float(ctrl.clip_min):g} for plotting.")
                if errors:
                    st.caption(f"Lyapunov sweep failures: {len(errors)}")
                last_pv = st.session_state.get(LyapunovDataKeys.LAST_PV, None)
                if last_pv is not None:
                    st.caption(f"Accumulated Lyapunov sweep up to {ctrl.sweep_param} = {float(last_pv):g}")

    st.divider()
    st.markdown("**Configuration**")
    _, cfg_center, _ = st.columns([1, 1.8, 1], gap="large")
    with cfg_center:
        save_sweep_cfg = st.button(
            "Save configuration",
            key="save_cfg_lya_tab3",
            use_container_width=True,
        )
        sweep_cfg_upload = st.file_uploader(
            "Upload SweepParamConfig.json",
            type=["json"],
            key="upload_sweep_cfg_tab3",
            help="Load settings from a previously exported sweep configuration file.",
        )
        apply_sweep_cfg = st.button(
            "Apply uploaded configuration",
            key="apply_sweep_cfg_tab3",
            disabled=sweep_cfg_upload is None,
            use_container_width=True,
        )

    if apply_sweep_cfg and sweep_cfg_upload is not None:
        try:
            loaded_sweep_cfg = json.loads(sweep_cfg_upload.getvalue().decode("utf-8"))
            if not isinstance(loaded_sweep_cfg, dict):
                raise ValueError("JSON root must be an object.")
            _apply_sweep_config_to_state(
                loaded_sweep_cfg,
                sweep_choices=list(ctrl.sweep_choices),
                var_names=list(ctrl.var_names),
                t0_default=float(ctrl.t0),
                tf_default=float(ctrl.tf),
                dt_default=float(ctrl.dt),
            )
            st.session_state[SweepDataKeys.CONFIG] = loaded_sweep_cfg
            st.success("Sweep configuration loaded. Applying settings...")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to load SweepParamConfig: {exc}")

    if save_sweep_cfg:
        _save_sweep_config(ctrl)
