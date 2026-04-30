from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import streamlit as st

from app.cache import solve_cached
from app.export_utils import build_static_config
from app.helpers import (
    apply_transient_cut,
    downsample_trajectory,
    parse_list_of_floats,
)
from app.logic.lyapunov_cached import compute_lyapunov_cached
from app.params import (
    CustomSystemDefinition,
    HenonHeilesParams,
    InitialConditions,
    IntegrationConfig,
    LorenzParams,
    LyapunovConfig,
    RosslerParams,
    SolverTolerances,
    SystemConfig,
)
from app.plots import plot_phase_2d, plot_phase_3d
from app.state import (
    MAX_PLOT_POINTS_DEFAULT,
    MAX_PLOT_POINTS_UI_MAX,
    MAX_STORE_STEPS_DEFAULT,
    PENDING_STATIC_CFG_KEY,
    PHASE_LINEWIDTH_DEFAULT,
    STATIC_CFG_APPLY_ERROR_KEY,
    STATIC_CFG_APPLY_SUCCESS_KEY,
)
from app.state.apply_config import apply_state_values, to_float
from app.ui.poincare_map_panel import render_poincare_map_panel


@dataclass(frozen=True)
class PhaseTabResult:
    system: SystemConfig
    integration: IntegrationConfig
    initial: InitialConditions
    solve_tols: SolverTolerances
    lyapunov_cfg: LyapunovConfig
    t: np.ndarray
    y: np.ndarray
    t_plot: np.ndarray
    y_plot: np.ndarray
    var_names: List[str]
    system_label: str
    max_plot_points: int
    transient_steps: int
    transient_cut_time: float
    plot_mode: str
    x_idx: int
    y_idx: int
    z_idx: int
    phase_linewidth: float
    lyapunov_transient_frac: float
    lyapunov_transient_steps: int
    qr_interval: float


def _axis_bounds(values: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        delta = max(1e-6, 0.05 * max(1.0, abs(vmin)))
        return vmin - delta, vmax + delta
    return vmin, vmax


def _square_xy_bounds(
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x0, x1 = float(x_bounds[0]), float(x_bounds[1])
    y0, y1 = float(y_bounds[0]), float(y_bounds[1])
    dx = max(1e-12, x1 - x0)
    dy = max(1e-12, y1 - y0)
    half_span = 0.5 * max(dx, dy)
    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)
    return (x_mid - half_span, x_mid + half_span), (y_mid - half_span, y_mid + half_span)


def render_phase_tab(
    tab: Any,
    *,
    system_key: str,
    n_vars: int,
    solver_kind_effective: str,
    y0_text: str,
    var_names: List[str],
    eq_lines: List[str],
    params_text: str,
    custom_auto_jac: bool,
    custom_use_jac: bool,
    system_label: str,
    app_name: str,
    repo_root: Path,
) -> PhaseTabResult:
    sigma = rho = beta = 0.0
    ross_a = ross_b = ross_c = 0.0
    hh_lambda = 1.0

    with tab:
        phase_col_controls, phase_col_plot = st.columns([1, 2], gap="large")

        with phase_col_controls:
            st.header("Integration")
            apply_state_values(
                {"t0_tab1": 0.0, "tf_tab1": 50.0, "dt_tab1": 0.01},
                only_missing=True,
            )
            t0 = st.number_input("initial time", step=1.0, key="t0_tab1")
            tf = st.number_input("final time", step=1.0, key="tf_tab1")
            dt = st.number_input("time step", step=0.01, format="%.5f", key="dt_tab1")

            st.divider()
            st.header("System parameters")

            if system_key == "lorenz":
                apply_state_values(
                    {"sigma": 10.0, "rho": 28.0, "beta": float(8.0 / 3.0)},
                    only_missing=True,
                )
                sigma = st.number_input("sigma", step=0.1, format="%.3f", key="sigma")
                rho = st.number_input("rho", step=0.5, format="%.3f", key="rho")
                beta = st.number_input("beta", step=0.05, format="%.4f", key="beta")
            elif system_key == "rossler":
                apply_state_values(
                    {"ross_a": 0.2, "ross_b": 0.2, "ross_c": 5.7},
                    only_missing=True,
                )
                ross_a = st.number_input("a", step=0.01, format="%.4f", key="ross_a")
                ross_b = st.number_input("b", step=0.01, format="%.4f", key="ross_b")
                ross_c = st.number_input("c", step=0.1, format="%.3f", key="ross_c")
            elif system_key == "henon_heiles":
                apply_state_values({"hh_lambda": 1.0}, only_missing=True)
                hh_lambda = st.number_input("lambda", step=0.05, format="%.4f", key="hh_lambda")
            else:
                st.caption("Custom: parameters are defined above.")

            st.divider()
            st.header("Plot settings")
            apply_state_values({"plot_mode_tab1": "2D phase plane"}, only_missing=True)
            plot_mode = st.selectbox(
                "Plot mode",
                ["2D phase plane", "3D phase plot"],
                key="plot_mode_tab1",
            )

            tc_c1, tc_c2 = st.columns([1.2, 1], gap="small")
            with tc_c1:
                apply_state_values({"transient_cut_time_tab1": 0.0}, only_missing=True)
                transient_cut_time = st.number_input(
                    "Transient cut (time to skip)",
                    min_value=0.0,
                    step=1.0,
                    format="%.6f",
                    key="transient_cut_time_tab1",
                    help=(
                        "Skips trajectory samples from t0 up to t0 + this duration before plotting/export. "
                        "The equivalent number of steps is shown on the right. Does not affect Lyapunov."
                    ),
                )
            with tc_c2:
                dt_abs = max(abs(float(dt)), 1e-12)
                transient_steps = int(max(0.0, float(transient_cut_time)) / dt_abs)
                st.metric("Equivalent transient steps", transient_steps)

            perf_c1, perf_c2 = st.columns([1, 1], gap="small")
            with perf_c1:
                apply_state_values({"max_plot_points_tab1": MAX_PLOT_POINTS_DEFAULT}, only_missing=True)
                max_plot_points = st.number_input(
                    "Max points per plot",
                    min_value=10_000,
                    max_value=MAX_PLOT_POINTS_UI_MAX,
                    step=10_000,
                    key="max_plot_points_tab1",
                    help=(
                        "Uniformly downsamples trajectories for plotting only. "
                        "Use larger values for denser, publication-oriented phase plots."
                    ),
                )
            with perf_c2:
                apply_state_values({"max_store_steps_tab1": MAX_STORE_STEPS_DEFAULT}, only_missing=True)
                max_store_steps_ui = st.number_input(
                    "Max stored trajectory samples (0=all)",
                    min_value=0,
                    max_value=5_000_000,
                    step=50_000,
                    key="max_store_steps_tab1",
                    help=(
                        "Limits in-memory trajectory samples to avoid Streamlit Cloud crashes. "
                        "Set 0 to keep the full trajectory in memory when RAM allows."
                    ),
                )

            st.divider()
            st.header("Lyapunov exponents calculation settings")
            apply_state_values({"qr_interval_tab1": 0.1}, only_missing=True)
            qr_interval = st.number_input(
                "QR interval (time)",
                min_value=1e-6,
                step=0.01,
                format="%.4f",
                key="qr_interval_tab1",
                help="Time between orthonormalizations during Lyapunov computation.",
            )
            lya_c1, lya_c2 = st.columns([1, 1], gap="small")
            with lya_c1:
                apply_state_values({"lya_transient_frac_tab1": 0.30}, only_missing=True)
                lyapunov_transient_frac = st.slider(
                    "Lyapunov transient fraction",
                    min_value=0.0,
                    max_value=0.99,
                    step=0.05,
                    key="lya_transient_frac_tab1",
                    help="Fraction of integration steps discarded before Lyapunov accumulation.",
                )
            with lya_c2:
                n_steps_est_lya = int(max(1.0, (float(tf) - float(t0)) / float(dt)))
                transient_steps_lya = int(lyapunov_transient_frac * n_steps_est_lya)
                st.metric("Lyapunov transient steps (estimated)", transient_steps_lya)
            compute_lya_btn = st.button("Compute Lyapunov exponents", key="compute_lya_tab1")

            st.divider()
            st.markdown("**Axis selection**")
            axis_options = [(f"{name} (index {i})", i) for i, name in enumerate(var_names)]
            idx_list = [o[1] for o in axis_options]
            apply_state_values({"phase_x_idx_tab1": 0}, only_missing=True)
            x_idx = st.selectbox(
                "x-axis",
                options=idx_list,
                format_func=lambda i: axis_options[i][0],
                key="phase_x_idx_tab1",
            )
            y_default = 1 if len(idx_list) > 1 else 0
            apply_state_values({"phase_y_idx_tab1": y_default}, only_missing=True)
            y_idx = st.selectbox(
                "y-axis",
                options=idx_list,
                format_func=lambda i: axis_options[i][0],
                key="phase_y_idx_tab1",
            )
            z_idx = 2 if len(idx_list) > 2 else 0
            if plot_mode == "3D phase plot":
                apply_state_values({"phase_z_idx_tab1": z_idx}, only_missing=True)
                z_idx = st.selectbox(
                    "z-axis",
                    options=idx_list,
                    format_func=lambda i: axis_options[i][0],
                    key="phase_z_idx_tab1",
                )

            st.divider()
            st.header("Solver tolerances")
            apply_state_values({"rtol": 1e-6, "atol": 1e-8}, only_missing=True)
            rtol = st.number_input(
                "relative tolerance (rtol)",
                min_value=0.0,
                step=1e-6,
                format="%.1e",
                key="rtol",
            )
            atol = st.number_input(
                "absolute tolerance (atol)",
                min_value=0.0,
                step=1e-8,
                format="%.1e",
                key="atol",
            )
            if str(solver_kind_effective) not in ("rk45", "dop853", "ivp"):
                st.caption("Note: rtol/atol are used only by RK45/DOP853.")
            solve_tols = SolverTolerances(rtol=float(rtol), atol=float(atol))

        st.divider()
        st.markdown("**Configuration**")
        static_cfg_apply_success = st.session_state.pop(STATIC_CFG_APPLY_SUCCESS_KEY, None)
        if static_cfg_apply_success:
            st.success(str(static_cfg_apply_success))
        static_cfg_apply_error = st.session_state.pop(STATIC_CFG_APPLY_ERROR_KEY, None)
        if static_cfg_apply_error:
            st.error(f"Failed to load StaticParamsConfig: {static_cfg_apply_error}")
        _, cfg_center_tab1, _ = st.columns([1, 1.8, 1], gap="large")
        with cfg_center_tab1:
            save_static_cfg = st.button(
                "Save configuration",
                key="save_static_cfg_tab1",
                use_container_width=True,
            )
            static_cfg_upload = st.file_uploader(
                "Upload StaticParamsConfig.json",
                type=["json"],
                key="upload_static_cfg_tab1",
                help="Load settings from a previously exported static configuration file.",
            )
            apply_static_cfg = st.button(
                "Apply uploaded configuration",
                key="apply_static_cfg_tab1",
                disabled=static_cfg_upload is None,
                use_container_width=True,
            )

        if apply_static_cfg and static_cfg_upload is not None:
            try:
                loaded_static_cfg = json.loads(static_cfg_upload.getvalue().decode("utf-8"))
                if not isinstance(loaded_static_cfg, dict):
                    raise ValueError("JSON root must be an object.")
                st.session_state.pop(STATIC_CFG_APPLY_SUCCESS_KEY, None)
                st.session_state.pop(STATIC_CFG_APPLY_ERROR_KEY, None)
                st.session_state[PENDING_STATIC_CFG_KEY] = loaded_static_cfg
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load StaticParamsConfig: {exc}")

    # Build configs
    y0 = parse_list_of_floats(y0_text, int(n_vars), label="y0")
    effective_params_text = params_text
    if system_key == "henon_heiles":
        effective_params_text = f"lambda={float(hh_lambda)}"
    phase_linewidth = max(
        0.001,
        to_float(
            st.session_state.get("phase_linewidth_tab1", PHASE_LINEWIDTH_DEFAULT),
            PHASE_LINEWIDTH_DEFAULT,
        ),
    )
    initial = InitialConditions(tuple(float(v) for v in y0))
    max_store_steps = int(max_store_steps_ui)
    if max_store_steps <= 0:
        max_store_steps = None
    integration = IntegrationConfig(
        t0=float(t0),
        tf=float(tf),
        dt=float(dt),
        solver_kind=str(solver_kind_effective),
        max_store_steps=max_store_steps,
    )
    system = SystemConfig(
        key=system_key,
        lorenz=LorenzParams(sigma=float(sigma), rho=float(rho), beta=float(beta)),
        rossler=RosslerParams(a=float(ross_a), b=float(ross_b), c=float(ross_c)),
        henon_heiles=HenonHeilesParams(lam=float(hh_lambda)),
        custom=CustomSystemDefinition(
            var_names=tuple(var_names),
            eq_lines=tuple(eq_lines),
            params_text=effective_params_text,
            auto_jacobian=bool(custom_auto_jac),
            use_jacobian=bool(custom_use_jac),
        ),
    )
    lyapunov_cfg = LyapunovConfig(
        transient_steps=int(transient_steps_lya),
        qr_interval=float(qr_interval),
    )

    if save_static_cfg:
        z_idx_val = int(z_idx) if plot_mode == "3D phase plot" else None
        static_config = build_static_config(
            app_name=app_name,
            repo_root=repo_root,
            system=system,
            integration=integration,
            initial=initial,
            solve_tols=solve_tols,
            plot_mode=plot_mode,
            x_idx=int(x_idx),
            y_idx=int(y_idx),
            z_idx=z_idx_val,
            phase_linewidth=float(phase_linewidth),
            transient_steps=int(transient_steps),
            lyapunov_transient_steps=int(transient_steps_lya),
            lyapunov_transient_frac=float(lyapunov_transient_frac),
            qr_interval=float(qr_interval),
        )
        st.session_state["static_config"] = static_config
        with phase_col_controls:
            st.success("Static configuration saved. Download from the Export tab.")

    t, y = solve_cached(
        system=system,
        integration=integration,
        initial=initial,
        solve_tols=solve_tols,
    )

    t_plot, y_plot = apply_transient_cut(t, y, int(transient_steps))
    N = max(0, min(int(transient_steps), max(0, y.shape[1] - 2)))
    max_plot_points_i = max(2, int(max_plot_points))
    t_plot_ds, y_plot_ds = downsample_trajectory(t_plot, y_plot, max_plot_points_i)

    with phase_col_plot:
        x_auto = _axis_bounds(y_plot_ds[int(x_idx), :])
        y_auto = _axis_bounds(y_plot_ds[int(y_idx), :])
        z_auto: Optional[Tuple[float, float]] = None
        if plot_mode == "3D phase plot":
            z_auto = _axis_bounds(y_plot_ds[int(z_idx), :])

        phase_bounds_sig = (
            str(plot_mode),
            int(x_idx),
            int(y_idx),
            int(z_idx) if plot_mode == "3D phase plot" else None,
            float(x_auto[0]),
            float(x_auto[1]),
            float(y_auto[0]),
            float(y_auto[1]),
            float(z_auto[0]) if z_auto is not None else None,
            float(z_auto[1]) if z_auto is not None else None,
        )
        if st.session_state.get("phase_bounds_sig_tab1") != phase_bounds_sig:
            st.session_state["phase_xlim_min_tab1"] = float(x_auto[0])
            st.session_state["phase_xlim_max_tab1"] = float(x_auto[1])
            st.session_state["phase_ylim_min_tab1"] = float(y_auto[0])
            st.session_state["phase_ylim_max_tab1"] = float(y_auto[1])
            if z_auto is not None:
                st.session_state["phase_zlim_min_tab1"] = float(z_auto[0])
                st.session_state["phase_zlim_max_tab1"] = float(z_auto[1])
            st.session_state["phase_bounds_sig_tab1"] = phase_bounds_sig

        x_view = (
            float(st.session_state.get("phase_xlim_min_tab1", x_auto[0])),
            float(st.session_state.get("phase_xlim_max_tab1", x_auto[1])),
        )
        y_view = (
            float(st.session_state.get("phase_ylim_min_tab1", y_auto[0])),
            float(st.session_state.get("phase_ylim_max_tab1", y_auto[1])),
        )
        z_view: Optional[Tuple[float, float]] = None
        if z_auto is not None:
            z_view = (
                float(st.session_state.get("phase_zlim_min_tab1", z_auto[0])),
                float(st.session_state.get("phase_zlim_max_tab1", z_auto[1])),
            )

        valid_xy = x_view[0] < x_view[1] and y_view[0] < y_view[1]
        valid_z = (z_view is None) or (z_view[0] < z_view[1])
        if not valid_xy or not valid_z:
            st.warning("Invalid axis limits detected. Reverting to data bounds.")
            x_view = x_auto
            y_view = y_auto
            if z_auto is not None:
                z_view = z_auto
            st.session_state["phase_xlim_min_tab1"] = float(x_view[0])
            st.session_state["phase_xlim_max_tab1"] = float(x_view[1])
            st.session_state["phase_ylim_min_tab1"] = float(y_view[0])
            st.session_state["phase_ylim_max_tab1"] = float(y_view[1])
            if z_view is not None:
                st.session_state["phase_zlim_min_tab1"] = float(z_view[0])
                st.session_state["phase_zlim_max_tab1"] = float(z_view[1])

        phase_square_axes = False
        if plot_mode == "2D phase plane":
            phase_square_axes = st.checkbox(
                "Square axes (equal x/y scale)",
                value=bool(st.session_state.get("phase_square_axes_tab1", False)),
                key="phase_square_axes_tab1",
                help="Use the same scale on both axes and keep the phase plot square.",
            )
            if phase_square_axes:
                x_view, y_view = _square_xy_bounds(x_view, y_view)
                st.session_state["phase_xlim_min_tab1"] = float(x_view[0])
                st.session_state["phase_xlim_max_tab1"] = float(x_view[1])
                st.session_state["phase_ylim_min_tab1"] = float(y_view[0])
                st.session_state["phase_ylim_max_tab1"] = float(y_view[1])

        if plot_mode == "2D phase plane":
            title = f"{system_label} – {var_names[int(y_idx)]} vs {var_names[int(x_idx)]}"
            fig = plot_phase_2d(
                y=y_plot_ds,
                i=int(x_idx),
                j=int(y_idx),
                title=title,
                xlabel=var_names[int(x_idx)],
                ylabel=var_names[int(y_idx)],
                linewidth=float(phase_linewidth),
            )
            ax = fig.axes[0]
            ax.set_xlim(float(x_view[0]), float(x_view[1]))
            ax.set_ylim(float(y_view[0]), float(y_view[1]))
            if phase_square_axes:
                ax.set_aspect("equal", adjustable="box")
            st.pyplot(fig, clear_figure=True)
        else:
            title = f"{system_label} – 3D phase ({var_names[int(x_idx)]}, {var_names[int(y_idx)]}, {var_names[int(z_idx)]})"
            fig = plot_phase_3d(
                y=y_plot_ds,
                i=int(x_idx),
                j=int(y_idx),
                k=int(z_idx),
                title=title,
                labels=(var_names[int(x_idx)], var_names[int(y_idx)], var_names[int(z_idx)]),
                linewidth=float(phase_linewidth),
            )
            ax3d = fig.axes[0]
            ax3d.set_xlim(float(x_view[0]), float(x_view[1]))
            ax3d.set_ylim(float(y_view[0]), float(y_view[1]))
            if z_view is not None and hasattr(ax3d, "set_zlim"):
                cast(Any, ax3d).set_zlim(float(z_view[0]), float(z_view[1]))
            st.pyplot(fig, clear_figure=True)

        preferred_poincare_axes = [int(x_idx), int(y_idx)]
        if plot_mode == "3D phase plot":
            preferred_poincare_axes.append(int(z_idx))
        render_poincare_map_panel(
            t=t_plot,
            y=y_plot,
            var_names=var_names,
            preferred_axes=preferred_poincare_axes,
            title_prefix=system_label,
        )

        st.markdown("**Phase style**")
        apply_state_values({"phase_linewidth_tab1": PHASE_LINEWIDTH_DEFAULT}, only_missing=True)
        st.number_input(
            "Phase line width",
            min_value=0.01,
            max_value=5.0,
            step=0.01,
            format="%.3f",
            key="phase_linewidth_tab1",
            help="Controls the trajectory line thickness in the phase diagram.",
        )

        st.markdown("**Axis limits (view window)**")
        if plot_mode == "2D phase plane":
            lim_c1, lim_c2, lim_c3, lim_c4 = st.columns([1, 1, 1, 1], gap="small")
            with lim_c1:
                st.number_input(f"{var_names[int(x_idx)]} min", format="%.6f", key="phase_xlim_min_tab1")
            with lim_c2:
                st.number_input(f"{var_names[int(x_idx)]} max", format="%.6f", key="phase_xlim_max_tab1")
            with lim_c3:
                st.number_input(f"{var_names[int(y_idx)]} min", format="%.6f", key="phase_ylim_min_tab1")
            with lim_c4:
                st.number_input(f"{var_names[int(y_idx)]} max", format="%.6f", key="phase_ylim_max_tab1")
        else:
            lim_c1, lim_c2, lim_c3 = st.columns([1, 1, 1], gap="small")
            with lim_c1:
                st.number_input(f"{var_names[int(x_idx)]} min", format="%.6f", key="phase_xlim_min_tab1")
                st.number_input(f"{var_names[int(x_idx)]} max", format="%.6f", key="phase_xlim_max_tab1")
            with lim_c2:
                st.number_input(f"{var_names[int(y_idx)]} min", format="%.6f", key="phase_ylim_min_tab1")
                st.number_input(f"{var_names[int(y_idx)]} max", format="%.6f", key="phase_ylim_max_tab1")
            with lim_c3:
                st.number_input(f"{var_names[int(z_idx)]} min", format="%.6f", key="phase_zlim_min_tab1")
                st.number_input(f"{var_names[int(z_idx)]} max", format="%.6f", key="phase_zlim_max_tab1")

        st.caption(
            f"Default bounds: {var_names[int(x_idx)]} [{x_auto[0]:.4g}, {x_auto[1]:.4g}], "
            f"{var_names[int(y_idx)]} [{y_auto[0]:.4g}, {y_auto[1]:.4g}]"
            + (
                f", {var_names[int(z_idx)]} [{z_auto[0]:.4g}, {z_auto[1]:.4g}]"
                if z_auto is not None
                else ""
            )
        )
        st.caption(
            f"Total steps: {len(t)} | stored: {len(t_plot)} | plotted: {len(t_plot_ds)} | "
            f"transient cut: {float(transient_cut_time):.3f} time ({N} steps) | "
            f"n_vars: {y.shape[0]} | t in [{t[0]:.2f}, {t[-1]:.2f}]"
        )
        est_steps = int(np.floor((float(tf) - float(t0)) / max(float(dt), 1e-12))) + 1
        if max_store_steps is not None and est_steps > len(t):
            st.caption(
                f"Storage cap active: kept {len(t)} of ~{est_steps} trajectory samples in memory."
            )

        st.divider()
        st.markdown("**Lyapunov exponents**")
        t_transient_lya = float(transient_steps_lya) * float(dt)
        total_time_lya = float(tf) - float(t0)
        t_measure_lya = total_time_lya - t_transient_lya
        lya_sig = (
            repr(system),
            repr(integration),
            repr(initial),
            repr(lyapunov_cfg),
            repr(solve_tols),
        )
        if t_measure_lya <= 0.0:
            st.warning(
                "Not enough time for Lyapunov measurement. "
                "Increase tf or reduce Lyapunov transient fraction."
            )
        else:
            if compute_lya_btn:
                try:
                    with st.spinner("Computing Lyapunov spectrum..."):
                        lambdas = compute_lyapunov_cached(
                            system=system,
                            integration=integration,
                            initial=initial,
                            lyapunov=lyapunov_cfg,
                            solve_tols=solve_tols,
                        )
                    st.session_state["lya_result_tab1"] = np.array(lambdas, dtype=float)
                    st.session_state["lya_result_sig"] = lya_sig
                except Exception as exc:
                    st.warning(f"Lyapunov computation failed: {exc}")

            lambdas = st.session_state.get("lya_result_tab1", None)
            sig_ok = st.session_state.get("lya_result_sig", None) == lya_sig
            if lambdas is not None and sig_ok:
                formatted = ", ".join(f"{val:.5f}" for val in lambdas)
                st.write(f"lambda = [{formatted}]")
                st.caption(
                    f"n={len(lambdas)} | t_transient={t_transient_lya:.3f} | "
                    f"t_measure={t_measure_lya:.3f} | lyapunov_transient_frac={lyapunov_transient_frac:.2f}"
                )
            else:
                st.info("Click 'Compute Lyapunov exponents' to run the calculation.")

    return PhaseTabResult(
        system=system,
        integration=integration,
        initial=initial,
        solve_tols=solve_tols,
        lyapunov_cfg=lyapunov_cfg,
        t=t,
        y=y,
        t_plot=t_plot,
        y_plot=y_plot,
        var_names=list(var_names),
        system_label=system_label,
        max_plot_points=max_plot_points_i,
        transient_steps=N,
        transient_cut_time=float(transient_cut_time),
        plot_mode=str(plot_mode),
        x_idx=int(x_idx),
        y_idx=int(y_idx),
        z_idx=int(z_idx),
        phase_linewidth=float(phase_linewidth),
        lyapunov_transient_frac=float(lyapunov_transient_frac),
        lyapunov_transient_steps=int(transient_steps_lya),
        qr_interval=float(qr_interval),
    )
