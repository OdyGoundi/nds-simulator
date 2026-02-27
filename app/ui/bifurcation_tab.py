import math
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from app.helpers import decimate_indices, downsample_xy, parse_params
from app.export_utils import build_sweep_config
from app.logic.bifurcation_sweep import _run_bifurcation_parallel
from app.logic.lyapunov_sweep import _run_lyapunov_sweep
from app.logic.reservoir_sampling import (
    ensure_xy_reservoir,
    get_xy_reservoir_points,
    make_xy_reservoir,
    update_xy_reservoir,
)
from app.logic.sweep_utils import (
    _default_worker_count,
    _sweep_settings_fingerprint,
)
from app.params import (
    InitialConditions,
    IntegrationConfig,
    LyapunovConfig,
    SolverTolerances,
    SweepRunConfig,
    SystemConfig,
)
from app.sweep import run_sweep_chunk
from core.poincare_sweep import PoincareConfig, SweepConfig
from core import numba_backend


COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
DT_WARNING_THRESHOLD = 0.05
DIRECTION_LABEL_BY_VALUE = {1: "+1 (up)", -1: "-1 (down)", 0: "0 (both)"}
OBSERVABLE_POINCARE_LABEL = "Poincaré crossings"
OBSERVABLE_EXTREMA_LABEL = "Local extrema (max/min)"
MAX_LYA_PLOT_POINTS = 200_000
MAX_SWEEP_ROWS_IN_MEMORY = 300_000
MAX_BIF_RESERVOIR_POINTS = 120_000
MAX_BIF_PLOT_POINTS = MAX_SWEEP_ROWS_IN_MEMORY + MAX_BIF_RESERVOIR_POINTS


def _axis_bounds(values: np.ndarray) -> tuple[float, float]:
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
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    x0, x1 = float(x_bounds[0]), float(x_bounds[1])
    y0, y1 = float(y_bounds[0]), float(y_bounds[1])
    dx = max(1e-12, x1 - x0)
    dy = max(1e-12, y1 - y0)
    half_span = 0.5 * max(dx, dy)
    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)
    return (x_mid - half_span, x_mid + half_span), (y_mid - half_span, y_mid + half_span)


def _to_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _clip_sweep_df(df: pd.DataFrame, max_rows: int) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    max_rows_i = max(1, int(max_rows))
    if len(df) <= max_rows_i:
        return df, pd.DataFrame(columns=df.columns), False
    drop_n = int(len(df) - max_rows_i)
    dropped = df.head(drop_n).reset_index(drop=True)
    kept = df.tail(max_rows_i).reset_index(drop=True)
    return kept, dropped, True


def _append_dropped_rows_to_reservoir(df_dropped: pd.DataFrame, x_col: str, y_col: str) -> None:
    if df_dropped is None or len(df_dropped) == 0:
        return
    if x_col not in df_dropped.columns or y_col not in df_dropped.columns:
        return
    x_old = np.asarray(df_dropped[x_col].to_numpy(), dtype=float)
    y_old = np.asarray(df_dropped[y_col].to_numpy(), dtype=float)
    reservoir_state = ensure_xy_reservoir(
        st.session_state.get("sweep_reservoir"),
        capacity=MAX_BIF_RESERVOIR_POINTS,
    )
    reservoir_state = update_xy_reservoir(
        reservoir_state,
        x_old,
        y_old,
    )
    st.session_state["sweep_reservoir"] = reservoir_state


def _apply_sweep_config_to_state(
    cfg: dict,
    *,
    sweep_choices: list[str],
    var_names: list[str],
    t0_default: float,
    tf_default: float,
    dt_default: float,
) -> None:
    sweep_obj = cfg.get("sweep")
    lyapunov_obj = cfg.get("lyapunov")
    if not isinstance(sweep_obj, dict):
        raise ValueError("Invalid config: missing 'sweep' block.")
    sweep_settings = sweep_obj.get("settings")
    if not isinstance(sweep_settings, dict):
        raise ValueError("Invalid config: missing 'sweep.settings' block.")

    sweep_param = str(sweep_settings.get("sweep_param", "")).strip()
    if sweep_param in sweep_choices:
        st.session_state["sw_param_tab3"] = sweep_param

    sweep_start = _to_float(sweep_settings.get("sweep_start", 0.0), 0.0)
    sweep_stop = _to_float(sweep_settings.get("sweep_stop", 50.0), 50.0)
    sweep_step = _to_float(sweep_settings.get("sweep_step", 0.1), 0.1)
    st.session_state["sw_start_tab3"] = float(sweep_start)
    st.session_state["sw_stop_tab3"] = float(sweep_stop)
    st.session_state["sw_step_tab3"] = float(sweep_step)
    st.session_state["sweep_stop_internal"] = float(sweep_stop)

    sweep_integration = sweep_settings.get("integration")
    if isinstance(sweep_integration, dict):
        dt_sweep = max(1e-6, _to_float(sweep_integration.get("dt", dt_default), dt_default))
        tf_sweep = max(float(t0_default) + 1e-6, _to_float(sweep_integration.get("tf", tf_default), tf_default))
        st.session_state["dt_sweep_tab3"] = float(dt_sweep)
        st.session_state["tf_sweep_tab3"] = float(tf_sweep)

    mode = sweep_settings.get("mode")
    if isinstance(mode, dict):
        warm_start = bool(mode.get("warm_start", False))
        st.session_state["sweep_mode_tab3"] = (
            "Continuation (warm start)" if warm_start else "Bifurcation (reset ICs)"
        )
        st.session_state["bif_parallel_tab3"] = bool(mode.get("parallel", False))
        workers_bif = mode.get("parallel_workers")
        if workers_bif is not None:
            st.session_state["bif_workers_tab3"] = max(1, _to_int(workers_bif, 1))

    sweep_solver = sweep_settings.get("solver")
    if isinstance(sweep_solver, dict):
        if "rtol" in sweep_solver:
            st.session_state["rtol_sweep_tab3"] = _to_float(sweep_solver.get("rtol"), 3e-4)
        if "atol" in sweep_solver:
            st.session_state["atol_sweep_tab3"] = _to_float(sweep_solver.get("atol"), 1e-6)

    poincare = sweep_settings.get("poincare")
    if isinstance(poincare, dict):
        section_var = str(poincare.get("section_var", "")).strip()
        if section_var in var_names:
            st.session_state["sec_var_tab3"] = section_var
        st.session_state["sec_val_tab3"] = _to_float(poincare.get("section_value", 0.0), 0.0)
        st.session_state["sec_expr_tab3"] = str(poincare.get("section_expr", "") or "")
        direction = _to_int(poincare.get("direction", 1), 1)
        st.session_state["sec_dir_tab3"] = DIRECTION_LABEL_BY_VALUE.get(direction, "+1 (up)")
        method = str(poincare.get("method", "crossing")).strip().lower()
        st.session_state["sec_method_tab3"] = "slab" if method == "slab" else "crossing"
        st.session_state["sec_tol_tab3"] = _to_float(poincare.get("tol", 1e-3), 1e-3)

    output = sweep_settings.get("output")
    if isinstance(output, dict):
        output_var = str(output.get("var", "")).strip()
        if output_var in var_names:
            st.session_state["out_var_tab3"] = output_var

    observable = str(sweep_settings.get("observable", "poincare") or "poincare").strip().lower()
    st.session_state["obs_kind_tab3"] = (
        OBSERVABLE_EXTREMA_LABEL if observable == "extrema" else OBSERVABLE_POINCARE_LABEL
    )
    extrema_kind = str(sweep_settings.get("extrema_kind", "max") or "max").strip().lower()
    if extrema_kind not in ("max", "min", "both"):
        extrema_kind = "max"
    st.session_state["ext_kind_tab3"] = extrema_kind

    transient = sweep_settings.get("transient")
    if isinstance(transient, dict) and "fraction" in transient:
        st.session_state["sw_transient_frac_tab3"] = float(
            max(0.0, min(0.95, _to_float(transient.get("fraction", 0.75), 0.75)))
        )

    run = sweep_settings.get("run")
    if isinstance(run, dict):
        st.session_state["early_stop_tab3"] = bool(run.get("early_stop", True))
        st.session_state["max_hits_tab3"] = max(10, _to_int(run.get("max_hits", 200), 200))
        st.session_state["chunk_time_tab3"] = max(0.1, _to_float(run.get("chunk_time", 2.0), 2.0))

    if isinstance(lyapunov_obj, dict):
        lyapunov_settings = lyapunov_obj.get("settings")
        if isinstance(lyapunov_settings, dict):
            st.session_state["qr_interval_lya_tab3"] = max(
                1e-6, _to_float(lyapunov_settings.get("qr_interval", 0.1), 0.1)
            )
            if "transient_fraction" in lyapunov_settings:
                st.session_state["lya_transient_frac_tab3"] = float(
                    max(0.0, min(0.95, _to_float(lyapunov_settings.get("transient_fraction", 0.30), 0.30)))
                )
            st.session_state["lya_parallel_tab3"] = bool(lyapunov_settings.get("parallel", False))
            lya_workers = lyapunov_settings.get("parallel_workers")
            if lya_workers is not None:
                st.session_state["lya_workers_tab3"] = max(1, _to_int(lya_workers, 1))
            clip = lyapunov_settings.get("clip")
            if isinstance(clip, dict):
                clip_enabled = bool(clip.get("enabled", False))
                st.session_state["clip_lyapunov_tab3"] = clip_enabled
                clip_min = clip.get("min", -50.0)
                if clip_min is not None:
                    st.session_state["clip_min_lyapunov_tab3"] = _to_float(clip_min, -50.0)

def _render_tab3_quick_guide() -> None:
    with st.expander("Quick guide: Parameter Sweep Analysis", expanded=False):
        st.markdown(
            """
**Recommended workflow**
1. In **Parameter sweep setup**, choose `Sweep param` and set `start`, `stop`, `step`.
2. In **Sweep performance settings**, choose `dt`, `final time`, `Sweep mode`, and solver tolerances.
3. For the left panel (**Bifurcation sweep settings**), set the Poincare section and click **Generate Bifurcation Diagram**.
4. For the right panel (**Lyapunov sweep settings**), set `QR interval` and click **Generate Lyapunov Diagram**.
5. After plotting, use **Axis limits (view window)** under the Lyapunov chart to inspect a different region without recomputation.
6. At the bottom **Configuration** section (centered), use **Save configuration** or upload/apply `SweepParamConfig.json`.
7. Use **Continue ...** only when settings are unchanged; otherwise click **Generate ...** to restart.
"""
        )
        st.markdown(
            """
**Mode selection**
- **Bifurcation (reset ICs)**: independent runs, standard reference diagrams.
- **Continuation (warm start)**: smoother/faster continuation, runs sequentially (parallel disabled).
"""
        )
        st.markdown(
            """
**Fast first pass**
- Start with a larger `step`, smaller `final time`, and lower `Max hits kept`.
- Increase resolution after you verify the overall structure.
"""
        )
def _init_sweep_state():
    if "sweep_stop_internal" not in st.session_state:
        st.session_state["sweep_stop_internal"] = float(
            st.session_state.get("sweep_stop_internal", 50.0)
        )
    if "sweep_acc_df" not in st.session_state:
        st.session_state["sweep_acc_df"] = None
    if "sweep_last_pv" not in st.session_state:
        st.session_state["sweep_last_pv"] = None
    if "sweep_boundaries" not in st.session_state:
        st.session_state["sweep_boundaries"] = []
    if "sweep_meta" not in st.session_state:
        st.session_state["sweep_meta"] = {}
    if "sweep_rows_clipped" not in st.session_state:
        st.session_state["sweep_rows_clipped"] = False
    st.session_state["sweep_reservoir"] = ensure_xy_reservoir(
        st.session_state.get("sweep_reservoir"),
        capacity=MAX_BIF_RESERVOIR_POINTS,
    )
    if "lya_acc_data" not in st.session_state:
        st.session_state["lya_acc_data"] = None
    if "lya_last_pv" not in st.session_state:
        st.session_state["lya_last_pv"] = None
    if "lya_meta" not in st.session_state:
        st.session_state["lya_meta"] = {}
    if "lya_boundaries" not in st.session_state:
        st.session_state["lya_boundaries"] = []


def render_bifurcation_tab(
    tab,
    *,
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    solve_tols: SolverTolerances,
    app_name: str,
    repo_root: Path,
):
    with tab:
        system_key = system.key
        t0 = float(integration.t0)
        tf = float(integration.tf)
        dt = float(integration.dt)
        numba_available = numba_backend.numba_available()
        var_names = list(system.custom.var_names)
        params_text = system.custom.params_text

        _init_sweep_state()

        if system_key == "lorenz":
            sweep_choices = ["sigma", "rho", "beta"]
        elif system_key == "rossler":
            sweep_choices = ["a", "b", "c"]
        elif system_key == "henon_heiles":
            sweep_choices = ["lambda"]
        else:
            try:
                sweep_choices = list(parse_params(params_text).keys())
            except Exception:
                sweep_choices = []

        if not sweep_choices:
            st.warning("No sweep parameters available (check parameters).")
            st.stop()

        _render_tab3_quick_guide()

        st.divider()

        top_c1, top_c2, top_c3 = st.columns([2, 2, 1], gap="large")
        with top_c1:
            st.markdown("**Parameter sweep setup**")
            p1c1, p1c2, p1c3, p1c4 = st.columns([1, 1, 1, 1], gap="small")
            with p1c1:
                sweep_param = st.selectbox("Sweep param", sweep_choices, index=0, key="sw_param_tab3")
            with p1c2:
                sweep_start = st.number_input("start", value=0.0, step=0.1, format="%.6f", key="sw_start_tab3")
            with p1c3:
                sweep_stop = st.number_input(
                    "stop",
                    value=float(st.session_state["sweep_stop_internal"]),
                    step=0.1,
                    format="%.6f",
                    key="sw_stop_tab3",
                )
            with p1c4:
                sweep_step = st.number_input("step", value=0.1, step=0.01, format="%.6f", key="sw_step_tab3")

        with top_c2:
            st.markdown("**Sweep performance settings**")
            p2c1, p2c2, p2c3 = st.columns([1, 1, 1], gap="small")
            with p2c1:
                dt_sweep = st.number_input(
                    "dt",
                    min_value=1e-6,
                    value=max(float(dt), 0.01),
                    step=0.01,
                    format="%.6f",
                    key="dt_sweep_tab3",
                    help="Time step used for sweep."
                )
            with p2c2:
                tf_sweep = st.number_input(
                    "final time",
                    min_value=float(t0) + 1e-6,
                    value=min(float(tf), 80.0),
                    step=5.0,
                    format="%.3f",
                    key="tf_sweep_tab3",
                    help="Final integration time for sweep."
                )
            with p2c3:
                sweep_mode = st.selectbox(
                    "Sweep mode",
                    ["Bifurcation (reset ICs)", "Continuation (warm start)"],
                    index=0,
                    key="sweep_mode_tab3",
                    help="Reset ICs = bibliography-style. Warm start = faster continuation."
                )

        with top_c3:
            st.markdown("**Sweep solver tolerances**")
            p3c1, p3c2 = st.columns([1, 1], gap="small")
            with p3c1:
                rtol_sweep = st.number_input(
                    "relative tolerance",
                    min_value=0.0,
                    value=3e-4,
                    step=1e-4,
                    format="%.1e",
                    key="rtol_sweep_tab3",
                )
            with p3c2:
                atol_sweep = st.number_input(
                    "absolute tolerance",
                    min_value=0.0,
                    value=1e-6,
                    step=1e-6,
                    format="%.1e",
                    key="atol_sweep_tab3",
                )

        warm_start = sweep_mode.startswith("Continuation")
        solve_tols_sweep = SolverTolerances(rtol=float(rtol_sweep), atol=float(atol_sweep))
        continue_stop = None
        continue_stop_lya = None

        st.divider()

        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            st.markdown("**Bifurcation sweep settings**")
            if not numba_available:
                st.caption("Numba backend unavailable; sweep runs in Python.")
            if system_key in ("lorenz", "rossler") and float(dt_sweep) >= float(DT_WARNING_THRESHOLD):
                st.warning(
                    f"Warning: dt >= {DT_WARNING_THRESHOLD:g} may be too large for stable "
                    f"{system_key.capitalize()} bifurcation sweeps."
                )
            parallel_bif_disabled = bool(warm_start)
            parallel_bif = st.checkbox(
                "Parallel sweep",
                value=False,
                disabled=parallel_bif_disabled,
                key="bif_parallel_tab3",
            )
            if warm_start:
                st.caption("Continuation mode: sequential (smooth).")
            elif parallel_bif:
                st.caption("Parallel independent mode: faster (no warm start).")
            else:
                st.caption("Independent mode: sequential.")

            cpu_count_bif = os.cpu_count() or 1
            max_workers_ui_bif = max(1, min(int(cpu_count_bif), 32))
            workers_default_bif = min(_default_worker_count(), max_workers_ui_bif)
            workers_bif = st.slider(
                "Workers",
                min_value=1,
                max_value=max_workers_ui_bif,
                value=workers_default_bif,
                step=1,
                key="bif_workers_tab3",
                disabled=parallel_bif_disabled or not parallel_bif,
            )

            observable = st.selectbox(
                "Observable",
                [OBSERVABLE_POINCARE_LABEL, OBSERVABLE_EXTREMA_LABEL],
                index=0,
                key="obs_kind_tab3",
            )
            use_extrema = observable.startswith("Local extrema")
            if use_extrema:
                extrema_kind = st.selectbox("Extrema", ["max", "min", "both"], index=0, key="ext_kind_tab3")
            else:
                extrema_kind = "max"

            st.markdown("**Poincaré section selection**")
            if use_extrema:
                st.caption("Section controls are disabled in extrema mode.")

            r1c1, r1c2, r1c3, r1c4 = st.columns([1, 1, 1, 1], gap="small")
            with r1c1:
                section_var = st.selectbox(
                    "Section var",
                    var_names,
                    index=0,
                    key="sec_var_tab3",
                    disabled=use_extrema,
                )
                section_index = var_names.index(section_var)
            with r1c2:
                section_value = st.number_input(
                    "Section value",
                    value=0.0,
                    step=0.1,
                    format="%.6f",
                    key="sec_val_tab3",
                    disabled=use_extrema,
                )
            with r1c3:
                direction_label = st.selectbox(
                    "Direction",
                    ["+1 (up)", "-1 (down)", "0 (both)"],
                    index=0,
                    key="sec_dir_tab3",
                    disabled=use_extrema,
                )
                direction = +1 if direction_label.startswith("+1") else (-1 if direction_label.startswith("-1") else 0)
            with r1c4:
                out_var = st.selectbox(
                    "Output var (plotted)",
                    var_names,
                    index=min(2, len(var_names) - 1),
                    key="out_var_tab3",
                )
                output_index = var_names.index(out_var)

            var_hint = ", ".join(var_names)
            param_hint = ", ".join(sweep_choices)
            section_expr = st.text_input(
                "Section equation (optional, overrides plane)",
                value="",
                key="sec_expr_tab3",
                help=(
                    f"Vars: {var_hint}. Params: {param_hint}."
                ),
                disabled=use_extrema,
            )
            if (not use_extrema) and str(section_expr).strip():
                st.caption("Using section equation; Section var/value is ignored.")

            st.divider()

            r2c1, r2c2, r2c3 = st.columns([1, 1, 1], gap="small")
            with r2c1:
                method = st.selectbox(
                    "Method",
                    ["crossing", "slab"],
                    index=0,
                    key="sec_method_tab3",
                    disabled=use_extrema,
                )
            with r2c2:
                tol = st.number_input(
                    "Tolerance (slab only)",
                    value=1e-3,
                    step=1e-3,
                    format="%.1e",
                    key="sec_tol_tab3",
                    disabled=use_extrema,
                )
            with r2c3:
                st.empty()

            r3c1, r3c2, r3c3 = st.columns([1, 1, 1], gap="small")
            with r3c1:
                early_stop = st.checkbox(
                    "Early stop (events)",
                    value=True,
                    key="early_stop_tab3",
                    disabled=use_extrema,
                    help="Stop each run after collecting enough Poincaré hits."
                )
            with r3c2:
                max_hits = st.number_input(
                    "Max hits kept",
                    min_value=10,
                    max_value=2000,
                    value=200,
                    step=10,
                    key="max_hits_tab3",
                    disabled=(not early_stop) and (not use_extrema),
                    help="Maximum number of hits kept per parameter value."
                )
            with r3c3:
                chunk_time = st.number_input(
                    "Chunk time",
                    min_value=0.1,
                    value=2.0,
                    step=0.5,
                    format="%.2f",
                    key="chunk_time_tab3",
                    disabled=use_extrema or (not early_stop),
                    help="Integration time window for event detection."
                )

            st.markdown("**Bifurcation discard (not plotted)**")
            r4c1, r4c2 = st.columns([1, 1], gap="small")
            with r4c1:
                transient_frac = st.slider(
                    "Transient fraction",
                    min_value=0.0,
                    max_value=0.95,
                    value=0.75,
                    step=0.05,
                    key="sw_transient_frac_tab3",
                    help="Fraction of sweep integration steps to discard before crossings."
                )
            with r4c2:
                n_steps_est = int(max(1.0, (float(tf_sweep) - float(t0)) / float(dt_sweep)))
                transient_steps_sweep = int(transient_frac * n_steps_est)
                st.metric("Transient steps (estimated)", transient_steps_sweep)

            bbtn1, bbtn2, bbtn3 = st.columns([1, 1, 1], gap="small")
            with bbtn1:
                run_new = st.button("Generate Bifurcation Diagram", type="primary", key="run_new_sweep")
            with bbtn2:
                reset_bif = st.button("Reset bifurcation data", type="secondary", key="reset_acc_bif")
            with bbtn3:
                run_cont = st.button("Continue Bifurcation", type="secondary", key="run_cont_sweep")
            if reset_bif:
                st.session_state["sweep_acc_df"] = None
                st.session_state["sweep_last_pv"] = None
                st.session_state["sweep_boundaries"] = []
                st.session_state["sweep_meta"] = {}
                st.session_state["sweep_rows_clipped"] = False
                st.session_state["sweep_reservoir"] = make_xy_reservoir(
                    capacity=MAX_BIF_RESERVOIR_POINTS
                )
                st.success("Bifurcation sweep cleared.")

            have_prev_bif = (
                st.session_state.get("sweep_acc_df", None) is not None and
                st.session_state.get("sweep_last_pv", None) is not None
            )
            if have_prev_bif:
                last_pv_ui = float(st.session_state["sweep_last_pv"])
                continue_stop = st.number_input(
                    f"Continue bifurcation to (stop) [{sweep_param}]",
                    min_value=last_pv_ui + float(sweep_step),
                    value=max(float(sweep_stop), last_pv_ui + float(sweep_step)),
                    step=float(sweep_step),
                    format="%.6f",
                    key="continue_stop_tab3",
                    help="Sets the new stop for Continue Bifurcation. Start is last_pv + step."
                )

        with right_col:
            st.markdown("**Lyapunov sweep settings**")
            parallel_disabled = bool(warm_start)
            parallel_lya = st.checkbox(
                "Parallel sweep",
                value=False,
                disabled=parallel_disabled,
                key="lya_parallel_tab3",
            )
            if warm_start:
                st.caption("Continuation mode: sequential (smooth).")
            elif parallel_lya:
                st.caption("Parallel independent mode: faster (no warm start).")
            else:
                st.caption("Independent mode: sequential.")

            cpu_count = os.cpu_count() or 1
            max_workers_ui = max(1, min(int(cpu_count), 32))
            workers_default = min(_default_worker_count(), max_workers_ui)
            workers = st.slider(
                "Workers",
                min_value=1,
                max_value=max_workers_ui,
                value=workers_default,
                step=1,
                key="lya_workers_tab3",
                disabled=parallel_disabled or not parallel_lya,
            )

            qr_interval_lya = st.number_input(
                "QR interval (time)",
                min_value=1e-6,
                value=0.1,
                step=0.01,
                format="%.4f",
                key="qr_interval_lya_tab3",
                help="Time between orthonormalizations during Lyapunov sweep.",
            )
            clip_lyapunov = st.checkbox(
                "Clip lower exponents",
                value=False,
                key="clip_lyapunov_tab3",
                help="Clamp very negative exponents for cleaner plots.",
            )
            clip_min = st.number_input(
                "Clip minimum",
                value=-50.0,
                step=1.0,
                format="%.3f",
                key="clip_min_lyapunov_tab3",
                disabled=not clip_lyapunov,
            )
            
            st.markdown("**Lyapunov transient cut**")
            ltc1, ltc2 = st.columns([1, 1], gap="small")
            with ltc1:
                transient_frac_lya = st.slider(
                    "Transient fraction",
                    min_value=0.0,
                    max_value=0.95,
                    value=0.30,
                    step=0.05,
                    key="lya_transient_frac_tab3",
                    help="Fraction of sweep integration steps discarded before Lyapunov computation.",
                )
            with ltc2:
                n_steps_est_lya = int(max(1.0, (float(tf_sweep) - float(t0)) / float(dt_sweep)))
                transient_steps_lya = int(transient_frac_lya * n_steps_est_lya)
                st.metric("Transient steps (estimated)", transient_steps_lya)

            lbtn1, lbtn2, lbtn3 = st.columns([1, 1, 1], gap="small")
            with lbtn1:
                run_lya = st.button("Generate Lyapunov Diagram", type="primary", key="run_lya_sweep")
            with lbtn2:
                reset_lya = st.button("Reset Lyapunov data", type="secondary", key="reset_acc_lya")
            with lbtn3:
                run_lya_cont = st.button("Continue Lyapunov", type="secondary", key="run_cont_lya")
            if reset_lya:
                st.session_state["lya_acc_data"] = None
                st.session_state["lya_last_pv"] = None
                st.session_state["lya_meta"] = {}
                st.session_state["lya_boundaries"] = []
                st.success("Lyapunov sweep cleared.")

            have_prev_lya = (
                st.session_state.get("lya_acc_data", None) is not None and
                st.session_state.get("lya_last_pv", None) is not None
            )
            if have_prev_lya:
                last_pv_ui = float(st.session_state["lya_last_pv"])
                continue_stop_lya = st.number_input(
                    f"Continue Lyapunov to (stop) [{sweep_param}]",
                    min_value=last_pv_ui + float(sweep_step),
                    value=max(float(sweep_stop), last_pv_ui + float(sweep_step)),
                    step=float(sweep_step),
                    format="%.6f",
                    key="continue_stop_lya_tab3",
                    help="Sets the new stop for Continue Lyapunov. Start is last_pv + step."
                )

        sweep_cfg = SweepConfig(
            param_name=str(sweep_param),
            start=float(sweep_start),
            stop=float(sweep_stop),
            step=float(sweep_step),
        )
        poincare_cfg = PoincareConfig(
            section_index=int(section_index),
            section_value=float(section_value),
            section_expr=str(section_expr).strip(),
            section_vars=tuple(var_names),
            direction=int(direction),
            method=str(method),
            tol=float(tol),
            transient_steps=int(transient_steps_sweep),
        )
        integration_sweep = IntegrationConfig(
            t0=float(t0),
            tf=float(tf_sweep),
            dt=float(dt_sweep),
            solver_kind=str(getattr(integration, "solver_kind", "ivp")),
        )
        run_cfg = SweepRunConfig(
            output_index=int(output_index),
            warm_start=bool(warm_start),
            max_hits=int(max_hits),
            early_stop=bool(early_stop),
            chunk_time=float(chunk_time),
        )
        lyapunov_cfg = LyapunovConfig(
            transient_steps=int(transient_steps_lya),
            qr_interval=float(qr_interval_lya),
        )

        parallel_bif_enabled = bool(parallel_bif and not parallel_bif_disabled)
        parallel_enabled = bool(parallel_lya and not parallel_disabled)

        sweep_meta = _sweep_settings_fingerprint(
            system=system,
            sweep=sweep_cfg,
            poincare=poincare_cfg,
            run_cfg=run_cfg,
            integration=integration_sweep,
            transient_frac=transient_frac,
            solve_tols=solve_tols_sweep,
            observable="extrema" if use_extrema else "poincare",
            extrema_kind=str(extrema_kind),
        )
        lya_meta = dict(sweep_meta)
        lya_meta.pop("transient_frac", None)
        lya_meta["lyapunov_transient_frac"] = float(transient_frac_lya)
        lya_meta["lyapunov_qr_interval"] = float(lyapunov_cfg.qr_interval)
        lya_meta["lyapunov_solver_kind"] = str(getattr(integration, "solver_kind", "ivp"))
        lya_meta["parallel"] = parallel_enabled
        lya_meta["parallel_workers"] = int(workers) if parallel_enabled else None

        def _save_sweep_config_now() -> None:
            sweep_config = build_sweep_config(
                app_name=app_name,
                repo_root=repo_root,
                system=system,
                integration=integration,
                initial=initial,
                solve_tols=solve_tols,
                sweep_param=str(sweep_param),
                sweep_start=float(sweep_start),
                sweep_stop=float(sweep_stop),
                sweep_step=float(sweep_step),
                dt_sweep=float(dt_sweep),
                tf_sweep=float(tf_sweep),
                section_var=str(section_var),
                section_index=int(section_index),
                section_value=float(section_value),
                section_expr=str(section_expr or ""),
                direction=int(direction),
                method=str(method),
                tol=float(tol),
                output_var=str(out_var),
                output_index=int(output_index),
                observable="extrema" if use_extrema else "poincare",
                extrema_kind=str(extrema_kind),
                transient_frac=float(transient_frac),
                transient_steps_est=int(transient_steps_sweep),
                warm_start=bool(warm_start),
                early_stop=bool(early_stop),
                max_hits=int(max_hits),
                chunk_time=float(chunk_time),
                parallel_bif=bool(parallel_bif),
                workers_bif=int(workers_bif),
                rtol_sweep=float(rtol_sweep),
                atol_sweep=float(atol_sweep),
                qr_interval_lya=float(qr_interval_lya),
                lyapunov_transient_frac=float(transient_frac_lya),
                lyapunov_transient_steps=int(transient_steps_lya),
                parallel_lya=bool(parallel_enabled),
                workers_lya=int(workers),
                clip_lyapunov=bool(clip_lyapunov),
                clip_min=float(clip_min),
                sweep_fingerprint=sweep_meta,
                lya_fingerprint=lya_meta,
            )
            st.session_state["sweep_config"] = sweep_config
            st.success("Sweep configuration saved. Download from the Export tab.")

        df_plot = None
        ycol = f"y{int(output_index)}"

        if run_new:
            st.session_state["sweep_acc_df"] = None
            st.session_state["sweep_last_pv"] = None
            st.session_state["sweep_boundaries"] = []
            st.session_state["sweep_meta"] = sweep_meta
            st.session_state["sweep_rows_clipped"] = False
            st.session_state["sweep_reservoir"] = make_xy_reservoir(
                capacity=MAX_BIF_RESERVOIR_POINTS
            )

            start_here = float(sweep_start)
            stop_here = float(sweep_stop)
            sweep_run = SweepConfig(
                param_name=str(sweep_param),
                start=float(start_here),
                stop=float(stop_here),
                step=float(sweep_step),
            )

            with st.spinner("Running sweep..."):
                if parallel_bif_enabled:
                    df_chunk = _run_bifurcation_parallel(
                        system=system,
                        integration=integration_sweep,
                        initial=initial,
                        sweep=sweep_run,
                        poincare=poincare_cfg,
                        observable="extrema" if use_extrema else "poincare",
                        extrema_kind=str(extrema_kind),
                        run_cfg=run_cfg,
                        solve_tols=solve_tols_sweep,
                        max_workers=int(workers_bif),
                    )
                else:
                    df_chunk = run_sweep_chunk(
                        system=system,
                        integration=integration_sweep,
                        initial=initial,
                        sweep=sweep_run,
                        poincare=poincare_cfg,
                        observable="extrema" if use_extrema else "poincare",
                        extrema_kind=str(extrema_kind),
                        run_cfg=run_cfg,
                        solve_tols=solve_tols_sweep,
                    )

            df_chunk = pd.DataFrame(df_chunk) if not isinstance(df_chunk, pd.DataFrame) else df_chunk
            df_chunk, dropped_new, clipped_new = _clip_sweep_df(df_chunk, MAX_SWEEP_ROWS_IN_MEMORY)
            if clipped_new:
                _append_dropped_rows_to_reservoir(dropped_new, str(sweep_param), str(ycol))
            st.session_state["sweep_rows_clipped"] = bool(clipped_new)
            st.session_state["sweep_acc_df"] = df_chunk
            st.session_state["sweep_last_pv"] = float(stop_here)

            st.session_state["last_sweep_df"] = df_chunk
            st.session_state["last_sweep_meta"] = st.session_state["sweep_meta"]

            df_plot = df_chunk

        elif run_cont:
            acc_df = st.session_state.get("sweep_acc_df", None)
            last_pv = st.session_state.get("sweep_last_pv", None)

            if acc_df is None or last_pv is None:
                st.warning("No previous sweep found. Run 'Generate Bifurcation Diagram' first.")
                st.stop()

            else:
                prev_meta = st.session_state.get("sweep_meta", {})
                now_meta = sweep_meta

                mismatches = []
                for k, v_prev in prev_meta.items():
                    if k not in now_meta:
                        continue
                    v_now = now_meta[k]
                    if isinstance(v_prev, float) and isinstance(v_now, float):
                        if not math.isclose(v_prev, v_now, rel_tol=0.0, abs_tol=1e-12):
                            mismatches.append(k)
                    else:
                        if v_prev != v_now:
                            mismatches.append(k)

                if mismatches:
                    st.error(
                        "Cannot continue: settings changed since last run. "
                        f"Changed: {', '.join(mismatches)}. Run 'Generate Bifurcation Diagram' to restart."
                    )
                else:
                    last_pv = float(st.session_state["sweep_last_pv"])
                    start_here = last_pv + float(sweep_step)

                    stop_here = float(continue_stop) if continue_stop is not None else float(sweep_stop)

                    st.session_state["sweep_stop_internal"] = stop_here

                    if start_here > stop_here + 1e-12:
                        st.warning("Nothing to continue: start is already beyond stop.")
                    else:
                        st.session_state["sweep_boundaries"].append(last_pv)
                        sweep_run = SweepConfig(
                            param_name=str(sweep_param),
                            start=float(start_here),
                            stop=float(stop_here),
                            step=float(sweep_step),
                        )

                        with st.spinner("Continuing sweep..."):
                            if parallel_bif_enabled:
                                df_chunk = _run_bifurcation_parallel(
                                    system=system,
                                    integration=integration_sweep,
                                    initial=initial,
                                    sweep=sweep_run,
                                    poincare=poincare_cfg,
                                    observable="extrema" if use_extrema else "poincare",
                                    extrema_kind=str(extrema_kind),
                                    run_cfg=run_cfg,
                                    solve_tols=solve_tols_sweep,
                                    max_workers=int(workers_bif),
                                )
                            else:
                                df_chunk = run_sweep_chunk(
                                    system=system,
                                    integration=integration_sweep,
                                    initial=initial,
                                    sweep=sweep_run,
                                    poincare=poincare_cfg,
                                    observable="extrema" if use_extrema else "poincare",
                                    extrema_kind=str(extrema_kind),
                                    run_cfg=run_cfg,
                                    solve_tols=solve_tols_sweep,
                                )

                        df_chunk = pd.DataFrame(df_chunk) if not isinstance(df_chunk, pd.DataFrame) else df_chunk
                        df_acc = st.session_state["sweep_acc_df"]
                        df_acc = pd.concat([df_acc, df_chunk], ignore_index=True)
                        df_acc, dropped_acc, clipped_acc = _clip_sweep_df(
                            df_acc,
                            MAX_SWEEP_ROWS_IN_MEMORY,
                        )
                        if clipped_acc:
                            _append_dropped_rows_to_reservoir(
                                dropped_acc,
                                str(sweep_param),
                                str(ycol),
                            )
                        st.session_state["sweep_rows_clipped"] = bool(
                            st.session_state.get("sweep_rows_clipped", False) or clipped_acc
                        )

                        st.session_state["sweep_acc_df"] = df_acc
                        st.session_state["sweep_last_pv"] = float(stop_here)

                        st.session_state["last_sweep_df"] = df_acc
                        st.session_state["last_sweep_meta"] = prev_meta

                        df_plot = df_acc

        if run_lya_cont:
            acc_data = st.session_state.get("lya_acc_data", None)
            last_pv = st.session_state.get("lya_last_pv", None)

            if acc_data is None or last_pv is None:
                st.warning("No previous Lyapunov sweep found. Run 'Generate Lyapunov Diagram' first.")
            else:
                prev_meta = st.session_state.get("lya_meta", {})
                now_meta = lya_meta

                mismatches = []
                for k, v_prev in prev_meta.items():
                    if k not in now_meta:
                        continue
                    v_now = now_meta[k]
                    if isinstance(v_prev, float) and isinstance(v_now, float):
                        if not math.isclose(v_prev, v_now, rel_tol=0.0, abs_tol=1e-12):
                            mismatches.append(k)
                    else:
                        if v_prev != v_now:
                            mismatches.append(k)

                if mismatches:
                    st.error(
                        "Cannot continue Lyapunov: settings changed since last run. "
                        f"Changed: {', '.join(mismatches)}. Run 'Generate Lyapunov Diagram' to restart."
                    )
                else:
                    start_here = float(last_pv) + float(sweep_step)
                    stop_here = float(continue_stop_lya) if continue_stop_lya is not None else float(sweep_stop)

                    if start_here > stop_here + 1e-12:
                        st.warning("Nothing to continue: start is already beyond stop.")
                    else:
                        st.session_state["lya_boundaries"].append(float(last_pv))
                        sweep_run = SweepConfig(
                            param_name=str(sweep_param),
                            start=float(start_here),
                            stop=float(stop_here),
                            step=float(sweep_step),
                        )
                        with st.spinner("Continuing Lyapunov sweep..."):
                            param_vals, lambdas_arr, errors = _run_lyapunov_sweep(
                                system=system,
                                integration=integration_sweep,
                                initial=initial,
                                sweep=sweep_run,
                                lyapunov=lyapunov_cfg,
                                solve_tols=solve_tols_sweep,
                                warm_start=bool(warm_start),
                                parallel=parallel_enabled,
                                max_workers=int(workers),
                            )

                        prev_vals = acc_data.get("param_vals", np.array([], dtype=float))
                        prev_lambdas = acc_data.get("lambdas", np.zeros((0, len(var_names))))
                        prev_errors = acc_data.get("errors", [])

                        new_vals = np.concatenate([prev_vals, param_vals])
                        new_lambdas = np.vstack([prev_lambdas, lambdas_arr])
                        new_errors = list(prev_errors) + list(errors)

                        st.session_state["lya_acc_data"] = {
                            "param_vals": new_vals,
                            "lambdas": new_lambdas,
                            "errors": new_errors,
                            "meta": dict(prev_meta),
                        }
                        st.session_state["lya_last_pv"] = float(stop_here)

        if run_lya:
            st.session_state["lya_acc_data"] = None
            st.session_state["lya_last_pv"] = None
            st.session_state["lya_boundaries"] = []
            st.session_state["lya_meta"] = lya_meta

            with st.spinner("Computing Lyapunov sweep..."):
                param_vals, lambdas_arr, errors = _run_lyapunov_sweep(
                    system=system,
                    integration=integration_sweep,
                    initial=initial,
                    sweep=sweep_cfg,
                    lyapunov=lyapunov_cfg,
                    solve_tols=solve_tols_sweep,
                    warm_start=bool(warm_start),
                    parallel=parallel_enabled,
                    max_workers=int(workers),
                )

            st.session_state["lya_acc_data"] = {
                "param_vals": param_vals,
                "lambdas": lambdas_arr,
                "errors": errors,
                "meta": dict(lya_meta),
            }
            st.session_state["lya_last_pv"] = float(sweep_stop)

            if errors:
                st.warning(f"Lyapunov sweep had {len(errors)} failures. Showing NaNs for those points.")

        if df_plot is None:
            df_plot = st.session_state.get("sweep_acc_df", None)

        with left_col:
            st.divider()
            if df_plot is None or len(df_plot) == 0:
                st.info("No sweep data yet. Click 'Generate Bifurcation Diagram' to start.")
            else:
                if not isinstance(df_plot, pd.DataFrame):
                    df_plot = pd.DataFrame(df_plot)

                x_vals = np.asarray(df_plot[sweep_param].to_numpy(), dtype=float)
                y_vals = np.asarray(df_plot[ycol].to_numpy(), dtype=float)
                reservoir_state = ensure_xy_reservoir(
                    st.session_state.get("sweep_reservoir"),
                    capacity=MAX_BIF_RESERVOIR_POINTS,
                )
                st.session_state["sweep_reservoir"] = reservoir_state
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
                x_start = float(sweep_start)
                x_stop_data = float(st.session_state.get("sweep_last_pv", sweep_stop))
                if not np.isfinite(x_stop_data):
                    x_stop_data = float(sweep_stop)
                if x_stop_data <= x_start:
                    x_stop_data = x_start + max(1e-12, float(abs(sweep_step)))
                x_auto = (float(x_start), float(x_stop_data))
                y_auto = _axis_bounds(y_bounds_data)

                bif_bounds_sig = (
                    str(sweep_param),
                    str(ycol),
                    int(x_vals.size),
                    int(x_hist_plot.size),
                    int(st.session_state["sweep_reservoir"].get("seen", 0)),
                    float(x_auto[0]),
                    float(x_auto[1]),
                    float(y_auto[0]),
                    float(y_auto[1]),
                )
                if st.session_state.get("bif_bounds_sig_tab3") != bif_bounds_sig:
                    st.session_state["bif_xlim_min_tab3"] = float(x_auto[0])
                    st.session_state["bif_xlim_max_tab3"] = float(x_auto[1])
                    st.session_state["bif_ylim_min_tab3"] = float(y_auto[0])
                    st.session_state["bif_ylim_max_tab3"] = float(y_auto[1])
                    st.session_state["bif_bounds_sig_tab3"] = bif_bounds_sig

                x_view = (
                    float(st.session_state.get("bif_xlim_min_tab3", x_auto[0])),
                    float(st.session_state.get("bif_xlim_max_tab3", x_auto[1])),
                )
                y_view = (
                    float(st.session_state.get("bif_ylim_min_tab3", y_auto[0])),
                    float(st.session_state.get("bif_ylim_max_tab3", y_auto[1])),
                )
                if not (x_view[0] < x_view[1] and y_view[0] < y_view[1]):
                    st.warning("Invalid bifurcation axis limits detected. Reverting to data bounds.")
                    x_view = x_auto
                    y_view = y_auto
                    st.session_state["bif_xlim_min_tab3"] = float(x_auto[0])
                    st.session_state["bif_xlim_max_tab3"] = float(x_auto[1])
                    st.session_state["bif_ylim_min_tab3"] = float(y_auto[0])
                    st.session_state["bif_ylim_max_tab3"] = float(y_auto[1])

                square_axes_bif = st.checkbox(
                    "Square axes (equal x/y scale)",
                    value=bool(st.session_state.get("bif_square_axes_tab3", False)),
                    key="bif_square_axes_tab3",
                    help="Use the same scale on both axes and keep this plot square.",
                )
                if square_axes_bif:
                    x_view, y_view = _square_xy_bounds(x_view, y_view)
                    st.session_state["bif_xlim_min_tab3"] = float(x_view[0])
                    st.session_state["bif_xlim_max_tab3"] = float(x_view[1])
                    st.session_state["bif_ylim_min_tab3"] = float(y_view[0])
                    st.session_state["bif_ylim_max_tab3"] = float(y_view[1])

                st.markdown("**Axis limits (view window)**")
                lim_c1, lim_c2, lim_c3, lim_c4 = st.columns([1, 1, 1, 1], gap="small")
                with lim_c1:
                    st.number_input(
                        f"{sweep_param} min",
                        format="%.6f",
                        key="bif_xlim_min_tab3",
                    )
                with lim_c2:
                    st.number_input(
                        f"{sweep_param} max",
                        format="%.6f",
                        key="bif_xlim_max_tab3",
                    )
                with lim_c3:
                    st.number_input(
                        f"{out_var} min",
                        format="%.6f",
                        key="bif_ylim_min_tab3",
                    )
                with lim_c4:
                    st.number_input(
                        f"{out_var} max",
                        format="%.6f",
                        key="bif_ylim_max_tab3",
                    )
                st.caption(
                    f"Default bounds: {sweep_param} [{x_auto[0]:.4g}, {x_auto[1]:.4g}], "
                    f"{out_var} [{y_auto[0]:.4g}, {y_auto[1]:.4g}]"
                )
                expected_params = 0
                if float(sweep_step) > 0:
                    expected_params = int(
                        np.floor((float(x_auto[1]) - float(x_auto[0])) / float(sweep_step) + 1e-12)
                    ) + 1
                observed_params = int(np.unique(np.round(x_vals, 12)).size)
                if expected_params > 0 and observed_params < expected_params:
                    missing_params = int(expected_params - observed_params)
                    st.caption(
                        f"Recent-buffer coverage: {observed_params}/{expected_params} parameter values have hits "
                        f"(missing {missing_params})."
                    )

                fig, ax = plt.subplots(figsize=(6.0, 3.2))
                fig.set_dpi(140)
                if x_hist_plot.size > 0:
                    ax.scatter(
                        x_hist_plot,
                        y_hist_plot,
                        s=1,
                        c="#7f7f7f",
                        marker=".",
                        linewidths=0,
                        alpha=0.22,
                    )
                ax.scatter(
                    x_vals,
                    y_vals,
                    s=2,
                    c="black",
                    marker=".",
                    linewidths=0,
                    alpha=0.8,
                )

                for x_sep in st.session_state.get("sweep_boundaries", []):
                    ax.axvline(float(x_sep), color="magenta", linewidth=0.3)

                ax.set_xlabel(sweep_param)
                if use_extrema:
                    ax.set_ylabel(f"{out_var} local extrema ({extrema_kind})")
                else:
                    section_label = str(section_expr).strip()
                    if section_label:
                        ax.set_ylabel(f"{out_var} on section ({section_label})")
                    else:
                        ax.set_ylabel(f"{out_var} on section ({section_var}={section_value})")
                ax.set_xlim(float(x_view[0]), float(x_view[1]))
                ax.set_ylim(float(y_view[0]), float(y_view[1]))
                if square_axes_bif:
                    ax.set_aspect("equal", adjustable="box")
                ax.grid(True, linewidth=0.3)
                st.pyplot(fig, clear_figure=True)
                total_plotted = int(x_hist_plot.size + x_vals.size)
                st.caption(
                    f"Plotted points: recent {len(x_vals):,} + reservoir {len(x_hist_plot):,} = {total_plotted:,}"
                )

                last_pv = st.session_state.get("sweep_last_pv", None)
                if last_pv is not None:
                    st.caption(f"Accumulated sweep up to {sweep_param} = {float(last_pv):g} | Rows: {len(df_plot)}")
                else:
                    try:
                        st.caption(f"Accumulated sweep | Rows: {len(df_plot)}")
                    except Exception:
                        pass
                if bool(st.session_state.get("sweep_rows_clipped", False)):
                    reservoir_seen = int(st.session_state["sweep_reservoir"].get("seen", 0))
                    st.caption(
                        f"Stored sweep rows are capped at {MAX_SWEEP_ROWS_IN_MEMORY:,} (recent full-resolution). "
                        f"Dropped history is kept as a reservoir sample up to {MAX_BIF_RESERVOIR_POINTS:,} "
                        f"points from {reservoir_seen:,} dropped rows."
                    )

        with right_col:
            st.divider()
            lya_data = st.session_state.get("lya_acc_data", None)
            if lya_data is None:
                st.info("No Lyapunov sweep data yet. Click 'Generate Lyapunov Diagram'.")
            else:
                param_vals = lya_data.get("param_vals", np.array([], dtype=float))
                lambdas_arr = lya_data.get("lambdas", np.zeros((0, len(var_names))))
                errors = lya_data.get("errors", [])

                if param_vals.size == 0 or lambdas_arr.size == 0:
                    st.info("No Lyapunov sweep data yet. Click 'Generate Lyapunov Diagram'.")
                else:
                    plot_lambdas = np.array(lambdas_arr, dtype=float)
                    if clip_lyapunov:
                        plot_lambdas = np.maximum(plot_lambdas, float(clip_min))
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
                        str(sweep_param),
                        bool(clip_lyapunov),
                        float(clip_min),
                        int(plot_lambdas.shape[0]),
                        int(plot_lambdas.shape[1]) if plot_lambdas.ndim == 2 else 0,
                        float(x_auto[0]),
                        float(x_auto[1]),
                        float(y_auto[0]),
                        float(y_auto[1]),
                    )
                    if st.session_state.get("lya_bounds_sig_tab3") != lya_bounds_sig:
                        st.session_state["lya_xlim_min_tab3"] = float(x_auto[0])
                        st.session_state["lya_xlim_max_tab3"] = float(x_auto[1])
                        st.session_state["lya_ylim_min_tab3"] = float(y_auto[0])
                        st.session_state["lya_ylim_max_tab3"] = float(y_auto[1])
                        st.session_state["lya_bounds_sig_tab3"] = lya_bounds_sig

                    x_view = (
                        float(st.session_state.get("lya_xlim_min_tab3", x_auto[0])),
                        float(st.session_state.get("lya_xlim_max_tab3", x_auto[1])),
                    )
                    y_view = (
                        float(st.session_state.get("lya_ylim_min_tab3", y_auto[0])),
                        float(st.session_state.get("lya_ylim_max_tab3", y_auto[1])),
                    )
                    if not (x_view[0] < x_view[1] and y_view[0] < y_view[1]):
                        st.warning("Invalid Lyapunov axis limits detected. Reverting to data bounds.")
                        x_view = x_auto
                        y_view = y_auto
                        st.session_state["lya_xlim_min_tab3"] = float(x_auto[0])
                        st.session_state["lya_xlim_max_tab3"] = float(x_auto[1])
                        st.session_state["lya_ylim_min_tab3"] = float(y_auto[0])
                        st.session_state["lya_ylim_max_tab3"] = float(y_auto[1])

                    square_axes_lya = st.checkbox(
                        "Square axes (equal x/y scale)",
                        value=bool(st.session_state.get("lya_square_axes_tab3", False)),
                        key="lya_square_axes_tab3",
                        help="Use the same scale on both axes and keep this plot square.",
                    )
                    if square_axes_lya:
                        x_view, y_view = _square_xy_bounds(x_view, y_view)
                        st.session_state["lya_xlim_min_tab3"] = float(x_view[0])
                        st.session_state["lya_xlim_max_tab3"] = float(x_view[1])
                        st.session_state["lya_ylim_min_tab3"] = float(y_view[0])
                        st.session_state["lya_ylim_max_tab3"] = float(y_view[1])

                    fig_lya, ax_lya = plt.subplots(figsize=(6.0, 3.2))
                    fig_lya.set_dpi(140)

                    n_exps = plot_lambdas_plot.shape[1]
                    for k in range(n_exps):
                        ax_lya.plot(
                            param_vals_plot,
                            plot_lambdas_plot[:, k],
                            color=COLORS[k % len(COLORS)],
                            linestyle="-",
                            linewidth=1.1,
                            label=f"lambda{k}",
                        )

                    for x_sep in st.session_state.get("lya_boundaries", []):
                        ax_lya.axvline(float(x_sep), color="magenta", linewidth=0.3)

                    ax_lya.set_xlabel(sweep_param)
                    ax_lya.set_ylabel("Lyapunov exponents")
                    ax_lya.set_xlim(float(x_view[0]), float(x_view[1]))
                    ax_lya.set_ylim(float(y_view[0]), float(y_view[1]))
                    if square_axes_lya:
                        ax_lya.set_aspect("equal", adjustable="box")
                    ax_lya.grid(True, linewidth=0.3)
                    ax_lya.legend(loc="best", fontsize=8)
                    st.pyplot(fig_lya, clear_figure=True)
                    st.caption(f"Plotted points: {len(param_vals_plot)}/{len(param_vals)}")

                    st.markdown("**Axis limits (view window)**")
                    lim_c1, lim_c2, lim_c3, lim_c4 = st.columns([1, 1, 1, 1], gap="small")
                    with lim_c1:
                        st.number_input(
                            f"{sweep_param} min",
                            format="%.6f",
                            key="lya_xlim_min_tab3",
                        )
                    with lim_c2:
                        st.number_input(
                            f"{sweep_param} max",
                            format="%.6f",
                            key="lya_xlim_max_tab3",
                        )
                    with lim_c3:
                        st.number_input(
                            "lambda min",
                            format="%.6f",
                            key="lya_ylim_min_tab3",
                        )
                    with lim_c4:
                        st.number_input(
                            "lambda max",
                            format="%.6f",
                            key="lya_ylim_max_tab3",
                        )
                    st.caption(
                        f"Default bounds: {sweep_param} [{x_auto[0]:.4g}, {x_auto[1]:.4g}], "
                        f"lambda [{y_auto[0]:.4g}, {y_auto[1]:.4g}]"
                    )

                    if clip_lyapunov:
                        st.caption(f"Clipped exponents below {float(clip_min):g} for plotting.")
                    if errors:
                        st.caption(f"Lyapunov sweep failures: {len(errors)}")
                    last_pv = st.session_state.get("lya_last_pv", None)
                    if last_pv is not None:
                        st.caption(f"Accumulated Lyapunov sweep up to {sweep_param} = {float(last_pv):g}")

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
                    sweep_choices=list(sweep_choices),
                    var_names=list(var_names),
                    t0_default=float(t0),
                    tf_default=float(tf),
                    dt_default=float(dt),
                )
                st.session_state["sweep_config"] = loaded_sweep_cfg
                st.success("Sweep configuration loaded. Applying settings...")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load SweepParamConfig: {exc}")

        if save_sweep_cfg:
            _save_sweep_config_now()
