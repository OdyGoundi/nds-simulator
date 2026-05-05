import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import streamlit as st

from app.helpers import parse_params
from app.logic.reservoir_sampling import make_xy_reservoir
from app.logic.sweep_utils import _default_worker_count, _sweep_settings_fingerprint
from app.params import (
    InitialConditions,
    IntegrationConfig,
    LyapunovConfig,
    SolverTolerances,
    SweepRunConfig,
    SystemConfig,
)
from app.services import get_builtin
from app.state import LyapunovDataKeys, SweepControlsKeys, SweepDataKeys
from app.services.sweep_state_service import (
    MAX_BIF_RESERVOIR_POINTS,
    _init_sweep_state,
)
from core.poincare_sweep import PoincareConfig, SweepConfig
from core import numba_backend


DT_WARNING_THRESHOLD = 0.05
OBSERVABLE_POINCARE_LABEL = "Poincaré crossings"
OBSERVABLE_EXTREMA_LABEL = "Local extrema (max/min)"


@dataclass
class SweepControlsResult:
    # Streamlit column handles reused by the plot panel
    left_col: Any
    right_col: Any

    # Pass-through context (needed by save-config in the plot panel)
    system: SystemConfig
    integration: IntegrationConfig
    initial: InitialConditions
    solve_tols: SolverTolerances
    app_name: str
    repo_root: Path
    var_names: list
    sweep_choices: list
    t0: float
    tf: float
    dt: float

    # Built config objects
    sweep_cfg: SweepConfig
    poincare_cfg: PoincareConfig
    integration_sweep: IntegrationConfig
    run_cfg: SweepRunConfig
    lyapunov_cfg: LyapunovConfig
    solve_tols_sweep: SolverTolerances
    sweep_meta: dict
    lya_meta: dict

    # Individual widget values (plot panel and save-config need these)
    sweep_param: str
    ycol: str
    out_var: str
    use_extrema: bool
    extrema_kind: str
    section_expr: str
    section_var: str
    section_value: float
    section_index: int
    output_index: int
    sweep_start: float
    sweep_stop: float
    sweep_step: float
    dt_sweep: float
    tf_sweep: float
    direction: int
    method: str
    tol: float
    transient_frac: float
    transient_steps_sweep: int
    transient_frac_lya: float
    transient_steps_lya: int
    warm_start: bool
    early_stop: bool
    max_hits: int
    chunk_time: float
    rtol_sweep: float
    atol_sweep: float
    qr_interval_lya: float
    clip_lyapunov: bool
    clip_min: float
    parallel_bif: bool
    workers_bif: int
    workers: int
    parallel_bif_enabled: bool
    parallel_enabled: bool

    # Button states
    run_new: bool
    run_cont: bool
    run_lya: bool
    run_lya_cont: bool
    continue_stop: Optional[float]
    continue_stop_lya: Optional[float]


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


def render_sweep_controls(
    *,
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    solve_tols: SolverTolerances,
    app_name: str,
    repo_root: Path,
) -> SweepControlsResult:
    # Caller is responsible for establishing the `with tab:` context.
    system_key = system.key
    t0 = float(integration.t0)
    tf = float(integration.tf)
    dt = float(integration.dt)
    numba_available = numba_backend.numba_available()
    var_names = list(system.custom.var_names)
    params_text = system.custom.params_text

    _init_sweep_state()

    if system_key == "custom":
        try:
            sweep_choices = list(parse_params(params_text).keys())
        except Exception:
            sweep_choices = []
    else:
        sweep_choices = list(get_builtin(system_key).param_names)

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
            sweep_param = st.selectbox("Sweep param", sweep_choices, index=0, key=SweepControlsKeys.PARAM)
        with p1c2:
            sweep_start = st.number_input("start", value=0.0, step=0.1, format="%.6f", key=SweepControlsKeys.START)
        with p1c3:
            sweep_stop = st.number_input(
                "stop",
                value=float(st.session_state[SweepDataKeys.STOP_INTERNAL]),
                step=0.1,
                format="%.6f",
                key=SweepControlsKeys.STOP,
            )
        with p1c4:
            sweep_step = st.number_input("step", value=0.1, step=0.01, format="%.6f", key=SweepControlsKeys.STEP)

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
                key=SweepControlsKeys.DT_SWEEP,
                help="Time step used for sweep."
            )
        with p2c2:
            tf_sweep = st.number_input(
                "final time",
                min_value=float(t0) + 1e-6,
                value=min(float(tf), 80.0),
                step=5.0,
                format="%.3f",
                key=SweepControlsKeys.TF_SWEEP,
                help="Final integration time for sweep."
            )
        with p2c3:
            sweep_mode = st.selectbox(
                "Sweep mode",
                ["Bifurcation (reset ICs)", "Continuation (warm start)"],
                index=0,
                key=SweepControlsKeys.MODE,
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
                key=SweepControlsKeys.RTOL_SWEEP,
            )
        with p3c2:
            atol_sweep = st.number_input(
                "absolute tolerance",
                min_value=0.0,
                value=1e-6,
                step=1e-6,
                format="%.1e",
                key=SweepControlsKeys.ATOL_SWEEP,
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
            key=SweepControlsKeys.BIF_PARALLEL,
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
            key=SweepControlsKeys.BIF_WORKERS,
            disabled=parallel_bif_disabled or not parallel_bif,
        )

        observable = st.selectbox(
            "Observable",
            [OBSERVABLE_POINCARE_LABEL, OBSERVABLE_EXTREMA_LABEL],
            index=0,
            key=SweepControlsKeys.OBS_KIND,
        )
        use_extrema = observable.startswith("Local extrema")
        if use_extrema:
            extrema_kind = st.selectbox("Extrema", ["max", "min", "both"], index=0, key=SweepControlsKeys.EXTREMA_KIND)
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
                key=SweepControlsKeys.SECTION_VAR,
                disabled=use_extrema,
            )
            section_index = var_names.index(section_var)
        with r1c2:
            section_value = st.number_input(
                "Section value",
                value=0.0,
                step=0.1,
                format="%.6f",
                key=SweepControlsKeys.SECTION_VAL,
                disabled=use_extrema,
            )
        with r1c3:
            direction_label = st.selectbox(
                "Direction",
                ["+1 (up)", "-1 (down)", "0 (both)"],
                index=0,
                key=SweepControlsKeys.SECTION_DIR,
                disabled=use_extrema,
            )
            direction = +1 if direction_label.startswith("+1") else (-1 if direction_label.startswith("-1") else 0)
        with r1c4:
            out_var = st.selectbox(
                "Output var (plotted)",
                var_names,
                index=min(2, len(var_names) - 1),
                key=SweepControlsKeys.OUTPUT_VAR,
            )
            output_index = var_names.index(out_var)

        var_hint = ", ".join(var_names)
        param_hint = ", ".join(sweep_choices)
        section_expr = st.text_input(
            "Section equation (optional, overrides plane)",
            value="",
            key=SweepControlsKeys.SECTION_EXPR,
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
                key=SweepControlsKeys.SECTION_METHOD,
                disabled=use_extrema,
            )
        with r2c2:
            tol = st.number_input(
                "Tolerance (slab only)",
                value=1e-3,
                step=1e-3,
                format="%.1e",
                key=SweepControlsKeys.SECTION_TOL,
                disabled=use_extrema,
            )
        with r2c3:
            st.empty()

        r3c1, r3c2, r3c3 = st.columns([1, 1, 1], gap="small")
        with r3c1:
            early_stop = st.checkbox(
                "Early stop (events)",
                value=True,
                key=SweepControlsKeys.EARLY_STOP,
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
                key=SweepControlsKeys.MAX_HITS,
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
                key=SweepControlsKeys.CHUNK_TIME,
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
                key=SweepControlsKeys.TRANSIENT_FRAC,
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
            st.session_state[SweepDataKeys.ACC_DF] = None
            st.session_state[SweepDataKeys.LAST_PV] = None
            st.session_state[SweepDataKeys.BOUNDARIES] = []
            st.session_state[SweepDataKeys.META] = {}
            st.session_state[SweepDataKeys.ROWS_CLIPPED] = False
            st.session_state[SweepDataKeys.RESERVOIR] = make_xy_reservoir(
                capacity=MAX_BIF_RESERVOIR_POINTS
            )
            st.success("Bifurcation sweep cleared.")

        have_prev_bif = (
            st.session_state.get(SweepDataKeys.ACC_DF, None) is not None and
            st.session_state.get(SweepDataKeys.LAST_PV, None) is not None
        )
        if have_prev_bif:
            last_pv_ui = float(st.session_state[SweepDataKeys.LAST_PV])
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
            key=SweepControlsKeys.LYA_PARALLEL,
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
            key=SweepControlsKeys.LYA_WORKERS,
            disabled=parallel_disabled or not parallel_lya,
        )

        qr_interval_lya = st.number_input(
            "QR interval (time)",
            min_value=1e-6,
            value=0.1,
            step=0.01,
            format="%.4f",
            key=SweepControlsKeys.QR_INTERVAL_LYA,
            help="Time between orthonormalizations during Lyapunov sweep.",
        )
        clip_lyapunov = st.checkbox(
            "Clip lower exponents",
            value=False,
            key=SweepControlsKeys.CLIP_LYAPUNOV,
            help="Clamp very negative exponents for cleaner plots.",
        )
        clip_min = st.number_input(
            "Clip minimum",
            value=-50.0,
            step=1.0,
            format="%.3f",
            key=SweepControlsKeys.CLIP_MIN_LYAPUNOV,
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
                key=SweepControlsKeys.LYA_TRANSIENT_FRAC,
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
            st.session_state[LyapunovDataKeys.ACC_DATA] = None
            st.session_state[LyapunovDataKeys.LAST_PV] = None
            st.session_state[LyapunovDataKeys.META] = {}
            st.session_state[LyapunovDataKeys.BOUNDARIES] = []
            st.success("Lyapunov sweep cleared.")

        have_prev_lya = (
            st.session_state.get(LyapunovDataKeys.ACC_DATA, None) is not None and
            st.session_state.get(LyapunovDataKeys.LAST_PV, None) is not None
        )
        if have_prev_lya:
            last_pv_ui = float(st.session_state[LyapunovDataKeys.LAST_PV])
            continue_stop_lya = st.number_input(
                f"Continue Lyapunov to (stop) [{sweep_param}]",
                min_value=last_pv_ui + float(sweep_step),
                value=max(float(sweep_stop), last_pv_ui + float(sweep_step)),
                step=float(sweep_step),
                format="%.6f",
                key="continue_stop_lya_tab3",
                help="Sets the new stop for Continue Lyapunov. Start is last_pv + step."
            )

    # Build config objects from collected widget values
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

    return SweepControlsResult(
        left_col=left_col,
        right_col=right_col,
        system=system,
        integration=integration,
        initial=initial,
        solve_tols=solve_tols,
        app_name=app_name,
        repo_root=repo_root,
        var_names=var_names,
        sweep_choices=sweep_choices,
        t0=t0,
        tf=tf,
        dt=dt,
        sweep_cfg=sweep_cfg,
        poincare_cfg=poincare_cfg,
        integration_sweep=integration_sweep,
        run_cfg=run_cfg,
        lyapunov_cfg=lyapunov_cfg,
        solve_tols_sweep=solve_tols_sweep,
        sweep_meta=sweep_meta,
        lya_meta=lya_meta,
        sweep_param=str(sweep_param),
        ycol=f"y{int(output_index)}",
        out_var=str(out_var),
        use_extrema=bool(use_extrema),
        extrema_kind=str(extrema_kind),
        section_expr=str(section_expr),
        section_var=str(section_var),
        section_value=float(section_value),
        section_index=int(section_index),
        output_index=int(output_index),
        sweep_start=float(sweep_start),
        sweep_stop=float(sweep_stop),
        sweep_step=float(sweep_step),
        dt_sweep=float(dt_sweep),
        tf_sweep=float(tf_sweep),
        direction=int(direction),
        method=str(method),
        tol=float(tol),
        transient_frac=float(transient_frac),
        transient_steps_sweep=int(transient_steps_sweep),
        transient_frac_lya=float(transient_frac_lya),
        transient_steps_lya=int(transient_steps_lya),
        warm_start=bool(warm_start),
        early_stop=bool(early_stop),
        max_hits=int(max_hits),
        chunk_time=float(chunk_time),
        rtol_sweep=float(rtol_sweep),
        atol_sweep=float(atol_sweep),
        qr_interval_lya=float(qr_interval_lya),
        clip_lyapunov=bool(clip_lyapunov),
        clip_min=float(clip_min),
        parallel_bif=bool(parallel_bif),
        workers_bif=int(workers_bif),
        workers=int(workers),
        parallel_bif_enabled=bool(parallel_bif_enabled),
        parallel_enabled=bool(parallel_enabled),
        run_new=bool(run_new),
        run_cont=bool(run_cont),
        run_lya=bool(run_lya),
        run_lya_cont=bool(run_lya_cont),
        continue_stop=continue_stop,
        continue_stop_lya=continue_stop_lya,
    )
