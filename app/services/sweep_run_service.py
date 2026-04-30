from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from app.logic.bifurcation_sweep import _run_bifurcation_parallel
from app.logic.lyapunov_sweep import _run_lyapunov_sweep
from app.logic.reservoir_sampling import make_xy_reservoir
from app.params import InitialConditions, IntegrationConfig, LyapunovConfig, SolverTolerances, SweepRunConfig, SystemConfig
from app.services.sweep_state_service import (
    MAX_BIF_RESERVOIR_POINTS,
    MAX_SWEEP_ROWS_IN_MEMORY,
    _append_dropped_rows_to_reservoir,
    _clip_sweep_df,
    _meta_mismatches,
)
from app.sweep import OBSERVABLE_EXTREMA, OBSERVABLE_POINCARE, run_sweep_chunk
from core.poincare_sweep import PoincareConfig, SweepConfig


def _run_bif_chunk(
    *,
    system: SystemConfig,
    integration_sweep: IntegrationConfig,
    initial: InitialConditions,
    sweep_run: SweepConfig,
    poincare_cfg: PoincareConfig,
    run_cfg: SweepRunConfig,
    solve_tols_sweep: SolverTolerances,
    observable: str,
    extrema_kind: str,
    parallel_bif_enabled: bool,
    workers_bif: int,
) -> pd.DataFrame:
    if parallel_bif_enabled:
        df_chunk = _run_bifurcation_parallel(
            system=system,
            integration=integration_sweep,
            initial=initial,
            sweep=sweep_run,
            poincare=poincare_cfg,
            observable=observable,
            extrema_kind=extrema_kind,
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
            observable=observable,
            extrema_kind=extrema_kind,
            run_cfg=run_cfg,
            solve_tols=solve_tols_sweep,
        )
    return df_chunk if isinstance(df_chunk, pd.DataFrame) else pd.DataFrame(df_chunk)


def execute_new_bif_sweep(
    *,
    system: SystemConfig,
    integration_sweep: IntegrationConfig,
    initial: InitialConditions,
    sweep_param: str,
    sweep_start: float,
    sweep_stop: float,
    sweep_step: float,
    poincare_cfg: PoincareConfig,
    run_cfg: SweepRunConfig,
    solve_tols_sweep: SolverTolerances,
    sweep_meta: Dict[str, Any],
    ycol: str,
    use_extrema: bool,
    extrema_kind: str,
    parallel_bif_enabled: bool,
    workers_bif: int,
) -> pd.DataFrame:
    st.session_state["sweep_acc_df"] = None
    st.session_state["sweep_last_pv"] = None
    st.session_state["sweep_boundaries"] = []
    st.session_state["sweep_meta"] = sweep_meta
    st.session_state["sweep_rows_clipped"] = False
    st.session_state["sweep_reservoir"] = make_xy_reservoir(capacity=MAX_BIF_RESERVOIR_POINTS)

    sweep_run = SweepConfig(
        param_name=sweep_param,
        start=float(sweep_start),
        stop=float(sweep_stop),
        step=float(sweep_step),
    )
    observable = OBSERVABLE_EXTREMA if use_extrema else OBSERVABLE_POINCARE

    with st.spinner("Running sweep..."):
        df_chunk = _run_bif_chunk(
            system=system,
            integration_sweep=integration_sweep,
            initial=initial,
            sweep_run=sweep_run,
            poincare_cfg=poincare_cfg,
            run_cfg=run_cfg,
            solve_tols_sweep=solve_tols_sweep,
            observable=observable,
            extrema_kind=extrema_kind,
            parallel_bif_enabled=parallel_bif_enabled,
            workers_bif=workers_bif,
        )

    df_chunk, dropped, clipped = _clip_sweep_df(df_chunk, MAX_SWEEP_ROWS_IN_MEMORY)
    if clipped:
        _append_dropped_rows_to_reservoir(dropped, sweep_param, ycol)
    st.session_state["sweep_rows_clipped"] = bool(clipped)
    st.session_state["sweep_acc_df"] = df_chunk
    st.session_state["sweep_last_pv"] = float(sweep_stop)
    st.session_state["last_sweep_df"] = df_chunk
    st.session_state["last_sweep_meta"] = sweep_meta
    return df_chunk


def execute_cont_bif_sweep(
    *,
    system: SystemConfig,
    integration_sweep: IntegrationConfig,
    initial: InitialConditions,
    sweep_param: str,
    sweep_step: float,
    sweep_stop: float,
    continue_stop: Optional[float],
    poincare_cfg: PoincareConfig,
    run_cfg: SweepRunConfig,
    solve_tols_sweep: SolverTolerances,
    sweep_meta: Dict[str, Any],
    ycol: str,
    use_extrema: bool,
    extrema_kind: str,
    parallel_bif_enabled: bool,
    workers_bif: int,
) -> Optional[pd.DataFrame]:
    prev_meta = st.session_state.get("sweep_meta", {})
    mismatches = _meta_mismatches(prev_meta, sweep_meta)
    if mismatches:
        st.error(
            "Cannot continue: settings changed since last run. "
            f"Changed: {', '.join(mismatches)}. Run 'Generate Bifurcation Diagram' to restart."
        )
        return None

    last_pv = float(st.session_state["sweep_last_pv"])
    start_here = last_pv + float(sweep_step)
    stop_here = float(continue_stop) if continue_stop is not None else float(sweep_stop)
    st.session_state["sweep_stop_internal"] = stop_here

    if start_here > stop_here + 1e-12:
        st.warning("Nothing to continue: start is already beyond stop.")
        return None

    st.session_state["sweep_boundaries"].append(last_pv)
    sweep_run = SweepConfig(
        param_name=sweep_param,
        start=float(start_here),
        stop=float(stop_here),
        step=float(sweep_step),
    )
    observable = OBSERVABLE_EXTREMA if use_extrema else OBSERVABLE_POINCARE

    with st.spinner("Continuing sweep..."):
        df_chunk = _run_bif_chunk(
            system=system,
            integration_sweep=integration_sweep,
            initial=initial,
            sweep_run=sweep_run,
            poincare_cfg=poincare_cfg,
            run_cfg=run_cfg,
            solve_tols_sweep=solve_tols_sweep,
            observable=observable,
            extrema_kind=extrema_kind,
            parallel_bif_enabled=parallel_bif_enabled,
            workers_bif=workers_bif,
        )

    df_acc = pd.concat([st.session_state["sweep_acc_df"], df_chunk], ignore_index=True)
    df_acc, dropped, clipped = _clip_sweep_df(df_acc, MAX_SWEEP_ROWS_IN_MEMORY)
    if clipped:
        _append_dropped_rows_to_reservoir(dropped, sweep_param, ycol)
    st.session_state["sweep_rows_clipped"] = bool(
        st.session_state.get("sweep_rows_clipped", False) or clipped
    )
    st.session_state["sweep_acc_df"] = df_acc
    st.session_state["sweep_last_pv"] = float(stop_here)
    st.session_state["last_sweep_df"] = df_acc
    st.session_state["last_sweep_meta"] = prev_meta
    return df_acc


def execute_new_lya_sweep(
    *,
    system: SystemConfig,
    integration_sweep: IntegrationConfig,
    initial: InitialConditions,
    sweep_cfg: SweepConfig,
    lyapunov_cfg: LyapunovConfig,
    solve_tols_sweep: SolverTolerances,
    sweep_stop: float,
    warm_start: bool,
    parallel_enabled: bool,
    workers: int,
    lya_meta: Dict[str, Any],
) -> None:
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


def execute_cont_lya_sweep(
    *,
    system: SystemConfig,
    integration_sweep: IntegrationConfig,
    initial: InitialConditions,
    lyapunov_cfg: LyapunovConfig,
    solve_tols_sweep: SolverTolerances,
    sweep_param: str,
    sweep_step: float,
    sweep_stop: float,
    continue_stop_lya: Optional[float],
    warm_start: bool,
    parallel_enabled: bool,
    workers: int,
    var_names: List[str],
    lya_meta: Dict[str, Any],
    acc_data: Dict[str, Any],
    last_pv: float,
) -> None:
    prev_meta = st.session_state.get("lya_meta", {})
    mismatches = _meta_mismatches(prev_meta, lya_meta)
    if mismatches:
        st.error(
            "Cannot continue Lyapunov: settings changed since last run. "
            f"Changed: {', '.join(mismatches)}. Run 'Generate Lyapunov Diagram' to restart."
        )
        return

    start_here = float(last_pv) + float(sweep_step)
    stop_here = float(continue_stop_lya) if continue_stop_lya is not None else float(sweep_stop)

    if start_here > stop_here + 1e-12:
        st.warning("Nothing to continue: start is already beyond stop.")
        return

    st.session_state["lya_boundaries"].append(float(last_pv))
    sweep_run = SweepConfig(
        param_name=sweep_param,
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

    st.session_state["lya_acc_data"] = {
        "param_vals": np.concatenate([prev_vals, param_vals]),
        "lambdas": np.vstack([prev_lambdas, lambdas_arr]),
        "errors": list(prev_errors) + list(errors),
        "meta": dict(prev_meta),
    }
    st.session_state["lya_last_pv"] = float(stop_here)
