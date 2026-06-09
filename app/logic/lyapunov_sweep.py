from typing import Any, Dict, List, Optional, Tuple

import concurrent.futures
import itertools
import numpy as np

from app.logic.sweep_utils import _chunk_param_values, _frange_inclusive
from app.services import (
    apply_to_solve_options,
    build_lyapunov_rhs_jac,
    build_numba_lyap_solver,
    extract_lyapunov_params,
    resolve_solver,
    resolve_time_window,
    run_lyapunov_numba,
    run_lyapunov_scipy,
    LyapunovTimeWindow,
)
from app.params import (
    InitialConditions,
    IntegrationConfig,
    LyapunovConfig,
    SolverTolerances,
    SystemConfig,
)
from core.poincare_sweep import SweepConfig


def _compute_one_step(
    *,
    system: SystemConfig,
    params: Dict[str, float],
    y0: np.ndarray,
    t0: float,
    dt: float,
    window: LyapunovTimeWindow,
    solve_options: Dict[str, Any],
    solver_kind: str,
    auto_switch_rk4: bool,
    lyap_nb: Optional[Any],
    param_names: Optional[List[str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Run one Lyapunov calculation at the given params dict.

    Uses the prebuilt Numba solver if provided, else falls back to SciPy.
    Returns (lambdas, x_final) — lambdas in raw order (caller sorts if needed).
    Raises on failure; the caller decides whether to NaN-fill or propagate.
    """
    if lyap_nb is not None and param_names is not None:
        return run_lyapunov_numba(lyap_nb, param_names, params, y0, t0, dt, window)
    rhs, jac = build_lyapunov_rhs_jac(system, params)
    return run_lyapunov_scipy(rhs, jac, y0, t0, dt, window, solve_options, solver_kind, auto_switch_rk4)


def _run_lyapunov_chunk(
    param_vals: np.ndarray,
    system: SystemConfig,
    base_params: Dict[str, float],
    sweep_param: str,
    y0_base: List[float],
    t0: float,
    dt: float,
    t_transient: float,
    t_measure: float,
    qr_every_steps: int,
    solver_kind: str,
    auto_switch_rk4: bool,
    solve_options: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Worker for parallel Lyapunov sweeps. No warm-start (independent chunks)."""
    y0_base_arr = np.array(y0_base, dtype=float)
    n_dim = y0_base_arr.shape[0]
    window = LyapunovTimeWindow(
        t_transient=float(t_transient),
        t_measure=float(t_measure),
        qr_every_steps=int(qr_every_steps),
    )

    lyap_nb: Optional[Any] = None
    param_names: Optional[List[str]] = None
    if str(solver_kind).lower() == "rk4":
        var_names = list(system.custom.var_names) if system.key == "custom" else []
        eq_lines = list(system.custom.eq_lines) if system.key == "custom" else []
        param_keys = list(base_params.keys()) if system.key == "custom" else []
        lyap_nb, param_names = build_numba_lyap_solver(
            system.key, var_names, eq_lines, param_keys,
            bool(system.custom.auto_jacobian) if system.key == "custom" else False,
            bool(system.custom.use_jacobian) if system.key == "custom" else False,
        )

    errors: List[str] = []
    lambdas_list: List[np.ndarray] = []
    for pv in param_vals:
        params = dict(base_params)
        params[str(sweep_param)] = float(pv)
        try:
            lambdas, _x_final = _compute_one_step(
                system=system, params=params, y0=y0_base_arr,
                t0=float(t0), dt=float(dt), window=window,
                solve_options=solve_options, solver_kind=solver_kind, auto_switch_rk4=auto_switch_rk4,
                lyap_nb=lyap_nb, param_names=param_names,
            )
            lambdas_list.append(np.sort(lambdas)[::-1])
        except Exception as exc:
            errors.append(f"{sweep_param}={float(pv):g}: {exc}")
            lambdas_list.append(np.full(n_dim, np.nan))

    lambdas_arr = np.vstack(lambdas_list) if lambdas_list else np.zeros((0, n_dim))
    return np.array(param_vals, dtype=float), lambdas_arr, errors


def _run_lyapunov_sweep(
    *,
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    sweep: SweepConfig,
    lyapunov: LyapunovConfig,
    solve_tols: SolverTolerances,
    warm_start: bool,
    parallel: bool,
    max_workers: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    param_vals = _frange_inclusive(float(sweep.start), float(sweep.stop), float(sweep.step))
    y0_base = np.array(initial.y0, dtype=float).copy()
    n_dim = y0_base.shape[0]
    base_params = extract_lyapunov_params(system)

    policy = resolve_solver(getattr(integration, "solver_kind", "ivp"))
    solver_kind = policy.kind
    auto_switch_rk4 = policy.auto_switch_rk4
    solve_options: Dict[str, Any] = dict(solve_tols.to_dict())
    apply_to_solve_options(policy, solve_options)
    window = resolve_time_window(integration, lyapunov)

    if parallel and not warm_start:
        param_chunks = _chunk_param_values(param_vals, max_workers)
        if not param_chunks:
            return param_vals, np.zeros((0, n_dim)), []

        workers = max(1, min(int(max_workers), int(param_vals.size)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(
                _run_lyapunov_chunk,
                param_chunks,
                itertools.repeat(system),
                itertools.repeat(base_params),
                itertools.repeat(str(sweep.param_name)),
                itertools.repeat(y0_base.tolist()),
                itertools.repeat(float(integration.t0)),
                itertools.repeat(float(integration.dt)),
                itertools.repeat(float(window.t_transient)),
                itertools.repeat(float(window.t_measure)),
                itertools.repeat(window.qr_every_steps),
                itertools.repeat(solver_kind),
                itertools.repeat(auto_switch_rk4),
                itertools.repeat(solve_options),
            ))

        param_out: List[np.ndarray] = []
        lambdas_out: List[np.ndarray] = []
        errors: List[str] = []
        for pv_chunk, lambdas_chunk, errors_chunk in results:
            param_out.append(pv_chunk)
            lambdas_out.append(lambdas_chunk)
            errors.extend(errors_chunk)

        param_vals_out = np.concatenate(param_out) if param_out else np.array([], dtype=float)
        lambdas_arr = np.vstack(lambdas_out) if lambdas_out else np.zeros((0, n_dim))
        return param_vals_out, lambdas_arr, errors

    # In-process loop with optional warm-start.
    lyap_nb: Optional[Any] = None
    param_names: Optional[List[str]] = None
    if solver_kind == "rk4":
        var_names = list(system.custom.var_names) if system.key == "custom" else []
        eq_lines = list(system.custom.eq_lines) if system.key == "custom" else []
        param_keys = list(base_params.keys()) if system.key == "custom" else []
        lyap_nb, param_names = build_numba_lyap_solver(
            system.key, var_names, eq_lines, param_keys,
            bool(system.custom.auto_jacobian) if system.key == "custom" else False,
            bool(system.custom.use_jacobian) if system.key == "custom" else False,
        )

    errors: List[str] = []
    lambdas_list: List[np.ndarray] = []
    y0_curr = y0_base.copy()
    for pv in param_vals:
        params = dict(base_params)
        params[str(sweep.param_name)] = float(pv)
        try:
            lambdas, x_final = _compute_one_step(
                system=system, params=params, y0=y0_curr,
                t0=float(integration.t0), dt=float(integration.dt), window=window,
                solve_options=solve_options, solver_kind=solver_kind, auto_switch_rk4=auto_switch_rk4,
                lyap_nb=lyap_nb, param_names=param_names,
            )
            lambdas_list.append(np.sort(lambdas)[::-1])
            y0_curr = x_final.copy() if warm_start else y0_base.copy()
        except Exception as exc:
            errors.append(f"{sweep.param_name}={float(pv):g}: {exc}")
            lambdas_list.append(np.full(n_dim, np.nan))
            y0_curr = y0_base.copy()

    lambdas_arr = np.vstack(lambdas_list) if lambdas_list else np.zeros((0, n_dim))
    return param_vals, lambdas_arr, errors
