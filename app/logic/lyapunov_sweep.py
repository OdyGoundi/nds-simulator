from typing import Any, Callable, Dict, List, Tuple

import concurrent.futures
import itertools
import numpy as np

from app.helpers import build_custom_rhs, build_custom_rhs_and_jacobian, parse_params
from app.logic.sweep_utils import _chunk_param_values, _frange_inclusive
from app.services import (
    get_builtin,
    resolve_solver,
    apply_to_solve_options,
    resolve_time_window,
    build_numba_lyap_solver,
)
from app.params import (
    InitialConditions,
    IntegrationConfig,
    LyapunovConfig,
    SolverTolerances,
    SystemConfig,
)
from core.lyapunov import JacFn, RhsFn, compute_lyapunov_spectrum
from core.poincare_sweep import SweepConfig


def _run_lyapunov_chunk(
    param_vals: np.ndarray,
    system_key: str,
    base_params: Dict[str, float],
    sweep_param: str,
    var_names: List[str],
    eq_lines: List[str],
    custom_auto_jac: bool,
    custom_use_jac: bool,
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
    y0_base_arr = np.array(y0_base, dtype=float)
    errors: List[str] = []
    lambdas_list: List[np.ndarray] = []
    lyap_nb = None
    param_names: List[str] | None = None

    if str(solver_kind).lower() == "rk4":
        param_keys = list(base_params.keys()) if system_key == "custom" else []
        lyap_nb, param_names = build_numba_lyap_solver(
            system_key, var_names, eq_lines, param_keys, custom_auto_jac, custom_use_jac
        )

    for pv in param_vals:
        params = dict(base_params)
        params[str(sweep_param)] = float(pv)

        if lyap_nb is not None and param_names is not None:
            try:
                params_arr = np.array([float(params[name]) for name in param_names], dtype=float)
                lambdas, _sums, _t_meas, n_qr, _x_final = lyap_nb(
                    y0_base_arr,
                    float(t0),
                    float(dt),
                    float(t_transient),
                    float(t_measure),
                    int(qr_every_steps),
                    float(1e-8),
                    params_arr,
                )
                if int(n_qr) <= 0:
                    raise ValueError("No QR steps performed. Increase t_measure or reduce qr_every_steps.")
                l_sorted = np.sort(np.array(lambdas, dtype=float))[::-1]
                lambdas_list.append(l_sorted)
            except Exception as exc:
                errors.append(f"{sweep_param}={float(pv):g}: {exc}")
                lambdas_list.append(np.full(y0_base_arr.shape[0], np.nan))
            continue

        rhs: RhsFn
        jac: JacFn | None
        if system_key == "custom":
            jac_custom = None
            if custom_auto_jac:
                rhs_custom, jac_custom = build_custom_rhs_and_jacobian(
                    var_names, eq_lines, params
                )
            else:
                rhs_custom = build_custom_rhs(var_names, eq_lines, params)

            def rhs_wrapped(tt: float, xx: np.ndarray) -> np.ndarray:
                return rhs_custom(tt, xx)

            rhs = rhs_wrapped
            jac = None
            if custom_auto_jac and custom_use_jac:
                if jac_custom is None:
                    raise RuntimeError("Analytic Jacobian requested but not available.")

                jac_custom_fn: Callable[[float, np.ndarray], np.ndarray] = jac_custom

                def jac_wrapped(tt: float, xx: np.ndarray) -> np.ndarray:
                    return jac_custom_fn(tt, xx)

                jac = jac_wrapped
        else:
            _adapter = get_builtin(system_key)
            rhs = _adapter.rhs_from_dict(params)
            jac = _adapter.jac_from_dict(params)

        try:
            res = compute_lyapunov_spectrum(
                rhs=rhs,
                x0=y0_base_arr,
                t0=float(t0),
                dt=float(dt),
                t_transient=float(t_transient),
                t_measure=float(t_measure),
                qr_every_steps=int(qr_every_steps),
                solve_options=solve_options,
                solver_kind=solver_kind,
                auto_switch_rk4=auto_switch_rk4,
                jac=jac,
            )
            l_sorted = np.sort(np.array(res.lambdas, dtype=float))[::-1]
            lambdas_list.append(l_sorted)
        except Exception as exc:
            errors.append(f"{sweep_param}={float(pv):g}: {exc}")
            lambdas_list.append(np.full(y0_base_arr.shape[0], np.nan))

    lambdas_arr = np.vstack(lambdas_list) if lambdas_list else np.zeros((0, y0_base_arr.shape[0]))
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
    y0_curr = y0_base.copy()

    if system.key == "custom":
        base_params = parse_params(system.custom.params_text)
    else:
        base_params = get_builtin(system.key).extract_params(system)

    policy = resolve_solver(getattr(integration, "solver_kind", "ivp"))
    solver_kind = policy.kind
    auto_switch_rk4 = policy.auto_switch_rk4
    solve_options: Dict[str, Any] = dict(solve_tols.to_dict())
    apply_to_solve_options(policy, solve_options)
    var_names = list(system.custom.var_names)
    eq_lines = list(system.custom.eq_lines)
    custom_auto_jac = bool(system.custom.auto_jacobian)
    custom_use_jac = bool(system.custom.use_jacobian)

    window = resolve_time_window(integration, lyapunov)
    t_transient = window.t_transient
    t_measure = window.t_measure
    qr_every_steps = window.qr_every_steps

    if parallel and not warm_start:
        param_chunks = _chunk_param_values(param_vals, max_workers)
        if not param_chunks:
            return param_vals, np.zeros((0, y0_base.shape[0])), []

        workers = max(1, min(int(max_workers), int(param_vals.size)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(
                _run_lyapunov_chunk,
                param_chunks,
                itertools.repeat(system.key),
                itertools.repeat(base_params),
                itertools.repeat(str(sweep.param_name)),
                itertools.repeat(var_names),
                itertools.repeat(eq_lines),
                itertools.repeat(custom_auto_jac),
                itertools.repeat(custom_use_jac),
                itertools.repeat(y0_base.tolist()),
                itertools.repeat(float(integration.t0)),
                itertools.repeat(float(integration.dt)),
                itertools.repeat(float(t_transient)),
                itertools.repeat(float(t_measure)),
                itertools.repeat(qr_every_steps),
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
        lambdas_arr = np.vstack(lambdas_out) if lambdas_out else np.zeros((0, y0_base.shape[0]))
        return param_vals_out, lambdas_arr, errors

    errors: List[str] = []
    lambdas_list: List[np.ndarray] = []
    lyap_nb = None
    param_names: List[str] | None = None

    if solver_kind == "rk4":
        param_keys = list(base_params.keys()) if system.key == "custom" else []
        lyap_nb, param_names = build_numba_lyap_solver(
            system.key, var_names, eq_lines, param_keys, custom_auto_jac, custom_use_jac
        )

    for pv in param_vals:
        params = dict(base_params)
        params[str(sweep.param_name)] = float(pv)

        if lyap_nb is not None and param_names is not None:
            try:
                params_arr = np.array([float(params[name]) for name in param_names], dtype=float)
                lambdas, _sums, _t_meas, n_qr, x_final = lyap_nb(
                    np.asarray(y0_curr, dtype=float),
                    float(integration.t0),
                    float(integration.dt),
                    float(t_transient),
                    float(t_measure),
                    int(qr_every_steps),
                    float(1e-8),
                    params_arr,
                )
                if int(n_qr) <= 0:
                    raise ValueError("No QR steps performed. Increase t_measure or reduce qr_every_steps.")
                l_sorted = np.sort(np.array(lambdas, dtype=float))[::-1]
                lambdas_list.append(l_sorted)
                if warm_start:
                    y0_curr = np.array(x_final, dtype=float).copy()
                else:
                    y0_curr = y0_base.copy()
            except Exception as exc:
                errors.append(f"{sweep.param_name}={float(pv):g}: {exc}")
                lambdas_list.append(np.full(y0_base.shape[0], np.nan))
                y0_curr = y0_base.copy()
            continue

        rhs: RhsFn
        jac: JacFn | None
        if system.key == "custom":
            jac_custom = None
            if custom_auto_jac:
                rhs_custom, jac_custom = build_custom_rhs_and_jacobian(
                    var_names, eq_lines, params
                )
            else:
                rhs_custom = build_custom_rhs(var_names, eq_lines, params)

            def rhs_wrapped(tt: float, xx: np.ndarray) -> np.ndarray:
                return rhs_custom(tt, xx)

            rhs = rhs_wrapped
            jac = None
            if custom_auto_jac and custom_use_jac:
                if jac_custom is None:
                    raise RuntimeError("Analytic Jacobian requested but not available.")

                jac_custom_fn: Callable[[float, np.ndarray], np.ndarray] = jac_custom

                def jac_wrapped(tt: float, xx: np.ndarray) -> np.ndarray:
                    return jac_custom_fn(tt, xx)

                jac = jac_wrapped
        else:
            _adapter = get_builtin(system.key)
            rhs = _adapter.rhs_from_dict(params)
            jac = _adapter.jac_from_dict(params)

        try:
            res = compute_lyapunov_spectrum(
                rhs=rhs,
                x0=y0_curr,
                t0=float(integration.t0),
                dt=float(integration.dt),
                t_transient=float(t_transient),
                t_measure=float(t_measure),
                qr_every_steps=qr_every_steps,
                solve_options=solve_options,
                solver_kind=solver_kind,
                auto_switch_rk4=auto_switch_rk4,
                jac=jac,
            )
            l_sorted = np.sort(np.array(res.lambdas, dtype=float))[::-1]
            lambdas_list.append(l_sorted)
            if warm_start:
                y0_curr = np.array(res.x_final, dtype=float).copy()
            else:
                y0_curr = y0_base.copy()
        except Exception as exc:
            errors.append(f"{sweep.param_name}={float(pv):g}: {exc}")
            lambdas_list.append(np.full(y0_base.shape[0], np.nan))
            y0_curr = y0_base.copy()

    lambdas_arr = np.vstack(lambdas_list) if lambdas_list else np.zeros((0, y0_base.shape[0]))
    return param_vals, lambdas_arr, errors
