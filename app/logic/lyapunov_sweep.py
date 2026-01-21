from typing import Dict, List, Tuple

import concurrent.futures
import itertools
import numpy as np

from app.helpers import build_custom_rhs, build_custom_rhs_and_jacobian, parse_params
from app.logic.sweep_utils import _chunk_param_values, _frange_inclusive
from app.params import (
    InitialConditions,
    IntegrationConfig,
    LyapunovConfig,
    SolverTolerances,
    SystemConfig,
)
from core.jacobians_fixed_systems import lorenz_jac, rossler_jac
from core.lorenz_system_rhs import lorenz_rhs
from core.rossler_system_rhs import rossler_rhs
from core.lyapunov import compute_lyapunov_spectrum
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
    solve_options: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    y0_base_arr = np.array(y0_base, dtype=float)
    errors: List[str] = []
    lambdas_list: List[np.ndarray] = []

    for pv in param_vals:
        params = dict(base_params)
        params[str(sweep_param)] = float(pv)

        if system_key == "lorenz":
            rhs = lambda tt, xx: lorenz_rhs(
                tt, xx, sigma=params["sigma"], rho=params["rho"], beta=params["beta"]
            )
            jac = lambda tt, xx: lorenz_jac(
                tt, xx, sigma=params["sigma"], rho=params["rho"], beta=params["beta"]
            )
        elif system_key == "rossler":
            rhs = lambda tt, xx: rossler_rhs(
                tt, xx, a=params["a"], b=params["b"], c=params["c"]
            )
            jac = lambda tt, xx: rossler_jac(
                tt, xx, a=params["a"], b=params["b"], c=params["c"]
            )
        else:
            if custom_auto_jac:
                rhs_custom, jac_custom = build_custom_rhs_and_jacobian(
                    var_names, eq_lines, params
                )
            else:
                rhs_custom = build_custom_rhs(var_names, eq_lines, params)
                jac_custom = None

            rhs = lambda tt, xx: rhs_custom(tt, xx)
            jac = (lambda tt, xx: jac_custom(tt, xx)) if (custom_auto_jac and custom_use_jac) else None

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

    if system.key == "lorenz":
        base_params = {
            "sigma": float(system.lorenz.sigma),
            "rho": float(system.lorenz.rho),
            "beta": float(system.lorenz.beta),
        }
    elif system.key == "rossler":
        base_params = {
            "a": float(system.rossler.a),
            "b": float(system.rossler.b),
            "c": float(system.rossler.c),
        }
    elif system.key == "custom":
        base_params = parse_params(system.custom.params_text)
    else:
        raise ValueError(f"Unknown system_key: {system.key}")

    solve_options = solve_tols.to_dict()
    var_names = list(system.custom.var_names)
    eq_lines = list(system.custom.eq_lines)
    custom_auto_jac = bool(system.custom.auto_jacobian)
    custom_use_jac = bool(system.custom.use_jacobian)

    total_time = float(integration.tf) - float(integration.t0)
    if lyapunov.keep_last_steps is not None:
        keep_steps = int(lyapunov.keep_last_steps)
        if keep_steps <= 0:
            raise ValueError("Lyapunov keep-last-steps must be > 0.")
        t_measure = min(total_time, float(keep_steps) * float(integration.dt))
        t_transient = max(0.0, total_time - t_measure)
    else:
        t_transient = float(lyapunov.transient_steps) * float(integration.dt)
        t_measure = total_time - t_transient
    if t_measure <= 0.0:
        if lyapunov.keep_last_steps is not None:
            raise ValueError("Not enough time for Lyapunov measurement. Increase tf or keep more steps.")
        raise ValueError("Not enough time for Lyapunov measurement. Increase tf or reduce transient cut.")

    if lyapunov.qr_interval <= 0.0:
        raise ValueError("Lyapunov QR interval must be > 0.")
    target_chunk = float(lyapunov.qr_interval)
    qr_every_steps = max(1, int(round(target_chunk / float(integration.dt))))

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

    for pv in param_vals:
        params = dict(base_params)
        params[str(sweep.param_name)] = float(pv)

        if system.key == "lorenz":
            rhs = lambda tt, xx: lorenz_rhs(
                tt, xx, sigma=params["sigma"], rho=params["rho"], beta=params["beta"]
            )
            jac = lambda tt, xx: lorenz_jac(
                tt, xx, sigma=params["sigma"], rho=params["rho"], beta=params["beta"]
            )
        elif system.key == "rossler":
            rhs = lambda tt, xx: rossler_rhs(
                tt, xx, a=params["a"], b=params["b"], c=params["c"]
            )
            jac = lambda tt, xx: rossler_jac(
                tt, xx, a=params["a"], b=params["b"], c=params["c"]
            )
        else:
            if custom_auto_jac:
                rhs_custom, jac_custom = build_custom_rhs_and_jacobian(
                    var_names, eq_lines, params
                )
            else:
                rhs_custom = build_custom_rhs(var_names, eq_lines, params)
                jac_custom = None

            rhs = lambda tt, xx: rhs_custom(tt, xx)
            jac = (lambda tt, xx: jac_custom(tt, xx)) if (custom_auto_jac and custom_use_jac) else None

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
