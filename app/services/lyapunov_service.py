from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.helpers import build_custom_rhs, build_custom_rhs_and_jacobian, parse_params
from app.params import (
    InitialConditions,
    IntegrationConfig,
    LyapunovConfig,
    SolverTolerances,
    SystemConfig,
)
from app.services.solver_policy import apply_to_solve_options, resolve_solver
from app.services.system_registry import get_builtin
from core.lyapunov import JacFn, RhsFn, compute_lyapunov_spectrum


@dataclass(frozen=True)
class LyapunovTimeWindow:
    t_transient: float
    t_measure: float
    qr_every_steps: int


def resolve_time_window(
    integration: IntegrationConfig,
    lyapunov: LyapunovConfig,
) -> LyapunovTimeWindow:
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
    qr_every_steps = max(1, int(round(float(lyapunov.qr_interval) / float(integration.dt))))
    return LyapunovTimeWindow(
        t_transient=t_transient,
        t_measure=t_measure,
        qr_every_steps=qr_every_steps,
    )


def build_numba_lyap_solver(
    system_key: str,
    var_names: List[str],
    eq_lines: List[str],
    param_keys: List[str],
    auto_jac: bool,
    use_jac: bool,
) -> Tuple[Optional[Any], Optional[List[str]]]:
    """Return (lyap_nb, param_names) or (None, None) if Numba is unavailable."""
    try:
        from core import numba_backend
        if not numba_backend.numba_available():
            return None, None
        if system_key in ("lorenz", "rossler", "henon_heiles"):
            rhs_nb, jac_nb, param_names_tpl = numba_backend.build_builtin_system(system_key)
            param_names = list(param_names_tpl)
            lyap_nb = numba_backend.build_lyapunov_solver(rhs_nb, jac_nb, use_fd_jac=False)
            return lyap_nb, param_names
        elif system_key == "custom":
            from app import numba_custom
            pnames = list(param_keys)
            if auto_jac and use_jac:
                rhs_nb, jac_nb = numba_custom.build_custom_numba_rhs_and_jacobian(
                    var_names, eq_lines, pnames
                )
                lyap_nb = numba_backend.build_lyapunov_solver(rhs_nb, jac_nb, use_fd_jac=False)
            else:
                rhs_nb = numba_custom.build_custom_numba_rhs(var_names, eq_lines, pnames)
                lyap_nb = numba_backend.build_lyapunov_solver(rhs_nb, None, use_fd_jac=True)
            return lyap_nb, pnames
        return None, None
    except Exception:
        return None, None


def compute_single_lyapunov(
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    lyapunov: LyapunovConfig,
    solve_tols: SolverTolerances,
) -> np.ndarray:
    policy = resolve_solver(getattr(integration, "solver_kind", "ivp"))
    solver_kind = policy.kind
    auto_switch_rk4 = policy.auto_switch_rk4
    solve_options: Dict[str, Any] = dict(solve_tols.to_dict())
    apply_to_solve_options(policy, solve_options)

    y0 = np.array(initial.y0, dtype=float)

    custom_params: Optional[Dict[str, float]] = None
    var_names: List[str] = []
    eq_lines: List[str] = []
    auto_jac = False
    use_jac = False
    rhs_fn: RhsFn
    jac_fn: Optional[JacFn]

    if system.key == "custom":
        var_names = list(system.custom.var_names)
        eq_lines = list(system.custom.eq_lines)
        custom_params = parse_params(system.custom.params_text)
        auto_jac = bool(system.custom.auto_jacobian)
        use_jac = bool(system.custom.use_jacobian)

        jac_custom_func = None
        if auto_jac:
            rhs_custom_func, jac_custom_func = build_custom_rhs_and_jacobian(
                var_names, eq_lines, custom_params
            )
        else:
            rhs_custom_func = build_custom_rhs(var_names, eq_lines, custom_params)

        def rhs_fn(tt, xx):
            return rhs_custom_func(tt, xx)

        jac_fn = None
        if auto_jac and use_jac:
            if jac_custom_func is None:
                raise RuntimeError("Analytic Jacobian requested but not available.")
            jac_fn = lambda tt, xx: jac_custom_func(tt, xx)
    else:
        adapter = get_builtin(system.key)
        rhs_fn = adapter.rhs_builder(system)
        jac_fn = adapter.jac_builder(system)

    window = resolve_time_window(integration, lyapunov)

    if solver_kind == "rk4":
        param_keys = list(custom_params.keys()) if custom_params is not None else []
        lyap_nb, param_names = build_numba_lyap_solver(
            system.key, var_names, eq_lines, param_keys, auto_jac, use_jac
        )
        if lyap_nb is not None and param_names is not None:
            try:
                params_dict = (
                    custom_params
                    if custom_params is not None
                    else get_builtin(system.key).extract_params(system)
                )
                params_arr = np.array(
                    [float(params_dict[name]) for name in param_names], dtype=float
                )
                lambdas, _sums, _t_meas, n_qr, _x_final = lyap_nb(
                    np.asarray(y0, dtype=float),
                    float(integration.t0),
                    float(integration.dt),
                    float(window.t_transient),
                    float(window.t_measure),
                    int(window.qr_every_steps),
                    float(1e-8),
                    params_arr,
                )
                if int(n_qr) <= 0:
                    raise ValueError("No QR steps performed. Increase t_measure or reduce qr_every_steps.")
                return np.asarray(lambdas, dtype=float)
            except Exception:
                pass

    result = compute_lyapunov_spectrum(
        rhs=rhs_fn,
        x0=y0,
        t0=float(integration.t0),
        dt=float(integration.dt),
        t_transient=float(window.t_transient),
        t_measure=float(window.t_measure),
        qr_every_steps=window.qr_every_steps,
        solve_options=solve_options,
        solver_kind=solver_kind,
        auto_switch_rk4=auto_switch_rk4,
        jac=jac_fn,
    )
    return result.lambdas
