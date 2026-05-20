from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

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


def extract_lyapunov_params(system: SystemConfig) -> Dict[str, float]:
    """Parameter dict for any system type — custom uses parse_params, built-in uses adapter."""
    if system.key == "custom":
        return parse_params(system.custom.params_text)
    return get_builtin(system.key).extract_params(system)


def build_lyapunov_rhs_jac(
    system: SystemConfig,
    params: Dict[str, float],
) -> Tuple[RhsFn, Optional[JacFn]]:
    """Build (rhs, jac) callables suitable for compute_lyapunov_spectrum.

    For custom systems: parses var_names/eq_lines from system.custom and respects
    auto_jacobian / use_jacobian flags. For built-in systems: dispatches via the
    system_registry adapter.
    """
    if system.key == "custom":
        var_names = list(system.custom.var_names)
        eq_lines = list(system.custom.eq_lines)
        auto_jac = bool(system.custom.auto_jacobian)
        use_jac = bool(system.custom.use_jacobian)

        jac_custom = None
        if auto_jac:
            rhs_custom, jac_custom = build_custom_rhs_and_jacobian(var_names, eq_lines, params)
        else:
            rhs_custom = build_custom_rhs(var_names, eq_lines, params)

        def rhs_wrapped(tt: float, xx: np.ndarray) -> np.ndarray:
            return rhs_custom(tt, xx)

        jac: Optional[JacFn] = None
        if auto_jac and use_jac:
            if jac_custom is None:
                raise RuntimeError("Analytic Jacobian requested but not available.")
            jac_custom_fn: Callable[[float, np.ndarray], np.ndarray] = jac_custom

            def jac_wrapped(tt: float, xx: np.ndarray) -> np.ndarray:
                return jac_custom_fn(tt, xx)

            jac = jac_wrapped

        return rhs_wrapped, jac

    adapter = get_builtin(system.key)
    return adapter.rhs_from_dict(params), adapter.jac_from_dict(params)


def run_lyapunov_numba(
    lyap_nb: Any,
    param_names: List[str],
    params: Dict[str, float],
    y0: np.ndarray,
    t0: float,
    dt: float,
    window: LyapunovTimeWindow,
) -> Tuple[np.ndarray, np.ndarray]:
    """Invoke the Numba Lyapunov solver. Raises if no QR steps happened.

    Returns (lambdas, x_final) — lambdas in the natural order produced by the solver.
    """
    params_arr = np.array([float(params[name]) for name in param_names], dtype=float)
    lambdas, _sums, _t_meas, n_qr, x_final = lyap_nb(
        np.asarray(y0, dtype=float),
        float(t0),
        float(dt),
        float(window.t_transient),
        float(window.t_measure),
        int(window.qr_every_steps),
        float(1e-8),
        params_arr,
    )
    if int(n_qr) <= 0:
        raise ValueError("No QR steps performed. Increase t_measure or reduce qr_every_steps.")
    return np.asarray(lambdas, dtype=float), np.asarray(x_final, dtype=float)


def run_lyapunov_scipy(
    rhs: RhsFn,
    jac: Optional[JacFn],
    y0: np.ndarray,
    t0: float,
    dt: float,
    window: LyapunovTimeWindow,
    solve_options: Dict[str, Any],
    solver_kind: str,
    auto_switch_rk4: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Invoke compute_lyapunov_spectrum and return (lambdas, x_final)."""
    res = compute_lyapunov_spectrum(
        rhs=rhs,
        x0=np.asarray(y0, dtype=float),
        t0=float(t0),
        dt=float(dt),
        t_transient=float(window.t_transient),
        t_measure=float(window.t_measure),
        qr_every_steps=int(window.qr_every_steps),
        solve_options=solve_options,
        solver_kind=solver_kind,
        auto_switch_rk4=auto_switch_rk4,
        jac=jac,
    )
    return np.asarray(res.lambdas, dtype=float), np.asarray(res.x_final, dtype=float)


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
    params = extract_lyapunov_params(system)
    window = resolve_time_window(integration, lyapunov)

    if solver_kind == "rk4":
        var_names = list(system.custom.var_names) if system.key == "custom" else []
        eq_lines = list(system.custom.eq_lines) if system.key == "custom" else []
        auto_jac = bool(system.custom.auto_jacobian) if system.key == "custom" else False
        use_jac = bool(system.custom.use_jacobian) if system.key == "custom" else False
        param_keys = list(params.keys()) if system.key == "custom" else []
        lyap_nb, param_names = build_numba_lyap_solver(
            system.key, var_names, eq_lines, param_keys, auto_jac, use_jac
        )
        if lyap_nb is not None and param_names is not None:
            try:
                lambdas, _x_final = run_lyapunov_numba(
                    lyap_nb, param_names, params, y0,
                    float(integration.t0), float(integration.dt), window,
                )
                return lambdas
            except Exception:
                pass

    rhs, jac = build_lyapunov_rhs_jac(system, params)
    lambdas, _x_final = run_lyapunov_scipy(
        rhs, jac, y0,
        float(integration.t0), float(integration.dt), window,
        solve_options, solver_kind, auto_switch_rk4,
    )
    return lambdas


def compute_lyapunov_cached(
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    lyapunov: LyapunovConfig,
    solve_tols: SolverTolerances,
) -> np.ndarray:
    """Cached wrapper for compute_single_lyapunov using Streamlit cache when available."""
    if st is None:
        # Streamlit not available; return direct computation
        return compute_single_lyapunov(system, integration, initial, lyapunov, solve_tols)

    # Use Streamlit caching at the service boundary
    @st.cache_data(show_spinner=False)
    def _cached_impl(
        sys_repr: str,
        integ_repr: str,
        init_repr: str,
        lyap_repr: str,
        tol_repr: str,
    ) -> np.ndarray:
        return compute_single_lyapunov(system, integration, initial, lyapunov, solve_tols)

    return _cached_impl(
        repr(system),
        repr(integration),
        repr(initial),
        repr(lyapunov),
        repr(solve_tols),
    )
