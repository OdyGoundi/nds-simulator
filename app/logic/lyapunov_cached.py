from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st

from app.helpers import build_custom_rhs, build_custom_rhs_and_jacobian, parse_params
from app.params import (
    InitialConditions,
    IntegrationConfig,
    LyapunovConfig,
    SolverTolerances,
    SystemConfig,
)
from app.services import get_builtin, resolve_solver, apply_to_solve_options
from core.lyapunov import compute_lyapunov_spectrum


@st.cache_data(show_spinner=False)
def compute_lyapunov_cached(
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

    rhs = None
    jac = None
    var_names: List[str] = []
    eq_lines: List[str] = []
    custom_params: Optional[Dict[str, float]] = None
    auto_jac = False
    use_jac = False

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

        def rhs_custom_wrapper(tt, xx):
            return rhs_custom_func(tt, xx)

        rhs = rhs_custom_wrapper
        jac = None
        if auto_jac and use_jac:
            if jac_custom_func is None:
                raise RuntimeError("Analytic Jacobian requested but not available.")
            jac = lambda tt, xx: jac_custom_func(tt, xx)
    else:
        adapter = get_builtin(system.key)
        rhs = adapter.rhs_builder(system)
        jac = adapter.jac_builder(system)

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

    if solver_kind == "rk4":
        try:
            from core import numba_backend
            if numba_backend.numba_available():
                lyap_nb = None
                params_arr = None
                if system.key in ("lorenz", "rossler", "henon_heiles"):
                    rhs_nb, jac_nb, param_names = numba_backend.build_builtin_system(system.key)
                    params_dict = get_builtin(system.key).extract_params(system)
                    params_arr = np.array(
                        [float(params_dict[name]) for name in param_names],
                        dtype=float,
                    )
                    lyap_nb = numba_backend.build_lyapunov_solver(rhs_nb, jac_nb, use_fd_jac=False)
                elif system.key == "custom":
                    from app import numba_custom
                    if custom_params is None:
                        raise ValueError("Custom parameters not initialized.")
                    param_names = list(custom_params.keys())
                    if auto_jac and use_jac:
                        rhs_nb, jac_nb = numba_custom.build_custom_numba_rhs_and_jacobian(
                            var_names, eq_lines, param_names
                        )
                        lyap_nb = numba_backend.build_lyapunov_solver(rhs_nb, jac_nb, use_fd_jac=False)
                    else:
                        rhs_nb = numba_custom.build_custom_numba_rhs(var_names, eq_lines, param_names)
                        lyap_nb = numba_backend.build_lyapunov_solver(rhs_nb, None, use_fd_jac=True)
                    params_arr = np.array([float(custom_params[name]) for name in param_names], dtype=float)

                if lyap_nb is not None and params_arr is not None:
                    lambdas, _sums, _t_meas, n_qr, _x_final = lyap_nb(
                        np.asarray(y0, dtype=float),
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
                    return np.asarray(lambdas, dtype=float)
        except Exception:
            pass

    result = compute_lyapunov_spectrum(
        rhs=rhs,
        x0=y0,
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
    return result.lambdas
