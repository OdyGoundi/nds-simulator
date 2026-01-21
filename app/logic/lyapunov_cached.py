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
from core.jacobians_fixed_systems import lorenz_jac, rossler_jac
from core.lorenz_system_rhs import lorenz_rhs
from core.rossler_system_rhs import rossler_rhs
from core.lyapunov import compute_lyapunov_spectrum


@st.cache_data(show_spinner=False)
def compute_lyapunov_cached(
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    lyapunov: LyapunovConfig,
    solve_tols: SolverTolerances,
) -> np.ndarray:
    solve_options = solve_tols.to_dict()
    y0 = np.array(initial.y0, dtype=float)

    if system.key == "lorenz":
        params = system.lorenz

        def rhs_lorenz(tt, xx):
            return lorenz_rhs(tt, xx, sigma=params.sigma, rho=params.rho, beta=params.beta)

        rhs = rhs_lorenz
        jac = lambda tt, xx: lorenz_jac(tt, xx, sigma=params.sigma, rho=params.rho, beta=params.beta)

    elif system.key == "rossler":
        params = system.rossler

        def rhs_rossler(tt, xx):
            return rossler_rhs(tt, xx, a=params.a, b=params.b, c=params.c)

        rhs = rhs_rossler
        jac = lambda tt, xx: rossler_jac(tt, xx, a=params.a, b=params.b, c=params.c)

    elif system.key == "custom":
        var_names = list(system.custom.var_names)
        eq_lines = list(system.custom.eq_lines)
        params = parse_params(system.custom.params_text)
        auto_jac = bool(system.custom.auto_jacobian)
        use_jac = bool(system.custom.use_jacobian)

        if auto_jac:
            rhs_custom_func, jac_custom_func = build_custom_rhs_and_jacobian(
                var_names, eq_lines, params
            )
        else:
            rhs_custom_func = build_custom_rhs(var_names, eq_lines, params)
            jac_custom_func = None

        def rhs_custom_wrapper(tt, xx):
            return rhs_custom_func(tt, xx)

        rhs = rhs_custom_wrapper
        jac = (lambda tt, xx: jac_custom_func(tt, xx)) if (auto_jac and use_jac) else None

    else:
        raise ValueError(f"Unknown system_key: {system.key}")

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

    result = compute_lyapunov_spectrum(
        rhs=rhs,
        x0=y0,
        t0=float(integration.t0),
        dt=float(integration.dt),
        t_transient=float(t_transient),
        t_measure=float(t_measure),
        qr_every_steps=qr_every_steps,
        solve_options=solve_options,
        jac=jac,
    )
    return result.lambdas
