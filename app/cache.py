from typing import Any, Dict, Tuple

import numpy as np
import streamlit as st

from core.henon_heiles_system_rhs import (
    henon_heiles_dp_dt,
    henon_heiles_dq_dt,
    henon_heiles_rhs,
)
from core.lorenz_system_rhs import lorenz_rhs
from core.rossler_system_rhs import rossler_rhs
from core.solver import integrate_system, integrate_system_rk4
from core.symplectic_solver import (
    integrate_system_symplectic_fr,
    integrate_system_symplectic_verlet,
)
from core.poincare_sweep import (
    poincare_section,
    sweep_poincare,
    sweep_poincare_events_ivp,
    PoincareConfig,
    SweepConfig,
)
from app.helpers import (
    build_custom_rhs,
    build_custom_symplectic_functions,
    parse_params,
)
from app.params import (
    InitialConditions,
    IntegrationConfig,
    SolverTolerances,
    SweepRunConfig,
    SystemConfig,
)


@st.cache_data(show_spinner=False)
def solve_cached(
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    solve_tols: SolverTolerances,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (t, y):
      t: shape (n_steps,)
      y: shape (n_vars, n_steps)
    """
    y0 = np.array(initial.y0, dtype=float)
    solve_options: Dict[str, Any] = dict(solve_tols.to_dict())
    solver_kind = str(getattr(integration, "solver_kind", "ivp")).lower()
    method = None
    if solver_kind in ("rk45", "ivp"):
        method = "RK45"
    elif solver_kind == "dop853":
        method = "DOP853"
    if method is not None:
        solve_options["method"] = method

    if system.key == "lorenz":
        params = system.lorenz

        def rhs(t, y):
            return lorenz_rhs(t, y, sigma=params.sigma, rho=params.rho, beta=params.beta)

    elif system.key == "rossler":
        params = system.rossler

        def rhs(t, y):
            return rossler_rhs(t, y, a=params.a, b=params.b, c=params.c)

    elif system.key == "henon_heiles":
        params = system.henon_heiles

        def rhs(t, y):
            return henon_heiles_rhs(t, y, lam=params.lam)

    elif system.key == "custom":
        custom = system.custom
        var_names = list(custom.var_names)
        eq_lines = list(custom.eq_lines)
        params = parse_params(custom.params_text)
        rhs = build_custom_rhs(var_names, eq_lines, params)

    else:
        raise ValueError(f"Unknown system_key: {system.key}")

    if solver_kind in ("symplectic_verlet", "symplectic_fr"):
        if system.key == "custom":
            dq_dt, dp_dt = build_custom_symplectic_functions(var_names, eq_lines, params)
        elif system.key == "henon_heiles":
            def dq_dt(t, p):
                return henon_heiles_dq_dt(t, p, lam=params.lam)

            def dp_dt(t, q):
                return henon_heiles_dp_dt(t, q, lam=params.lam)
        else:
            raise ValueError("Symplectic solvers require Hamiltonian systems.")
        if solver_kind == "symplectic_verlet":
            sol = integrate_system_symplectic_verlet(
                rhs,
                t_span=(integration.t0, integration.tf),
                y0=y0,
                t_step=integration.dt,
                dp_dt=dp_dt,
                dq_dt=dq_dt,
            )
        else:
            sol = integrate_system_symplectic_fr(
                rhs,
                t_span=(integration.t0, integration.tf),
                y0=y0,
                t_step=integration.dt,
                dp_dt=dp_dt,
                dq_dt=dq_dt,
            )
    elif solver_kind == "rk4":
        sol = integrate_system_rk4(
            rhs,
            t_span=(integration.t0, integration.tf),
            y0=y0,
            t_step=integration.dt,
        )
    else:
        sol = integrate_system(
            rhs,
            t_span=(integration.t0, integration.tf),
            y0=y0,
            t_step=integration.dt,
            **solve_options,
        )
    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.t, sol.y


@st.cache_data(show_spinner=False)
def sweep_cached(
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    sweep: SweepConfig,
    poincare: PoincareConfig,
    run_cfg: SweepRunConfig,
    solve_tols: SolverTolerances,
    solver_kind: str = "ivp",
):
    import pandas as pd

    y0 = np.array(initial.y0, dtype=float)
    solve_options: Dict[str, Any] = dict(solve_tols.to_dict())
    solver_kind = str(solver_kind).lower()
    method = None
    if solver_kind in ("rk45", "ivp"):
        method = "RK45"
    elif solver_kind == "dop853":
        method = "DOP853"
    if method is not None:
        solve_options["method"] = method
    sweep_solver_kind = "ivp" if solver_kind in ("ivp", "rk45", "dop853") else solver_kind

    # Build base rhs + base_params (everything except swept param)
    if system.key == "lorenz":
        rhs_fn = lorenz_rhs
        base_params = {
            "sigma": float(system.lorenz.sigma),
            "rho": float(system.lorenz.rho),
            "beta": float(system.lorenz.beta),
        }
    elif system.key == "rossler":
        rhs_fn = rossler_rhs
        base_params = {
            "a": float(system.rossler.a),
            "b": float(system.rossler.b),
            "c": float(system.rossler.c),
        }
    elif system.key == "henon_heiles":
        rhs_fn = henon_heiles_rhs
        base_params = {
            "lambda": float(system.henon_heiles.lam),
        }
    elif system.key == "custom":
        custom = system.custom
        var_names = list(custom.var_names)
        eq_lines = list(custom.eq_lines)
        params = parse_params(custom.params_text)
        rhs_user = build_custom_rhs(var_names, eq_lines, params)

        rhs_fn = None  # handled below
        base_params = dict(params)
    else:
        raise ValueError(f"Unknown system_key: {system.key}")

    # -----------------------
    # EARLY RETURN: custom
    # -----------------------
    if system.key == "custom":
        custom = system.custom
        var_names = list(custom.var_names)
        eq_lines = list(custom.eq_lines)
        base_params = parse_params(custom.params_text)

        # Generate inclusive sweep values (safe for floats)
        if sweep.step <= 0:
            raise ValueError("Sweep step must be > 0.")
        n = int(np.floor((sweep.stop - sweep.start) / sweep.step + 1e-12)) + 1
        param_vals = sweep.start + sweep.step * np.arange(n, dtype=float)
        param_vals = param_vals[param_vals <= sweep.stop + 1e-12]

        rows = []
        ycol = f"y{int(run_cfg.output_index)}"

        for pv in param_vals:
            params2 = dict(base_params)
            params2[sweep.param_name] = float(pv)

            rhs2 = build_custom_rhs(var_names, eq_lines, params2)

            sol = integrate_system(
                rhs2,
                t_span=(integration.t0, integration.tf),
                y0=y0,
                t_step=integration.dt,
                **solve_options,
            )

            if not sol.success:
                continue

            t_hits, y_hits = poincare_section(sol.t, sol.y, poincare, params=params2)

            # Keep only last K Poincaré hits (bibliography-style)
            MAX_HITS = 100

            if t_hits.size > MAX_HITS:
                t_hits = t_hits[-MAX_HITS:]
                y_hits = y_hits[:, -MAX_HITS:]

            if t_hits.size == 0:
                continue

            for j in range(t_hits.size):
                rows.append({
                    sweep.param_name: float(pv),
                    "t_hit": float(t_hits[j]),
                    ycol: float(y_hits[int(run_cfg.output_index), j]),
                })

        return pd.DataFrame(rows)

    # -----------------------
    # Non-custom: use sweep_poincare
    # -----------------------
    if system.key == "lorenz":
        rhs_fn = lorenz_rhs
        base_params = {
            "sigma": float(system.lorenz.sigma),
            "rho": float(system.lorenz.rho),
            "beta": float(system.lorenz.beta),
        }
    elif system.key == "rossler":
        rhs_fn = rossler_rhs
        base_params = {
            "a": float(system.rossler.a),
            "b": float(system.rossler.b),
            "c": float(system.rossler.c),
        }
    elif system.key == "henon_heiles":
        rhs_fn = henon_heiles_rhs
        base_params = {
            "lambda": float(system.henon_heiles.lam),
        }
    else:
        raise ValueError(f"Unknown system_key: {system.key}")

    # Event-based fast path (only for ivp + crossing)
    if sweep_solver_kind == "ivp" and str(poincare.method).lower() == "crossing":
        df = sweep_poincare_events_ivp(
            rhs=rhs_fn,
            y0=tuple(y0),
            t_span=(float(integration.t0), float(integration.tf)),
            base_params=base_params,
            sweep=sweep,
            poincare=poincare,
            t_step=float(integration.dt),
            solve_options=solve_options,
            output_indices=[int(run_cfg.output_index)],
            include_all_state=False,
            warm_start=bool(run_cfg.warm_start),
            max_hits=int(run_cfg.max_hits),
            early_stop=bool(run_cfg.early_stop),
            chunk_time=float(run_cfg.chunk_time),
        )

    else:
        df = sweep_poincare(
            rhs=rhs_fn,
            y0=tuple(y0),
            t_span=(float(integration.t0), float(integration.tf)),
            base_params=base_params,
            sweep=sweep,
            poincare=poincare,
            solver_kind=str(sweep_solver_kind),
            t_step=float(integration.dt),
            solve_options=solve_options,
            output_indices=[int(run_cfg.output_index)],
            include_all_state=False,
            warm_start=bool(run_cfg.warm_start),
        )

    return df
