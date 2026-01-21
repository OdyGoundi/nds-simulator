from typing import Tuple

import numpy as np
import streamlit as st

from core.lorenz_system_rhs import lorenz_rhs
from core.rossler_system_rhs import rossler_rhs
from core.solver import integrate_system
from core.poincare_sweep import (
    poincare_section,
    sweep_poincare,
    sweep_poincare_events_ivp,
    PoincareConfig,
    SweepConfig,
)
from app.helpers import parse_params, build_custom_rhs
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
    solve_options = solve_tols.to_dict()

    if system.key == "lorenz":
        params = system.lorenz

        def rhs(t, y):
            return lorenz_rhs(t, y, sigma=params.sigma, rho=params.rho, beta=params.beta)

    elif system.key == "rossler":
        params = system.rossler

        def rhs(t, y):
            return rossler_rhs(t, y, a=params.a, b=params.b, c=params.c)

    elif system.key == "custom":
        custom = system.custom
        var_names = list(custom.var_names)
        eq_lines = list(custom.eq_lines)
        params = parse_params(custom.params_text)
        rhs = build_custom_rhs(var_names, eq_lines, params)

    else:
        raise ValueError(f"Unknown system_key: {system.key}")

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
    solve_options = solve_tols.to_dict()

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
    else:
        raise ValueError(f"Unknown system_key: {system.key}")

    # Event-based fast path (only for ivp + crossing)
    if str(solver_kind).lower() == "ivp" and str(poincare.method).lower() == "crossing":
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
            solver_kind=str(solver_kind),
            t_step=float(integration.dt),
            solve_options=solve_options,
            output_indices=[int(run_cfg.output_index)],
            include_all_state=False,
            warm_start=bool(run_cfg.warm_start),
        )

    return df
