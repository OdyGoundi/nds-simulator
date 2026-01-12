from typing import Dict, List, Optional, Tuple

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


@st.cache_data(show_spinner=False)
def solve_cached(system_key: str,
                 t0: float, tf: float, dt: float,
                 y0_tuple: Tuple[float, ...],
                 # Lorenz:
                 sigma: float, rho: float, beta: float,
                 # Rossler:
                 ross_a: float, ross_b: float, ross_c: float,
                 # Custom:
                 var_names_tuple: Tuple[str, ...],
                 eq_lines_tuple: Tuple[str, ...],
                 params_text: str,
                 solve_options: Optional[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (t, y):
      t: shape (n_steps,)
      y: shape (n_vars, n_steps)
    """
    y0 = np.array(y0_tuple, dtype=float)
    solve_options = dict(solve_options or {})

    if system_key == "lorenz":
        def rhs(t, y):
            return lorenz_rhs(t, y, sigma=sigma, rho=rho, beta=beta)

    elif system_key == "rossler":
        def rhs(t, y):
            return rossler_rhs(t, y, a=ross_a, b=ross_b, c=ross_c)

    elif system_key == "custom":
        var_names = list(var_names_tuple)
        eq_lines = list(eq_lines_tuple)
        params = parse_params(params_text)
        rhs = build_custom_rhs(var_names, eq_lines, params)

    else:
        raise ValueError(f"Unknown system_key: {system_key}")

    opts = dict(solve_options or {})
    sol = integrate_system(rhs, t_span=(t0, tf), y0=y0, t_step=dt, **opts)
    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.t, sol.y


@st.cache_data(show_spinner=False)
def sweep_cached(
    system_key: str,
    t0: float, tf: float, dt: float,
    y0_tuple: Tuple[float, ...],
    # built-in params
    sigma: float, rho: float, beta: float,
    ross_a: float, ross_b: float, ross_c: float,
    # custom definitions
    var_names_tuple: Tuple[str, ...],
    eq_lines_tuple: Tuple[str, ...],
    params_text: str,
    # sweep + poincare settings
    sweep_param: str, sweep_start: float, sweep_stop: float, sweep_step: float,
    section_index: int, section_value: float, direction: int,
    method: str, tol: float, transient_steps: int,
    # output selection
    output_index: int,
    solve_options: Optional[Dict[str, float]],
    solver_kind: str = "ivp",
    warm_start: bool = False,
    max_hits: int = 100,
    early_stop: bool = True,
    chunk_time: float = 2.0,
):
    import pandas as pd

    y0 = np.array(y0_tuple, dtype=float)

    # Build base rhs + base_params (everything except swept param)
    if system_key == "lorenz":
        rhs_fn = lorenz_rhs
        base_params = {"sigma": float(sigma), "rho": float(rho), "beta": float(beta)}
    elif system_key == "rossler":
        rhs_fn = rossler_rhs
        base_params = {"a": float(ross_a), "b": float(ross_b), "c": float(ross_c)}
    elif system_key == "custom":
        var_names = list(var_names_tuple)
        eq_lines = list(eq_lines_tuple)
        params = parse_params(params_text)
        rhs_user = build_custom_rhs(var_names, eq_lines, params)

        rhs_fn = None  # handled below
        base_params = dict(params)
    else:
        raise ValueError(f"Unknown system_key: {system_key}")

    sweep = SweepConfig(param_name=str(sweep_param), start=float(sweep_start),
                        stop=float(sweep_stop), step=float(sweep_step))

    transient_steps_sweep = int(transient_steps)
    poincare = PoincareConfig(
        section_index=int(section_index),
        section_value=float(section_value),
        direction=int(direction),
        method=str(method),
        tol=float(tol),
        transient_steps=transient_steps_sweep,
    )

    # -----------------------
    # EARLY RETURN: custom
    # -----------------------
    if system_key == "custom":
        var_names = list(var_names_tuple)
        eq_lines = list(eq_lines_tuple)
        base_params = parse_params(params_text)

        # Generate inclusive sweep values (safe for floats)
        if sweep.step <= 0:
            raise ValueError("Sweep step must be > 0.")
        n = int(np.floor((sweep.stop - sweep.start) / sweep.step + 1e-12)) + 1
        param_vals = sweep.start + sweep.step * np.arange(n, dtype=float)
        param_vals = param_vals[param_vals <= sweep.stop + 1e-12]

        rows = []
        ycol = f"y{int(output_index)}"

        for pv in param_vals:
            params2 = dict(base_params)
            params2[sweep.param_name] = float(pv)

            rhs2 = build_custom_rhs(var_names, eq_lines, params2)

            sol = integrate_system(rhs2, t_span=(t0, tf), y0=y0, t_step=dt, **solve_options)

            if not sol.success:
                continue

            t_hits, y_hits = poincare_section(sol.t, sol.y, poincare)

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
                    ycol: float(y_hits[int(output_index), j]),
                })

        return pd.DataFrame(rows)

    # -----------------------
    # Non-custom: use sweep_poincare
    # -----------------------
    if system_key == "lorenz":
        rhs_fn = lorenz_rhs
        base_params = {"sigma": float(sigma), "rho": float(rho), "beta": float(beta)}
    elif system_key == "rossler":
        rhs_fn = rossler_rhs
        base_params = {"a": float(ross_a), "b": float(ross_b), "c": float(ross_c)}
    else:
        raise ValueError(f"Unknown system_key: {system_key}")

    # Event-based fast path (only for ivp + crossing)
    if str(solver_kind).lower() == "ivp" and str(method).lower() == "crossing":
        df = sweep_poincare_events_ivp(
            rhs=rhs_fn,
            y0=tuple(y0),
            t_span=(float(t0), float(tf)),
            base_params=base_params,
            sweep=sweep,
            poincare=poincare,
            t_step=float(dt),
            solve_options=solve_options,
            output_indices=[int(output_index)],
            include_all_state=False,
            warm_start=bool(warm_start),
            max_hits=int(max_hits),
            early_stop=bool(early_stop),
            chunk_time=float(chunk_time),
        )

    else:
        df = sweep_poincare(
            rhs=rhs_fn,
            y0=tuple(y0),
            t_span=(float(t0), float(tf)),
            base_params=base_params,
            sweep=sweep,
            poincare=poincare,
            solver_kind=str(solver_kind),
            t_step=float(dt),
            solve_options=solve_options,
            output_indices=[int(output_index)],
            include_all_state=False,
            warm_start=bool(warm_start),
        )

    return df
