from typing import Any, Callable, Dict, List, Tuple

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
from core.symplectic_solver import integrate_system_symplectic_fr
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
from app.services import get_builtin


@st.cache_data(show_spinner=False, max_entries=2)
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
    max_store_steps_obj = getattr(integration, "max_store_steps", None)
    max_store_steps: int | None = None
    if max_store_steps_obj is not None:
        try:
            max_store_steps_i = int(max_store_steps_obj)
            if max_store_steps_i > 0:
                max_store_steps = max_store_steps_i
        except Exception:
            max_store_steps = None

    solve_options: Dict[str, Any] = dict(solve_tols.to_dict())
    solver_kind = str(getattr(integration, "solver_kind", "ivp")).lower()
    if solver_kind == "symplectic_verlet":
        solver_kind = "symplectic_fr"
    method = None
    if solver_kind in ("rk45", "ivp"):
        method = "RK45"
    elif solver_kind == "dop853":
        method = "DOP853"
    if method is not None:
        solve_options["method"] = method

    rhs_fn = None
    var_names: List[str] = []
    eq_lines: List[str] = []
    custom_params: Dict[str, float] | None = None
    if system.key == "custom":
        custom = system.custom
        var_names = list(custom.var_names)
        eq_lines = list(custom.eq_lines)
        custom_params = parse_params(custom.params_text)
        rhs_fn = build_custom_rhs(var_names, eq_lines, custom_params)
    else:
        rhs_fn = get_builtin(system.key).rhs_builder(system)

    if solver_kind == "symplectic_fr":
        if y0.size % 2 != 0:
            raise ValueError("Symplectic solvers require an even number of variables [q..., p...].")
        if system.key not in ("custom", "henon_heiles"):
            raise ValueError("Symplectic solvers require Hamiltonian systems.")
        try:
            from core import numba_backend
            if numba_backend.numba_available():
                if system.key == "henon_heiles":
                    dq_dt_nb, dp_dt_nb, param_names = numba_backend.build_builtin_symplectic(system.key)
                    params_dict = get_builtin(system.key).extract_params(system)
                    params_arr = np.array(
                        [float(params_dict[name]) for name in param_names],
                        dtype=float,
                    )
                else:
                    from app import numba_custom
                    if custom_params is None:
                        raise ValueError("Custom parameters not initialized.")
                    param_names = list(custom_params.keys())
                    dq_dt_nb, dp_dt_nb = numba_custom.build_custom_numba_symplectic_functions(
                        var_names, eq_lines, param_names
                    )
                    params_arr = np.array([float(custom_params[name]) for name in param_names], dtype=float)
                fr_nb = numba_backend.build_symplectic_fr_integrator(dq_dt_nb, dp_dt_nb)
                t_arr, y_arr = fr_nb(
                    y0,
                    float(integration.t0),
                    float(integration.tf),
                    float(integration.dt),
                    int(max_store_steps) if max_store_steps is not None else 0,
                    params_arr,
                )
                return t_arr, y_arr
        except Exception:
            pass

        dq_dt_fn: Callable[[float, np.ndarray], np.ndarray]
        dp_dt_fn: Callable[[float, np.ndarray], np.ndarray]
        if system.key == "custom":
            if custom_params is None:
                raise ValueError("Custom parameters not initialized.")
            dq_dt_fn, dp_dt_fn = build_custom_symplectic_functions(var_names, eq_lines, custom_params)
        elif system.key == "henon_heiles":
            dq_dt_fn, dp_dt_fn = get_builtin(system.key).dq_dp_builder(system)
        else:
            raise ValueError("Symplectic solvers require Hamiltonian systems.")
        sol = integrate_system_symplectic_fr(
            rhs_fn,
            t_span=(integration.t0, integration.tf),
            y0=y0,
            t_step=integration.dt,
            max_store_steps=max_store_steps,
            dp_dt=dp_dt_fn,
            dq_dt=dq_dt_fn,
        )
    elif solver_kind == "rk4":
        try:
            from core import numba_backend
            if numba_backend.numba_available():
                if system.key in ("lorenz", "rossler", "henon_heiles"):
                    rhs_nb, _jac_nb, param_names = numba_backend.build_builtin_system(system.key)
                    params_dict = get_builtin(system.key).extract_params(system)
                    params_arr = np.array(
                        [float(params_dict[name]) for name in param_names],
                        dtype=float,
                    )
                    rk4_nb = numba_backend.build_rk4_integrator(rhs_nb)
                    t_arr, y_arr = rk4_nb(
                        y0,
                        float(integration.t0),
                        float(integration.tf),
                        float(integration.dt),
                        int(max_store_steps) if max_store_steps is not None else 0,
                        params_arr,
                    )
                    return t_arr, y_arr
                if system.key == "custom":
                    from app import numba_custom
                    if custom_params is None:
                        raise ValueError("Custom parameters not initialized.")
                    param_names = list(custom_params.keys())
                    rhs_nb = numba_custom.build_custom_numba_rhs(var_names, eq_lines, param_names)
                    params_arr = np.array([float(custom_params[name]) for name in param_names], dtype=float)
                    rk4_nb = numba_backend.build_rk4_integrator(rhs_nb)
                    t_arr, y_arr = rk4_nb(
                        y0,
                        float(integration.t0),
                        float(integration.tf),
                        float(integration.dt),
                        int(max_store_steps) if max_store_steps is not None else 0,
                        params_arr,
                    )
                    return t_arr, y_arr
        except Exception:
            pass
        sol = integrate_system_rk4(
            rhs_fn,
            t_span=(integration.t0, integration.tf),
            y0=y0,
            t_step=integration.dt,
            max_store_steps=max_store_steps,
        )
    else:
        sol = integrate_system(
            rhs_fn,
            t_span=(integration.t0, integration.tf),
            y0=y0,
            t_step=integration.dt,
            max_store_steps=max_store_steps,
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
    if system.key == "custom":
        custom = system.custom
        var_names = list(custom.var_names)
        eq_lines = list(custom.eq_lines)
        params = parse_params(custom.params_text)
        rhs_user = build_custom_rhs(var_names, eq_lines, params)

        rhs_fn = None  # handled below
        base_params = dict(params)
    else:
        adapter = get_builtin(system.key)
        rhs_fn = adapter.rhs_fn
        base_params = adapter.extract_params(system)

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
    adapter = get_builtin(system.key)
    rhs_fn = adapter.rhs_fn
    base_params = adapter.extract_params(system)

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
