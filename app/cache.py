from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

from core.solver import integrate_system, integrate_system_rk4
from core.symplectic_solver import integrate_system_symplectic_fr
from app.helpers import (
    build_custom_rhs,
    build_custom_symplectic_functions,
    parse_params,
)
from app.params import (
    InitialConditions,
    IntegrationConfig,
    SolverTolerances,
    SystemConfig,
)
from app.services import get_builtin, resolve_solver, apply_to_solve_options


def _max_store_steps(integration: IntegrationConfig) -> Optional[int]:
    raw = getattr(integration, "max_store_steps", None)
    if raw is None:
        return None
    try:
        n = int(raw)
    except Exception:
        return None
    return n if n > 0 else None


def _custom_state(system: SystemConfig) -> Tuple[List[str], List[str], Optional[Dict[str, float]]]:
    """For custom systems return (var_names, eq_lines, parsed_params); else empty/None."""
    if system.key != "custom":
        return [], [], None
    custom = system.custom
    return list(custom.var_names), list(custom.eq_lines), parse_params(custom.params_text)


def _params_array_for_numba(
    system: SystemConfig,
    custom_params: Optional[Dict[str, float]],
    param_names: List[str],
) -> np.ndarray:
    """Order params according to the Numba kernel's expected param_names."""
    if system.key == "custom":
        if custom_params is None:
            raise ValueError("Custom parameters not initialized.")
        source = custom_params
    else:
        source = get_builtin(system.key).extract_params(system)
    return np.array([float(source[name]) for name in param_names], dtype=float)


def _try_numba_rk4(
    system: SystemConfig,
    integration: IntegrationConfig,
    y0: np.ndarray,
    max_store_steps: Optional[int],
    var_names: List[str],
    eq_lines: List[str],
    custom_params: Optional[Dict[str, float]],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Compile and run the Numba RK4 kernel. Return None if unavailable/unsupported."""
    try:
        from core import numba_backend
        if not numba_backend.numba_available():
            return None

        if system.key in ("lorenz", "rossler", "henon_heiles"):
            rhs_nb, _jac_nb, param_names_tpl = numba_backend.build_builtin_system(system.key)
            param_names = list(param_names_tpl)
        elif system.key == "custom":
            if custom_params is None:
                raise ValueError("Custom parameters not initialized.")
            from app import numba_custom
            param_names = list(custom_params.keys())
            rhs_nb = numba_custom.build_custom_numba_rhs(var_names, eq_lines, param_names)
        else:
            return None

        params_arr = _params_array_for_numba(system, custom_params, param_names)
        rk4_nb = numba_backend.build_rk4_integrator(rhs_nb)
        return rk4_nb(
            y0,
            float(integration.t0),
            float(integration.tf),
            float(integration.dt),
            int(max_store_steps) if max_store_steps is not None else 0,
            params_arr,
        )
    except Exception:
        return None


def _try_numba_symplectic(
    system: SystemConfig,
    integration: IntegrationConfig,
    y0: np.ndarray,
    max_store_steps: Optional[int],
    var_names: List[str],
    eq_lines: List[str],
    custom_params: Optional[Dict[str, float]],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Compile and run the Numba symplectic kernel. Return None if unavailable."""
    try:
        from core import numba_backend
        if not numba_backend.numba_available():
            return None

        if system.key == "henon_heiles":
            dq_dt_nb, dp_dt_nb, param_names_tpl = numba_backend.build_builtin_symplectic(system.key)
            param_names = list(param_names_tpl)
        elif system.key == "custom":
            if custom_params is None:
                raise ValueError("Custom parameters not initialized.")
            from app import numba_custom
            param_names = list(custom_params.keys())
            dq_dt_nb, dp_dt_nb = numba_custom.build_custom_numba_symplectic_functions(
                var_names, eq_lines, param_names
            )
        else:
            return None

        params_arr = _params_array_for_numba(system, custom_params, param_names)
        fr_nb = numba_backend.build_symplectic_fr_integrator(dq_dt_nb, dp_dt_nb)
        return fr_nb(
            y0,
            float(integration.t0),
            float(integration.tf),
            float(integration.dt),
            int(max_store_steps) if max_store_steps is not None else 0,
            params_arr,
        )
    except Exception:
        return None


def _build_scalar_symplectic_funcs(
    system: SystemConfig,
    var_names: List[str],
    eq_lines: List[str],
    custom_params: Optional[Dict[str, float]],
) -> Tuple[Callable[[float, np.ndarray], np.ndarray], Callable[[float, np.ndarray], np.ndarray]]:
    if system.key == "custom":
        if custom_params is None:
            raise ValueError("Custom parameters not initialized.")
        return build_custom_symplectic_functions(var_names, eq_lines, custom_params)
    adapter = get_builtin(system.key)
    if adapter.dq_dp_builder is None:
        raise ValueError("Symplectic solvers require Hamiltonian systems.")
    return adapter.dq_dp_builder(system)


@st.cache_data(show_spinner=False, max_entries=2)
def solve_cached(
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    solve_tols: SolverTolerances,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (t, y) — t shape (n_steps,), y shape (n_vars, n_steps)."""
    y0 = np.array(initial.y0, dtype=float)
    max_store_steps = _max_store_steps(integration)

    policy = resolve_solver(getattr(integration, "solver_kind", "ivp"))
    solver_kind = policy.kind
    solve_options: Dict[str, Any] = dict(solve_tols.to_dict())
    apply_to_solve_options(policy, solve_options)

    var_names, eq_lines, custom_params = _custom_state(system)
    if system.key == "custom":
        if custom_params is None:
            raise ValueError("Custom parameters not initialized.")
        rhs_fn = build_custom_rhs(var_names, eq_lines, custom_params)
    else:
        rhs_fn = get_builtin(system.key).rhs_builder(system)

    if solver_kind == "symplectic_fr":
        if y0.size % 2 != 0:
            raise ValueError("Symplectic solvers require an even number of variables [q..., p...].")
        if system.key not in ("custom", "henon_heiles"):
            raise ValueError("Symplectic solvers require Hamiltonian systems.")

        nb_result = _try_numba_symplectic(
            system, integration, y0, max_store_steps, var_names, eq_lines, custom_params
        )
        if nb_result is not None:
            return nb_result

        dq_dt_fn, dp_dt_fn = _build_scalar_symplectic_funcs(
            system, var_names, eq_lines, custom_params
        )
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
        nb_result = _try_numba_rk4(
            system, integration, y0, max_store_steps, var_names, eq_lines, custom_params
        )
        if nb_result is not None:
            return nb_result
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
