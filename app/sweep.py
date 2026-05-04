from collections import deque
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None

from core.henon_heiles_system_rhs import (
    henon_heiles_dp_dt,
    henon_heiles_dq_dt,
    henon_heiles_rhs,
)
from core.poincare_sweep import (
    poincare_section,
    sweep_poincare,
    sweep_poincare_events_ivp,
    PoincareConfig,
    SweepConfig,
)
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
    SweepRunConfig,
    SystemConfig,
)
from app.services import get_builtin, resolve_solver, apply_to_solve_options

DEFAULT_MAX_KEEP = 100
OBSERVABLE_POINCARE = "poincare"
OBSERVABLE_EXTREMA = "extrema"
EXTREMA_KINDS = {"max", "min", "both"}
MAX_SWEEP_ROWS_BUDGET = 300_000


def _local_extrema_indices(y: np.ndarray, kind: str) -> np.ndarray:
    """
    Return indices i where y[i] is a local extremum.
    kind: "max", "min", "both"
    """
    if y.size < 3:
        return np.array([], dtype=int)

    y0 = y[:-2]
    y1 = y[1:-1]
    y2 = y[2:]

    if kind == "max":
        mask = (y1 > y0) & (y1 > y2)
    elif kind == "min":
        mask = (y1 < y0) & (y1 < y2)
    elif kind == "both":
        mask = ((y1 > y0) & (y1 > y2)) | ((y1 < y0) & (y1 < y2))
    else:
        raise ValueError(f"Unknown extrema kind: {kind}")

    return np.where(mask)[0] + 1  # shift because we used y[1:-1]


def _normalize_observable(observable: str) -> str:
    obs = str(observable or OBSERVABLE_POINCARE).strip().lower()
    if obs not in (OBSERVABLE_POINCARE, OBSERVABLE_EXTREMA):
        raise ValueError(f"Unknown observable: {observable}")
    return obs


def _normalize_extrema_kind(extrema_kind: str) -> str:
    kind = str(extrema_kind or "max").strip().lower()
    if kind not in EXTREMA_KINDS:
        raise ValueError(f"Unknown extrema kind: {extrema_kind}")
    return kind


def _estimate_param_count(sweep: SweepConfig) -> int:
    step = float(sweep.step)
    if step <= 0:
        return 0
    span = float(sweep.stop) - float(sweep.start)
    if span < 0:
        return 0
    return int(np.floor(span / step + 1e-12)) + 1


def _effective_max_hits(max_hits: int, sweep: SweepConfig, max_rows_budget: int = MAX_SWEEP_ROWS_BUDGET) -> int:
    max_hits_i = max(1, int(max_hits))
    n_params = max(1, _estimate_param_count(sweep))
    max_hits_budget = max(1, int(max_rows_budget) // n_params)
    return max(1, min(max_hits_i, max_hits_budget))


def _param_values(sweep: SweepConfig) -> np.ndarray:
    """Inclusive parameter grid from sweep start/stop/step."""
    if float(sweep.step) <= 0:
        raise ValueError("Sweep step must be > 0.")
    n = int(np.floor((float(sweep.stop) - float(sweep.start)) / float(sweep.step) + 1e-12)) + 1
    vals = float(sweep.start) + float(sweep.step) * np.arange(n, dtype=float)
    return vals[vals <= float(sweep.stop) + 1e-12]


def _clip_rows_result(rows_obj, max_rows: int = MAX_SWEEP_ROWS_BUDGET):
    max_rows_i = max(1, int(max_rows))
    if pd is not None and isinstance(rows_obj, pd.DataFrame):
        if len(rows_obj) <= max_rows_i:
            return rows_obj
        return rows_obj.tail(max_rows_i).reset_index(drop=True)
    rows_list = list(rows_obj)
    if len(rows_list) <= max_rows_i:
        return rows_list
    return rows_list[-max_rows_i:]


def _collect_observable_hits(
    *,
    t_arr: np.ndarray,
    y_arr: np.ndarray,
    output_index: int,
    max_keep: int,
    observable: str,
    extrema_kind: str,
    poincare: PoincareConfig,
    params: Dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    t_data = np.asarray(t_arr, dtype=float).ravel()
    y_data = np.asarray(y_arr, dtype=float)
    if y_data.ndim != 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    out_idx = int(output_index)
    if out_idx < 0 or out_idx >= int(y_data.shape[0]):
        raise ValueError(f"output_index out of bounds: {out_idx}")

    max_keep_safe = max(1, int(max_keep))

    if observable == OBSERVABLE_EXTREMA:
        n_steps = min(int(t_data.size), int(y_data.shape[1]))
        if n_steps <= 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        transient_steps = max(0, int(getattr(poincare, "transient_steps", 0)))
        if transient_steps >= n_steps:
            return np.array([], dtype=float), np.array([], dtype=float)
        t_use = np.asarray(t_data[:n_steps], dtype=float)[transient_steps:]
        y_use = np.asarray(y_data[out_idx, :n_steps], dtype=float)[transient_steps:]
        if t_use.size != y_use.size:
            n_pair = min(int(t_use.size), int(y_use.size))
            t_use = t_use[:n_pair]
            y_use = y_use[:n_pair]
        idx = _local_extrema_indices(y_use, extrema_kind)
        if idx.size > max_keep_safe:
            idx = idx[-max_keep_safe:]
        return np.asarray(t_use[idx], dtype=float), np.asarray(y_use[idx], dtype=float)

    t_hits, y_hits = poincare_section(t_data, y_data, poincare, params=params)
    t_hits_arr = np.asarray(t_hits, dtype=float)
    if t_hits_arr.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    y_hits_arr = np.asarray(y_hits, dtype=float)
    y_out = np.asarray(y_hits_arr[out_idx, :], dtype=float)
    if t_hits_arr.size != y_out.size:
        n_pair = min(int(t_hits_arr.size), int(y_out.size))
        t_hits_arr = t_hits_arr[:n_pair]
        y_out = y_out[:n_pair]
    if t_hits_arr.size > max_keep_safe:
        t_hits_arr = t_hits_arr[-max_keep_safe:]
        y_out = y_out[-max_keep_safe:]
    return t_hits_arr, y_out


def _append_hit_rows(rows: deque, sweep: SweepConfig, pv: float, ycol: str, t_hits: np.ndarray, y_hits: np.ndarray) -> None:
    for j in range(t_hits.size):
        rows.append({
            str(sweep.param_name): float(pv),
            "t_hit": float(t_hits[j]),
            ycol: float(y_hits[j]),
        })


def _run_custom_chunk(
    *,
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    sweep: SweepConfig,
    poincare: PoincareConfig,
    observable_lc: str,
    extrema_kind_lc: str,
    run_cfg: SweepRunConfig,
    solve_options_any: Dict[str, Any],
    sweep_solver_kind: str,
    max_hits_effective: int,
) -> List[dict]:
    base_params = parse_params(system.custom.params_text)
    param_vals = _param_values(sweep)

    rows: deque = deque(maxlen=MAX_SWEEP_ROWS_BUDGET)
    ycol = f"y{int(run_cfg.output_index)}"
    max_keep = int(max_hits_effective)

    y0_base = np.array(initial.y0, dtype=float).copy()
    y0_curr = y0_base.copy()
    param_names = list(base_params.keys())

    use_numba_rk4 = False
    rk4_nb = None
    use_numba_symplectic = False
    symplectic_nb = None

    if sweep_solver_kind == "rk4":
        try:
            from core import numba_backend
            if numba_backend.numba_available():
                from app import numba_custom
                rhs_nb = numba_custom.build_custom_numba_rhs(
                    list(system.custom.var_names),
                    list(system.custom.eq_lines),
                    param_names,
                )
                rk4_nb = numba_backend.build_rk4_integrator(rhs_nb)
                use_numba_rk4 = True
        except Exception:
            use_numba_rk4 = False
    if sweep_solver_kind == "symplectic_fr":
        try:
            from core import numba_backend
            if numba_backend.numba_available():
                from app import numba_custom
                dq_dt_nb, dp_dt_nb = numba_custom.build_custom_numba_symplectic_functions(
                    list(system.custom.var_names),
                    list(system.custom.eq_lines),
                    param_names,
                )
                symplectic_nb = numba_backend.build_symplectic_fr_integrator(dq_dt_nb, dp_dt_nb)
                use_numba_symplectic = True
        except Exception:
            use_numba_symplectic = False

    for pv in param_vals:
        params2 = dict(base_params)
        params2[str(sweep.param_name)] = float(pv)

        if sweep_solver_kind == "symplectic_fr":
            if use_numba_symplectic and symplectic_nb is not None:
                params_arr = np.array([float(params2[name]) for name in param_names], dtype=float)
                t_arr, y_arr = symplectic_nb(
                    y0_curr,
                    float(integration.t0),
                    float(integration.tf),
                    float(integration.dt),
                    0,
                    params_arr,
                )
            else:
                dq_dt_fn, dp_dt_fn = build_custom_symplectic_functions(
                    list(system.custom.var_names),
                    list(system.custom.eq_lines),
                    params2,
                )
                sol = integrate_system_symplectic_fr(
                    None,
                    t_span=(float(integration.t0), float(integration.tf)),
                    y0=y0_curr,
                    t_step=float(integration.dt),
                    dp_dt=dp_dt_fn,
                    dq_dt=dq_dt_fn,
                )
                if not sol.success:
                    if not run_cfg.warm_start:
                        y0_curr = y0_base.copy()
                    continue
                t_arr = sol.t
                y_arr = sol.y
        elif use_numba_rk4 and rk4_nb is not None:
            params_arr = np.array([float(params2[name]) for name in param_names], dtype=float)
            t_arr, y_arr = rk4_nb(
                y0_curr,
                float(integration.t0),
                float(integration.tf),
                float(integration.dt),
                0,
                params_arr,
            )
        else:
            rhs2 = build_custom_rhs(
                list(system.custom.var_names),
                list(system.custom.eq_lines),
                params2,
            )

            if sweep_solver_kind == "rk4":
                sol = integrate_system_rk4(
                    rhs2,
                    t_span=(float(integration.t0), float(integration.tf)),
                    y0=y0_curr,
                    t_step=float(integration.dt),
                )
            else:
                sol = integrate_system(
                    rhs2,
                    t_span=(float(integration.t0), float(integration.tf)),
                    y0=y0_curr,
                    t_step=float(integration.dt),
                    **solve_options_any,
                )
            if not sol.success:
                if not run_cfg.warm_start:
                    y0_curr = y0_base.copy()
                continue
            t_arr = sol.t
            y_arr = sol.y

        if run_cfg.warm_start:
            y0_curr = np.array(y_arr[:, -1], dtype=float).copy()
        else:
            y0_curr = y0_base.copy()

        t_hits, y_hits = _collect_observable_hits(
            t_arr=t_arr,
            y_arr=y_arr,
            output_index=int(run_cfg.output_index),
            max_keep=max_keep,
            observable=observable_lc,
            extrema_kind=extrema_kind_lc,
            poincare=poincare,
            params=params2,
        )
        if t_hits.size == 0:
            continue
        _append_hit_rows(rows, sweep, pv, ycol, t_hits, y_hits)

    return list(rows)


def _run_symplectic_builtin_chunk(
    *,
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    sweep: SweepConfig,
    poincare: PoincareConfig,
    observable_lc: str,
    extrema_kind_lc: str,
    run_cfg: SweepRunConfig,
    base_params: Dict[str, float],
    max_hits_effective: int,
) -> List[dict]:
    if system.key != "henon_heiles":
        raise ValueError("Symplectic sweeps require a Hamiltonian system (Henon-Heiles or custom).")

    param_vals = _param_values(sweep)

    rows: deque = deque(maxlen=MAX_SWEEP_ROWS_BUDGET)
    ycol = f"y{int(run_cfg.output_index)}"
    max_keep = int(max_hits_effective)

    y0_base = np.array(initial.y0, dtype=float).copy()
    y0_curr = y0_base.copy()

    use_numba_symplectic = False
    symplectic_nb = None
    param_names_list = ["lambda"]

    try:
        from core import numba_backend
        if numba_backend.numba_available():
            dq_dt_nb, dp_dt_nb, param_names = numba_backend.build_builtin_symplectic(system.key)
            param_names_list = list(param_names)
            symplectic_nb = numba_backend.build_symplectic_fr_integrator(dq_dt_nb, dp_dt_nb)
            use_numba_symplectic = True
    except Exception:
        use_numba_symplectic = False

    for pv in param_vals:
        params2 = dict(base_params)
        params2[str(sweep.param_name)] = float(pv)

        if use_numba_symplectic and symplectic_nb is not None:
            params_arr = np.array([float(params2[name]) for name in param_names_list], dtype=float)
            t_arr, y_arr = symplectic_nb(
                y0_curr,
                float(integration.t0),
                float(integration.tf),
                float(integration.dt),
                0,
                params_arr,
            )
        else:
            lam = float(params2["lambda"])

            def dq_dt_hh(t, p):
                return henon_heiles_dq_dt(t, p, lam=lam)

            def dp_dt_hh(t, q):
                return henon_heiles_dp_dt(t, q, lam=lam)

            sol = integrate_system_symplectic_fr(
                henon_heiles_rhs,
                t_span=(float(integration.t0), float(integration.tf)),
                y0=y0_curr,
                t_step=float(integration.dt),
                dp_dt=dp_dt_hh,
                dq_dt=dq_dt_hh,
            )
            if not sol.success:
                if not run_cfg.warm_start:
                    y0_curr = y0_base.copy()
                continue
            t_arr = sol.t
            y_arr = sol.y

        if run_cfg.warm_start:
            y0_curr = np.array(y_arr[:, -1], dtype=float).copy()
        else:
            y0_curr = y0_base.copy()

        t_hits, y_hits = _collect_observable_hits(
            t_arr=t_arr,
            y_arr=y_arr,
            output_index=int(run_cfg.output_index),
            max_keep=max_keep,
            observable=observable_lc,
            extrema_kind=extrema_kind_lc,
            poincare=poincare,
            params=params2,
        )
        if t_hits.size == 0:
            continue
        _append_hit_rows(rows, sweep, pv, ycol, t_hits, y_hits)

    return list(rows)


def _run_builtin_extrema_chunk(
    *,
    integration: IntegrationConfig,
    initial: InitialConditions,
    sweep: SweepConfig,
    poincare: PoincareConfig,
    observable_lc: str,
    extrema_kind_lc: str,
    run_cfg: SweepRunConfig,
    solve_options_any: Dict[str, Any],
    sweep_solver_kind: str,
    rhs_fn: Callable,
    base_params: Dict[str, float],
    max_hits_effective: int,
) -> List[dict]:
    param_vals = _param_values(sweep)

    rows: deque = deque(maxlen=MAX_SWEEP_ROWS_BUDGET)
    ycol = f"y{int(run_cfg.output_index)}"
    max_keep = int(max_hits_effective)

    y0_base = np.array(initial.y0, dtype=float).copy()
    y0_curr = y0_base.copy()

    for pv in param_vals:
        params2 = dict(base_params)
        params2[str(sweep.param_name)] = float(pv)

        rhs_eval = lambda t, y, _params=params2: rhs_fn(t, y, **_params)
        if sweep_solver_kind == "rk4":
            sol = integrate_system_rk4(
                rhs_eval,
                t_span=(float(integration.t0), float(integration.tf)),
                y0=y0_curr,
                t_step=float(integration.dt),
            )
        else:
            sol = integrate_system(
                rhs_eval,
                t_span=(float(integration.t0), float(integration.tf)),
                y0=y0_curr,
                t_step=float(integration.dt),
                **solve_options_any,
            )
        if not sol.success:
            if not run_cfg.warm_start:
                y0_curr = y0_base.copy()
            continue

        if run_cfg.warm_start:
            y0_curr = np.asarray(sol.y[:, -1], dtype=float).copy()
        else:
            y0_curr = y0_base.copy()

        t_hits, y_hits = _collect_observable_hits(
            t_arr=sol.t,
            y_arr=sol.y,
            output_index=int(run_cfg.output_index),
            max_keep=max_keep,
            observable=observable_lc,
            extrema_kind=extrema_kind_lc,
            poincare=poincare,
            params=params2,
        )
        if t_hits.size == 0:
            continue
        _append_hit_rows(rows, sweep, pv, ycol, t_hits, y_hits)

    return list(rows)


def _try_builtin_numba_poincare(
    *,
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    sweep: SweepConfig,
    poincare: PoincareConfig,
    run_cfg: SweepRunConfig,
    sweep_solver_kind: str,
    adapter,
    base_params: Dict[str, float],
    max_hits_effective: int,
):
    """Attempt the JIT Poincaré sweep path. Return None if not applicable."""
    if sweep_solver_kind != "rk4":
        return None
    if str(poincare.section_expr or "").strip() != "":
        return None
    method_lc = str(poincare.method or "").lower()
    if method_lc not in ("crossing", "slab"):
        return None

    dim = adapter.dimension
    if int(poincare.section_index) < 0 or int(poincare.section_index) >= dim:
        return None
    if int(run_cfg.output_index) < 0 or int(run_cfg.output_index) >= dim:
        return None
    if int(poincare.direction) not in (-1, 0, 1):
        return None

    try:
        from core import numba_backend
        if not numba_backend.numba_available():
            return None
        rhs_nb, _jac_nb, param_names = numba_backend.build_builtin_system(system.key)
        param_names_list = list(param_names)
        sweep_param_name = str(sweep.param_name)
        if sweep_param_name not in param_names_list:
            return None
        base_params_arr = np.array(
            [float(base_params[name]) for name in param_names_list],
            dtype=float,
        )
        param_index = param_names_list.index(sweep_param_name)
        sweep_nb = numba_backend.build_poincare_sweep_rk4(rhs_nb)
        method_id = 0 if method_lc == "crossing" else 1
        max_keep = int(max_hits_effective)
        params_out, t_hit, y_hit, count = sweep_nb(
            np.asarray(initial.y0, dtype=float),
            float(integration.t0),
            float(integration.tf),
            float(integration.dt),
            base_params_arr,
            int(param_index),
            float(sweep.start),
            float(sweep.stop),
            float(sweep.step),
            int(poincare.section_index),
            float(poincare.section_value),
            int(poincare.direction),
            int(method_id),
            float(poincare.tol),
            int(poincare.transient_steps),
            int(run_cfg.output_index),
            bool(run_cfg.warm_start),
            int(max_keep),
        )
        params_out = np.asarray(params_out[:count], dtype=float)
        t_hit = np.asarray(t_hit[:count], dtype=float)
        y_hit = np.asarray(y_hit[:count], dtype=float)
        ycol = f"y{int(run_cfg.output_index)}"
        if pd is not None:
            df_nb = pd.DataFrame({
                str(sweep.param_name): params_out,
                "t_hit": t_hit,
                ycol: y_hit,
            })
            return _clip_rows_result(df_nb, MAX_SWEEP_ROWS_BUDGET)
        return _clip_rows_result([
            {str(sweep.param_name): float(p), "t_hit": float(t), ycol: float(y)}
            for p, t, y in zip(params_out, t_hit, y_hit)
        ], MAX_SWEEP_ROWS_BUDGET)
    except Exception:
        return None


def _run_builtin_poincare_chunk(
    *,
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    sweep: SweepConfig,
    poincare: PoincareConfig,
    run_cfg: SweepRunConfig,
    solve_options_any: Dict[str, Any],
    sweep_solver_kind: str,
    adapter,
    rhs_fn: Callable,
    base_params: Dict[str, float],
    max_hits_effective: int,
):
    nb_result = _try_builtin_numba_poincare(
        system=system,
        integration=integration,
        initial=initial,
        sweep=sweep,
        poincare=poincare,
        run_cfg=run_cfg,
        sweep_solver_kind=sweep_solver_kind,
        adapter=adapter,
        base_params=base_params,
        max_hits_effective=max_hits_effective,
    )
    if nb_result is not None:
        return nb_result

    # IVP + crossing fast path via SciPy events
    if str(poincare.method).lower() == "crossing" and sweep_solver_kind == "ivp":
        rows_ivp = sweep_poincare_events_ivp(
            rhs=rhs_fn,
            y0=tuple(initial.y0),
            t_span=(float(integration.t0), float(integration.tf)),
            base_params=base_params,
            sweep=sweep,
            poincare=poincare,
            t_step=float(integration.dt),
            solve_options=solve_options_any,
            output_indices=[int(run_cfg.output_index)],
            include_all_state=False,
            warm_start=bool(run_cfg.warm_start),
            max_hits=int(max_hits_effective),
            early_stop=bool(run_cfg.early_stop),
            chunk_time=float(run_cfg.chunk_time),
        )
        return _clip_rows_result(rows_ivp, MAX_SWEEP_ROWS_BUDGET)

    rows_std = sweep_poincare(
        rhs=rhs_fn,
        y0=tuple(initial.y0),
        t_span=(float(integration.t0), float(integration.tf)),
        base_params=base_params,
        sweep=sweep,
        poincare=poincare,
        solver_kind=sweep_solver_kind,
        t_step=float(integration.dt),
        solve_options=solve_options_any,
        output_indices=[int(run_cfg.output_index)],
        include_all_state=False,
        warm_start=bool(run_cfg.warm_start),
    )
    return _clip_rows_result(rows_std, MAX_SWEEP_ROWS_BUDGET)


def run_sweep_chunk(
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    sweep: SweepConfig,
    poincare: PoincareConfig,
    observable: str,
    extrema_kind: str,
    run_cfg: SweepRunConfig,
    solve_tols: SolverTolerances,
    solve_options: Optional[Dict[str, Any]] = None,
):
    if solve_options is None:
        solve_options = dict(solve_tols.to_dict())
    else:
        solve_options = dict(solve_options)
    solve_options_any: Dict[str, Any] = dict(solve_options)
    policy = resolve_solver(getattr(integration, "solver_kind", "ivp"))
    sweep_solver_kind = policy.sweep_kind
    apply_to_solve_options(policy, solve_options_any)
    observable_lc = _normalize_observable(observable)
    extrema_kind_lc = _normalize_extrema_kind(extrema_kind)
    max_hits_user = int(run_cfg.max_hits) if run_cfg.max_hits is not None else DEFAULT_MAX_KEEP
    max_hits_effective = _effective_max_hits(max_hits_user, sweep, MAX_SWEEP_ROWS_BUDGET)

    if system.key == "custom":
        return _run_custom_chunk(
            system=system,
            integration=integration,
            initial=initial,
            sweep=sweep,
            poincare=poincare,
            observable_lc=observable_lc,
            extrema_kind_lc=extrema_kind_lc,
            run_cfg=run_cfg,
            solve_options_any=solve_options_any,
            sweep_solver_kind=sweep_solver_kind,
            max_hits_effective=max_hits_effective,
        )

    adapter = get_builtin(system.key)
    rhs_fn = adapter.rhs_fn
    base_params = adapter.extract_params(system)

    if sweep_solver_kind == "symplectic_fr":
        return _run_symplectic_builtin_chunk(
            system=system,
            integration=integration,
            initial=initial,
            sweep=sweep,
            poincare=poincare,
            observable_lc=observable_lc,
            extrema_kind_lc=extrema_kind_lc,
            run_cfg=run_cfg,
            base_params=base_params,
            max_hits_effective=max_hits_effective,
        )

    if observable_lc == OBSERVABLE_EXTREMA:
        return _run_builtin_extrema_chunk(
            integration=integration,
            initial=initial,
            sweep=sweep,
            poincare=poincare,
            observable_lc=observable_lc,
            extrema_kind_lc=extrema_kind_lc,
            run_cfg=run_cfg,
            solve_options_any=solve_options_any,
            sweep_solver_kind=sweep_solver_kind,
            rhs_fn=rhs_fn,
            base_params=base_params,
            max_hits_effective=max_hits_effective,
        )

    return _run_builtin_poincare_chunk(
        system=system,
        integration=integration,
        initial=initial,
        sweep=sweep,
        poincare=poincare,
        run_cfg=run_cfg,
        solve_options_any=solve_options_any,
        sweep_solver_kind=sweep_solver_kind,
        adapter=adapter,
        rhs_fn=rhs_fn,
        base_params=base_params,
        max_hits_effective=max_hits_effective,
    )
