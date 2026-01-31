from typing import Any, Dict, Optional

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
from core.lorenz_system_rhs import lorenz_rhs
from core.rossler_system_rhs import rossler_rhs
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

DEFAULT_MAX_KEEP = 100


def run_sweep_chunk(
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    sweep: SweepConfig,
    poincare: PoincareConfig,
    run_cfg: SweepRunConfig,
    solve_tols: SolverTolerances,
    solve_options: Optional[Dict[str, Any]] = None,
):
    if solve_options is None:
        solve_options = dict(solve_tols.to_dict())
    else:
        solve_options = dict(solve_options)
    solve_options_any: Dict[str, Any] = dict(solve_options)
    solver_kind = str(getattr(integration, "solver_kind", "ivp")).lower()
    if solver_kind == "symplectic_verlet":
        solver_kind = "symplectic_fr"
    method = None
    if solver_kind in ("rk45", "ivp"):
        method = "RK45"
    elif solver_kind == "dop853":
        method = "DOP853"
    if method is not None:
        solve_options_any["method"] = method
    if solver_kind == "rk4":
        sweep_solver_kind = "rk4"
    elif solver_kind == "symplectic_fr":
        sweep_solver_kind = "symplectic_fr"
    else:
        sweep_solver_kind = "ivp"

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
        base_params = parse_params(system.custom.params_text)

        # manual sweep so swept param overrides correctly
        if float(sweep.step) <= 0:
            raise ValueError("Sweep step must be > 0.")
        n = int(np.floor((float(sweep.stop) - float(sweep.start)) / float(sweep.step) + 1e-12)) + 1
        param_vals = float(sweep.start) + float(sweep.step) * np.arange(n, dtype=float)
        param_vals = param_vals[param_vals <= float(sweep.stop) + 1e-12]

        rows = []
        ycol = f"y{int(run_cfg.output_index)}"
        max_keep = int(run_cfg.max_hits) if run_cfg.max_hits is not None else DEFAULT_MAX_KEEP

        y0_base = np.array(initial.y0, dtype=float).copy()
        y0_curr = y0_base.copy()
        use_numba_rk4 = False
        rk4_nb = None
        use_numba_symplectic = False
        symplectic_nb = None
        param_names = list(base_params.keys())

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

            t_hits, y_hits = poincare_section(t_arr, y_arr, poincare, params=params2)

            if t_hits.size > max_keep:
                t_hits = t_hits[-max_keep:]
                y_hits = y_hits[:, -max_keep:]

            if t_hits.size == 0:
                continue

            for j in range(t_hits.size):
                rows.append({
                    str(sweep.param_name): float(pv),
                    "t_hit": float(t_hits[j]),
                    ycol: float(y_hits[int(run_cfg.output_index), j]),
                })

        return rows

    else:
        raise ValueError(f"Unknown system_key: {system.key}")

    if sweep_solver_kind == "symplectic_fr":
        if system.key != "henon_heiles":
            raise ValueError("Symplectic sweeps require a Hamiltonian system (Henon-Heiles or custom).")

        if float(sweep.step) <= 0:
            raise ValueError("Sweep step must be > 0.")
        n = int(np.floor((float(sweep.stop) - float(sweep.start)) / float(sweep.step) + 1e-12)) + 1
        param_vals = float(sweep.start) + float(sweep.step) * np.arange(n, dtype=float)
        param_vals = param_vals[param_vals <= float(sweep.stop) + 1e-12]

        rows = []
        ycol = f"y{int(run_cfg.output_index)}"
        max_keep = int(run_cfg.max_hits) if run_cfg.max_hits is not None else DEFAULT_MAX_KEEP

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

            t_hits, y_hits = poincare_section(t_arr, y_arr, poincare, params=params2)

            if t_hits.size > max_keep:
                t_hits = t_hits[-max_keep:]
                y_hits = y_hits[:, -max_keep:]

            if t_hits.size == 0:
                continue

            for j in range(t_hits.size):
                rows.append({
                    str(sweep.param_name): float(pv),
                    "t_hit": float(t_hits[j]),
                    ycol: float(y_hits[int(run_cfg.output_index), j]),
                })

        return rows

    use_numba_sweep = (
        sweep_solver_kind == "rk4"
        and str(poincare.section_expr or "").strip() == ""
    )
    method_lc = str(poincare.method or "").lower()
    if use_numba_sweep and method_lc not in ("crossing", "slab"):
        use_numba_sweep = False

    if use_numba_sweep:
        dim = 3 if system.key in ("lorenz", "rossler") else 4
        if int(poincare.section_index) < 0 or int(poincare.section_index) >= dim:
            use_numba_sweep = False
        if int(run_cfg.output_index) < 0 or int(run_cfg.output_index) >= dim:
            use_numba_sweep = False
        if int(poincare.direction) not in (-1, 0, 1):
            use_numba_sweep = False

    if use_numba_sweep:
        try:
            from core import numba_backend
            if numba_backend.numba_available():
                rhs_nb, _jac_nb, param_names = numba_backend.build_builtin_system(system.key)
                param_names_list = list(param_names)
                sweep_param_name = str(sweep.param_name)
                if sweep_param_name in param_names_list:
                    base_params_arr = np.array(
                        [float(base_params[name]) for name in param_names_list],
                        dtype=float,
                    )
                    param_index = param_names_list.index(sweep_param_name)
                    sweep_nb = numba_backend.build_poincare_sweep_rk4(rhs_nb)
                    method_id = 0 if method_lc == "crossing" else 1
                    max_keep = int(run_cfg.max_hits) if run_cfg.max_hits is not None else 100
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
                        return pd.DataFrame({
                            str(sweep.param_name): params_out,
                            "t_hit": t_hit,
                            ycol: y_hit,
                        })
                    return [
                        {str(sweep.param_name): float(p), "t_hit": float(t), ycol: float(y)}
                        for p, t, y in zip(params_out, t_hit, y_hit)
                    ]
        except Exception:
            pass

    # use fast events only for ivp+crossing
    if str(poincare.method).lower() == "crossing" and sweep_solver_kind == "ivp":
        return sweep_poincare_events_ivp(
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
            max_hits=int(run_cfg.max_hits),
            early_stop=bool(run_cfg.early_stop),
            chunk_time=float(run_cfg.chunk_time),
        )

    return sweep_poincare(
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
