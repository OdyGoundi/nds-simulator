from typing import Optional

import numpy as np

from core.lorenz_system_rhs import lorenz_rhs
from core.rossler_system_rhs import rossler_rhs
from core.poincare_sweep import (
    poincare_section,
    sweep_poincare,
    sweep_poincare_events_ivp,
    PoincareConfig,
    SweepConfig,
)
from core.solver import integrate_system
from app.helpers import parse_params, build_custom_rhs
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
    solve_options: Optional[dict] = None,
):
    if solve_options is None:
        solve_options = solve_tols.to_dict()

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

        for pv in param_vals:
            params2 = dict(base_params)
            params2[str(sweep.param_name)] = float(pv)

            rhs2 = build_custom_rhs(
                list(system.custom.var_names),
                list(system.custom.eq_lines),
                params2,
            )

            sol = integrate_system(
                rhs2,
                t_span=(float(integration.t0), float(integration.tf)),
                y0=y0_curr,
                t_step=float(integration.dt),
                **solve_options,
            )
            if not sol.success:
                if not run_cfg.warm_start:
                    y0_curr = y0_base.copy()
                continue
            if run_cfg.warm_start:
                y0_curr = np.array(sol.y[:, -1], dtype=float).copy()
            else:
                y0_curr = y0_base.copy()

            t_hits, y_hits = poincare_section(sol.t, sol.y, poincare, params=params2)

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

    # use fast events only for ivp+crossing
    if str(poincare.method).lower() == "crossing":
        return sweep_poincare_events_ivp(
            rhs=rhs_fn,
            y0=tuple(initial.y0),
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

    return sweep_poincare(
        rhs=rhs_fn,
        y0=tuple(initial.y0),
        t_span=(float(integration.t0), float(integration.tf)),
        base_params=base_params,
        sweep=sweep,
        poincare=poincare,
        solver_kind="ivp",
        t_step=float(integration.dt),
        solve_options=solve_options,
        output_indices=[int(run_cfg.output_index)],
        include_all_state=False,
        warm_start=bool(run_cfg.warm_start),
    )
