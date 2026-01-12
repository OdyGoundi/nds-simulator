from typing import List

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

DEFAULT_SOLVE_OPTIONS = {"rtol": 3e-4, "atol": 1e-6}
DEFAULT_MAX_KEEP = 100

def run_sweep_chunk(
    system_key: str,
    t0: float, tf: float, dt: float,
    y0: np.ndarray,
    sigma: float, rho: float, beta: float,
    ross_a: float, ross_b: float, ross_c: float,
    var_names: List[str],
    eq_lines: List[str],
    params_text: str,
    sweep_param: str,
    sweep_start: float, sweep_stop: float, sweep_step: float,
    section_index: int, section_value: float, direction: int,
    method: str, tol: float, transient_steps: int,
    output_index: int,
    warm_start: bool,
    max_hits: int,
    early_stop: bool,
    chunk_time: float,
):
    import pandas as pd

    # build rhs_fn + base_params
    if system_key == "lorenz":
        rhs_fn = lorenz_rhs
        base_params = {"sigma": float(sigma), "rho": float(rho), "beta": float(beta)}
    elif system_key == "rossler":
        rhs_fn = rossler_rhs
        base_params = {"a": float(ross_a), "b": float(ross_b), "c": float(ross_c)}
    elif system_key == "custom":
        base_params = parse_params(params_text)

        poincare = PoincareConfig(
            section_index=int(section_index),
            section_value=float(section_value),
            direction=int(direction),
            method=str(method),
            tol=float(tol),
            transient_steps=int(transient_steps),
        )

        # manual sweep so swept param overrides correctly
        if float(sweep_step) <= 0:
            raise ValueError("Sweep step must be > 0.")
        n = int(np.floor((float(sweep_stop) - float(sweep_start)) / float(sweep_step) + 1e-12)) + 1
        param_vals = float(sweep_start) + float(sweep_step) * np.arange(n, dtype=float)
        param_vals = param_vals[param_vals <= float(sweep_stop) + 1e-12]

        rows = []
        ycol = f"y{int(output_index)}"
        max_keep = int(max_hits) if max_hits is not None else DEFAULT_MAX_KEEP

        for pv in param_vals:
            params2 = dict(base_params)
            params2[str(sweep_param)] = float(pv)

            rhs2 = build_custom_rhs(var_names, eq_lines, params2)

            sol = integrate_system(rhs2, t_span=(float(t0), float(tf)), y0=y0, t_step=float(dt))
            if not sol.success:
                continue

            t_hits, y_hits = poincare_section(sol.t, sol.y, poincare)

            if t_hits.size > max_keep:
                t_hits = t_hits[-max_keep:]
                y_hits = y_hits[:, -max_keep:]

            if t_hits.size == 0:
                continue

            for j in range(t_hits.size):
                rows.append({
                    str(sweep_param): float(pv),
                    "t_hit": float(t_hits[j]),
                    ycol: float(y_hits[int(output_index), j]),
                })

        return rows

    else:
        raise ValueError(f"Unknown system_key: {system_key}")

    poincare = PoincareConfig(
        section_index=int(section_index),
        section_value=float(section_value),
        direction=int(direction),
        method=str(method),
        tol=float(tol),
        transient_steps=int(transient_steps),
    )

    sweep = SweepConfig(
        param_name=str(sweep_param),
        start=float(sweep_start),
        stop=float(sweep_stop),
        step=float(sweep_step),
    )

    solve_options = DEFAULT_SOLVE_OPTIONS

    # use fast events only for ivp+crossing
    if str(method).lower() == "crossing":
        return sweep_poincare_events_ivp(
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

    return sweep_poincare(
        rhs=rhs_fn,
        y0=tuple(y0),
        t_span=(float(t0), float(tf)),
        base_params=base_params,
        sweep=sweep,
        poincare=poincare,
        solver_kind="ivp",
        t_step=float(dt),
        solve_options=solve_options,
        output_indices=[int(output_index)],
        include_all_state=False,
    )
