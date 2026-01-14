from typing import List

import concurrent.futures
import itertools
import numpy as np
import pandas as pd

from app.logic.sweep_utils import _chunk_param_values, _frange_inclusive
from app.params import (
    InitialConditions,
    IntegrationConfig,
    SolverTolerances,
    SweepRunConfig,
    SystemConfig,
)
from app.sweep import run_sweep_chunk
from core.poincare_sweep import PoincareConfig, SweepConfig


def _run_bifurcation_chunk(
    param_vals: np.ndarray,
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    sweep_param: str,
    sweep_step: float,
    poincare: PoincareConfig,
    run_cfg: SweepRunConfig,
    solve_tols: SolverTolerances,
) -> List[dict]:
    if param_vals.size == 0:
        return []
    sweep_run = SweepConfig(
        param_name=str(sweep_param),
        start=float(param_vals[0]),
        stop=float(param_vals[-1]),
        step=float(sweep_step),
    )
    rows = run_sweep_chunk(
        system=system,
        integration=integration,
        initial=initial,
        sweep=sweep_run,
        poincare=poincare,
        run_cfg=run_cfg,
        solve_tols=solve_tols,
    )
    if rows is None:
        return []
    if isinstance(rows, pd.DataFrame):
        return rows.to_dict(orient="records")
    return list(rows)


def _run_bifurcation_parallel(
    *,
    system: SystemConfig,
    integration: IntegrationConfig,
    initial: InitialConditions,
    sweep: SweepConfig,
    poincare: PoincareConfig,
    run_cfg: SweepRunConfig,
    solve_tols: SolverTolerances,
    max_workers: int,
) -> List[dict]:
    param_vals = _frange_inclusive(float(sweep.start), float(sweep.stop), float(sweep.step))
    param_chunks = _chunk_param_values(param_vals, max_workers)
    if not param_chunks:
        return []

    workers = max(1, min(int(max_workers), int(param_vals.size)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(
            _run_bifurcation_chunk,
            param_chunks,
            itertools.repeat(system),
            itertools.repeat(integration),
            itertools.repeat(initial),
            itertools.repeat(str(sweep.param_name)),
            itertools.repeat(float(sweep.step)),
            itertools.repeat(poincare),
            itertools.repeat(run_cfg),
            itertools.repeat(solve_tols),
        ))

    rows: List[dict] = []
    for chunk_rows in results:
        rows.extend(chunk_rows)
    return rows
