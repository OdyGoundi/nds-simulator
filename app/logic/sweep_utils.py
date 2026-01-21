from typing import Dict, List

import math
import os
import numpy as np

from app.params import IntegrationConfig, SolverTolerances, SweepRunConfig, SystemConfig
from core.poincare_sweep import PoincareConfig, SweepConfig


def _is_streamlit_cloud() -> bool:
    env = os.environ
    if env.get("STREAMLIT_RUNTIME_ENV", "").lower() == "cloud":
        return True
    if env.get("STREAMLIT_CLOUD", "").lower() in ("1", "true", "yes"):
        return True
    if env.get("STREAMLIT_SHARING", "").lower() in ("1", "true", "yes"):
        return True
    addr = env.get("STREAMLIT_SERVER_ADDRESS", "")
    if addr.endswith("streamlit.app"):
        return True
    return False


def _default_worker_count() -> int:
    physical = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
    except Exception:
        physical = None
    cpu_count = physical or (os.cpu_count() or 1)
    return max(1, min(int(cpu_count), 8))


def _chunk_param_values(param_vals: np.ndarray, max_workers: int) -> List[np.ndarray]:
    if param_vals.size == 0:
        return []
    workers = max(1, int(max_workers))
    target_chunks = max(1, workers * 4)
    chunk_size = max(1, int(math.ceil(param_vals.size / target_chunks)))
    return [param_vals[i:i + chunk_size] for i in range(0, param_vals.size, chunk_size)]


def _frange_inclusive(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0.0:
        raise ValueError("Sweep step must be > 0.")
    n = int(np.floor((stop - start) / step + 1e-12)) + 1
    vals = start + step * np.arange(n, dtype=float)
    return vals[vals <= stop + 1e-12]


def _sweep_settings_fingerprint(
    system: SystemConfig,
    sweep: SweepConfig,
    poincare: PoincareConfig,
    run_cfg: SweepRunConfig,
    integration: IntegrationConfig,
    transient_frac: float,
    solve_tols: SolverTolerances,
) -> Dict[str, object]:
    return {
        "system_key": system.key,
        "sweep_param": str(sweep.param_name),
        "sweep_step": float(sweep.step),
        "section_index": int(poincare.section_index),
        "section_value": float(poincare.section_value),
        "section_expr": str(poincare.section_expr or ""),
        "direction": int(poincare.direction),
        "method": str(poincare.method),
        "tol": float(poincare.tol),
        "output_index": int(run_cfg.output_index),
        "tf_sweep": float(integration.tf),
        "dt_sweep": float(integration.dt),
        "solver_kind": str(getattr(integration, "solver_kind", "ivp")),
        "transient_frac": float(transient_frac),
        "max_hits": int(run_cfg.max_hits),
        "early_stop": bool(run_cfg.early_stop),
        "chunk_time": float(run_cfg.chunk_time),
        "warm_start": bool(run_cfg.warm_start),
        "rtol": float(solve_tols.rtol),
        "atol": float(solve_tols.atol),
    }
