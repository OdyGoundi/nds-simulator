import sys
from pathlib import Path

# Add project root to sys.path (so `import core...` works when running from tests/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import time
import numpy as np
import matplotlib.pyplot as plt

from core.poincare_sweep import PoincareConfig, SweepConfig, sweep_poincare
from core.lorenz_system_rhs import lorenz_rhs


def run_once_ivp(n_steps: int) -> None:
    """
    Run sweep_poincare with solver_kind='ivp' for a given effective number of output steps.
    We keep sweep size P=1 to isolate scaling vs n_steps.
    """
    t0, tf = 0.0, 10.0
    t_span = (t0, tf)

    # Make t_step consistent with requested n_steps
    t_step = (tf - t0) / float(n_steps)

    # Single sweep value (P=1) to focus on time-resolution cost
    sweep = SweepConfig(param_name="rho", start=28.0, stop=28.0, step=1.0)

    poincare = PoincareConfig(
        section_index=0,     # x = 0 section
        section_value=0.0,
        direction=+1,
        method="crossing",
        transient_steps=0,
    )

    y0 = [1.0, 1.0, 1.0]
    base_params = {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}

    # IMPORTANT: keep solve_options fixed across runs
    solve_options = dict(method="RK45", rtol=1e-6, atol=1e-9)

    _ = sweep_poincare(
        rhs=lorenz_rhs,
        y0=y0,
        t_span=t_span,
        base_params=base_params,
        sweep=sweep,
        poincare=poincare,
        solver_kind="ivp",
        t_step=t_step,
        solve_options=solve_options,
        output_indices=[1],
        include_all_state=False,
    )


def benchmark(step_list, repeats: int = 7):
    """
    Returns:
      steps: np.ndarray
      times: np.ndarray  (median of repeats)
    """
    # Warm-up to reduce one-time overhead noise
    run_once_ivp(int(step_list[0]))

    times = []
    for n_steps in step_list:
        samples = []
        for _ in range(repeats):
            t_start = time.perf_counter()
            run_once_ivp(int(n_steps))
            t_end = time.perf_counter()
            samples.append(t_end - t_start)
        times.append(float(np.median(samples)))
    return np.array(step_list, dtype=int), np.array(times, dtype=float)


if __name__ == "__main__":
    # 100 -> 10000 (log-spaced)
    step_list = np.unique(np.logspace(2, 4, 12).astype(int))

    steps, times = benchmark(step_list, repeats=7)

    # Linear plot
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(steps, times, marker="o")
    ax.set_xlabel("Requested output steps N (via t_step)")
    ax.set_ylabel("Runtime [s] (median)")
    ax.set_title("IVP (solve_ivp) runtime vs requested output steps")
    ax.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.show()

    # Log-log plot + fitted exponent
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.loglog(steps, times, marker="o")
    ax.set_xlabel("N (log)")
    ax.set_ylabel("Runtime [s] (log)")
    ax.set_title("IVP scaling (log–log)")
    ax.grid(True, which="both", linewidth=0.3)
    plt.tight_layout()
    plt.show()

    # Fit exponent p in time ~ N^p
    p = np.polyfit(np.log(steps), np.log(times), 1)[0]
    print(f"Estimated scaling: O(N^{p:.2f})")
