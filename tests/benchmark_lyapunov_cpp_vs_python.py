import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path (so `import core...` works when running from tests/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import core.lyapunov as lyap
from core import numba_backend


def lorenz_rhs(sigma: float, rho: float, beta: float):
    def rhs(tt: float, xx: np.ndarray) -> np.ndarray:
        x, y, z = float(xx[0]), float(xx[1]), float(xx[2])
        return np.array(
            [
                sigma * (y - x),
                x * (rho - z) - y,
                x * y - beta * z,
            ],
            dtype=float,
        )

    return rhs


def lorenz_jac(sigma: float, rho: float, beta: float):
    def jac(tt: float, xx: np.ndarray) -> np.ndarray:
        x, y, z = float(xx[0]), float(xx[1]), float(xx[2])
        return np.array(
            [
                [-sigma, sigma, 0.0],
                [rho - z, -1.0, -x],
                [y, x, -beta],
            ],
            dtype=float,
        )

    return jac


def _compute_python(kwargs):
    return lyap.compute_lyapunov_spectrum(**kwargs)


def _time_runs(fn, runs: int) -> float:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.mean(times))


def _compute_numba(
    lyap_nb,
    x0: np.ndarray,
    t0: float,
    dt: float,
    t_transient: float,
    t_measure: float,
    qr_every_steps: int,
    fd_eps: float,
    params_arr: np.ndarray,
):
    lambdas, _sums, _t_meas, _n_qr, _x_final = lyap_nb(
        x0,
        float(t0),
        float(dt),
        float(t_transient),
        float(t_measure),
        int(qr_every_steps),
        float(fd_eps),
        params_arr,
    )
    return lambdas


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Lyapunov Numba vs Python (RK4).")
    parser.add_argument("--runs", type=int, default=5, help="Timed runs per backend.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per backend.")
    parser.add_argument("--dt", type=float, default=0.01, help="RK4 time step.")
    parser.add_argument("--t-transient", type=float, default=500.0, help="Transient time.")
    parser.add_argument("--t-measure", type=float, default=1000.0, help="Measurement time (ignored during t-final sweep).")
    parser.add_argument("--t-final-start", type=int, default=1000, help="Sweep start for total final time (t_transient + t_measure).")
    parser.add_argument("--t-final-stop", type=int, default=10000, help="Sweep stop for total final time (inclusive when aligned with step).")
    parser.add_argument("--t-final-step", type=int, default=1000, help="Sweep step for total final time.")
    parser.add_argument("--qr-every-steps", type=int, default=10, help="QR interval in steps.")
    parser.add_argument("--no-jac", action="store_true", help="Use finite-difference Jacobian.")
    parser.add_argument("--no-numba", action="store_true", help="Disable Numba backend.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting the results.")
    parser.add_argument("--plot-path", type=str, default=None, help="Optional path to save the plot instead of showing it.")
    args = parser.parse_args()

    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    rhs = lorenz_rhs(sigma, rho, beta)
    jac = None if args.no_jac else lorenz_jac(sigma, rho, beta)

    x0 = np.array([1.0, 1.0, 1.0], dtype=float)

    kwargs_base = dict(
        rhs=rhs,
        x0=x0,
        t0=0.0,
        dt=float(args.dt),
        t_transient=float(args.t_transient),
        qr_every_steps=int(args.qr_every_steps),
        solver_kind="rk4",
        jac=jac,
    )

    if args.t_final_step <= 0:
        raise ValueError("t-final-step must be > 0.")
    if args.t_final_stop < args.t_final_start:
        raise ValueError("t-final-stop must be >= t-final-start.")

    final_times = list(range(args.t_final_start, args.t_final_stop + 1, args.t_final_step))

    print("Lyapunov benchmark (RK4) - Lorenz system")
    print(f"dt={args.dt}, t_transient={args.t_transient}, qr_every_steps={args.qr_every_steps}")
    print(f"final time sweep: {final_times[0]}..{final_times[-1]} step {args.t_final_step}")
    print("jacobian:", "FD" if args.no_jac else "analytic")
    print()

    numba_available = (not args.no_numba) and numba_backend.numba_available()
    if not numba_available:
        print("Numba backend not available; install numba to benchmark Numba.")

    py_times = []
    numba_times = []

    lyap_nb = None
    params_arr = None
    if numba_available:
        rhs_nb, jac_nb, param_names = numba_backend.build_builtin_system("lorenz")
        if args.no_jac:
            lyap_nb = numba_backend.build_lyapunov_solver(rhs_nb, None, use_fd_jac=True)
        else:
            lyap_nb = numba_backend.build_lyapunov_solver(rhs_nb, jac_nb, use_fd_jac=False)
        params_arr = np.array([sigma, rho, beta], dtype=float)

    for t_final in final_times:
        t_measure = float(t_final) - float(args.t_transient)
        if t_measure <= 0.0:
            raise ValueError(f"t_final={t_final} must be > t_transient={args.t_transient}.")

        kwargs = dict(kwargs_base, t_measure=t_measure)

        # Warmup
        for _ in range(args.warmup):
            _compute_python(kwargs)
        if numba_available and lyap_nb is not None and params_arr is not None:
            for _ in range(args.warmup):
                _compute_numba(
                    lyap_nb,
                    x0,
                    0.0,
                    float(args.dt),
                    float(args.t_transient),
                    float(t_measure),
                    int(args.qr_every_steps),
                    1e-8,
                    params_arr,
                )

        py_time = _time_runs(lambda: _compute_python(kwargs), args.runs)
        py_times.append(py_time)
        print(f"t_final={t_final:5d} (t_measure={t_measure:.1f}) | Python RK4 avg: {py_time:.4f} s")

        if numba_available and lyap_nb is not None and params_arr is not None:
            numba_time = _time_runs(
                lambda: _compute_numba(
                    lyap_nb,
                    x0,
                    0.0,
                    float(args.dt),
                    float(args.t_transient),
                    float(t_measure),
                    int(args.qr_every_steps),
                    1e-8,
                    params_arr,
                ),
                args.runs,
            )
            numba_times.append(numba_time)
            speedup = py_time / numba_time if numba_time > 0 else float("inf")
            print(f"t_final={t_final:5d} (t_measure={t_measure:.1f}) | Numba RK4 avg:  {numba_time:.4f} s | Speedup: {speedup:.2f}x")

    if not args.no_plot:
        plt.figure(figsize=(9, 5), dpi=150)
        plt.plot(final_times, py_times, marker="o", color="tab:blue", label="Python")
        if numba_available:
            plt.plot(final_times, numba_times, marker="o", color="tab:green", label="Numba")
        plt.xlabel("Final time")
        plt.ylabel("Execution time (s)")
        plt.title("Lyapunov RK4: Final time vs execution time")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if args.plot_path:
            plt.savefig(args.plot_path, dpi=200)
            print(f"Saved plot to: {args.plot_path}")
        else:
            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
