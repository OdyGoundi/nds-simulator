#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import sys
import time
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path (so `import core...` works when running from tests/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import core.lyapunov as lyap
from core import numba_backend


DEFAULT_CPP_DIR = Path.home() / "Documents" / "nds_cpp_backend_archive_2026-01-30" / "cpp_core"


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


def _compute_cpp(
    cpp_mod: ModuleType,
    *,
    rhs,
    x0: np.ndarray,
    t0: float,
    dt: float,
    t_transient: float,
    t_measure: float,
    qr_every_steps: int,
    jac,
    fd_eps: float,
):
    out = cpp_mod.compute_lyapunov_spectrum(
        rhs=rhs,
        x0=np.asarray(x0, dtype=float),
        t0=float(t0),
        dt=float(dt),
        t_transient=float(t_transient),
        t_measure=float(t_measure),
        qr_every_steps=int(qr_every_steps),
        jac=jac,
        fd_eps=float(fd_eps),
    )
    return out["lambdas"]


def _time_runs(fn, runs: int) -> float:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.mean(times))


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        out.append(p)
    return out


def _collect_cpp_search_paths(cpp_dir: Path) -> list[Path]:
    paths: list[Path] = []

    roots = [cpp_dir]
    if (cpp_dir / "cpp_core").is_dir():
        roots.append(cpp_dir / "cpp_core")

    for root in roots:
        if not root.is_dir():
            continue
        paths.append(root)
        for p in root.glob("build/lib*"):
            if p.is_dir():
                paths.append(p)
        for pattern in ("nlds_cpp*.so", "nlds_cpp*.pyd"):
            for p in root.rglob(pattern):
                paths.append(p.parent)

    return _dedupe_paths(paths)


def _import_nlds_cpp(cpp_dir: Path) -> tuple[ModuleType | None, str]:
    try:
        return importlib.import_module("nlds_cpp"), "imported from current Python environment"
    except Exception as e:
        env_err = e

    search_paths = _collect_cpp_search_paths(cpp_dir)
    for p in search_paths:
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)

    try:
        mod = importlib.import_module("nlds_cpp")
        return mod, f"imported after adding search paths under {cpp_dir}"
    except Exception as e:
        if search_paths:
            attempted = ", ".join(str(p) for p in search_paths)
            msg = (
                f"failed to import nlds_cpp (env error: {env_err}; "
                f"search error: {e}; attempted paths: {attempted})"
            )
            return None, msg
        return None, f"failed to import nlds_cpp (env error: {env_err}; search error: {e})"


def _resolve_cpp_build_dir(cpp_dir: Path) -> Path:
    if (cpp_dir / "setup.py").is_file():
        return cpp_dir
    nested = cpp_dir / "cpp_core"
    if (nested / "setup.py").is_file():
        return nested
    return cpp_dir


def _save_csv(
    csv_path: Path,
    final_times: list[int],
    t_measures: list[float],
    py_times: list[float],
    numba_times: list[float | None],
    cpp_times: list[float | None],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t_final",
                "t_measure",
                "runtime_s_python",
                "runtime_s_numba",
                "runtime_s_cpp",
                "speedup_numba_vs_python",
                "speedup_cpp_vs_python",
            ]
        )
        for i, t_final in enumerate(final_times):
            py_t = py_times[i]
            nb_t = numba_times[i]
            cpp_t = cpp_times[i]
            nb_speed = (py_t / nb_t) if (nb_t is not None and nb_t > 0) else None
            cpp_speed = (py_t / cpp_t) if (cpp_t is not None and cpp_t > 0) else None
            w.writerow(
                [
                    int(t_final),
                    float(t_measures[i]),
                    float(py_t),
                    "" if nb_t is None else float(nb_t),
                    "" if cpp_t is None else float(cpp_t),
                    "" if nb_speed is None else float(nb_speed),
                    "" if cpp_speed is None else float(cpp_speed),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Lyapunov runtime: Python vs Numba vs C++ (RK4, Lorenz)."
    )
    parser.add_argument("--runs", type=int, default=5, help="Timed runs per backend.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per backend.")
    parser.add_argument("--dt", type=float, default=0.01, help="RK4 time step.")
    parser.add_argument("--t-transient", type=float, default=500.0, help="Transient time.")
    parser.add_argument(
        "--t-measure",
        type=float,
        default=1000.0,
        help="Measurement time (used only if t-final sweep is disabled).",
    )
    parser.add_argument(
        "--t-final-start",
        type=int,
        default=1000,
        help="Sweep start for total final time (t_transient + t_measure).",
    )
    parser.add_argument(
        "--t-final-stop",
        type=int,
        default=10000,
        help="Sweep stop for total final time (inclusive when aligned with step).",
    )
    parser.add_argument("--t-final-step", type=int, default=1000, help="Sweep step for total final time.")
    parser.add_argument("--qr-every-steps", type=int, default=10, help="QR interval in steps.")
    parser.add_argument("--fd-eps", type=float, default=1e-8, help="Finite-difference epsilon.")
    parser.add_argument("--no-jac", action="store_true", help="Use finite-difference Jacobian.")
    parser.add_argument("--no-numba", action="store_true", help="Disable Numba backend.")
    parser.add_argument("--no-cpp", action="store_true", help="Disable C++ backend.")
    parser.add_argument(
        "--cpp-dir",
        type=str,
        default=str(DEFAULT_CPP_DIR),
        help="Path to archived C++ backend folder (archive root or cpp_core dir).",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting the results.")
    parser.add_argument("--plot-path", type=str, default=None, help="Optional path to save the plot.")
    parser.add_argument("--csv-path", type=str, default=None, help="Optional path to save benchmark CSV.")
    args = parser.parse_args()

    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    rhs = lorenz_rhs(sigma, rho, beta)
    jac = None if args.no_jac else lorenz_jac(sigma, rho, beta)
    x0 = np.array([1.0, 1.0, 1.0], dtype=float)

    if args.t_final_step <= 0:
        raise ValueError("t-final-step must be > 0.")
    if args.t_final_stop < args.t_final_start:
        raise ValueError("t-final-stop must be >= t-final-start.")

    final_times = list(range(args.t_final_start, args.t_final_stop + 1, args.t_final_step))
    t_measures = [float(t_final) - float(args.t_transient) for t_final in final_times]
    if any(tm <= 0.0 for tm in t_measures):
        raise ValueError("All t_final values must be > t_transient.")

    print("Lyapunov benchmark (RK4) - Lorenz system")
    print(f"dt={args.dt}, t_transient={args.t_transient}, qr_every_steps={args.qr_every_steps}, fd_eps={args.fd_eps}")
    print(f"final time sweep: {final_times[0]}..{final_times[-1]} step {args.t_final_step}")
    print("jacobian:", "FD" if args.no_jac else "analytic")
    print()

    numba_available = (not args.no_numba) and numba_backend.numba_available()
    if not numba_available:
        print("Numba backend unavailable/disabled.")
    else:
        print("Numba backend available.")

    cpp_mod: ModuleType | None = None
    cpp_available = False
    cpp_dir = Path(args.cpp_dir).expanduser()
    cpp_build_dir = _resolve_cpp_build_dir(cpp_dir)
    if not args.no_cpp:
        cpp_mod, cpp_info = _import_nlds_cpp(cpp_dir)
        cpp_available = cpp_mod is not None
        if cpp_available:
            print(f"C++ backend available ({cpp_info}).")
        else:
            print("C++ backend unavailable.")
            print(f"  reason: {cpp_info}")
            print(f"  build hint: cd {cpp_build_dir} && python3 setup.py build_ext --inplace")
    else:
        print("C++ backend disabled by flag.")
    print()

    py_times: list[float] = []
    numba_times: list[float | None] = []
    cpp_times: list[float | None] = []

    lyap_nb = None
    params_arr = None
    if numba_available:
        rhs_nb, jac_nb, _param_names = numba_backend.build_builtin_system("lorenz")
        if args.no_jac:
            lyap_nb = numba_backend.build_lyapunov_solver(rhs_nb, None, use_fd_jac=True)
        else:
            lyap_nb = numba_backend.build_lyapunov_solver(rhs_nb, jac_nb, use_fd_jac=False)
        params_arr = np.array([sigma, rho, beta], dtype=float)

    kwargs_base = dict(
        rhs=rhs,
        x0=x0,
        t0=0.0,
        dt=float(args.dt),
        t_transient=float(args.t_transient),
        qr_every_steps=int(args.qr_every_steps),
        fd_eps=float(args.fd_eps),
        solver_kind="rk4",
        jac=jac,
    )

    for t_final, t_measure in zip(final_times, t_measures):
        kwargs = dict(kwargs_base, t_measure=float(t_measure))

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
                    float(args.fd_eps),
                    params_arr,
                )

        if cpp_available and cpp_mod is not None:
            for _ in range(args.warmup):
                _compute_cpp(
                    cpp_mod,
                    rhs=rhs,
                    x0=x0,
                    t0=0.0,
                    dt=float(args.dt),
                    t_transient=float(args.t_transient),
                    t_measure=float(t_measure),
                    qr_every_steps=int(args.qr_every_steps),
                    jac=jac,
                    fd_eps=float(args.fd_eps),
                )

        py_time = _time_runs(lambda: _compute_python(kwargs), args.runs)
        py_times.append(py_time)
        print(f"t_final={t_final:5d} (t_measure={t_measure:.1f}) | Python avg: {py_time:.4f} s")

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
                    float(args.fd_eps),
                    params_arr,
                ),
                args.runs,
            )
            numba_times.append(numba_time)
            speedup = py_time / numba_time if numba_time > 0 else float("inf")
            print(f"t_final={t_final:5d} (t_measure={t_measure:.1f}) | Numba  avg: {numba_time:.4f} s | Speedup: {speedup:.2f}x")
        else:
            numba_times.append(None)

        if cpp_available and cpp_mod is not None:
            cpp_time = _time_runs(
                lambda: _compute_cpp(
                    cpp_mod,
                    rhs=rhs,
                    x0=x0,
                    t0=0.0,
                    dt=float(args.dt),
                    t_transient=float(args.t_transient),
                    t_measure=float(t_measure),
                    qr_every_steps=int(args.qr_every_steps),
                    jac=jac,
                    fd_eps=float(args.fd_eps),
                ),
                args.runs,
            )
            cpp_times.append(cpp_time)
            speedup = py_time / cpp_time if cpp_time > 0 else float("inf")
            print(f"t_final={t_final:5d} (t_measure={t_measure:.1f}) | C++    avg: {cpp_time:.4f} s | Speedup: {speedup:.2f}x")
        else:
            cpp_times.append(None)

    if args.csv_path:
        csv_path = Path(args.csv_path).expanduser()
        _save_csv(csv_path, final_times, t_measures, py_times, numba_times, cpp_times)
        print(f"\nSaved CSV to: {csv_path}")

    if not args.no_plot:
        plt.figure(figsize=(9, 5), dpi=150)
        plt.plot(final_times, py_times, marker="o", color="tab:blue", label="Python")
        if any(v is not None for v in numba_times):
            nb_vals = [np.nan if v is None else v for v in numba_times]
            plt.plot(final_times, nb_vals, marker="o", color="tab:green", label="Numba")
        if any(v is not None for v in cpp_times):
            cpp_vals = [np.nan if v is None else v for v in cpp_times]
            plt.plot(final_times, cpp_vals, marker="o", color="tab:orange", label="C++")
        plt.xlabel("Final time")
        plt.ylabel("Execution time (s)")
        plt.title("Lyapunov RK4 runtime benchmark: Python vs Numba vs C++")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if args.plot_path:
            out = Path(args.plot_path).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=220)
            print(f"Saved plot to: {out}")
        else:
            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
