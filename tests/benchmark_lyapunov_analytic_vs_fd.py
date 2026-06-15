#!/usr/bin/env python3
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import sympy as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.henon_heiles_system_rhs import henon_heiles_rhs
from core.jacobians_fixed_systems import henon_heiles_jac, lorenz_jac, rossler_jac
from core.lorenz_system_rhs import lorenz_rhs
from core.lyapunov import compute_lyapunov_spectrum
from core.rossler_system_rhs import rossler_rhs


SAFE_FUNCS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "abs": sp.Abs,
}


def parse_params_text(text: str) -> dict[str, float]:
    params: dict[str, float] = {}
    for line in (text or "").splitlines():
        s = line.replace("\u00a0", " ").strip()
        if not s:
            continue
        if "=" not in s:
            raise ValueError(f"Parameter line must be name=value. Got: {line!r}")
        name, val = s.split("=", 1)
        name = name.replace("\u00a0", " ").strip()
        val = val.replace("\u00a0", " ").strip()
        if name.lower() == "t":
            raise ValueError("Parameter name 't' is reserved.")
        params[name] = float(val)
    return params


def build_custom_rhs(var_names: list[str], eq_lines: list[str], params: dict[str, float]):
    n = len(var_names)
    if len(eq_lines) != n:
        raise ValueError(f"Need exactly {n} equations. Got {len(eq_lines)}.")

    t_sym = sp.Symbol("t")
    var_syms = sp.symbols(var_names)
    param_syms = {k: sp.Symbol(k) for k in params.keys()}
    locals_dict = {
        **SAFE_FUNCS,
        "t": t_sym,
        **{name: sym for name, sym in zip(var_names, var_syms)},
        **param_syms,
    }

    exprs = []
    for i, line in enumerate(eq_lines):
        s = (line or "").strip()
        if not s:
            raise ValueError(f"Equation {i + 1} is empty.")
        exprs.append(sp.sympify(s, locals=locals_dict))

    args = [t_sym] + list(var_syms) + [param_syms[k] for k in params.keys()]
    f_rhs = sp.lambdify(args, exprs, modules=["numpy"])
    param_values = [float(params[k]) for k in params.keys()]

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        vals = [float(t)] + list(np.asarray(y, dtype=float)) + param_values
        out = f_rhs(*vals)
        return np.array(out, dtype=float)

    return rhs


def build_custom_rhs_and_jacobian(var_names: list[str], eq_lines: list[str], params: dict[str, float]):
    n = len(var_names)
    if len(eq_lines) != n:
        raise ValueError(f"Need exactly {n} equations. Got {len(eq_lines)}.")

    t_sym = sp.Symbol("t")
    var_syms = sp.symbols(var_names)
    param_syms = {k: sp.Symbol(k) for k in params.keys()}
    locals_dict = {
        **SAFE_FUNCS,
        "t": t_sym,
        **{name: sym for name, sym in zip(var_names, var_syms)},
        **param_syms,
    }

    exprs = []
    for i, line in enumerate(eq_lines):
        s = (line or "").strip()
        if not s:
            raise ValueError(f"Equation {i + 1} is empty.")
        exprs.append(sp.sympify(s, locals=locals_dict))

    jac_expr = sp.Matrix(exprs).jacobian(var_syms)
    args = [t_sym] + list(var_syms) + [param_syms[k] for k in params.keys()]
    f_rhs = sp.lambdify(args, exprs, modules=["numpy"])
    f_jac = sp.lambdify(args, jac_expr, modules=["numpy"])
    param_values = [float(params[k]) for k in params.keys()]

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        vals = [float(t)] + list(np.asarray(y, dtype=float)) + param_values
        out = f_rhs(*vals)
        return np.array(out, dtype=float)

    def jac(t: float, y: np.ndarray) -> np.ndarray:
        vals = [float(t)] + list(np.asarray(y, dtype=float)) + param_values
        out = f_jac(*vals)
        return np.array(out, dtype=float)

    return rhs, jac


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_time_grid(args: argparse.Namespace) -> list[float]:
    if args.t_final_list:
        vals = []
        for raw in args.t_final_list.split(","):
            s = raw.strip()
            if not s:
                continue
            vals.append(float(s))
        if not vals:
            raise ValueError("Empty --t-final-list.")
        vals = sorted(set(vals))
        return vals

    start = float(args.t_final_start)
    stop = float(args.t_final_stop)
    step = float(args.t_final_step)
    if step <= 0.0:
        raise ValueError("--t-final-step must be > 0.")
    if stop < start:
        raise ValueError("--t-final-stop must be >= --t-final-start.")

    n = int(np.floor((stop - start) / step + 1e-12)) + 1
    vals = start + step * np.arange(n, dtype=float)
    vals = vals[vals <= stop + 1e-12]
    return [float(v) for v in vals]


def measurement_window(lyap_settings: dict, t0: float, tf: float, dt: float) -> tuple[float, float]:
    total = float(tf) - float(t0)
    if total <= 0.0:
        raise ValueError("tf must be > t0.")

    t_transient = lyap_settings.get("t_transient", None)
    if t_transient is None:
        transient_steps = lyap_settings.get("transient_steps", None)
        transient_fraction = lyap_settings.get("transient_fraction", None)
        if transient_steps is not None:
            t_transient = float(transient_steps) * float(dt)
        elif transient_fraction is not None:
            t_transient = float(transient_fraction) * total
        else:
            t_transient = 0.0

    t_transient = float(t_transient)
    t_measure = total - t_transient
    if t_measure <= 0.0:
        raise ValueError(
            f"Invalid Lyapunov window: tf={tf:.3f}, t_transient={t_transient:.3f} leaves no measure time."
        )
    return t_transient, t_measure


def build_rhs_and_jac(sys_cfg: dict, param_name: str | None, param_value: float | None):
    key = str(sys_cfg.get("system_key", "")).lower().strip()
    params = sys_cfg.get("params") or {}
    params_text = sys_cfg.get("params_text")
    if not params and isinstance(params_text, str) and params_text.strip():
        params = parse_params_text(params_text)
    params = {str(k): float(v) for k, v in (params or {}).items()}

    if param_value is not None and not param_name:
        raise ValueError("--param-value requires --param-name.")
    if param_name:
        if param_name not in params and param_value is None:
            raise ValueError(f"Parameter {param_name!r} not found in config.")
        if param_value is not None:
            params[param_name] = float(param_value)

    if key == "lorenz":
        return (
            lambda tt, xx: lorenz_rhs(tt, xx, **params),
            lambda tt, xx: lorenz_jac(tt, xx, **params),
            params,
        )
    if key == "rossler":
        return (
            lambda tt, xx: rossler_rhs(tt, xx, **params),
            lambda tt, xx: rossler_jac(tt, xx, **params),
            params,
        )
    if key == "henon_heiles":
        return (
            lambda tt, xx: henon_heiles_rhs(tt, xx, **params),
            lambda tt, xx: henon_heiles_jac(tt, xx, **params),
            params,
        )
    if key == "custom":
        var_names = list(sys_cfg.get("var_names") or [])
        eq_lines = list(sys_cfg.get("eq_lines") or [])
        if len(var_names) == 0 or len(eq_lines) == 0:
            raise ValueError("Custom system requires var_names and eq_lines.")
        if len(eq_lines) != len(var_names):
            raise ValueError("Custom system: eq_lines must match var_names length.")
        auto_jac = bool(sys_cfg.get("auto_jacobian", False))
        use_jac = bool(sys_cfg.get("use_jacobian", False))
        if auto_jac:
            rhs, jac = build_custom_rhs_and_jacobian(var_names, eq_lines, params)
            if not use_jac:
                jac = None
        else:
            rhs = build_custom_rhs(var_names, eq_lines, params)
            jac = None
        return rhs, jac, params

    raise ValueError(f"Unknown system_key: {key!r}")


def benchmark_lambda_max(compute_once, runs: int, warmup: int) -> tuple[float, float]:
    for _ in range(max(0, int(warmup))):
        compute_once()

    times = []
    values = []
    for _ in range(max(1, int(runs))):
        t0 = time.perf_counter()
        val = float(compute_once())
        t1 = time.perf_counter()
        values.append(val)
        times.append(t1 - t0)

    return float(np.mean(values)), float(np.mean(times))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Lyapunov lambda_max for analytic vs finite-difference Jacobian "
            "across final times using a StaticParamsConfig."
        )
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "tests" / "StaticParamsConfig_5.json"),
        help="Path to StaticParamsConfig JSON.",
    )
    parser.add_argument(
        "--param-name",
        default="c",
        help="Parameter name to fix (single-parameter benchmark).",
    )
    parser.add_argument(
        "--param-value",
        type=float,
        default=None,
        help="Optional override value for --param-name.",
    )
    parser.add_argument("--t-final-list", default="", help="Comma-separated final-time list (overrides range args).")
    parser.add_argument("--t-final-start", type=float, default=4000.0)
    parser.add_argument("--t-final-stop", type=float, default=10000.0)
    parser.add_argument("--t-final-step", type=float, default=1000.0)
    parser.add_argument("--runs", type=int, default=3, help="Timed runs per method/time.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per method/time.")
    parser.add_argument("--dt", type=float, default=None, help="Override Lyapunov integration dt for benchmarking.")
    parser.add_argument("--fd-eps", type=float, default=None, help="Override finite-difference epsilon.")
    parser.add_argument("--qr-every-steps", type=int, default=None, help="Override QR steps.")
    parser.add_argument("--solver-kind", default="", help="Override solver kind (rk4/ivp/rk45/dop853).")
    parser.add_argument(
        "--save-prefix",
        default="tests/lyapunov_benchmark_StaticParamsConfig_5",
        help="Prefix for outputs: *_accuracy.png, *_runtime.png, *.csv",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    sys_cfg = cfg.get("system") or {}
    integ = cfg.get("integration") or {}
    lyap_cfg = cfg.get("lyapunov") or {}
    lyap_settings = lyap_cfg.get("settings") or {}

    t0 = float(integ.get("t0", 0.0))
    dt_integration = float(integ.get("dt", 0.01))
    y0 = np.asarray(integ.get("y0", []), dtype=float)
    if y0.size == 0:
        raise ValueError("integration.y0 is missing or empty.")

    lyap_dt = float(args.dt) if args.dt is not None else float(lyap_settings.get("dt", dt_integration))
    solve_options = dict(integ.get("solve_options") or {})
    solver_kind = str(args.solver_kind).strip().lower() or str(integ.get("solver_kind", "ivp")).lower()
    if solver_kind == "ivp":
        solver_kind = "rk45"
    if solver_kind == "rk45":
        solve_options.setdefault("method", "RK45")
    elif solver_kind == "dop853":
        solve_options.setdefault("method", "DOP853")

    qr_every_steps = args.qr_every_steps
    if qr_every_steps is None:
        qr_every_steps = int(lyap_settings.get("qr_every_steps", 1))
        qr_interval = float(lyap_settings.get("qr_interval", 0.0))
        if qr_interval > 0.0:
            qr_every_steps = max(1, int(round(qr_interval / lyap_dt)))
    qr_every_steps = max(1, int(qr_every_steps))

    fd_eps = float(args.fd_eps) if args.fd_eps is not None else float(lyap_settings.get("fd_eps", 1e-8))

    rhs_raw, jac_raw, params = build_rhs_and_jac(
        sys_cfg,
        param_name=str(args.param_name).strip() or None,
        param_value=args.param_value,
    )
    if jac_raw is None:
        raise ValueError(
            "Analytic Jacobian is not available in this config/system. "
            "Enable auto_jacobian+use_jacobian or use a system with built-in jacobian."
        )

    def rhs_fn(tt: float, xx: np.ndarray) -> np.ndarray:
        return np.asarray(rhs_raw(tt, xx), dtype=float)

    def jac_fn(tt: float, xx: np.ndarray) -> np.ndarray:
        return np.asarray(jac_raw(tt, xx), dtype=float)

    t_finals = parse_time_grid(args)

    prefix = Path(args.save_prefix)
    if not prefix.is_absolute():
        prefix = PROJECT_ROOT / prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    print("Lyapunov Jacobian benchmark (analytic vs finite-difference)")
    print(f"config: {args.config}")
    if args.param_name:
        if args.param_name in params:
            print(f"fixed parameter: {args.param_name}={params[args.param_name]:.8g}")
        else:
            print(f"fixed parameter: {args.param_name} (not in params map)")
    print(
        f"solver={solver_kind} | dt={lyap_dt:g} | qr_every_steps={qr_every_steps} | "
        f"fd_eps={fd_eps:g} | runs={args.runs} | warmup={args.warmup}"
    )

    for tf in t_finals:
        t_transient, t_measure = measurement_window(lyap_settings, t0=t0, tf=tf, dt=lyap_dt)

        common_kwargs = dict(
            rhs=rhs_fn,
            x0=y0,
            t0=t0,
            dt=lyap_dt,
            t_transient=t_transient,
            t_measure=t_measure,
            qr_every_steps=qr_every_steps,
            solve_options=solve_options,
            solver_kind=solver_kind,
            fd_eps=fd_eps,
        )

        def compute_analytic():
            res = compute_lyapunov_spectrum(jac=jac_fn, **common_kwargs)
            return float(np.max(np.asarray(res.lambdas, dtype=float)))

        def compute_fd():
            res = compute_lyapunov_spectrum(jac=None, **common_kwargs)
            return float(np.max(np.asarray(res.lambdas, dtype=float)))

        lmax_analytic, t_analytic = benchmark_lambda_max(compute_analytic, runs=args.runs, warmup=args.warmup)
        lmax_fd, t_fd = benchmark_lambda_max(compute_fd, runs=args.runs, warmup=args.warmup)

        abs_err = abs(lmax_fd - lmax_analytic)
        rel_err = abs_err / max(abs(lmax_analytic), 1e-12)

        row = {
            "tf": float(tf),
            "t_transient": float(t_transient),
            "t_measure": float(t_measure),
            "lambda_max_analytic": float(lmax_analytic),
            "lambda_max_fd": float(lmax_fd),
            "abs_error_fd_vs_analytic": float(abs_err),
            "rel_error_fd_vs_analytic": float(rel_err),
            "runtime_s_analytic": float(t_analytic),
            "runtime_s_fd": float(t_fd),
        }
        rows.append(row)

        print(
            f"tf={tf:8.2f} | lmax_an={lmax_analytic:+.8f} | lmax_fd={lmax_fd:+.8f} | "
            f"|Δ|={abs_err:.3e} | t_an={t_analytic:.4f}s | t_fd={t_fd:.4f}s"
        )

    tf_arr = np.array([r["tf"] for r in rows], dtype=float)
    lmax_an_arr = np.array([r["lambda_max_analytic"] for r in rows], dtype=float)
    lmax_fd_arr = np.array([r["lambda_max_fd"] for r in rows], dtype=float)
    abs_err_arr = np.array([r["abs_error_fd_vs_analytic"] for r in rows], dtype=float)
    t_an_arr = np.array([r["runtime_s_analytic"] for r in rows], dtype=float)
    t_fd_arr = np.array([r["runtime_s_fd"] for r in rows], dtype=float)

    fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
    ax1.plot(tf_arr, lmax_an_arr, marker="o", color="tab:blue", label="lambda_max (analytic)")
    ax1.plot(tf_arr, lmax_fd_arr, marker="s", color="tab:orange", label="lambda_max (finite-difference)")
    ax1.set_xlabel("Final time (tf)")
    ax1.set_ylabel("lambda_max")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(tf_arr, abs_err_arr, marker="^", linestyle="--", color="tab:red", label="|Δlambda_max| (FD vs analytic)")
    ax2.set_ylabel("|Δlambda_max|")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    param_label = ""
    if args.param_name and args.param_name in params:
        param_label = f" | fixed {args.param_name}={params[args.param_name]:.6g}"
    ax1.set_title("Lyapunov accuracy: analytic vs finite-difference Jacobian" + param_label)
    fig1.tight_layout()
    accuracy_path = prefix.with_name(prefix.name + "_accuracy.png")
    fig1.savefig(accuracy_path, dpi=220)
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(tf_arr, t_an_arr, marker="o", color="tab:blue", label="runtime analytic")
    ax.plot(tf_arr, t_fd_arr, marker="s", color="tab:orange", label="runtime finite-difference")
    ax.set_xlabel("Final time (tf)")
    ax.set_ylabel("Mean execution time (s)")
    ax.set_title("Lyapunov runtime benchmark: analytic vs finite-difference Jacobian" + param_label)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig2.tight_layout()
    runtime_path = prefix.with_name(prefix.name + "_runtime.png")
    fig2.savefig(runtime_path, dpi=220)
    plt.close(fig2)

    csv_path = prefix.with_suffix(".csv")
    fieldnames = [
        "tf",
        "t_transient",
        "t_measure",
        "lambda_max_analytic",
        "lambda_max_fd",
        "abs_error_fd_vs_analytic",
        "rel_error_fd_vs_analytic",
        "runtime_s_analytic",
        "runtime_s_fd",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved accuracy plot: {accuracy_path}")
    print(f"Saved runtime  plot: {runtime_path}")
    print(f"Saved benchmark csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
