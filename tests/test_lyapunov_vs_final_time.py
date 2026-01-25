#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.helpers import build_custom_rhs, build_custom_rhs_and_jacobian, parse_params
from core.henon_heiles_system_rhs import henon_heiles_rhs
from core.jacobians_fixed_systems import henon_heiles_jac, lorenz_jac, rossler_jac
from core.lorenz_system_rhs import lorenz_rhs
from core.lyapunov import compute_lyapunov_spectrum
from core.rossler_system_rhs import rossler_rhs


TF_LIST = [
    5000.0,
    7500.0,
    10000.0,
    12500.0,
    15000.0,
    20000.0,
    30000.0,
    40000.0,
    50000.0,
    60000.0,
    100000.0,
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_rhs_and_jac(sys_cfg: dict):
    key = str(sys_cfg.get("system_key", "")).lower().strip()
    params = sys_cfg.get("params") or {}
    params_text = sys_cfg.get("params_text")
    if not params and isinstance(params_text, str) and params_text.strip():
        params = parse_params(params_text)
    params = {str(k): float(v) for k, v in (params or {}).items()}

    if key == "lorenz":
        return (
            lambda t, y: lorenz_rhs(t, y, **params),
            lambda t, y: lorenz_jac(t, y, **params),
            params,
        )
    if key == "rossler":
        return (
            lambda t, y: rossler_rhs(t, y, **params),
            lambda t, y: rossler_jac(t, y, **params),
            params,
        )
    if key == "henon_heiles":
        return (
            lambda t, y: henon_heiles_rhs(t, y, **params),
            lambda t, y: henon_heiles_jac(t, y, **params),
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


def lyapunov_window_for_tf(cfg: dict, t0: float, tf: float, dt: float) -> tuple[float, float]:
    lyap_cfg = cfg.get("lyapunov") or {}
    s = lyap_cfg.get("settings") or {}
    t_transient = s.get("t_transient", None)
    if t_transient is None:
        transient_steps = s.get("transient_steps", None)
        transient_fraction = s.get("transient_fraction", None)
        if transient_steps is not None:
            t_transient = float(transient_steps) * float(dt)
        elif transient_fraction is not None:
            t_transient = float(transient_fraction) * (float(tf) - float(t0))
        else:
            t_transient = 0.0

    t_transient = float(t_transient)
    # Use remaining time after transient to see convergence vs final time.
    t_measure = float(tf) - float(t0) - t_transient
    if t_measure <= 0.0:
        raise ValueError("Not enough time for Lyapunov measurement.")
    return t_transient, t_measure


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute Lyapunov spectrum vs final time using a shared config."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "tests" / "StaticParamsConfig(5).json"),
        help="Path to StaticParamsConfig JSON.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    sys_cfg = cfg.get("system") or {}
    integ = cfg.get("integration") or {}

    t0 = float(integ.get("t0", 0.0))
    dt = float(integ.get("dt", 0.01))
    y0 = np.asarray(integ.get("y0", []), dtype=float)
    solver_kind = str(integ.get("solver_kind", "rk45")).lower().strip()
    solve_opts = dict(integ.get("solve_options") or {})

    if solver_kind == "ivp":
        solver_kind = "rk45"
    if solver_kind == "rk45":
        solve_opts.setdefault("method", "RK45")
    elif solver_kind == "dop853":
        solve_opts.setdefault("method", "DOP853")

    rhs_raw, jac_raw, _params = build_rhs_and_jac(sys_cfg)

    def rhs_fn(tt: float, xx: np.ndarray) -> np.ndarray:
        return rhs_raw(tt, xx)

    lyap_cfg = cfg.get("lyapunov") or {}
    s = lyap_cfg.get("settings") or {}
    lyap_dt = float(s.get("dt", dt))
    qr_every_steps = int(s.get("qr_every_steps", 1))
    qr_interval = float(s.get("qr_interval", 0.0))
    if qr_interval > 0.0:
        qr_every_steps = max(1, int(round(qr_interval / lyap_dt)))
    if qr_every_steps <= 0:
        qr_every_steps = 1

    jac_mode = str(s.get("jacobian", "")).lower().strip()
    fd_eps = float(s.get("fd_eps", 1e-8))
    jac_to_use = None
    if jac_mode == "analytic" and jac_raw is not None:
        def jac_fn(tt: float, xx: np.ndarray) -> np.ndarray:
            return jac_raw(tt, xx)
        jac_to_use = jac_fn

    results = []
    for tf in TF_LIST:
        t_transient, t_measure = lyapunov_window_for_tf(cfg, t0, tf, lyap_dt)
        res = compute_lyapunov_spectrum(
            rhs=rhs_fn,
            x0=y0,
            t0=t0,
            dt=lyap_dt,
            t_transient=t_transient,
            t_measure=t_measure,
            qr_every_steps=qr_every_steps,
            solve_options=solve_opts,
            jac=jac_to_use,
            fd_eps=fd_eps,
        )
        lambdas = res.lambdas
        lmax = float(np.max(lambdas))
        results.append((tf, lmax, lambdas))

        lambdas_str = np.array2string(lambdas, precision=6, separator=", ")
        print(
            f"tf={tf:.1f} | t_transient={t_transient:.3f} | "
            f"t_measure={t_measure:.3f} | lmax={lmax:.6f} | lambdas={lambdas_str}"
        )

    print("tf,lambda_max")
    for tf, lmax, _ in results:
        print(f"{tf:.1f},{lmax:.8f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
