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
from core.henon_heiles_system_rhs import (
    henon_heiles_dp_dt,
    henon_heiles_dq_dt,
    henon_heiles_rhs,
)
from core.jacobians_fixed_systems import henon_heiles_jac, lorenz_jac, rossler_jac
from core.lorenz_system_rhs import lorenz_rhs
from core.lyapunov import compute_lyapunov_spectrum
from core.rossler_system_rhs import rossler_rhs
from core.solver import integrate_system, integrate_system_rk4
from core.symplectic_solver import integrate_system_symplectic_fr, integrate_system_symplectic_verlet


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


def measurement_window(cfg: dict, t0: float, tf: float, dt: float) -> tuple[float, float]:
    lyap_cfg = cfg.get("lyapunov") or {}
    s = lyap_cfg.get("settings") or {}
    t_transient = s.get("t_transient", None)
    t_measure = s.get("t_measure", None)
    total_time = float(tf) - float(t0)

    if t_transient is None or t_measure is None:
        transient_steps = s.get("transient_steps", None)
        transient_fraction = s.get("transient_fraction", None)
        if transient_steps is not None:
            t_transient = float(transient_steps) * float(dt)
        elif transient_fraction is not None:
            t_transient = float(transient_fraction) * total_time
        else:
            t_transient = 0.0
        t_measure = total_time - float(t_transient)

    t_transient = float(t_transient)
    t_measure = float(t_measure)
    t_start = float(t0) + t_transient
    t_end = t_start + t_measure
    if t_end > float(tf) + 1e-12:
        raise ValueError("Measurement window exceeds integration interval.")
    return t_start, t_end


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check sum of Lyapunov exponents vs mean divergence in the same window."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "tests" / "StaticParamsConfig(5).json"),
        help="Path to StaticParamsConfig JSON.",
    )
    parser.add_argument("--tol-abs", type=float, default=5e-3)
    parser.add_argument("--tol-rel", type=float, default=0.1)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    sys_cfg = cfg.get("system") or {}
    integ = cfg.get("integration") or {}

    t0 = float(integ.get("t0", 0.0))
    tf = float(integ.get("tf", 1.0))
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

    if solver_kind in ("symplectic_verlet", "symplectic_fr"):
        if sys_cfg.get("system_key", "").lower().strip() == "henon_heiles":
            lam = float((_params or {}).get("lambda", 1.0))

            def dq_dt(t, p):
                return henon_heiles_dq_dt(t, p, lam=lam)

            def dp_dt(t, q):
                return henon_heiles_dp_dt(t, q, lam=lam)

        else:
            raise ValueError("Symplectic solver requested for non-Hamiltonian system.")

        if solver_kind == "symplectic_verlet":
            sol = integrate_system_symplectic_verlet(
                rhs_fn, (t0, tf), y0, t_step=dt, dp_dt=dp_dt, dq_dt=dq_dt
            )
        else:
            sol = integrate_system_symplectic_fr(
                rhs_fn, (t0, tf), y0, t_step=dt, dp_dt=dp_dt, dq_dt=dq_dt
            )
    elif solver_kind == "rk4":
        sol = integrate_system_rk4(rhs_fn, (t0, tf), y0, t_step=dt)
    else:
        sol = integrate_system(rhs_fn, (t0, tf), y0, t_step=dt, **solve_opts)

    if not sol.success:
        raise RuntimeError(sol.message)

    t = sol.t
    Y = sol.y
    t_start, t_end = measurement_window(cfg, t0, tf, dt)

    idx_start = int(np.searchsorted(t, t_start, side="left"))
    idx_end = int(np.searchsorted(t, t_end, side="right"))
    if idx_end <= idx_start:
        raise ValueError("Measurement window is empty after indexing.")

    var_names = list(sys_cfg.get("var_names") or [])
    if "x3" in var_names:
        x3_idx = var_names.index("x3")
    elif Y.shape[0] >= 3:
        x3_idx = 2
    else:
        raise ValueError("Cannot locate x3 for divergence check.")

    x3_window = Y[x3_idx, idx_start:idx_end]
    mean_div = float(np.mean(2.0 * x3_window))

    lyap_cfg = cfg.get("lyapunov") or {}
    s = lyap_cfg.get("settings") or {}
    t_transient = float(t_start - t0)
    t_measure = float(t_end - t_start)
    qr_every_steps = int(s.get("qr_every_steps", 1))
    qr_interval = float(s.get("qr_interval", 0.0))
    if qr_every_steps <= 0:
        qr_every_steps = 1
    if qr_interval > 0.0 and qr_every_steps == 1:
        qr_every_steps = max(1, int(round(qr_interval / dt)))

    jac_mode = str(s.get("jacobian", "")).lower().strip()
    fd_eps = float(s.get("fd_eps", 1e-8))
    jac_to_use = None
    if jac_mode == "analytic" and jac_raw is not None:
        def jac_fn(tt: float, xx: np.ndarray) -> np.ndarray:
            return jac_raw(tt, xx)
        jac_to_use = jac_fn

    result = compute_lyapunov_spectrum(
        rhs=rhs_fn,
        x0=y0,
        t0=t0,
        dt=dt,
        t_transient=t_transient,
        t_measure=t_measure,
        qr_every_steps=qr_every_steps,
        solve_options=solve_opts,
        jac=jac_to_use,
        fd_eps=fd_eps,
    )
    sum_lambda = float(np.sum(result.lambdas))

    diff = abs(sum_lambda - mean_div)
    allowed = max(float(args.tol_abs), float(args.tol_rel) * max(1.0, abs(mean_div)))

    print(f"t_window=[{t_start:.6f}, {t_end:.6f}] | samples={idx_end - idx_start}")
    print(f"sum_lambda={sum_lambda:.6f}")
    print(f"mean_div={mean_div:.6f}")
    print(f"diff={diff:.6f} | allowed={allowed:.6f}")

    if diff > allowed:
        raise SystemExit("Mismatch between sum of exponents and mean divergence.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
