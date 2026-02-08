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
from core import numba_backend
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
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_python_rhs_and_jac(sys_cfg: dict):
    key = str(sys_cfg.get("system_key", "")).lower().strip()
    params = sys_cfg.get("params") or {}
    params_text = sys_cfg.get("params_text")
    if not params and isinstance(params_text, str) and params_text.strip():
        params = parse_params(params_text)
    params = {str(k): float(v) for k, v in (params or {}).items()}

    if key == "lorenz":
        def rhs(tt: float, xx: np.ndarray) -> np.ndarray:
            return lorenz_rhs(tt, xx, **params)

        def jac(tt: float, xx: np.ndarray) -> np.ndarray:
            return lorenz_jac(tt, xx, **params)

        return rhs, jac
    if key == "rossler":
        def rhs(tt: float, xx: np.ndarray) -> np.ndarray:
            return rossler_rhs(tt, xx, **params)

        def jac(tt: float, xx: np.ndarray) -> np.ndarray:
            return rossler_jac(tt, xx, **params)

        return rhs, jac
    if key == "henon_heiles":
        def rhs(tt: float, xx: np.ndarray) -> np.ndarray:
            return henon_heiles_rhs(tt, xx, **params)

        def jac(tt: float, xx: np.ndarray) -> np.ndarray:
            return henon_heiles_jac(tt, xx, **params)

        return rhs, jac
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
            rhs_raw, jac_raw = build_custom_rhs_and_jacobian(var_names, eq_lines, params)
            if not use_jac:
                jac_raw = None
        else:
            rhs_raw = build_custom_rhs(var_names, eq_lines, params)
            jac_raw = None

        def rhs(tt: float, xx: np.ndarray) -> np.ndarray:
            return rhs_raw(tt, xx)

        if jac_raw is None:
            return rhs, None

        def jac(tt: float, xx: np.ndarray) -> np.ndarray:
            return jac_raw(tt, xx)

        return rhs, jac

    raise ValueError(f"Unknown system_key: {key!r}")


def build_numba_rhs_and_params(sys_cfg: dict):
    key = str(sys_cfg.get("system_key", "")).lower().strip()
    params = sys_cfg.get("params") or {}
    params_text = sys_cfg.get("params_text")
    if not params and isinstance(params_text, str) and params_text.strip():
        params = parse_params(params_text)
    params = {str(k): float(v) for k, v in (params or {}).items()}

    if key in ("lorenz", "rossler", "henon_heiles"):
        rhs_nb, jac_nb, param_names = numba_backend.build_builtin_system(key)
        values = []
        for name in param_names:
            if name in params:
                values.append(float(params[name]))
            elif name == "lambda" and "lam" in params:
                values.append(float(params["lam"]))
            else:
                raise ValueError(f"Missing parameter {name!r} for system {key!r}.")
        params_arr = np.array(values, dtype=float)
        return rhs_nb, jac_nb, params_arr, False

    if key == "custom":
        var_names = list(sys_cfg.get("var_names") or [])
        eq_lines = list(sys_cfg.get("eq_lines") or [])
        if len(var_names) == 0 or len(eq_lines) == 0:
            raise ValueError("Custom system requires var_names and eq_lines.")
        if len(eq_lines) != len(var_names):
            raise ValueError("Custom system: eq_lines must match var_names length.")
        param_names = list(params.keys())
        if len(param_names) == 0:
            raise ValueError("Custom system requires parameters for Numba compilation.")
        auto_jac = bool(sys_cfg.get("auto_jacobian", False))
        use_jac = bool(sys_cfg.get("use_jacobian", False))
        from app import numba_custom
        if auto_jac and use_jac:
            rhs_nb, jac_nb = numba_custom.build_custom_numba_rhs_and_jacobian(
                var_names, eq_lines, param_names
            )
            use_fd_jac = False
        else:
            rhs_nb = numba_custom.build_custom_numba_rhs(var_names, eq_lines, param_names)
            jac_nb = None
            use_fd_jac = True
        params_arr = np.array([float(params[name]) for name in param_names], dtype=float)
        return rhs_nb, jac_nb, params_arr, use_fd_jac

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
    if not numba_backend.numba_available():
        print("[SKIP] numba backend not available.")
        return 0

    rhs_nb, jac_nb, params_arr, use_fd_jac = build_numba_rhs_and_params(sys_cfg)
    rhs_py, jac_py = build_python_rhs_and_jac(sys_cfg)

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
    lyap_nb = numba_backend.build_lyapunov_solver(
        rhs_nb,
        jac_nb,
        use_fd_jac=use_fd_jac,
    )

    results = []
    for tf in TF_LIST:
        t_transient, t_measure = lyapunov_window_for_tf(cfg, t0, tf, lyap_dt)
        lambdas, _sums, _t_meas, n_qr, _x_final = lyap_nb(
            np.asarray(y0, dtype=float),
            float(t0),
            float(lyap_dt),
            float(t_transient),
            float(t_measure),
            int(qr_every_steps),
            float(fd_eps),
            params_arr,
        )
        if int(n_qr) <= 0:
            raise ValueError("No QR steps performed. Increase t_measure or reduce qr_every_steps.")
        lambdas = np.asarray(lambdas, dtype=float)
        lmax = float(np.max(lambdas))
        results.append((tf, lmax, lambdas))

        lambdas_str = np.array2string(lambdas, precision=6, separator=", ")
        print(
            f"tf={tf:.1f} | t_transient={t_transient:.3f} | "
            f"t_measure={t_measure:.3f} | lmax={lmax:.6f} | lambdas={lambdas_str}"
        )

        if tf in (5000.0, 10000.0):
            jac_to_use = None
            if jac_mode == "analytic" and jac_py is not None:
                jac_to_use = jac_py
            res_py = compute_lyapunov_spectrum(
                rhs=rhs_py,
                x0=y0,
                t0=t0,
                dt=lyap_dt,
                t_transient=t_transient,
                t_measure=t_measure,
                qr_every_steps=qr_every_steps,
                jac=jac_to_use,
                fd_eps=fd_eps,
                solver_kind="rk4",
            )
            lambdas_py = np.asarray(res_py.lambdas, dtype=float)
            lmax_py = float(np.max(lambdas_py))
            lambdas_py_str = np.array2string(lambdas_py, precision=6, separator=", ")
            print(
                f"[COMPARE] tf={tf:.1f} | lmax_numba={lmax:.6f} | "
                f"lmax_rk4={lmax_py:.6f} | lambdas_numba={lambdas_str} | "
                f"lambdas_rk4={lambdas_py_str}"
            )

    print("tf,lambda_max")
    for tf, lmax, _ in results:
        print(f"{tf:.1f},{lmax:.8f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
