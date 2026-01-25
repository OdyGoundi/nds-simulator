#!/usr/bin/env python3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.lyapunov as lyap


def _print_result(name: str, passed: bool, details: str) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {details}")


def _fake_solve_ivp_factory(call_counter, nfev: int = 100):
    def fake_solve_ivp(rhs, t_span, y0, t_eval=None, **opts):
        call_counter["count"] += 1
        y0_arr = np.asarray(y0, dtype=float)
        if t_eval is None:
            t_eval = [t_span[1]]
        y = y0_arr.reshape(-1, 1)
        return SimpleNamespace(success=True, message="ok", y=y, nfev=nfev)
    return fake_solve_ivp


def _run_test_auto_switch_to_rk4() -> bool:
    call_counter = {"count": 0}
    fake = _fake_solve_ivp_factory(call_counter, nfev=100)
    real = lyap.solve_ivp
    lyap.solve_ivp = fake
    try:
        a = 0.1

        def rhs(tt: float, xx: np.ndarray) -> np.ndarray:
            return np.array([a * xx[0]], dtype=float)

        def jac(tt: float, xx: np.ndarray) -> np.ndarray:
            return np.array([[a]], dtype=float)

        lyap.compute_lyapunov_spectrum(
            rhs=rhs,
            x0=np.array([1.0], dtype=float),
            t0=0.0,
            dt=0.1,
            t_transient=0.0,
            t_measure=0.2,
            qr_every_steps=1,
            solve_options={"rtol": 1e-6, "atol": 1e-9},
            jac=jac,
            solver_kind="rk45",
            auto_switch_rk4=True,
        )
    finally:
        lyap.solve_ivp = real

    passed = call_counter["count"] == 1
    _print_result("auto_switch_rk4", passed, f"solve_ivp_calls={call_counter['count']}")
    return passed


def _run_test_force_rk4_no_solve_ivp() -> bool:
    call_counter = {"count": 0}
    fake = _fake_solve_ivp_factory(call_counter, nfev=100)
    real = lyap.solve_ivp
    lyap.solve_ivp = fake
    try:
        a = 0.1

        def rhs(tt: float, xx: np.ndarray) -> np.ndarray:
            return np.array([a * xx[0]], dtype=float)

        def jac(tt: float, xx: np.ndarray) -> np.ndarray:
            return np.array([[a]], dtype=float)

        lyap.compute_lyapunov_spectrum(
            rhs=rhs,
            x0=np.array([1.0], dtype=float),
            t0=0.0,
            dt=0.1,
            t_transient=0.0,
            t_measure=0.2,
            qr_every_steps=1,
            solve_options={"rtol": 1e-6, "atol": 1e-9},
            jac=jac,
            solver_kind="rk4",
            auto_switch_rk4=False,
        )
    finally:
        lyap.solve_ivp = real

    passed = call_counter["count"] == 0
    _print_result("force_rk4_no_ivp", passed, f"solve_ivp_calls={call_counter['count']}")
    return passed


def main() -> int:
    passed = True
    passed &= _run_test_auto_switch_to_rk4()
    passed &= _run_test_force_rk4_no_solve_ivp()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
