import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core import numba_backend
from core.jacobians_fixed_systems import lorenz_jac
from core.lorenz_system_rhs import lorenz_rhs
from core.lyapunov import compute_lyapunov_spectrum


def _print_result(name: str, passed: bool, details: str) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {details}")


def _compute_python(x0, dt, t_transient, t_measure, qr_every_steps, sigma, rho, beta):
    def rhs(tt: float, xx: np.ndarray) -> np.ndarray:
        return lorenz_rhs(tt, xx, sigma=sigma, rho=rho, beta=beta)

    def jac(tt: float, xx: np.ndarray) -> np.ndarray:
        return lorenz_jac(tt, xx, sigma=sigma, rho=rho, beta=beta)

    return compute_lyapunov_spectrum(
        rhs=rhs,
        x0=x0,
        t0=0.0,
        dt=dt,
        t_transient=t_transient,
        t_measure=t_measure,
        jac=jac,
        qr_every_steps=qr_every_steps,
        solver_kind="rk4",
    )


def _compute_numba(x0, dt, t_transient, t_measure, qr_every_steps, sigma, rho, beta):
    rhs_nb, jac_nb, _param_names = numba_backend.build_builtin_system("lorenz")
    lyap_nb = numba_backend.build_lyapunov_solver(rhs_nb, jac_nb, use_fd_jac=False)
    params_arr = np.array([sigma, rho, beta], dtype=float)
    lambdas, _sums, _t_meas, _n_qr, _x_final = lyap_nb(
        np.asarray(x0, dtype=float),
        0.0,
        float(dt),
        float(t_transient),
        float(t_measure),
        int(qr_every_steps),
        float(1e-8),
        params_arr,
    )
    return np.asarray(lambdas, dtype=float)


def main() -> int:
    if not numba_backend.numba_available():
        print("[SKIP] numba backend not available.")
        return 0

    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    x0 = np.array([1.0, 1.0, 1.0], dtype=float)
    dt = 0.01
    t_transient = 50.0
    t_measure = 100.0
    qr_every_steps = 10

    res_py = _compute_python(x0, dt, t_transient, t_measure, qr_every_steps, sigma, rho, beta)
    res_nb = _compute_numba(x0, dt, t_transient, t_measure, qr_every_steps, sigma, rho, beta)

    l_py = np.sort(np.array(res_py.lambdas, dtype=float))[::-1]
    l_nb = np.sort(np.array(res_nb, dtype=float))[::-1]

    max_err = float(np.max(np.abs(l_py - l_nb)))
    tol = 1e-2
    passed = max_err <= tol
    _print_result(
        "lyapunov_numba_vs_python",
        passed,
        f"max_err={max_err:.3e}, tol={tol:.1e}, py={l_py}, nb={l_nb}",
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
