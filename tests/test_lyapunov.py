import sys
from pathlib import Path

import numpy as np

# Add project root to sys.path (so `import core...` works when running from tests/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.lyapunov import compute_lyapunov_spectrum


def _print_result(name: str, passed: bool, details: str) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {details}")


def _run_test_linear_1d() -> bool:
    """
    dx/dt = a xx -> Lyapunov exponent should be a.
    """
    a = 0.7

    def rhs(tt: float, xx: np.ndarray) -> np.ndarray:
        return np.array([a * xx[0]], dtype=float)

    def jac(tt: float, xx: np.ndarray) -> np.ndarray:
        return np.array([[a]], dtype=float)

    res = compute_lyapunov_spectrum(
        rhs=rhs,
        x0=np.array([1.0], dtype=float),
        t0=0.0,
        dt=0.05,
        t_transient=0.0,
        t_measure=8.0,
        jac=jac,
        qr_every_steps=1,
        solve_options={"rtol": 1e-6, "atol": 1e-9},
    )

    actual = float(res.lambdas[0])
    passed = abs(actual - a) < 5e-2
    _print_result(
        "linear_1d",
        passed,
        f"expected ~{a:.3f}, got {actual:.3f}",
    )
    return passed


def _run_test_linear_2d_fd() -> bool:
    """
    2D diagonal linear system with finite-difference Jacobian.
    dx/dt = a xx, dy/dt = b y -> exponents {a, b}.
    """
    a = 0.3
    b = -0.9

    def rhs(tt: float, xx: np.ndarray) -> np.ndarray:
        return np.array([a * xx[0], b * xx[1]], dtype=float)

    res = compute_lyapunov_spectrum(
        rhs=rhs,
        x0=np.array([1.0, -1.0], dtype=float),
        t0=0.0,
        dt=0.05,
        t_transient=0.0,
        t_measure=8.0,
        jac=None,
        fd_eps=1e-7,
        qr_every_steps=1,
        solve_options={"rtol": 1e-6, "atol": 1e-9},
    )

    exps = np.sort(res.lambdas)
    expected = np.sort(np.array([a, b], dtype=float))
    max_err = float(np.max(np.abs(exps - expected)))
    passed = max_err < 7e-2
    _print_result(
        "linear_2d_fd",
        passed,
        f"expected ~{expected}, got {exps}, max_err={max_err:.3f}",
    )
    return passed


def _run_test_harmonic_oscillator() -> bool:
    """
    Harmonic oscillator: xx'' + xx = 0 -> Lyapunov exponents ~0.
    """
    def rhs(tt: float, xx: np.ndarray) -> np.ndarray:
        # xx[0] = position, xx[1] = velocity
        return np.array([xx[1], -xx[0]], dtype=float)

    def jac(tt: float, xx: np.ndarray) -> np.ndarray:
        return np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float)

    res = compute_lyapunov_spectrum(
        rhs=rhs,
        x0=np.array([1.0, 0.0], dtype=float),
        t0=0.0,
        dt=0.05,
        t_transient=0.0,
        t_measure=10.0,
        jac=jac,
        qr_every_steps=1,
        solve_options={"rtol": 1e-6, "atol": 1e-9},
    )

    max_abs = float(np.max(np.abs(res.lambdas)))
    passed = max_abs < 5e-2
    _print_result(
        "harmonic_oscillator",
        passed,
        f"expected ~0, got {res.lambdas}, max_abs={max_abs:.3f}",
    )
    return passed


def main() -> int:
    tests = [
        _run_test_linear_1d,
        _run_test_linear_2d_fd,
        _run_test_harmonic_oscillator,
    ]
    results = [fn() for fn in tests]
    passed = all(results)
    summary = "ALL TESTS PASSED" if passed else "SOME TESTS FAILED"
    print(summary)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
