
"""
Lyapunov spectrum for nD continuous-time ODEs using:
  - variational equations (tangent dynamics)
  - periodic QR re-orthonormalization (numerically stable Gram–Schmidt)

Core-only numerics (no plotting, no I/O, no Streamlit).

Implementation detail:
- Because QR must happen frequently, we integrate in short chunks:
    [t, t+chunk_dt] -> take final state -> QR -> continue
- Default uses scipy.solve_ivp; optional fixed-step RK4 can be selected.

Typing note (Pyright/Pylance):
- We use Protocol-based callables so parameter-name checking does not break.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Protocol, Literal, overload

import numpy as np

try:
    from scipy.integrate import solve_ivp
except Exception:  # pragma: no cover
    solve_ivp = None


# ----------------------------
# Protocol-based callable types (Pyright-friendly)
# ----------------------------
class RhsFn(Protocol):
    def __call__(self, tt: float, xx: np.ndarray) -> np.ndarray: ...


class JacFn(Protocol):
    def __call__(self, tt: float, xx: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class LyapunovResult:
    """Results of Lyapunov spectrum estimation."""
    lambdas: np.ndarray        # (n,)
    sums_log: np.ndarray       # (n,) accumulated sum of ln|R_ii|
    t_meas: float              # effective measurement time
    n_qr: int                  # number of QR updates performed
    x_final: np.ndarray        # (n,) final state after measurement


# ----------------------------
# Jacobian helpers
# ----------------------------
def finite_difference_jacobian(
    rhs: RhsFn,
    t: float,
    x: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Central finite-difference Jacobian:
      J[:, j] = (f(x+e_j eps) - f(x-e_j eps)) / (2 eps)
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    J = np.zeros((n, n), dtype=float)

    f0 = rhs(t, x)
    if f0.shape != (n,):
        raise ValueError("rhs(t, x) must return shape (n,)")

    for j in range(n):
        dx = np.zeros(n, dtype=float)
        dx[j] = eps
        fp = rhs(t, x + dx)
        fm = rhs(t, x - dx)
        J[:, j] = (fp - fm) / (2.0 * eps)

    return J


# ----------------------------
# Internal utilities
# ----------------------------
def _pack_augmented(x: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Pack (x, Q) into a flat augmented vector y_aug."""
    n = x.size
    return np.concatenate([x, Q.reshape(n * n)], axis=0)


def _unpack_augmented(y_aug: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unpack y_aug into (x, Q)."""
    x = y_aug[:n].copy()
    Q = y_aug[n:].reshape((n, n)).copy()
    return x, Q


def _qr_accumulate(Q: np.ndarray, sums_log: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    QR decomposition with accumulation of ln|diag(R)|.
    Returns (Q_new, sums_log_updated).
    """
    Q_new, R = np.linalg.qr(Q)
    diag = np.diag(R)

    # Avoid log(0) if singular/underflow appears.
    diag_abs = np.maximum(np.abs(diag), 1e-300)
    sums_log = sums_log + np.log(diag_abs)

    return Q_new, sums_log


def _rk4_step(rhs, t: float, y: np.ndarray, h: float) -> np.ndarray:
    k1 = rhs(t, y)
    k2 = rhs(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = rhs(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = rhs(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _integrate_chunk_rk4(
    rhs_aug,
    t0: float,
    y0: np.ndarray,
    t1: float,
    dt: float,
) -> np.ndarray:
    if dt <= 0:
        raise ValueError("dt must be > 0 for RK4 integration.")
    duration = float(t1) - float(t0)
    if duration < 0:
        raise ValueError("t1 must be >= t0.")
    if duration <= 1e-15:
        return np.asarray(y0, dtype=float).copy()

    t = float(t0)
    y = np.asarray(y0, dtype=float).copy()
    n_full = int(np.floor(duration / float(dt)))
    for _ in range(n_full):
        y = _rk4_step(rhs_aug, t, y, float(dt))
        t += float(dt)

    rem = float(t1) - t
    if rem > 1e-15:
        y = _rk4_step(rhs_aug, t, y, rem)

    return y


def _rk4_cost_estimate(t0: float, t1: float, dt: float) -> int:
    duration = float(t1) - float(t0)
    if duration <= 0:
        return 0
    n_steps = int(np.ceil(duration / float(dt)))
    n_steps = max(1, n_steps)
    return int(4 * n_steps)


@overload
def _integrate_chunk_ivp(
    rhs_aug,
    t0: float,
    y0: np.ndarray,
    t1: float,
    solve_options: Optional[Dict[str, Any]] = None,
    *,
    return_nfev: Literal[False] = False,
) -> np.ndarray: ...


@overload
def _integrate_chunk_ivp(
    rhs_aug,
    t0: float,
    y0: np.ndarray,
    t1: float,
    solve_options: Optional[Dict[str, Any]] = None,
    *,
    return_nfev: Literal[True],
) -> Tuple[np.ndarray, int]: ...


def _integrate_chunk_ivp(
    rhs_aug,
    t0: float,
    y0: np.ndarray,
    t1: float,
    solve_options: Optional[Dict[str, Any]] = None,
    *,
    return_nfev: bool = False,
) -> np.ndarray | Tuple[np.ndarray, int]:
    """
    Integrate one chunk with solve_ivp and return y(t1).
    Uses t_eval=[t1] to reduce overhead.
    """
    if solve_ivp is None:
        raise RuntimeError("scipy is required (solve_ivp not available).")

    opts = dict(solve_options or {})
    if "method" not in opts:
        opts["method"] = "RK45"
    opts["max_step"] = float(t1) - float(t0)

    sol = solve_ivp(
        rhs_aug,
        (float(t0), float(t1)),
        y0,
        t_eval=[float(t1)],
        **opts,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    y_out = sol.y[:, -1]
    if return_nfev:
        return y_out, int(getattr(sol, "nfev", 0))
    return y_out


# ----------------------------
# Public API
# ----------------------------
def compute_lyapunov_spectrum(
    rhs: RhsFn,
    x0: np.ndarray,
    t0: float,
    dt: float,
    t_transient: float,
    t_measure: float,
    *,
    jac: Optional[JacFn] = None,
    fd_eps: float = 1e-8,
    qr_every_steps: int = 1,
    solve_options: Optional[Dict[str, Any]] = None,
    solver_kind: str = "ivp",
    auto_switch_rk4: bool = False,
    rk4_cost_ratio: float = 1.0,
    seed_frame: Optional[np.ndarray] = None,
) -> LyapunovResult:
    """
    Compute the full Lyapunov spectrum (n exponents) for an nD ODE.

    Parameters
    ----------
    rhs
        f(t, x) -> dx/dt, shape (n,). (May be a closure capturing parameters.)
    x0
        Initial condition, shape (n,).
    t0
        Start time.
    dt
        Base time increment that defines chunking/QR frequency.
        (solve_ivp still uses adaptive internal steps.)
    t_transient
        Time to discard (no accumulation).
    t_measure
        Time to accumulate ln|R_ii|.
    jac
        Optional analytic Jacobian J(t, x) -> shape (n,n).
        If None, finite differences are used.
    fd_eps
        FD epsilon (only if jac is None).
    qr_every_steps
        Perform QR every (qr_every_steps * dt) units of time.
    solve_options
        Passed to solve_ivp (e.g., {"method":"RK45","rtol":..., "atol":...}).
        Ignored when using RK4.
    solver_kind
        If "rk4", uses fixed-step RK4 with step size dt. Otherwise uses solve_ivp.
    auto_switch_rk4
        If True, switches from solve_ivp to RK4 when solve_ivp's function-eval
        cost exceeds the RK4 estimate for a chunk.
    rk4_cost_ratio
        Multiplicative threshold for the RK4 cost estimate (e.g., 1.2).
    seed_frame
        Optional initial perturbation frame Q0 (n,n). If None, identity is used.

    Returns
    -------
    LyapunovResult
    """
    x = np.asarray(x0, dtype=float).copy()
    n = x.size

    if n < 1:
        raise ValueError("x0 must have at least one component.")
    if dt <= 0:
        raise ValueError("dt must be > 0.")
    if t_transient < 0:
        raise ValueError("t_transient must be >= 0.")
    if t_measure <= 0:
        raise ValueError("t_measure must be > 0.")
    if qr_every_steps < 1:
        raise ValueError("qr_every_steps must be >= 1.")
    if rk4_cost_ratio <= 0:
        raise ValueError("rk4_cost_ratio must be > 0.")

    # Jacobian provider
    if jac is None:
        def jac_fn(tt: float, xx: np.ndarray) -> np.ndarray:
            return finite_difference_jacobian(rhs, tt, xx, eps=fd_eps)
    else:
        jac_fn = jac

    # Initial orthonormal frame (columns are perturbation vectors)
    if seed_frame is None:
        Q = np.eye(n, dtype=float)
    else:
        Q = np.asarray(seed_frame, dtype=float)
        if Q.shape != (n, n):
            raise ValueError("seed_frame must have shape (n, n).")
        Q, _ = np.linalg.qr(Q)

    # Augmented RHS: y_aug = [x, vec(Q)]
    def rhs_aug(tt: float, y_aug: np.ndarray) -> np.ndarray:
        xx, QQ = _unpack_augmented(y_aug, n)
        dx = rhs(tt, xx)
        J = jac_fn(tt, xx)
        dQ = J @ QQ
        return _pack_augmented(dx, dQ)

    chunk_dt = float(qr_every_steps) * float(dt)
    solver_kind_norm = str(solver_kind or "ivp").lower()
    use_rk4 = solver_kind_norm == "rk4"
    auto_switch = bool(auto_switch_rk4) and not use_rk4

    def _integrate_chunk(t_start: float, y_start: np.ndarray, t_end: float) -> np.ndarray:
        nonlocal use_rk4
        if use_rk4:
            return _integrate_chunk_rk4(rhs_aug, t_start, y_start, t_end, dt)
        if auto_switch:
            y_next, nfev = _integrate_chunk_ivp(
                rhs_aug,
                t_start,
                y_start,
                t_end,
                solve_options=solve_options,
                return_nfev=True,
            )
            rk4_cost = _rk4_cost_estimate(t_start, t_end, dt)
            if rk4_cost > 0 and nfev > rk4_cost_ratio * rk4_cost:
                use_rk4 = True
            return y_next
        return _integrate_chunk_ivp(rhs_aug, t_start, y_start, t_end, solve_options=solve_options)

    # -----------------------
    # 1) Transient phase
    # -----------------------
    t = float(t0)
    y_aug = _pack_augmented(x, Q)

    n_full = int(np.floor(t_transient / chunk_dt))
    rem = float(t_transient) - n_full * chunk_dt

    for _ in range(n_full):
        y_aug = _integrate_chunk(t, y_aug, t + chunk_dt)
        t += chunk_dt
        x, Q = _unpack_augmented(y_aug, n)
        Q, _ = np.linalg.qr(Q)  # keep conditioning
        y_aug = _pack_augmented(x, Q)

    if rem > 1e-15:
        y_aug = _integrate_chunk(t, y_aug, t + rem)
        t += rem
        x, Q = _unpack_augmented(y_aug, n)
        Q, _ = np.linalg.qr(Q)
        y_aug = _pack_augmented(x, Q)

    # -----------------------
    # 2) Measurement phase
    # -----------------------
    sums_log = np.zeros(n, dtype=float)
    n_qr = 0

    n_full = int(np.floor(t_measure / chunk_dt))
    rem = float(t_measure) - n_full * chunk_dt

    for _ in range(n_full):
        y_aug = _integrate_chunk(t, y_aug, t + chunk_dt)
        t += chunk_dt

        x, Q = _unpack_augmented(y_aug, n)
        Q, sums_log = _qr_accumulate(Q, sums_log)
        n_qr += 1
        y_aug = _pack_augmented(x, Q)

    if rem > 1e-15:
        y_aug = _integrate_chunk(t, y_aug, t + rem)
        t += rem

        x, Q = _unpack_augmented(y_aug, n)
        Q, sums_log = _qr_accumulate(Q, sums_log)
        n_qr += 1
        y_aug = _pack_augmented(x, Q)

    if n_qr == 0:
        raise RuntimeError("No QR steps performed. Increase t_measure or reduce qr_every_steps.")

    lambdas = sums_log / float(t_measure)

    return LyapunovResult(
        lambdas=lambdas,
        sums_log=sums_log,
        t_meas=float(t_measure),
        n_qr=n_qr,
        x_final=x.copy(),
    )
