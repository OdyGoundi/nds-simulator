from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from ._common import require_numba
from ._linalg import _matmul, _qr_orthonormalize

_LYAP_CACHE: Dict[Tuple[int, Optional[int], bool], Callable] = {}


def build_lyapunov_solver(rhs_nb: Callable, jac_nb: Optional[Callable], use_fd_jac: bool) -> Callable:
    nb = require_numba()
    key = (id(rhs_nb), id(jac_nb) if jac_nb is not None else None, bool(use_fd_jac))
    cached = _LYAP_CACHE.get(key)
    if cached is not None:
        return cached

    if use_fd_jac:

        @nb.njit(cache=True, fastmath=True)
        def _integrate_chunk_fd(x: np.ndarray, Q: np.ndarray, t0: float, t1: float,
                                dt: float, fd_eps: float, p: np.ndarray):
            duration = t1 - t0
            if duration <= 1e-15:
                return x, Q, t0
            n_full = int(math.floor(duration / dt))
            t = t0
            for _ in range(n_full):
                x, Q = _rk4_step_fd(x, Q, t, dt, fd_eps, p)
                t += dt
            rem = t1 - t
            if rem > 1e-15:
                x, Q = _rk4_step_fd(x, Q, t, rem, fd_eps, p)
                t += rem
            return x, Q, t

        @nb.njit(cache=True, fastmath=True)
        def _rk4_step_fd(x: np.ndarray, Q: np.ndarray, t: float, h: float, fd_eps: float, p: np.ndarray):
            k1_x, k1_Q = _rhs_aug_fd(t, x, Q, fd_eps, p)
            x2 = x + 0.5 * h * k1_x
            Q2 = Q + 0.5 * h * k1_Q
            k2_x, k2_Q = _rhs_aug_fd(t + 0.5 * h, x2, Q2, fd_eps, p)
            x3 = x + 0.5 * h * k2_x
            Q3 = Q + 0.5 * h * k2_Q
            k3_x, k3_Q = _rhs_aug_fd(t + 0.5 * h, x3, Q3, fd_eps, p)
            x4 = x + h * k3_x
            Q4 = Q + h * k3_Q
            k4_x, k4_Q = _rhs_aug_fd(t + h, x4, Q4, fd_eps, p)
            x_new = x + (h / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
            Q_new = Q + (h / 6.0) * (k1_Q + 2.0 * k2_Q + 2.0 * k3_Q + k4_Q)
            return x_new, Q_new

        @nb.njit(cache=True, fastmath=True)
        def _rhs_aug_fd(t: float, x: np.ndarray, Q: np.ndarray, fd_eps: float, p: np.ndarray):
            dx = rhs_nb(t, x, p)
            n = x.size
            J = np.empty((n, n), dtype=np.float64)
            for j in range(n):
                x_pert = x.copy()
                x_pert[j] = x_pert[j] + fd_eps
                fp = rhs_nb(t, x_pert, p)
                x_pert[j] = x[j] - fd_eps
                fm = rhs_nb(t, x_pert, p)
                for i in range(n):
                    J[i, j] = (fp[i] - fm[i]) / (2.0 * fd_eps)
            dQ = _matmul(J, Q)
            return dx, dQ

        @nb.njit(cache=True, fastmath=True)
        def _lyap(x0: np.ndarray, t0: float, dt: float, t_transient: float, t_measure: float,
                  qr_every_steps: int, fd_eps: float, p: np.ndarray):
            n = x0.size
            x = x0.copy()
            Q = np.eye(n, dtype=np.float64)
            dummy = np.zeros(n, dtype=np.float64)

            chunk_dt = float(qr_every_steps) * float(dt)
            t = float(t0)

            n_full = int(math.floor(t_transient / chunk_dt))
            rem = t_transient - n_full * chunk_dt

            for _ in range(n_full):
                x, Q, t = _integrate_chunk_fd(x, Q, t, t + chunk_dt, dt, fd_eps, p)
                _qr_orthonormalize(Q, dummy, False)

            if rem > 1e-15:
                x, Q, t = _integrate_chunk_fd(x, Q, t, t + rem, dt, fd_eps, p)
                _qr_orthonormalize(Q, dummy, False)

            sums_log = np.zeros(n, dtype=np.float64)
            n_qr = 0

            n_full = int(math.floor(t_measure / chunk_dt))
            rem = t_measure - n_full * chunk_dt

            for _ in range(n_full):
                x, Q, t = _integrate_chunk_fd(x, Q, t, t + chunk_dt, dt, fd_eps, p)
                _qr_orthonormalize(Q, sums_log, True)
                n_qr += 1

            if rem > 1e-15:
                x, Q, t = _integrate_chunk_fd(x, Q, t, t + rem, dt, fd_eps, p)
                _qr_orthonormalize(Q, sums_log, True)
                n_qr += 1

            lambdas = sums_log / t_measure
            return lambdas, sums_log, t_measure, n_qr, x

    else:
        if jac_nb is None:
            raise ValueError("jac_nb must be provided when use_fd_jac is False.")

        @nb.njit(cache=True, fastmath=True)
        def _integrate_chunk(x: np.ndarray, Q: np.ndarray, t0: float, t1: float,
                             dt: float, p: np.ndarray):
            duration = t1 - t0
            if duration <= 1e-15:
                return x, Q, t0
            n_full = int(math.floor(duration / dt))
            t = t0
            for _ in range(n_full):
                x, Q = _rk4_step(x, Q, t, dt, p)
                t += dt
            rem = t1 - t
            if rem > 1e-15:
                x, Q = _rk4_step(x, Q, t, rem, p)
                t += rem
            return x, Q, t

        @nb.njit(cache=True, fastmath=True)
        def _rk4_step(x: np.ndarray, Q: np.ndarray, t: float, h: float, p: np.ndarray):
            dx1 = rhs_nb(t, x, p)
            J1 = jac_nb(t, x, p)
            dQ1 = _matmul(J1, Q)

            x2 = x + 0.5 * h * dx1
            Q2 = Q + 0.5 * h * dQ1
            dx2 = rhs_nb(t + 0.5 * h, x2, p)
            J2 = jac_nb(t + 0.5 * h, x2, p)
            dQ2 = _matmul(J2, Q2)

            x3 = x + 0.5 * h * dx2
            Q3 = Q + 0.5 * h * dQ2
            dx3 = rhs_nb(t + 0.5 * h, x3, p)
            J3 = jac_nb(t + 0.5 * h, x3, p)
            dQ3 = _matmul(J3, Q3)

            x4 = x + h * dx3
            Q4 = Q + h * dQ3
            dx4 = rhs_nb(t + h, x4, p)
            J4 = jac_nb(t + h, x4, p)
            dQ4 = _matmul(J4, Q4)

            x_new = x + (h / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
            Q_new = Q + (h / 6.0) * (dQ1 + 2.0 * dQ2 + 2.0 * dQ3 + dQ4)
            return x_new, Q_new

        @nb.njit(cache=True, fastmath=True)
        def _lyap(x0: np.ndarray, t0: float, dt: float, t_transient: float, t_measure: float,
                  qr_every_steps: int, fd_eps: float, p: np.ndarray):
            n = x0.size
            x = x0.copy()
            Q = np.eye(n, dtype=np.float64)
            dummy = np.zeros(n, dtype=np.float64)

            chunk_dt = float(qr_every_steps) * float(dt)
            t = float(t0)

            n_full = int(math.floor(t_transient / chunk_dt))
            rem = t_transient - n_full * chunk_dt

            for _ in range(n_full):
                x, Q, t = _integrate_chunk(x, Q, t, t + chunk_dt, dt, p)
                _qr_orthonormalize(Q, dummy, False)

            if rem > 1e-15:
                x, Q, t = _integrate_chunk(x, Q, t, t + rem, dt, p)
                _qr_orthonormalize(Q, dummy, False)

            sums_log = np.zeros(n, dtype=np.float64)
            n_qr = 0

            n_full = int(math.floor(t_measure / chunk_dt))
            rem = t_measure - n_full * chunk_dt

            for _ in range(n_full):
                x, Q, t = _integrate_chunk(x, Q, t, t + chunk_dt, dt, p)
                _qr_orthonormalize(Q, sums_log, True)
                n_qr += 1

            if rem > 1e-15:
                x, Q, t = _integrate_chunk(x, Q, t, t + rem, dt, p)
                _qr_orthonormalize(Q, sums_log, True)
                n_qr += 1

            lambdas = sums_log / t_measure
            return lambdas, sums_log, t_measure, n_qr, x

    _LYAP_CACHE[key] = _lyap
    return _lyap
