from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

try:
    import numba as nb  # type: ignore
except Exception:  # pragma: no cover
    nb = None


def numba_available() -> bool:
    return nb is not None


def require_numba():
    if nb is None:  # pragma: no cover
        raise RuntimeError("Numba is required for the Numba backend.")
    return nb


_BUILTIN_CACHE: Dict[str, Tuple[Callable, Callable, Tuple[str, ...]]] = {}
_SYMPLECTIC_CACHE: Dict[str, Tuple[Callable, Callable, Tuple[str, ...]]] = {}
_RK4_CACHE: Dict[int, Callable] = {}
_SYMPLECTIC_FR_CACHE: Dict[Tuple[int, int], Callable] = {}
_SWEEP_CACHE: Dict[int, Callable] = {}
_LYAP_CACHE: Dict[Tuple[int, Optional[int], bool], Callable] = {}

if TYPE_CHECKING:
    _matmul: Callable[[np.ndarray, np.ndarray], np.ndarray]
    _qr_orthonormalize: Callable[[np.ndarray, np.ndarray, bool], None]


if nb is not None:

    @nb.njit(cache=True, fastmath=True)
    def _matmul_nb(J: np.ndarray, Q: np.ndarray) -> np.ndarray:
        n = J.shape[0]
        out = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += J[i, k] * Q[k, j]
                out[i, j] = s
        return out

    @nb.njit(cache=True, fastmath=True)
    def _qr_orthonormalize_nb(Q: np.ndarray, sums_log: np.ndarray, update_sums: bool) -> None:
        n = Q.shape[0]
        for j in range(n):
            for i in range(j):
                r = 0.0
                for k in range(n):
                    r += Q[k, i] * Q[k, j]
                for k in range(n):
                    Q[k, j] -= r * Q[k, i]

            norm = 0.0
            for k in range(n):
                norm += Q[k, j] * Q[k, j]
            norm = math.sqrt(norm)
            if norm < 1e-300:
                norm = 1e-300
            if update_sums:
                sums_log[j] += math.log(norm)
            inv = 1.0 / norm
            for k in range(n):
                Q[k, j] *= inv

    _matmul = _matmul_nb
    _qr_orthonormalize = _qr_orthonormalize_nb

else:  # pragma: no cover

    def _matmul_fallback(J: np.ndarray, Q: np.ndarray) -> np.ndarray:
        raise RuntimeError("Numba is required for the Numba backend.")

    def _qr_orthonormalize_fallback(Q: np.ndarray, sums_log: np.ndarray, update_sums: bool) -> None:
        raise RuntimeError("Numba is required for the Numba backend.")

    _matmul = _matmul_fallback
    _qr_orthonormalize = _qr_orthonormalize_fallback


def build_builtin_system(system_key: str) -> Tuple[Callable, Callable, Tuple[str, ...]]:
    nb = require_numba()
    key = str(system_key).lower()
    cached = _BUILTIN_CACHE.get(key)
    if cached is not None:
        return cached

    if key == "lorenz":
        param_names = ("sigma", "rho", "beta")

        def rhs(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
            sigma = p[0]
            rho = p[1]
            beta = p[2]
            x = y[0]
            yv = y[1]
            z = y[2]
            out = np.empty(3, dtype=np.float64)
            out[0] = sigma * (yv - x)
            out[1] = x * (rho - z) - yv
            out[2] = x * yv - beta * z
            return out

        def jac(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
            sigma = p[0]
            rho = p[1]
            beta = p[2]
            x = y[0]
            yv = y[1]
            z = y[2]
            out = np.empty((3, 3), dtype=np.float64)
            out[0, 0] = -sigma
            out[0, 1] = sigma
            out[0, 2] = 0.0
            out[1, 0] = rho - z
            out[1, 1] = -1.0
            out[1, 2] = -x
            out[2, 0] = yv
            out[2, 1] = x
            out[2, 2] = -beta
            return out

    elif key == "rossler":
        param_names = ("a", "b", "c")

        def rhs(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
            a = p[0]
            b = p[1]
            c = p[2]
            x = y[0]
            yv = y[1]
            z = y[2]
            out = np.empty(3, dtype=np.float64)
            out[0] = -yv - z
            out[1] = x + a * yv
            out[2] = b + z * (x - c)
            return out

        def jac(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
            a = p[0]
            c = p[2]
            x = y[0]
            z = y[2]
            out = np.empty((3, 3), dtype=np.float64)
            out[0, 0] = 0.0
            out[0, 1] = -1.0
            out[0, 2] = -1.0
            out[1, 0] = 1.0
            out[1, 1] = a
            out[1, 2] = 0.0
            out[2, 0] = z
            out[2, 1] = 0.0
            out[2, 2] = x - c
            return out

    elif key == "henon_heiles":
        param_names = ("lambda",)

        def rhs(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
            lam = p[0]
            q1 = y[0]
            q2 = y[1]
            p1 = y[2]
            p2 = y[3]
            out = np.empty(4, dtype=np.float64)
            out[0] = p1
            out[1] = p2
            out[2] = -(q1 + 2.0 * lam * q1 * q2)
            out[3] = -(q2 + lam * (q1 * q1 - q2 * q2))
            return out

        def jac(t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
            lam = p[0]
            q1 = y[0]
            q2 = y[1]
            out = np.empty((4, 4), dtype=np.float64)
            out[0, 0] = 0.0
            out[0, 1] = 0.0
            out[0, 2] = 1.0
            out[0, 3] = 0.0
            out[1, 0] = 0.0
            out[1, 1] = 0.0
            out[1, 2] = 0.0
            out[1, 3] = 1.0
            out[2, 0] = -(1.0 + 2.0 * lam * q2)
            out[2, 1] = -(2.0 * lam * q1)
            out[2, 2] = 0.0
            out[2, 3] = 0.0
            out[3, 0] = -(2.0 * lam * q1)
            out[3, 1] = -(1.0 - 2.0 * lam * q2)
            out[3, 2] = 0.0
            out[3, 3] = 0.0
            return out

    else:
        raise ValueError(f"Unsupported system_key for Numba backend: {system_key}")

    rhs_nb = nb.njit(cache=True, fastmath=True)(rhs)
    jac_nb = nb.njit(cache=True, fastmath=True)(jac)
    _BUILTIN_CACHE[key] = (rhs_nb, jac_nb, param_names)
    return rhs_nb, jac_nb, param_names


def build_builtin_symplectic(system_key: str) -> Tuple[Callable, Callable, Tuple[str, ...]]:
    nb = require_numba()
    key = str(system_key).lower()
    cached = _SYMPLECTIC_CACHE.get(key)
    if cached is not None:
        return cached

    if key == "henon_heiles":
        param_names = ("lambda",)

        def dq_dt(t: float, p: np.ndarray, params: np.ndarray) -> np.ndarray:
            out = np.empty(2, dtype=np.float64)
            out[0] = p[0]
            out[1] = p[1]
            return out

        def dp_dt(t: float, q: np.ndarray, params: np.ndarray) -> np.ndarray:
            lam = params[0]
            q1 = q[0]
            q2 = q[1]
            out = np.empty(2, dtype=np.float64)
            out[0] = -(q1 + 2.0 * lam * q1 * q2)
            out[1] = -(q2 + lam * (q1 * q1 - q2 * q2))
            return out
    else:
        raise ValueError(f"Unsupported system_key for symplectic Numba backend: {system_key}")

    dq_dt_nb = nb.njit(cache=True, fastmath=True)(dq_dt)
    dp_dt_nb = nb.njit(cache=True, fastmath=True)(dp_dt)
    _SYMPLECTIC_CACHE[key] = (dq_dt_nb, dp_dt_nb, param_names)
    return dq_dt_nb, dp_dt_nb, param_names


def build_rk4_integrator(rhs_nb: Callable) -> Callable:
    nb = require_numba()
    key = id(rhs_nb)
    cached = _RK4_CACHE.get(key)
    if cached is not None:
        return cached

    @nb.njit(cache=True, fastmath=True)
    def _integrate(y0: np.ndarray, t0: float, tf: float, dt: float, max_steps: int, p: np.ndarray):
        if dt <= 0.0:
            return np.zeros(0, dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
        if max_steps > 0:
            n_steps = max_steps
        else:
            n_steps = int(math.floor((tf - t0) / dt)) + 1
            if n_steps < 1:
                n_steps = 1
        n = y0.size
        t_arr = np.empty(n_steps, dtype=np.float64)
        y_arr = np.empty((n, n_steps), dtype=np.float64)

        t = t0
        y = y0.copy()
        for i in range(n_steps):
            t_arr[i] = t
            for k in range(n):
                y_arr[k, i] = y[k]
            if i == n_steps - 1:
                break

            k1 = rhs_nb(t, y, p)
            k2 = rhs_nb(t + 0.5 * dt, y + 0.5 * dt * k1, p)
            k3 = rhs_nb(t + 0.5 * dt, y + 0.5 * dt * k2, p)
            k4 = rhs_nb(t + dt, y + dt * k3, p)
            y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t = t + dt

        return t_arr, y_arr

    _RK4_CACHE[key] = _integrate
    return _integrate


def build_symplectic_fr_integrator(dq_dt_nb: Callable, dp_dt_nb: Callable) -> Callable:
    nb = require_numba()
    key = (id(dq_dt_nb), id(dp_dt_nb))
    cached = _SYMPLECTIC_FR_CACHE.get(key)
    if cached is not None:
        return cached

    @nb.njit(cache=True, fastmath=True)
    def _verlet_step(t: float, q: np.ndarray, p: np.ndarray, h: float, params: np.ndarray):
        dp1 = dp_dt_nb(t, q, params)
        p_half = p + 0.5 * h * dp1
        dq1 = dq_dt_nb(t, p_half, params)
        q_new = q + h * dq1
        dp2 = dp_dt_nb(t + h, q_new, params)
        p_new = p_half + 0.5 * h * dp2
        return t + h, q_new, p_new

    @nb.njit(cache=True, fastmath=True)
    def _integrate(y0: np.ndarray, t0: float, tf: float, dt: float, max_steps: int, params: np.ndarray):
        if dt <= 0.0:
            return np.zeros(0, dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
        if max_steps > 0:
            n_steps = max_steps
        else:
            n_steps = int(math.floor((tf - t0) / dt)) + 1
            if n_steps < 1:
                n_steps = 1

        n = y0.size
        n_q = n // 2
        t_arr = np.empty(n_steps, dtype=np.float64)
        y_arr = np.empty((n, n_steps), dtype=np.float64)

        q = y0[:n_q].copy()
        p = y0[n_q:].copy()
        t = t0

        w = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
        c1 = w
        c2 = 1.0 - 2.0 * w
        c3 = w

        for i in range(n_steps):
            t_arr[i] = t
            for k in range(n_q):
                y_arr[k, i] = q[k]
                y_arr[n_q + k, i] = p[k]
            if i == n_steps - 1:
                break

            t, q, p = _verlet_step(t, q, p, c1 * dt, params)
            t, q, p = _verlet_step(t, q, p, c2 * dt, params)
            t, q, p = _verlet_step(t, q, p, c3 * dt, params)

        return t_arr, y_arr

    _SYMPLECTIC_FR_CACHE[key] = _integrate
    return _integrate


def build_poincare_sweep_rk4(rhs_nb: Callable) -> Callable:
    nb = require_numba()
    key = id(rhs_nb)
    cached = _SWEEP_CACHE.get(key)
    if cached is not None:
        return cached

    @nb.njit(cache=True, fastmath=True)
    def _sweep(
        y0: np.ndarray,
        t0: float,
        tf: float,
        dt: float,
        base_params: np.ndarray,
        sweep_param_index: int,
        sweep_start: float,
        sweep_stop: float,
        sweep_step: float,
        section_index: int,
        section_value: float,
        direction: int,
        method_id: int,
        tol: float,
        transient_steps: int,
        output_index: int,
        warm_start: bool,
        max_hits: int,
    ):
        if dt <= 0.0 or sweep_step <= 0.0:
            return (
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                0,
            )

        n_steps = int(math.floor((tf - t0) / dt)) + 1
        if n_steps < 2:
            return (
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                0,
            )

        n_vals = int(math.floor((sweep_stop - sweep_start) / sweep_step + 1e-12)) + 1
        if n_vals < 1:
            return (
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                0,
            )

        max_hits_local = max_hits if max_hits > 0 else 1
        max_total = n_vals * max_hits_local
        param_out = np.empty(max_total, dtype=np.float64)
        t_out = np.empty(max_total, dtype=np.float64)
        y_out = np.empty(max_total, dtype=np.float64)
        out_count = 0

        y_init = y0.copy()
        for idx in range(n_vals):
            pv = sweep_start + sweep_step * idx
            if pv > sweep_stop + 1e-12:
                break

            params = base_params.copy()
            params[sweep_param_index] = pv

            y = y_init.copy()
            t = t0
            prev_y = y.copy()
            prev_t = t
            prev_ds = prev_y[section_index] - section_value
            hits = 0

            for step in range(1, n_steps):
                k1 = rhs_nb(t, y, params)
                k2 = rhs_nb(t + 0.5 * dt, y + 0.5 * dt * k1, params)
                k3 = rhs_nb(t + 0.5 * dt, y + 0.5 * dt * k2, params)
                k4 = rhs_nb(t + dt, y + dt * k3, params)
                y_next = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                t_next = t + dt

                if step > transient_steps:
                    curr_ds = y_next[section_index] - section_value
                    if method_id == 0:
                        # crossing
                        cond = False
                        if direction == 1:
                            cond = (prev_ds < 0.0) and (curr_ds >= 0.0)
                        elif direction == -1:
                            cond = (prev_ds > 0.0) and (curr_ds <= 0.0)
                        else:
                            cond = (prev_ds == 0.0) or (curr_ds == 0.0) or (prev_ds * curr_ds < 0.0)

                        if cond and hits < max_hits_local:
                            denom = curr_ds - prev_ds
                            if denom == 0.0:
                                alpha = 1.0
                            else:
                                alpha = (0.0 - prev_ds) / denom
                            if alpha < 0.0:
                                alpha = 0.0
                            elif alpha > 1.0:
                                alpha = 1.0
                            th = prev_t + alpha * (t_next - prev_t)
                            yh = prev_y[output_index] + alpha * (
                                y_next[output_index] - prev_y[output_index]
                            )
                            param_out[out_count] = pv
                            t_out[out_count] = th
                            y_out[out_count] = yh
                            out_count += 1
                            hits += 1
                    else:
                        # slab
                        if math.fabs(curr_ds) <= tol:
                            cond = True
                            if direction != 0:
                                deriv = (curr_ds - prev_ds) / (t_next - prev_t)
                                if direction == 1:
                                    cond = deriv > 0.0
                                else:
                                    cond = deriv < 0.0
                            if cond and hits < max_hits_local:
                                param_out[out_count] = pv
                                t_out[out_count] = t_next
                                y_out[out_count] = y_next[output_index]
                                out_count += 1
                                hits += 1
                    prev_ds = curr_ds
                else:
                    prev_ds = y_next[section_index] - section_value

                prev_y = y_next
                prev_t = t_next
                y = y_next
                t = t_next

            if warm_start:
                y_init = y.copy()

        return param_out, t_out, y_out, out_count

    _SWEEP_CACHE[key] = _sweep
    return _sweep


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
