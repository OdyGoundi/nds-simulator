from __future__ import annotations

import math
from typing import Callable, Dict, Tuple

import numpy as np

from ._common import require_numba

_RK4_CACHE: Dict[int, Callable] = {}
_SYMPLECTIC_FR_CACHE: Dict[Tuple[int, int], Callable] = {}


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
        n_steps = int(math.floor((tf - t0) / dt)) + 1
        if n_steps < 1:
            n_steps = 1

        if max_steps > 0:
            n_store = max_steps
        else:
            n_store = n_steps
        if n_steps <= 1:
            n_store = 1
        else:
            if n_store < 2:
                n_store = 2
            if n_store > n_steps:
                n_store = n_steps

        stride = 1
        if n_store < n_steps:
            stride = int(math.ceil((n_steps - 1) / float(n_store - 1)))
            if stride < 1:
                stride = 1

        n = y0.size
        t_arr = np.empty(n_store, dtype=np.float64)
        y_arr = np.empty((n, n_store), dtype=np.float64)

        t = t0
        y = y0.copy()
        out_count = 0
        for i in range(n_steps):
            should_store = (
                i == 0
                or i == n_steps - 1
                or stride == 1
                or (i % stride == 0)
            )
            if should_store:
                if out_count < n_store:
                    t_arr[out_count] = t
                    for k in range(n):
                        y_arr[k, out_count] = y[k]
                    out_count += 1
                else:
                    t_arr[n_store - 1] = t
                    for k in range(n):
                        y_arr[k, n_store - 1] = y[k]

            if i == n_steps - 1:
                break

            k1 = rhs_nb(t, y, p)
            k2 = rhs_nb(t + 0.5 * dt, y + 0.5 * dt * k1, p)
            k3 = rhs_nb(t + 0.5 * dt, y + 0.5 * dt * k2, p)
            k4 = rhs_nb(t + dt, y + dt * k3, p)
            y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t = t + dt

        return t_arr[:out_count], y_arr[:, :out_count]

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
        n_steps = int(math.floor((tf - t0) / dt)) + 1
        if n_steps < 1:
            n_steps = 1

        if max_steps > 0:
            n_store = max_steps
        else:
            n_store = n_steps
        if n_steps <= 1:
            n_store = 1
        else:
            if n_store < 2:
                n_store = 2
            if n_store > n_steps:
                n_store = n_steps

        stride = 1
        if n_store < n_steps:
            stride = int(math.ceil((n_steps - 1) / float(n_store - 1)))
            if stride < 1:
                stride = 1

        n = y0.size
        n_q = n // 2
        t_arr = np.empty(n_store, dtype=np.float64)
        y_arr = np.empty((n, n_store), dtype=np.float64)

        q = y0[:n_q].copy()
        p = y0[n_q:].copy()
        t = t0
        out_count = 0

        w = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
        c1 = w
        c2 = 1.0 - 2.0 * w
        c3 = w

        for i in range(n_steps):
            should_store = (
                i == 0
                or i == n_steps - 1
                or stride == 1
                or (i % stride == 0)
            )
            if should_store:
                if out_count < n_store:
                    t_arr[out_count] = t
                    for k in range(n_q):
                        y_arr[k, out_count] = q[k]
                        y_arr[n_q + k, out_count] = p[k]
                    out_count += 1
                else:
                    t_arr[n_store - 1] = t
                    for k in range(n_q):
                        y_arr[k, n_store - 1] = q[k]
                        y_arr[n_q + k, n_store - 1] = p[k]
            if i == n_steps - 1:
                break

            t, q, p = _verlet_step(t, q, p, c1 * dt, params)
            t, q, p = _verlet_step(t, q, p, c2 * dt, params)
            t, q, p = _verlet_step(t, q, p, c3 * dt, params)

        return t_arr[:out_count], y_arr[:, :out_count]

    _SYMPLECTIC_FR_CACHE[key] = _integrate
    return _integrate
