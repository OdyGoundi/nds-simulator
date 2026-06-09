from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

from ._common import require_numba

_BUILTIN_CACHE: Dict[str, Tuple[Callable, Callable, Tuple[str, ...]]] = {}
_SYMPLECTIC_CACHE: Dict[str, Tuple[Callable, Callable, Tuple[str, ...]]] = {}


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
