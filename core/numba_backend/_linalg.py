from __future__ import annotations

import math
from typing import Callable, TYPE_CHECKING

import numpy as np

from ._common import nb

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
