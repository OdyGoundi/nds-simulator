"""Axis-bound helpers shared by phase, sweep, and Lyapunov plots."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def axis_bounds(values: np.ndarray) -> Tuple[float, float]:
    """Tight bounds around finite values, with a small pad if all values are equal."""
    arr = np.asarray(values, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        delta = max(1e-6, 0.05 * max(1.0, abs(vmin)))
        return vmin - delta, vmax + delta
    return vmin, vmax


def square_xy_bounds(
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Expand bounds to a common square span centered on each axis midpoint."""
    x0, x1 = float(x_bounds[0]), float(x_bounds[1])
    y0, y1 = float(y_bounds[0]), float(y_bounds[1])
    dx = max(1e-12, x1 - x0)
    dy = max(1e-12, y1 - y0)
    half_span = 0.5 * max(dx, dy)
    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)
    return (x_mid - half_span, x_mid + half_span), (y_mid - half_span, y_mid + half_span)
