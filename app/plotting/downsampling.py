from typing import Tuple

import numpy as np


def decimate_indices(n_points: int, max_points: int) -> np.ndarray:
    n = int(max(0, n_points))
    if n <= 0:
        return np.array([], dtype=np.int64)
    max_pts = int(max(2, max_points))
    if n <= max_pts:
        return np.arange(n, dtype=np.int64)
    idx = np.linspace(0, n - 1, num=max_pts, dtype=np.int64)
    idx[-1] = n - 1
    return np.unique(idx)


def downsample_trajectory(t: np.ndarray, y: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    t_arr = np.asarray(t, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim != 2:
        raise ValueError("y must be shape (n_vars, n_steps)")
    n = min(int(t_arr.size), int(y_arr.shape[1]))
    if n <= 0:
        return np.array([], dtype=float), np.zeros((int(y_arr.shape[0]), 0), dtype=float)
    t_use = t_arr[:n]
    y_use = y_arr[:, :n]
    idx = decimate_indices(n, max_points)
    return t_use[idx], y_use[:, idx]


def downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    n = min(int(x_arr.size), int(y_arr.size))
    if n <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    x_use = x_arr[:n]
    y_use = y_arr[:n]
    idx = decimate_indices(n, max_points)
    return x_use[idx], y_use[idx]
