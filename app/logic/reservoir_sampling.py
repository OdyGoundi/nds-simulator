from __future__ import annotations

from typing import Any

import numpy as np


def make_xy_reservoir(*, capacity: int, seed: int | None = None) -> dict[str, Any]:
    cap = max(0, int(capacity))
    return {
        "capacity": cap,
        "seen": 0,
        "x": np.empty(0, dtype=float),
        "y": np.empty(0, dtype=float),
        "keys": np.empty(0, dtype=float),
        "rng": np.random.default_rng(seed),
    }


def ensure_xy_reservoir(
    state: dict[str, Any] | None,
    *,
    capacity: int,
    seed: int | None = None,
) -> dict[str, Any]:
    cap = max(0, int(capacity))
    if not isinstance(state, dict):
        return make_xy_reservoir(capacity=cap, seed=seed)
    if int(state.get("capacity", -1)) != cap:
        return make_xy_reservoir(capacity=cap, seed=seed)

    rng = state.get("rng")
    if not isinstance(rng, np.random.Generator):
        state["rng"] = np.random.default_rng(seed)
    return state


def get_xy_reservoir_points(state: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(state, dict):
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    x = np.asarray(state.get("x", np.empty(0, dtype=float)), dtype=float).ravel()
    y = np.asarray(state.get("y", np.empty(0, dtype=float)), dtype=float).ravel()
    if x.size != y.size:
        n = min(int(x.size), int(y.size))
        x = x[:n]
        y = y[:n]
    return x, y


def update_xy_reservoir(
    state: dict[str, Any],
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    batch_size: int = 100_000,
) -> dict[str, Any]:
    x_new = np.asarray(x_values, dtype=float).ravel()
    y_new = np.asarray(y_values, dtype=float).ravel()
    n_new = min(int(x_new.size), int(y_new.size))
    if n_new <= 0:
        return state
    x_new = x_new[:n_new]
    y_new = y_new[:n_new]

    cap = max(0, int(state.get("capacity", 0)))
    seen = int(state.get("seen", 0))
    seen += n_new
    state["seen"] = seen
    if cap <= 0:
        state["x"] = np.empty(0, dtype=float)
        state["y"] = np.empty(0, dtype=float)
        state["keys"] = np.empty(0, dtype=float)
        return state

    x_res = np.asarray(state.get("x", np.empty(0, dtype=float)), dtype=float).ravel()
    y_res = np.asarray(state.get("y", np.empty(0, dtype=float)), dtype=float).ravel()
    keys_res = np.asarray(state.get("keys", np.empty(0, dtype=float)), dtype=float).ravel()
    n_old = min(int(x_res.size), int(y_res.size), int(keys_res.size))
    x_res = x_res[:n_old]
    y_res = y_res[:n_old]
    keys_res = keys_res[:n_old]

    rng = state.get("rng")
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng()
        state["rng"] = rng

    step = max(1, int(batch_size))
    for i0 in range(0, n_new, step):
        i1 = min(i0 + step, n_new)
        x_chunk = x_new[i0:i1]
        y_chunk = y_new[i0:i1]
        k_chunk = rng.random(i1 - i0)

        if x_res.size == 0:
            x_all = x_chunk
            y_all = y_chunk
            k_all = k_chunk
        else:
            x_all = np.concatenate((x_res, x_chunk), axis=0)
            y_all = np.concatenate((y_res, y_chunk), axis=0)
            k_all = np.concatenate((keys_res, k_chunk), axis=0)

        if x_all.size > cap:
            cut = int(x_all.size - cap)
            keep_idx = np.argpartition(k_all, cut)[cut:]
            x_res = x_all[keep_idx]
            y_res = y_all[keep_idx]
            keys_res = k_all[keep_idx]
        else:
            x_res = x_all
            y_res = y_all
            keys_res = k_all

    state["x"] = x_res
    state["y"] = y_res
    state["keys"] = keys_res
    return state
