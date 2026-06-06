from __future__ import annotations

import math
from typing import Callable, Dict

import numpy as np

from ._common import require_numba

_SWEEP_CACHE: Dict[int, Callable] = {}


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
        descending: bool,
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
            idx_eff = (n_vals - 1 - idx) if descending else idx
            pv = sweep_start + sweep_step * idx_eff
            if not descending and pv > sweep_stop + 1e-12:
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
