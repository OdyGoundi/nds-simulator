from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.poincare_sweep import PoincareConfig, poincare_section


DIRECTION_LABEL_BY_VALUE = {
    +1: "Upward (+)",
    0: "Both (±)",
    -1: "Downward (-)",
}
DIRECTION_VALUE_BY_LABEL = {label: value for value, label in DIRECTION_LABEL_BY_VALUE.items()}


@dataclass(frozen=True)
class PoincareMapConfig:
    section_index: int
    section_value: float = 0.0
    direction: int = +1
    axis_pair: tuple[int, int] | None = None
    max_points: int = 5000


@dataclass(frozen=True)
class PoincareMapResult:
    section_index: int
    section_value: float
    direction: int
    axis_pair: tuple[int, int]
    hit_count: int
    display_count: int
    x_values: np.ndarray
    y_values: np.ndarray


def remaining_axes(n_dim: int, section_index: int) -> tuple[int, ...]:
    n_dim_i = int(n_dim)
    section_i = int(section_index)
    if n_dim_i < 1:
        raise ValueError("n_dim must be positive.")
    if section_i < 0 or section_i >= n_dim_i:
        raise ValueError("section_index out of range.")
    return tuple(i for i in range(n_dim_i) if i != section_i)


def default_axis_pair(
    n_dim: int,
    section_index: int,
    preferred_axes: Sequence[int] = (),
) -> tuple[int, int]:
    axes = list(remaining_axes(n_dim, section_index))
    if len(axes) < 2:
        raise ValueError("A Poincaré map needs at least two axes outside the section plane.")

    chosen: list[int] = []
    for axis in preferred_axes:
        axis_i = int(axis)
        if axis_i in axes and axis_i not in chosen:
            chosen.append(axis_i)
        if len(chosen) == 2:
            return chosen[0], chosen[1]

    for axis_i in axes:
        if axis_i not in chosen:
            chosen.append(axis_i)
        if len(chosen) == 2:
            return chosen[0], chosen[1]

    raise ValueError("Unable to choose a valid axis pair for the Poincaré map.")


def axis_pair_options(
    n_dim: int,
    section_index: int,
    preferred_axes: Sequence[int] = (),
) -> tuple[tuple[int, int], ...]:
    axes = remaining_axes(n_dim, section_index)
    if len(axes) < 2:
        return ()

    default_pair = default_axis_pair(n_dim, section_index, preferred_axes=preferred_axes)
    ordered_pairs: list[tuple[int, int]] = [default_pair]
    for axis_x in axes:
        for axis_y in axes:
            if axis_x == axis_y:
                continue
            pair = (int(axis_x), int(axis_y))
            if pair not in ordered_pairs:
                ordered_pairs.append(pair)
    return tuple(ordered_pairs)


def compute_poincare_map(
    t: np.ndarray,
    y: np.ndarray,
    cfg: PoincareMapConfig,
    *,
    preferred_axes: Sequence[int] = (),
) -> PoincareMapResult:
    t_arr = np.asarray(t, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim != 2:
        raise ValueError("y must have shape (n_vars, n_steps).")

    n_dim = int(y_arr.shape[0])
    axis_options = axis_pair_options(n_dim, int(cfg.section_index), preferred_axes=preferred_axes)
    if not axis_options:
        raise ValueError("Poincaré map rendering requires at least 3 state variables.")

    axis_pair = cfg.axis_pair or axis_options[0]
    axis_x, axis_y = int(axis_pair[0]), int(axis_pair[1])
    if axis_x == axis_y:
        raise ValueError("axis_pair must contain two different axes.")
    if axis_pair not in axis_options:
        raise ValueError("axis_pair is not valid for the selected section plane.")

    poincare_cfg = PoincareConfig(
        section_index=int(cfg.section_index),
        section_value=float(cfg.section_value),
        direction=int(cfg.direction),
        method="crossing",
        tol=1e-6,
        transient_steps=0,
    )
    t_hits, y_hits = poincare_section(t_arr, y_arr, poincare_cfg, params=None)

    hit_count = int(t_hits.size)
    if hit_count == 0:
        return PoincareMapResult(
            section_index=int(cfg.section_index),
            section_value=float(cfg.section_value),
            direction=int(cfg.direction),
            axis_pair=(axis_x, axis_y),
            hit_count=0,
            display_count=0,
            x_values=np.array([], dtype=float),
            y_values=np.array([], dtype=float),
        )

    max_points = max(1, int(cfg.max_points))
    if hit_count > max_points:
        keep_idx = np.linspace(0, hit_count - 1, max_points, dtype=int)
    else:
        keep_idx = np.arange(hit_count, dtype=int)

    return PoincareMapResult(
        section_index=int(cfg.section_index),
        section_value=float(cfg.section_value),
        direction=int(cfg.direction),
        axis_pair=(axis_x, axis_y),
        hit_count=hit_count,
        display_count=int(keep_idx.size),
        x_values=np.asarray(y_hits[axis_x, keep_idx], dtype=float),
        y_values=np.asarray(y_hits[axis_y, keep_idx], dtype=float),
    )
