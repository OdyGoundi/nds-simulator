"""Bifurcation diagram plot."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from app.plotting.settings import (
    BIFURCATION_DEFAULTS,
    PlotSettings,
    apply_axis_settings,
)


def plot_bifurcation(
    *,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    x_history: Optional[np.ndarray] = None,
    y_history: Optional[np.ndarray] = None,
    boundaries: Sequence[float] = (),
    xlabel: str,
    ylabel: str,
    x_view: Tuple[float, float],
    y_view: Tuple[float, float],
    settings: Optional[PlotSettings] = None,
):
    """Scatter the recent batch over a faded reservoir history, plus boundary lines."""
    s = settings or BIFURCATION_DEFAULTS
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    fig.set_dpi(140)

    marker_size = max(1.0, float(s.linewidth) * 6.0)

    if x_history is not None and y_history is not None and x_history.size > 0:
        ax.scatter(
            x_history, y_history,
            s=marker_size, c=s.color, marker=".", linewidths=0, alpha=0.8,
        )
    ax.scatter(
        x_vals, y_vals,
        s=marker_size, c=s.color, marker=".", linewidths=0, alpha=0.8,
    )

    for x_sep in boundaries:
        ax.axvline(float(x_sep), color="magenta", linewidth=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(float(x_view[0]), float(x_view[1]))
    ax.set_ylim(float(y_view[0]), float(y_view[1]))
    apply_axis_settings(ax, s, has_z=False)
    return fig
