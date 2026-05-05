"""Bifurcation diagram plot."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


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
):
    """Scatter the recent batch over a faded reservoir history, plus boundary lines."""
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    fig.set_dpi(140)

    if x_history is not None and y_history is not None and x_history.size > 0:
        ax.scatter(
            x_history, y_history,
            s=2, c="black", marker=".", linewidths=0, alpha=0.8,
        )
    ax.scatter(
        x_vals, y_vals,
        s=2, c="black", marker=".", linewidths=0, alpha=0.8,
    )

    for x_sep in boundaries:
        ax.axvline(float(x_sep), color="magenta", linewidth=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(float(x_view[0]), float(x_view[1]))
    ax.set_ylim(float(y_view[0]), float(y_view[1]))
    ax.grid(True, linewidth=0.3)
    return fig
