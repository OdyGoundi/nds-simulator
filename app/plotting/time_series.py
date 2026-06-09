"""Time-series plots (multi-variable overlay and per-variable single)."""
from __future__ import annotations

from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_time_series(
    t: np.ndarray,
    y: np.ndarray,
    indices: Sequence[int],
    var_names: Sequence[str],
    title: str,
):
    """Overlay several variables vs time on a single axes."""
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    fig.set_dpi(140)
    for i in indices:
        ax.plot(t, y[i, :], linewidth=0.9, label=var_names[i])
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.grid(True, linewidth=0.3)
    ax.legend(loc="best")
    return fig


def plot_single_variable(
    t: np.ndarray,
    y_var: np.ndarray,
    var_name: str,
    title: str,
    color: Optional[str] = None,
):
    """One variable vs time, in its own figure."""
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    fig.set_dpi(140)
    ax.plot(t, y_var, linewidth=0.9, label=var_name, color=color)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("t", fontsize=9)
    ax.set_ylabel(var_name, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, linewidth=0.3)
    ax.legend(loc="best")
    return fig
