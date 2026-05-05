"""Phase-portrait plots (2D/3D)."""
from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_phase_2d(
    y: np.ndarray,
    i: int,
    j: int,
    title: str,
    xlabel: str,
    ylabel: str,
    linewidth: float = 0.07,
):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    fig.set_dpi(150)
    ax.plot(y[i, :], y[j, :], linewidth=float(linewidth))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, linewidth=0.3)
    ax.set_aspect("equal", adjustable="box")
    return fig


def plot_phase_3d(
    y: np.ndarray,
    i: int,
    j: int,
    k: int,
    title: str,
    labels: Tuple[str, str, str],
    linewidth: float = 0.07,
):
    fig = plt.figure(figsize=(3.2, 3.2))
    fig.set_dpi(150)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(y[i, :], y[j, :], y[k, :], linewidth=float(linewidth))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(labels[0], fontsize=9)
    ax.set_ylabel(labels[1], fontsize=9)
    ax.set_zlabel(labels[2], fontsize=9)
    ax.tick_params(labelsize=8)
    return fig
