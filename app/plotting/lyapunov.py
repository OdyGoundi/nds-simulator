"""Lyapunov-spectrum sweep plot."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from app.plotting.settings import (
    LYAPUNOV_DEFAULTS,
    PlotSettings,
    apply_axis_settings,
)
from app.plotting.style import LINE_COLORS


def plot_lyapunov_sweep(
    *,
    param_vals: np.ndarray,
    lambdas: np.ndarray,
    boundaries: Sequence[float] = (),
    xlabel: str,
    x_view: Tuple[float, float],
    y_view: Tuple[float, float],
    settings: Optional[PlotSettings] = None,
):
    """Plot Lyapunov exponents (one line per exponent) over the swept parameter.

    The ``settings.color`` field is intentionally ignored — this is a multi-line
    plot that uses the shared ``LINE_COLORS`` palette to disambiguate exponents.
    """
    s = settings or LYAPUNOV_DEFAULTS
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    fig.set_dpi(140)

    lambdas = np.asarray(lambdas, dtype=float)
    if lambdas.ndim == 1:
        lambdas = lambdas[:, None]
    n_exps = lambdas.shape[1]

    for k in range(n_exps):
        ax.plot(
            param_vals,
            lambdas[:, k],
            color=LINE_COLORS[k % len(LINE_COLORS)],
            linestyle="-",
            linewidth=float(s.linewidth),
            label=f"lambda{k}",
        )

    for x_sep in boundaries:
        ax.axvline(float(x_sep), color="magenta", linewidth=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Lyapunov exponents")
    ax.set_xlim(float(x_view[0]), float(x_view[1]))
    ax.set_ylim(float(y_view[0]), float(y_view[1]))
    apply_axis_settings(ax, s, has_z=False)
    ax.legend(loc="best", fontsize=8)
    return fig
