"""Per-plot user-configurable settings + helper to apply them to a Matplotlib axis.

Stored in ``st.session_state`` as a single ``PlotSettings`` value per plot.
The companion ``app/plotting/dialog.py`` renders a popup to edit it.
"""
from __future__ import annotations

from dataclasses import dataclass

from matplotlib.ticker import MaxNLocator, ScalarFormatter


class _MantissaScalarFormatter(ScalarFormatter):
    """ScalarFormatter that lets the caller pick the mantissa decimals.

    When the order of magnitude is outside ``set_powerlimits`` (we set
    >= 100 to trigger), values are shown as ``M.mm`` and ``× 10^N`` is
    rendered as the axis offsetText (next to the axis label).
    """

    def __init__(self, decimals: int) -> None:
        super().__init__(useMathText=True, useOffset=False)
        self._decimals = max(0, int(decimals))
        self.set_powerlimits((-3, 2))

    def _set_format(self) -> None:  # type: ignore[override]
        self.format = f"%.{self._decimals}f"
        if self._useMathText:
            self.format = r"$\mathdefault{%s}$" % self.format


@dataclass(frozen=True)
class PlotSettings:
    color: str = "#1f77b4"
    linewidth: float = 1.0
    grid: bool = True
    tick_density: int = 6
    decimals: int = 2
    square_axis: bool = False


PHASE_2D_DEFAULTS = PlotSettings(
    color="#1f77b4", linewidth=0.07, grid=True, tick_density=6, decimals=2, square_axis=True
)
PHASE_3D_DEFAULTS = PlotSettings(
    color="#1f77b4", linewidth=0.07, grid=True, tick_density=6, decimals=2, square_axis=False
)
BIFURCATION_DEFAULTS = PlotSettings(
    color="#000000", linewidth=0.3, grid=True, tick_density=6, decimals=2, square_axis=False
)
LYAPUNOV_DEFAULTS = PlotSettings(
    color="#1f77b4", linewidth=1.1, grid=True, tick_density=6, decimals=2, square_axis=False
)


def apply_axis_settings(ax, settings: PlotSettings, *, has_z: bool = False) -> None:
    """Apply grid, tick density, decimal formatting (scientific for |v|>=100),
    and (2D-only) square aspect."""
    if settings.grid:
        ax.grid(True, linewidth=0.3)
    else:
        ax.grid(False)

    n_ticks = max(2, int(settings.tick_density))
    decimals = int(settings.decimals)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
    # Each axis needs its own formatter instance — ScalarFormatter holds
    # per-axis state (data scale, current offset).
    ax.xaxis.set_major_formatter(_MantissaScalarFormatter(decimals))
    ax.yaxis.set_major_formatter(_MantissaScalarFormatter(decimals))

    if has_z and hasattr(ax, "zaxis"):
        ax.zaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
        ax.zaxis.set_major_formatter(_MantissaScalarFormatter(decimals))

    if settings.square_axis and not has_z:
        ax.set_aspect("equal", adjustable="box")
