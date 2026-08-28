"""Scatter plots for the communication/sensing association statistic."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ..metrics import association_ratio
from ..simulator import LargeScaleSimulationResult
from ._common import finish_figure


def _paired_finite(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("paired scatter inputs must have the same shape")
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _should_use_log_scale(
    values: np.ndarray,
    *,
    bulk_quantiles: tuple[float, float] = (0.10, 0.90),
    max_bulk_fraction: float = 0.25,
    min_dynamic_ratio: float = 100.0,
    min_samples: int = 10,
) -> bool:
    """Return whether a log *display* scale improves readability.

    The decision is deliberately independent of the association computation.
    Association metrics are always evaluated on the original linear values.

    A log axis is selected only when all displayed values are strictly
    positive, the total multiplicative range is large, and the central 80% of
    the samples occupies only a small fraction of the full linear range.  This
    identifies the intended case where a few outliers stretch a linear axis and
    visually compress the bulk of the scatter cloud.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < min_samples or np.any(values <= 0.0):
        return False

    v_min = float(np.min(values))
    v_max = float(np.max(values))
    if not np.isfinite(v_min) or not np.isfinite(v_max) or v_max <= v_min:
        return False

    q_low, q_high = np.quantile(values, bulk_quantiles)
    full_span = v_max - v_min
    bulk_span = float(q_high - q_low)
    bulk_fraction = bulk_span / full_span
    dynamic_ratio = v_max / v_min

    return bool(
        bulk_fraction <= max_bulk_fraction
        and dynamic_ratio >= min_dynamic_ratio
    )


def _resolve_display_scale(values: np.ndarray, requested: str) -> str:
    requested = str(requested).strip().lower()
    if requested not in {"auto", "linear", "log"}:
        raise ValueError("display scale must be 'auto', 'linear', or 'log'")
    if requested == "auto":
        return "log" if _should_use_log_scale(values) else "linear"
    if requested == "log" and np.any(np.asarray(values, dtype=float) <= 0.0):
        raise ValueError("log display scale requires strictly positive values")
    return requested


def plot_corr_scatter(
    x,
    y,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    s: float = 60,
    ax=None,
    show: bool = False,
    grid_alpha: float = 0.3,
    xscale: str = "auto",
    yscale: str = "auto",
    save_path: str | Path | None = None,
):
    """Scatter paired samples and show the association ratio.

    IMPORTANT: the association ratio is always computed from the
    original finite *linear* values.  ``xscale`` and ``yscale`` only control
    how those same samples are displayed.  In ``"auto"`` mode, a logarithmic
    display is selected only when outliers materially compress the bulk of the
    data on a linear axis.

    No per-point labels and no Pearson coefficient are displayed.
    """
    # These are the raw finite linear values used by the association metric.
    x_linear, y_linear = _paired_finite(x, y)
    if len(x_linear) == 0:
        raise ValueError("no finite paired samples are available for this scatter plot")

    association = association_ratio(x_linear, y_linear)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # Plot the same raw values. Matplotlib's axis transform changes only the
    # visualization; no logarithm is ever passed to the association metric.
    ax.scatter(x_linear, y_linear, s=s)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    resolved_xscale = _resolve_display_scale(x_linear, xscale)
    resolved_yscale = _resolve_display_scale(y_linear, yscale)
    ax.set_xscale(resolved_xscale)
    ax.set_yscale(resolved_yscale)   
    ax.grid(True, alpha=grid_alpha)

    legend_handle = Line2D(
        [],
        [],
        linestyle="none",
        label=rf"Association Ratio = {association:.4f}",
    )
    ax.legend(handles=[legend_handle], frameon=False)

    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)
    return fig, ax


def _require_large_scale(result) -> LargeScaleSimulationResult:
    if not isinstance(result, LargeScaleSimulationResult):
        raise ValueError("association scatter plots are defined for large-scale results")
    return result


def plot_interference_association_scatter(
    result: LargeScaleSimulationResult,
    *,
    ax=None,
    show: bool = False,
    xscale: str = "auto",
    yscale: str = "auto",
    save_path: str | Path | None = None,
):
    result = _require_large_scale(result)
    return plot_corr_scatter(
        result.bs_metrics["bar_communication_interference"],
        result.bs_metrics["bar_sensing_interference"],
        title="Communication - Sensing Interference",
        xlabel="Communication Interference",
        ylabel="Sensing Interference",
        ax=ax,
        show=show,
        xscale=xscale,
        yscale=yscale,
        save_path=save_path,
    )


def plot_sinr_association_scatter(
    result: LargeScaleSimulationResult,
    *,
    ax=None,
    show: bool = False,
    xscale: str = "auto",
    yscale: str = "auto",
    save_path: str | Path | None = None,
):
    result = _require_large_scale(result)
    return plot_corr_scatter(
        result.bs_metrics["bar_communication_sinr"],
        result.bs_metrics["bar_sensing_sinr"],
        title="Communication - Sensing SINR",
        xlabel="Communication SINR",
        ylabel="Sensing SINR",
        ax=ax,
        show=show,
        xscale=xscale,
        yscale=yscale,
        save_path=save_path,
    )


def plot_filter_queue_association_scatter(
    result: LargeScaleSimulationResult,
    *,
    ax=None,
    show: bool = False,
    xscale: str = "auto",
    yscale: str = "auto",
    save_path: str | Path | None = None,
):
    result = _require_large_scale(result)
    return plot_corr_scatter(
        result.bs_metrics["bar_workload"],
        result.bs_metrics["bar_covariance_trace"],
        title="Queue Workload vs Filter Covariance Trace",
        xlabel="Queue Workload",
        ylabel=r"$\mathrm{Tr}(\Sigma)$",
        ax=ax,
        show=show,
        xscale=xscale,
        yscale=yscale,
        save_path=save_path,
    )
