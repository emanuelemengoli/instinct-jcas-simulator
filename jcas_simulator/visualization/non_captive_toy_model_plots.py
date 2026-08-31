"""Visualizations specific to the supplied non-captive tracking experiment."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from ..non_captive_toy_model import NonCaptiveToyModelSimulationResult
from ._common import finish_figure


def plot_non_captive_toy_model_estimation_comparison(
    result: NonCaptiveToyModelSimulationResult,
    *,
    ax=None,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Reproduce the JCAS-vs-sensing-only estimation comparison.

    The plotted data are direct result fields corresponding to the source's
    ``smooth_err_P``, ``smooth_err_P_bl`` and ``X_rel[0,:]``.  No simulation,
    SNR, filtering, or RNG operation is performed here.
    """
    if not isinstance(result, NonCaptiveToyModelSimulationResult) or result.mode != "non_captive_toy_model":
        raise ValueError("non-captive estimation comparison requires a non-captive result")

    if ax is None:
        fig, ax1 = plt.subplots(figsize=(8, 5.5))
    else:
        ax1 = ax
        fig = ax1.figure
    ax2 = ax1.twinx()

    time = np.asarray(result.simulation_times, dtype=float)
    # The source's np.log behavior is retained.  Suppress only the expected warning
    # at exact zero errors; -inf remains visible in the data.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_jcas = np.log(result.smoothed_position_error_jcas)
        log_baseline = np.log(result.smoothed_position_error_sensing_only)

    ax1.plot(time, log_jcas, label="Error JCAS")
    ax1.plot(time, log_baseline, label="Error Sensing only framework")
    ax1.set_ylabel("Error")
    ax1.set_xlabel("Time")

    delta = float(result.metadata["delta"])
    T = int(result.metadata["horizon"])
    normalization = delta * T

    ax2.plot(
        time,
        result.relative_state[0, :] / normalization,
        color="tab:green",
        label="Relative distance of object from BS",
    )
    ax2.set_ylabel("Relative distance of object")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)
    return fig, (ax1, ax2)
