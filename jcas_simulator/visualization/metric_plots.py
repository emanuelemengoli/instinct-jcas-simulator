"""KDE plots consuming metric arrays already produced by the simulator."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal, Mapping

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..non_captive import NonCaptiveSimulationResult
from ..simulator import LargeScaleSimulationResult
from ._common import (
    deterministic_subsample,
    finite_samples,
    finish_figure,
    mser_truncation_index,
    resolve_steady_state,
)

#: A steady-state cutoff: a fixed sample count, a fraction in [0, 1), or
#: ``"auto"`` to estimate it via the Marginal Standard Error Rule.
SteadyState = int | float | Literal["auto"]


def _ensemble_start(series: Mapping[str, np.ndarray], *, drop_initial: bool) -> tuple[int, bool]:
    """MSER truncation index shared by every entity, from their ensemble mean.

    A per-entity series is individually noisy; averaging across entities at
    each time step first gives the detector a much less noisy trajectory to
    work with, and gives every entity in ``series`` the same truncation point
    (rather than each independently — and possibly inconsistently — picking
    its own). Returns ``(index, steady_state_detected)``; see
    :func:`~.._common.mser_truncation_index`.
    """
    arrays = []
    for key in sorted(series):
        values = np.asarray(series[key], dtype=float).reshape(-1)
        if drop_initial and len(values):
            values = values[1:]
        arrays.append(values)
    lengths = [len(a) for a in arrays]
    if not arrays or min(lengths) == 0:
        return 0, False
    min_len = min(lengths)
    # A given time step can be non-finite for every entity at once (e.g.
    # before any of them has a first measurement); nanmean legitimately warns
    # and returns NaN for that column, which mser_truncation_index already
    # handles as a non-finite batch.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ensemble_mean = np.nanmean(np.vstack([a[:min_len] for a in arrays]), axis=0)
    return mser_truncation_index(ensemble_mean)


def _pool_dict_series(
    series: Mapping[str, np.ndarray],
    *,
    steady_state: SteadyState,
    drop_initial: bool,
    time_slice: slice | None,
) -> tuple[np.ndarray, bool]:
    if steady_state == "auto":
        auto_start, detected = _ensemble_start(series, drop_initial=drop_initial)
    else:
        auto_start, detected = None, True
    pooled: list[np.ndarray] = []
    for key in sorted(series):
        values = np.asarray(series[key], dtype=float).reshape(-1)
        if drop_initial and len(values):
            values = values[1:]
        start = auto_start if auto_start is not None else resolve_steady_state(len(values), steady_state)
        values = values[start:]
        if time_slice is not None:
            values = values[time_slice]
        pooled.append(values)
    if not pooled:
        return np.empty(0, dtype=float), detected
    return finite_samples(np.concatenate(pooled)), detected


def _prepare_array(
    values: np.ndarray,
    *,
    steady_state: SteadyState,
    time_slice: slice | None,
) -> tuple[np.ndarray, bool]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if steady_state == "auto":
        start, detected = mser_truncation_index(values)
    else:
        start, detected = resolve_steady_state(len(values), steady_state), True
    values = values[start:]
    if time_slice is not None:
        values = values[time_slice]
    return finite_samples(values), detected


def _steady_state_title(base_title: str, failed_labels: list[str]) -> str:
    """Title for a steady-state plot; drop the phrase if detection failed.

    When ``steady_state`` isn't ``"auto"``, or ``"auto"`` succeeded for every
    plotted series, the title states "in steady state" as before. When
    auto-detection failed for one or more series, saying so in the title
    would misrepresent the figure — the full run is shown instead of a
    steady-state window — so the title drops the phrase; see
    :func:`_add_steady_state_caption` for the accompanying explanation.
    Call this *before* ``fig.tight_layout()`` so the title's margin is
    accounted for, matching a plain (non-annotated) title.
    """
    if not failed_labels:
        return f"{base_title} in steady state"
    return base_title


def _add_steady_state_caption(fig, failed_labels: list[str]) -> None:
    """Add a below-figure caption when steady-state detection failed for some series.

    Deliberately not baked into the legend: a legend entry is a clean data
    label a reader would want to reuse as-is (e.g. cropped into a
    publication figure), not a place for methodological caveats. Real margin
    is reserved for the caption (not just left to ``bbox_inches`` cropping at
    save time) so it isn't clipped when the figure is only shown or
    returned, not saved. Call this *after* ``fig.tight_layout()``, since it
    adjusts margins tight_layout would otherwise not know to reserve.
    """
    if not failed_labels:
        return
    caption = (
        "Steady state not detected for "
        + ", ".join(failed_labels)
        + " — full run shown instead."
    )
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=9, style="italic", wrap=True)


def _plot_seaborn_kde(
    ax,
    samples: np.ndarray,
    *,
    label: str,
    cumulative: bool,
    grid_points: int,
) -> None:
    """Plot already-selected samples using :func:`seaborn.kdeplot` only."""
    samples = finite_samples(samples)
    if len(samples) == 0:
        raise ValueError("no finite samples are available for this plot")
    if grid_points < 64:
        raise ValueError("grid_points must be at least 64")

    if np.ptp(samples) == 0.0:
        # A degenerate (zero-variance) sample set has no density for
        # seaborn.kdeplot to estimate; it silently draws nothing, leaving the
        # legend empty. A labeled vertical line is the honest picture of a
        # point-mass distribution, and is a real outcome here — e.g. a
        # steady-state communication queue that is empty at every sample.
        # axvline draws in a mixed (data-x, axes-fraction-y) transform, so on
        # an otherwise-empty axes it contributes nothing for autoscale to
        # work with: the view silently collapses to matplotlib's default unit
        # box and the line sits invisibly on the left edge. Give the axes an
        # explicit, visible window around the value in that case.
        had_data = ax.has_data()
        value = float(samples[0])
        ax.axvline(value, label=label, linestyle="--")
        if not had_data:
            margin = max(abs(value), 1.0) * 0.5
            ax.set_xlim(value - margin, value + margin)
        return

    sns.kdeplot(
        x=samples,
        ax=ax,
        label=label,
        fill=not cumulative,
        cumulative=cumulative,
        gridsize=grid_points,
        warn_singular=False,
    )


def plot_sinr_kde(
    result: LargeScaleSimulationResult | NonCaptiveSimulationResult,
    *,
    series: str = "both",
    steady_state: SteadyState = 0.4,
    time_slice: slice | None = None,
    db: bool = True,
    cumulative: bool = False,
    max_samples: int | None = None,
    grid_points: int = 512,
    ax=None,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot KDEs of stored SINR/SNR samples with ``seaborn.kdeplot``.

    Large-scale results pool samples across all UEs/SOs and selected time steps
    in steady state. Non-captive results use the SNR histories produced by the
    non-captive simulation. No SINR/SNR is recomputed by this function.

    ``steady_state`` picks where that steady-state window begins: a fixed
    sample count, a fraction in [0, 1), or ``"auto"`` to estimate it with the
    Marginal Standard Error Rule instead of an arbitrary fixed cutoff.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5.5))
    else:
        fig = ax.figure

    if isinstance(result, LargeScaleSimulationResult):
        aliases = {
            "communication": "communication", "com": "communication",
            "sensing": "sensing", "sen": "sensing", "rad": "sensing",
            "both": "both",
        }
        choice = aliases.get(series.lower())
        if choice is None:
            raise ValueError("series must be 'communication', 'sensing', or 'both'")
        selected: list[tuple[str, np.ndarray, bool]] = []
        if choice in {"communication", "both"}:
            samples, detected = _pool_dict_series(
                result.communication_sinr,
                steady_state=steady_state,
                drop_initial=False,
                time_slice=time_slice,
            )
            selected.append(("Communication SINR", samples, detected))
        if choice in {"sensing", "both"}:
            samples, detected = _pool_dict_series(
                result.sensing_sinr,
                steady_state=steady_state,
                drop_initial=False,
                time_slice=time_slice,
            )
            selected.append(("Sensing SINR", samples, detected))
    elif isinstance(result, NonCaptiveSimulationResult):
        aliases = {
            "jcas": "jcas", "non_captive": "jcas",
            "sensing_only": "sensing_only", "sensing": "sensing_only",
            "both": "both",
        }
        choice = aliases.get(series.lower())
        if choice is None:
            raise ValueError(
                "for non-captive results, series must be 'jcas', 'sensing_only', or 'both'"
            )
        selected = []
        if choice in {"jcas", "both"}:
            samples, detected = _prepare_array(result.jcas_snr, steady_state=steady_state, time_slice=time_slice)
            selected.append(("Non-captive JCAS SNR", samples, detected))
        if choice in {"sensing_only", "both"}:
            samples, detected = _prepare_array(
                result.sensing_only_snr, steady_state=steady_state, time_slice=time_slice
            )
            selected.append(("Sensing-only SNR", samples, detected))
    else:
        raise TypeError("unsupported simulation result type")

    failed_labels: list[str] = []
    for label, samples, detected in selected:
        samples = deterministic_subsample(samples, max_samples)
        if db:
            samples = 10.0 * np.log10(np.maximum(samples, 1.0e-12))
            label += " (dB)"
        if not detected:
            failed_labels.append(label)
        _plot_seaborn_kde(
            ax,
            samples,
            label=label,
            cumulative=cumulative,
            grid_points=grid_points,
        )

    ax.set_xlabel("SINR (dB)" if db else "SINR (linear scale)")
    ax.set_ylabel("Cumulative probability" if cumulative else "Density")
    ax.set_title(_steady_state_title("SINR distribution", failed_labels))
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _add_steady_state_caption(fig, failed_labels)
    finish_figure(fig, save_path=save_path, show=show)
    return fig, ax


def plot_covariance_trace_kde(
    result: LargeScaleSimulationResult | NonCaptiveSimulationResult,
    *,
    steady_state: SteadyState = 0.4,
    time_slice: slice | None = None,
    cumulative: bool = False,
    max_samples: int | None = None,
    grid_points: int = 512,
    ax=None,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot KDEs of ``Tr(Sigma)`` with ``seaborn.kdeplot``.

    ``steady_state`` accepts ``"auto"`` (Marginal Standard Error Rule) in
    addition to a fixed sample count or a fraction in [0, 1); see
    :func:`plot_sinr_kde`.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5.5))
    else:
        fig = ax.figure

    failed_labels: list[str] = []
    if isinstance(result, LargeScaleSimulationResult):
        samples, detected = _pool_dict_series(
            result.covariance_traces,
            steady_state=steady_state,
            drop_initial=True,
            time_slice=time_slice,
        )
        samples = deterministic_subsample(samples, max_samples)
        if not detected:
            failed_labels.append("KF/EKF covariance trace")
        _plot_seaborn_kde(
            ax,
            samples,
            label="KF/EKF covariance trace",
            cumulative=cumulative,
            grid_points=grid_points,
        )
    elif isinstance(result, NonCaptiveSimulationResult):
        jcas_trace = np.trace(result.jcas_covariance, axis1=0, axis2=1)
        baseline_trace = np.trace(result.sensing_only_covariance, axis1=0, axis2=1)
        for label, values in (
            ("Non-captive JCAS covariance trace", jcas_trace),
            ("Sensing-only covariance trace", baseline_trace),
        ):
            samples, detected = _prepare_array(values, steady_state=steady_state, time_slice=time_slice)
            samples = deterministic_subsample(samples, max_samples)
            if not detected:
                failed_labels.append(label)
            _plot_seaborn_kde(
                ax,
                samples,
                label=label,
                cumulative=cumulative,
                grid_points=grid_points,
            )
    else:
        raise TypeError("unsupported simulation result type")

    ax.set_xlabel(r"$\mathrm{Tr}(\Sigma)$")
    ax.set_ylabel("Cumulative probability" if cumulative else "Density")
    ax.set_title(_steady_state_title("Filter covariance-trace distribution", failed_labels))
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _add_steady_state_caption(fig, failed_labels)
    finish_figure(fig, save_path=save_path, show=show)
    return fig, ax


def plot_workload_kde(
    result: LargeScaleSimulationResult | NonCaptiveSimulationResult,
    *,
    steady_state: SteadyState = 0.4,
    time_slice: slice | None = None,
    cumulative: bool = False,
    max_samples: int | None = None,
    grid_points: int = 512,
    ax=None,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot the KDE of stored Lindley workloads with ``seaborn.kdeplot``.

    ``steady_state`` accepts ``"auto"`` (Marginal Standard Error Rule) in
    addition to a fixed sample count or a fraction in [0, 1); see
    :func:`plot_sinr_kde`.
    """
    if not isinstance(result, LargeScaleSimulationResult):
        raise ValueError("the supplied non-captive model has no communication queue/workload output")
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5.5))
    else:
        fig = ax.figure

    samples, detected = _pool_dict_series(
        result.queue_workloads,
        steady_state=steady_state,
        drop_initial=True,
        time_slice=time_slice,
    )
    samples = deterministic_subsample(samples, max_samples)
    failed_labels = [] if detected else ["Queue workload"]
    _plot_seaborn_kde(
        ax,
        samples,
        label="Queue workload",
        cumulative=cumulative,
        grid_points=grid_points,
    )
    ax.set_xlabel("Workload")
    ax.set_ylabel("Cumulative probability" if cumulative else "Density")
    ax.set_title(_steady_state_title("Communication workload distribution", failed_labels))
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _add_steady_state_caption(fig, failed_labels)
    finish_figure(fig, save_path=save_path, show=show)
    return fig, ax
