"""Shared helpers for the optional Matplotlib visualization layer."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def finish_figure(fig, *, save_path: str | Path | None, show: bool) -> None:
    """Optionally save/show a figure without touching simulation state or RNGs."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()


def resolve_steady_state(length: int, steady_state: int | float) -> int:
    """Resolve the steady-state start index; a float in [0, 1) is a fraction.

    Only handles an explicit numeric ``steady_state``; the ``"auto"`` case is
    resolved separately by :func:`mser_truncation_index`, which needs the
    actual samples rather than just a length.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    if isinstance(steady_state, float):
        if not 0.0 <= steady_state < 1.0:
            raise ValueError("fractional steady_state must lie in [0, 1)")
        return min(length, int(round(steady_state * length)))
    start = int(steady_state)
    if start < 0:
        raise ValueError("steady_state must be non-negative")
    return min(length, start)


def mser_truncation_index(values: np.ndarray, *, batch_size: int = 5) -> tuple[int, bool]:
    """Estimate where a transient ends using the Marginal Standard Error Rule (MSER-5).

    ``values`` is a single trajectory, typically an ensemble mean over
    entities at each time step rather than any one entity's own noisy
    series. The series is batched into non-overlapping groups of
    ``batch_size`` samples (reducing serial correlation, per White 1997)
    and, for every candidate truncation point ``d`` (in units of batches),
    scored by the scaled variance of the remaining batches around their own
    mean:

        MSER(d) = var(batches[d:]) / len(batches[d:])

    The ``d`` minimizing that score is a first candidate (converted back to
    an index into the original, unbatched series). Candidates are restricted
    to discarding at most half the batches: without that cap, MSER trivially
    "prefers" truncating almost everything, since a near-empty remainder has
    near-zero variance regardless of whether the process has converged.

    That cap alone is not enough: MSER's score also rewards a short *recent*
    window of a series that is not stationary but merely drifting (e.g. the
    workload of an unstable queue, arrival rate >= service rate), since a
    short window of a smooth trend has low variance around its own —
    drifting — mean. To guard against mistaking "still transient" for
    "converged," the retained window is checked for a residual linear trend
    (an ordinary-least-squares slope test at the 5% level); if one remains,
    the candidate is rejected and every sample is kept instead, since
    discarding further would fabricate a steady state rather than remove a
    transient.

    Returns ``(index, steady_state_detected)``. ``steady_state_detected`` is
    ``False`` — with ``index=0``, i.e. keep everything — whenever there are
    too few samples to batch meaningfully, every batch is non-finite, or the
    residual-trend check above rejects the candidate; callers should surface
    that to the user rather than silently using the full series.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    values = np.asarray(values, dtype=float).reshape(-1)
    n_batches = len(values) // batch_size
    if n_batches < 2:
        return 0, False

    # A batch can be entirely non-finite (e.g. a quantity undefined before an
    # entity's first update); nanmean legitimately warns and returns NaN for
    # it, which the finite-batch handling below already expects.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        batches = np.nanmean(
            values[: n_batches * batch_size].reshape(n_batches, batch_size), axis=1
        )
    finite = np.isfinite(batches)
    if not np.any(finite):
        return 0, False
    # Drop any leading non-finite batches (e.g. a quantity undefined before an
    # entity's first update): MSER only needs to search among the rest.
    first_finite = int(np.argmax(finite))
    batches = batches[first_finite:]
    n_batches = len(batches)
    if n_batches < 2:
        return 0, False

    max_d = n_batches // 2
    best_d, best_score = 0, np.inf
    for d in range(0, max_d + 1):
        remaining = batches[d:]
        remaining = remaining[np.isfinite(remaining)]
        if len(remaining) < 2:
            break
        score = np.mean((remaining - remaining.mean()) ** 2) / len(remaining)
        if score < best_score:
            best_score = score
            best_d = d

    retained = batches[best_d:]
    retained = retained[np.isfinite(retained)]
    if len(retained) >= 4:
        regression = stats.linregress(np.arange(len(retained)), retained)
        if regression.pvalue < 0.05:
            return 0, False

    return (first_finite + best_d) * batch_size, True


def finite_samples(values: Iterable[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def deterministic_subsample(values: np.ndarray, max_samples: int | None) -> np.ndarray:
    """Deterministically thin samples without consuming a random-number stream."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if max_samples is None or len(values) <= max_samples:
        return values
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    indices = np.linspace(0, len(values) - 1, num=max_samples, dtype=int)
    return values[indices]
