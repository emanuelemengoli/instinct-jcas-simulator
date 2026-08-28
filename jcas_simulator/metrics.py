"""Association metrics used to quantify communication/sensing coupling."""

from __future__ import annotations

import numpy as np


def association_ratio(x: np.ndarray, y: np.ndarray) -> float:
    """Return E[XY] / (E[X]E[Y]).

    This is the quantity used by the baseline. Inputs are the original
    linear-valued quantities; plotting transformations (for example log axes)
    must never be applied before calling this function. It is deliberately not
    called a Pearson correlation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return float("nan")
    x = x[mask]
    y = y[mask]
    denominator = float(np.mean(x) * np.mean(y))
    if abs(denominator) < 1.0e-30:
        return float("nan")
    return float(np.mean(x * y) / denominator)


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
