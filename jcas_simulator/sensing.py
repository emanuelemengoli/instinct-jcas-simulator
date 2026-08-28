"""Sensing observation and SINR-dependent noise construction."""

from __future__ import annotations

import numpy as np

from .config import ObservationConfig, RegionConfig
from .filtering import LinearObservation, ObservationModel, RangeBearingObservation


def make_observation_model(
    config: ObservationConfig,
    state_dim: int,
    sensor_position: np.ndarray,
    *,
    region: RegionConfig | None = None,
) -> ObservationModel:
    """Construct an observation model using the configured spatial geometry."""
    if config.kind == "linear":
        m = int(config.observation_dim)
        if m > state_dim:
            raise ValueError("linear observation dimension cannot exceed state dimension")
        h = np.zeros((m, state_dim), dtype=float)
        h[:m, :m] = config.linear_scale * np.eye(m)
        reference = np.zeros(state_dim, dtype=float)
        reference[:2] = np.asarray(sensor_position, dtype=float)[:2]
        return LinearObservation(h, reference=reference, region=region)
    if config.kind == "range_bearing":
        return RangeBearingObservation(
            sensor_position,
            state_dim,
            include_range_rate=False,
            region=region,
        )
    if config.kind == "range_bearing_rate":
        return RangeBearingObservation(
            sensor_position,
            state_dim,
            include_range_rate=True,
            region=region,
        )
    raise ValueError(f"unsupported observation model: {config.kind}")


def measurement_covariance(config: ObservationConfig, sinr: float) -> np.ndarray:
    effective_sinr = max(float(sinr), config.min_sinr)
    if config.max_sinr is not None:
        effective_sinr = min(effective_sinr, config.max_sinr)
    if config.kind == "linear":
        m = int(config.observation_dim)
        return np.eye(m) / (m * effective_sinr)

    if config.kind == "range_bearing":
        base = np.diag([config.range_std ** 2, config.bearing_std_rad ** 2])
    elif config.kind == "range_bearing_rate":
        base = np.diag(
            [
                config.range_std ** 2,
                config.bearing_std_rad ** 2,
                config.range_rate_std ** 2,
            ]
        )
    else:
        raise ValueError(f"unsupported observation model: {config.kind}")
    return base / effective_sinr
