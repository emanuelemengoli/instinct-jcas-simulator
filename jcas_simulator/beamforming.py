"""Configurable beamforming models.

This large-scale simulator does not implement array precoders/combiners.
Its active beamforming model is a rotating 2-D sector antenna: one of
``2**log2_beams`` unit-vector directions is active at a time, and a link gets
one of two scalar gains depending on whether its direction lies inside the
current sector.  This module preserves that mathematical behavior while
keeping it separate from the physical channel model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .config import BeamformingConfig


def db_to_linear_power(gain_db: float) -> float:
    """Convert a dB power gain to linear scale."""
    return float(10.0 ** (float(gain_db) / 10.0))


@dataclass
class DirectionalSectorBeamformer:
    """Rotating directional beam model.

    ``n_beams = 2**log2_beams`` equally spaced directions over ``[0, 2*pi)``;
    main-lobe gain applies within half a beamwidth, otherwise side-lobe gain.
    ``advance`` moves exactly one sector per logical simulation slot.
    """

    config: BeamformingConfig
    current_beam_index: int

    def __post_init__(self) -> None:
        self.n_beams = 2 ** int(self.config.log2_beams)
        self.beamwidth_rad = 2.0 * math.pi / self.n_beams
        self.current_beam_index %= self.n_beams

    @property
    def reference_vector(self) -> np.ndarray:
        angle = self.current_beam_index * self.beamwidth_rad
        return np.array([math.cos(angle), math.sin(angle)], dtype=float)

    def advance(self) -> None:
        self.current_beam_index = (self.current_beam_index + 1) % self.n_beams

    @staticmethod
    def relative_angle(v1: np.ndarray, v2: np.ndarray) -> float:
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        dot = float(np.dot(v1, v2))
        return abs(math.atan2(cross, dot))

    def gain_db(self, transmitter_position: np.ndarray, target_position: np.ndarray) -> float:
        direction = np.asarray(target_position, dtype=float) - np.asarray(
            transmitter_position, dtype=float
        )
        # At coincident locations the old atan2(0, 0) path evaluates to zero,
        # hence the main-lobe gain. Preserve that convention.
        angle = self.relative_angle(direction, self.reference_vector)
        if angle <= self.beamwidth_rad / 2.0:
            return float(self.config.main_lobe_gain_db)
        return float(self.config.side_lobe_gain_db)

    def gain_linear(self, transmitter_position: np.ndarray, target_position: np.ndarray) -> float:
        return db_to_linear_power(self.gain_db(transmitter_position, target_position))


class UnityBeamformer:
    """Disabled-beamforming model with no RNG and unit gain."""

    current_beam_index = -1

    def advance(self) -> None:
        return None

    def gain_db(self, transmitter_position: np.ndarray, target_position: np.ndarray) -> float:
        return 0.0

    def gain_linear(self, transmitter_position: np.ndarray, target_position: np.ndarray) -> float:
        return 1.0
