"""Logical-time communication/sensing scheduling.

The active ``jcas`` path executes communication and sensing in every
simulation tick, with no UL/DL TDD frame.  The scheduler therefore uses a
single joint phase as the default profile, while providing an explicit
deterministic phase interface for future or user-configured time division
without scattering time-index conditions through the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import TDDConfig, TDDPhaseConfig


@dataclass(frozen=True)
class TDDState:
    phase_name: str
    communication_active: bool
    sensing_active: bool
    frame_slot: int


class TDDScheduler:
    """Deterministic cyclic scheduler driven only by logical slot index."""

    def __init__(self, config: TDDConfig):
        self.config = config
        if config.enabled:
            self._expanded = tuple(
                phase
                for phase in config.phases
                for _ in range(int(phase.duration_slots))
            )
        else:
            self._expanded = (
                TDDPhaseConfig(
                    name="joint",
                    duration_slots=1,
                    communication_active=True,
                    sensing_active=True,
                ),
            )

    @property
    def frame_length_slots(self) -> int:
        return len(self._expanded)

    def state_at(self, slot: int) -> TDDState:
        if slot < 0:
            raise ValueError("slot must be non-negative")
        frame_slot = int(slot) % len(self._expanded)
        phase = self._expanded[frame_slot]
        return TDDState(
            phase_name=phase.name,
            communication_active=bool(phase.communication_active),
            sensing_active=bool(phase.sensing_active),
            frame_slot=frame_slot,
        )
