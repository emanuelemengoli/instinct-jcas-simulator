"""Modular simulator for communication/sensing coupling in large-scale JCAS networks."""

from .config import (
    BeamformingConfig,
    ChannelConfig,
    CommunicationConfig,
    NonCaptiveToyModelConfig,
    FilterConfig,
    MotionConfig,
    NetworkConfig,
    ObservationConfig,
    PointProcessConfig,
    PopulationConfig,
    RegionConfig,
    SimulationConfig,
    TDDConfig,
    TDDPhaseConfig,
)
from .channel import (
    ExponentialPowerLawChannel,
    PhysicalChannel,
    available_channel_models,
    make_channel,
    register_channel_model,
)
from .simulator import JCASSimulator, LargeScaleJCASSimulator, LargeScaleSimulationResult

__all__ = [
    "BeamformingConfig",
    "ChannelConfig",
    "PhysicalChannel",
    "ExponentialPowerLawChannel",
    "make_channel",
    "register_channel_model",
    "available_channel_models",
    "CommunicationConfig",
    "NonCaptiveToyModelConfig",
    "FilterConfig",
    "MotionConfig",
    "NetworkConfig",
    "ObservationConfig",
    "PointProcessConfig",
    "PopulationConfig",
    "RegionConfig",
    "SimulationConfig",
    "TDDConfig",
    "TDDPhaseConfig",
    "JCASSimulator",
    "LargeScaleJCASSimulator",
    "LargeScaleSimulationResult",
]
