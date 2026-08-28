"""Central, serialisable configuration for the JCAS simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class RegionConfig:
    """Rectangular simulation window, in metres.

    This simulator is toroidal-only: every spatial subsystem (Voronoi
    tessellation, mobility, filtering, beamforming, handover) assumes the
    rectangular flat-torus (minimum-image) metric.  ``distance_model`` is kept
    as an explicit field for readability but ``"toroidal"`` is the only
    accepted value; a non-toroidal value is rejected at construction.
    """

    width: float = 4500.0
    height: float = 4000.0
    center_x: float = 0.0
    center_y: float = 0.0
    distance_model: Literal["toroidal"] = "toroidal"

    def __post_init__(self) -> None:
        if self.distance_model != "toroidal":
            raise ValueError(
                "RegionConfig.distance_model must be 'toroidal'; this simulator "
                "only supports the rectangular-torus (minimum-image) metric "
                f"(got {self.distance_model!r})"
            )

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.center_x - self.width / 2.0,
            self.center_y - self.height / 2.0,
            self.center_x + self.width / 2.0,
            self.center_y + self.height / 2.0,
        )


@dataclass(frozen=True)
class PointProcessConfig:
    """Base-station point-process configuration.

    ``kind='ppp'`` gives a homogeneous Poisson point process on the window.
    ``fixed_count`` can be set for controlled experiments; conditional on the
    count, locations are still i.i.d. uniform.
    """

    kind: Literal["ppp"] = "ppp"
    intensity_per_m2: float = 30.0 / 18_000_000.0
    fixed_count: int | None = None
    minimum_count: int = 1


@dataclass(frozen=True)
class PopulationConfig:
    """Per-cell UE/SO population model."""

    count_model: Literal["poisson", "fixed"] = "poisson"
    mean_per_cell: float = 1.0
    fixed_per_cell: int | None = None
    minimum_per_cell: int = 1
    placement: Literal["uniform", "gaussian_around_bs"] = "uniform"
    placement_std_fraction: float = 0.20


@dataclass(frozen=True)
class NetworkConfig:
    region: RegionConfig = field(default_factory=RegionConfig)
    base_stations: PointProcessConfig = field(default_factory=PointProcessConfig)
    ue_population: PopulationConfig = field(default_factory=PopulationConfig)
    so_population: PopulationConfig = field(default_factory=PopulationConfig)


@dataclass(frozen=True)
class MotionConfig:
    """State/mobility model used by UEs or sensing objects.

    ``process_noise_std`` is a standard deviation. The corresponding process
    covariance used by both state simulation and filtering is ``std**2 * I``.
    """

    kind: Literal["static", "gauss_markov", "constant_speed"] = "gauss_markov"
    state_dim: Literal[2, 4] = 2
    rho: float = 0.9
    process_noise_std: float = 1.0
    initial_speed_min_mps: float = 0.0
    initial_speed_max_mps: float = 0.0
    center_on_serving_bs: bool = True


@dataclass(frozen=True)
class BeamformingConfig:
    """Directional-sector beamforming configuration.

    This simulator models a scalar directional antenna gain rather than an
    array precoder/combiner; ``sector`` reproduces that model.
    """

    enabled: bool = False
    model: Literal["sector"] = "sector"
    main_lobe_gain_db: float = 23.0
    side_lobe_gain_db: float = 3.0
    log2_beams: int = 4
    initial_beam_index: int | None = None


@dataclass(frozen=True)
class TDDPhaseConfig:
    """One logical scheduling phase.

    The scheduler distinguishes communication and sensing activity, but it
    does not implement separate uplink/downlink equations.  These booleans
    therefore describe the operations that are active in a slot.
    """

    name: str = "joint"
    duration_slots: int = 1
    communication_active: bool = True
    sensing_active: bool = True


@dataclass(frozen=True)
class TDDConfig:
    """Explicit logical-time communication/sensing schedule.

    ``enabled=False`` reproduces the default behavior: both communication and
    sensing are evaluated every tick.  When enabled, the configured phases
    repeat cyclically.
    """

    enabled: bool = False
    phases: tuple[TDDPhaseConfig, ...] = field(
        default_factory=lambda: (TDDPhaseConfig(),)
    )


@dataclass(frozen=True)
class ChannelConfig:
    """Physical-channel selection and parameters.

    ``model="rt"`` preserves the current ray-tracing-derived channel.
    ``model="exponential"`` uses the Rayleigh-fading path-loss law
    ``H * max(d_min, d)**(-alpha)`` with ``H ~ Exp(mean)`` — an exponentially
    distributed power gain is exactly Rayleigh fading.

    The remaining transmit-power/noise/sensing parameters are shared by the
    large-scale simulator regardless of which physical channel is selected.
    """

    model: str = "rt"

    # Rayleigh fading + power-law path loss.
    exponential_path_loss_exponent: float = 2.7
    exponential_fading_mean: float = 1.0
    exponential_min_distance_m: float = 1.0

    # RTChannel parameters.
    carrier_frequency_hz: float = 1.0e9
    speed_of_light_mps: float = 299_792_458.0
    mean_delay_s: float = 7.4311e-7
    mean_gain_db: float = -83.274
    covariance_delay_delay: float = 4.1110e-13
    covariance_delay_gain: float = -3.2769e-6
    covariance_gain_gain: float = 533.3182
    path_count_prefactor: float = 532.5255
    path_count_decay_per_m: float = 0.0036
    min_paths: int = 0

    # Physical-support handling for the D4.3 Gaussian excess-delay fit.
    # The fitted bivariate Gaussian has unbounded support although physical
    # excess delay is non-negative.  When enabled, a negative delay causes the
    # complete (delay, gain) pair to be rejected and redrawn, preserving their
    # fitted joint dependence rather than clipping the delay alone.
    rt_reject_negative_delay: bool = True
    rt_max_rejection_rounds: int = 10_000

    transmit_power_dbm: float = 46.0
    noise_psd_dbm_per_hz: float = -125.0
    bandwidth_hz: float = 20.0e6
    sensing_two_way: bool = True
    # ``auto`` preserves the exact square-law sensing gain for the exponential
    # channel and uses the physical monostatic radar normalization for RT.
    # Set explicitly to override that channel-dependent default.
    sensing_gain_model: Literal["auto", "square_law", "radar_equation"] = "auto"
    radar_cross_section: float = 1.0  # physical sigma [m^2] for radar_equation

    # Power-domain denominator floor used only by SINR calculations.
    epsilon: float = 1.0e-15


@dataclass(frozen=True)
class CommunicationConfig:
    arrival_rate: float = 1.0
    service_scale: float = 1.0


@dataclass(frozen=True)
class ObservationConfig:
    """Observation model and SINR-dependent measurement noise."""

    kind: Literal["linear", "range_bearing", "range_bearing_rate"] = "range_bearing"
    observation_dim: Literal[2, 3] = 2
    linear_scale: float = 1.0
    # Base standard deviations before scaling by 1/sqrt(SINR).
    range_std: float = 0.5
    bearing_std_rad: float = 0.017453292519943295  # 1 degree
    range_rate_std: float = 0.1
    min_sinr: float = 1.0e-12
    max_sinr: float | None = None


@dataclass(frozen=True)
class FilterConfig:
    """Filtering configuration.

    ``kind='auto'`` selects KF for a linear observation and EKF otherwise.
    """

    kind: Literal["auto", "kf", "ekf"] = "auto"
    initial_covariance_scale: float = 1.0
    initial_state_error_std: float = 0.0
    observation: ObservationConfig = field(default_factory=ObservationConfig)


@dataclass(frozen=True)
class NonCaptiveConfig:
    """Parameters of the supplied non-captive JCAS tracking experiment."""

    horizon: int = 10_000
    delta: float = 1.0
    bs_relative_positions: tuple[float, ...] = (0.0, 0.1, 0.3, 0.6, 0.9, 1.0)
    position_process_variance: float = 50.0
    gain_process_variance: float = 50.0
    path_loss_exponent: float = 1.2
    gain_rho: float = 0.9
    distance_epsilon: float = 1.0
    smoothing_window: int = 1000


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level simulator configuration."""

    master_seed: int = 142
    operation_mode: Literal["non_cooperative", "non_captive"] = "non_cooperative"
    horizon: int = 100
    time_step_s: float = 1.0
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ue_motion: MotionConfig = field(default_factory=MotionConfig)
    so_motion: MotionConfig = field(default_factory=MotionConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    beamforming: BeamformingConfig = field(default_factory=BeamformingConfig)
    tdd: TDDConfig = field(default_factory=TDDConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    non_captive: NonCaptiveConfig = field(default_factory=NonCaptiveConfig)
    handover_enabled: bool = False
    # Hysteresis margin, in metres, for nearest-BS reassignment.  A serving-BS
    # switch only happens when a competitor is closer than the current serving
    # BS by more than this margin, which stops an entity sitting on a Voronoi
    # edge from ping-ponging (and rebuilding its dynamics) every tick.  The
    # default 0.0 reproduces plain instantaneous nearest-BS handover.
    handover_margin_m: float = 0.0

    def validate(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.time_step_s <= 0:
            raise ValueError("time_step_s must be positive")
        if self.network.region.width <= 0 or self.network.region.height <= 0:
            raise ValueError("simulation-region dimensions must be positive")
        if self.network.region.distance_model != "toroidal":
            raise ValueError(
                "network.region.distance_model must be 'toroidal'; the euclidean "
                "distance model is not supported by this simulator"
            )
        if self.handover_margin_m < 0:
            raise ValueError("handover_margin_m must be non-negative")
        if self.network.base_stations.intensity_per_m2 < 0:
            raise ValueError("base-station intensity must be non-negative")
        for name, pop in (
            ("ue_population", self.network.ue_population),
            ("so_population", self.network.so_population),
        ):
            if pop.mean_per_cell < 0:
                raise ValueError(f"{name}.mean_per_cell must be non-negative")
            if pop.minimum_per_cell < 0:
                raise ValueError(f"{name}.minimum_per_cell must be non-negative")
            if pop.placement_std_fraction <= 0:
                raise ValueError(f"{name}.placement_std_fraction must be positive")
        for name, motion in (("ue_motion", self.ue_motion), ("so_motion", self.so_motion)):
            if not 0 <= motion.rho <= 1.0:
                raise ValueError(f"{name}.rho must lie in [0, 1]")
            if motion.process_noise_std < 0:
                raise ValueError(f"{name}.process_noise_std must be non-negative")

        channel = self.channel
        if channel.sensing_gain_model not in {"auto", "square_law", "radar_equation"}:
            raise ValueError("unsupported channel.sensing_gain_model")
        if channel.carrier_frequency_hz <= 0:
            raise ValueError("channel.carrier_frequency_hz must be positive")
        if channel.speed_of_light_mps <= 0:
            raise ValueError("channel.speed_of_light_mps must be positive")
        if channel.radar_cross_section < 0:
            raise ValueError("channel.radar_cross_section must be non-negative")
        if channel.min_paths < 0:
            raise ValueError("channel.min_paths must be non-negative")
        if channel.rt_max_rejection_rounds <= 0:
            raise ValueError("channel.rt_max_rejection_rounds must be positive")
        for name, motion in (("ue_motion", self.ue_motion), ("so_motion", self.so_motion)):
            if motion.state_dim not in (2, 4):
                raise ValueError(f"{name}.state_dim must be 2 or 4")
        if not str(self.channel.model).strip():
            raise ValueError("channel.model must be non-empty")
        if self.channel.exponential_path_loss_exponent <= 0:
            raise ValueError("channel.exponential_path_loss_exponent must be positive")
        if self.channel.exponential_fading_mean <= 0:
            raise ValueError("channel.exponential_fading_mean must be positive")
        if self.channel.exponential_min_distance_m <= 0:
            raise ValueError("channel.exponential_min_distance_m must be positive")
        if self.beamforming.log2_beams < 0:
            raise ValueError("beamforming.log2_beams must be non-negative")
        n_beams = 2 ** self.beamforming.log2_beams
        if self.beamforming.initial_beam_index is not None and not (
            0 <= self.beamforming.initial_beam_index < n_beams
        ):
            raise ValueError("beamforming.initial_beam_index is outside the beam codebook")
        if not self.tdd.phases:
            raise ValueError("tdd.phases must contain at least one phase")
        for phase in self.tdd.phases:
            if phase.duration_slots <= 0:
                raise ValueError("TDD phase durations must be positive")
            if not phase.name:
                raise ValueError("TDD phase names must be non-empty")
            if not (phase.communication_active or phase.sensing_active):
                raise ValueError("a TDD phase must activate communication and/or sensing")
        if self.operation_mode == "non_captive" and (
            self.beamforming.enabled or self.tdd.enabled
        ):
            raise ValueError(
                "the supplied non-captive tracking model contains no RTChannel, "
                "sector-beamforming, or active TDD equations; enabling large-scale "
                "beamforming/TDD in non_captive mode would change that model"
            )
        if self.beamforming.enabled and str(self.channel.model).strip().lower() == "rt":
            raise ValueError(
                "channel.model='rt' is incompatible with beamforming.enabled=True: "
                "RTChannel.power_gain is an isotropic, distance-only power gain (it "
                "already contains its own Friis path loss, fitted multipath and "
                "measurement-campaign normalisation), so multiplying it by a "
                "directional sector-beam gain is not physically meaningful. Use "
                "channel.model='exponential' with beamforming."
            )
        if self.filtering.initial_covariance_scale < 0:
            raise ValueError("initial_covariance_scale must be non-negative")
        if self.communication.arrival_rate < 0:
            raise ValueError("arrival_rate must be non-negative")
