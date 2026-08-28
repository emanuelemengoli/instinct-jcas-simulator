"""Simulation orchestration for large-scale and supplied non-captive JCAS modes."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Callable
import numpy as np

from .beamforming import DirectionalSectorBeamformer, UnityBeamformer
from .communication import LindleyQueue
from .config import RegionConfig, SimulationConfig
from .non_captive import NonCaptiveSimulationResult, run_supplied_non_captive_model
from .filtering import ExtendedKalmanFilter, KalmanFilter
from .geometry import distance, toroidal_displacement, wrap_state
from .metrics import association_ratio, pearson_correlation
from .mobility import DynamicsModel, initial_state, make_dynamics
from .network import Network, NetworkEntity, generate_network
from .rng import RNGManager
from .channel import PhysicalChannel, RTChannel, dbm_to_watts, make_channel, noise_power_watts
from .scheduling import TDDScheduler
from .sensing import make_observation_model, measurement_covariance


@dataclass
class LargeScaleSimulationResult:
    mode: str
    network: Network
    queue_workloads: dict[str, np.ndarray]
    communication_sinr: dict[str, np.ndarray]
    sensing_sinr: dict[str, np.ndarray]
    covariance_norms: dict[str, np.ndarray]
    covariance_traces: dict[str, np.ndarray]
    ue_trajectories: dict[str, np.ndarray]
    so_trajectories: dict[str, np.ndarray]
    simulation_times: np.ndarray
    region: RegionConfig
    bs_metrics: dict[str, np.ndarray]
    association: dict[str, dict[str, float]]
    tdd_phase_names: tuple[str, ...] = ()
    communication_active: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    sensing_active: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    beam_indices: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=int))
    communication_effective_gains: dict[str, np.ndarray] = field(default_factory=dict)
    sensing_effective_gains: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "n_base_stations": len(self.network.base_stations),
            "n_ues": len(self.network.ues),
            "n_sensing_objects": len(self.network.sensing_objects),
            "beamforming_enabled": bool(self.metadata.get("beamforming_enabled", False)),
            "tdd_enabled": bool(self.metadata.get("tdd_enabled", False)),
            "association": self.association,
        }


class LargeScaleJCASSimulator:
    """Deterministic logical-time simulation of the stochastic JCAS network."""

    def __init__(
        self,
        config: SimulationConfig,
        *,
        ue_dynamics_factory: Callable = make_dynamics,
        so_dynamics_factory: Callable = make_dynamics,
        observation_factory: Callable = make_observation_model,
        channel: PhysicalChannel | None = None,
    ):
        config.validate()
        self.config = config
        self.rngs = RNGManager(config.master_seed)
        self.network = generate_network(config.network, self.rngs)
        self.ue_dynamics_factory = ue_dynamics_factory
        self.so_dynamics_factory = so_dynamics_factory
        self.observation_factory = observation_factory
        self.channel = make_channel(config.channel) if channel is None else channel
        if config.beamforming.enabled and isinstance(self.channel, RTChannel):
            raise ValueError(
                "beamforming.enabled=True requires a non-isotropic physical channel; "
                "RTChannel.power_gain is distance-only and cannot be combined with a "
                "directional sector-beam gain. Use the exponential channel model."
            )
        self.tx_power_w = dbm_to_watts(config.channel.transmit_power_dbm)
        self.noise_power_w = noise_power_watts(config.channel)
        self.scheduler = TDDScheduler(config.tdd)

        self.ue_dynamics: dict[str, DynamicsModel] = {}
        self.so_dynamics: dict[str, DynamicsModel] = {}
        self.queues: dict[str, LindleyQueue] = {}
        self.filters: dict[str, KalmanFilter | ExtendedKalmanFilter] = {}
        self.beamformers: dict[int, DirectionalSectorBeamformer | UnityBeamformer] = {}
        self._initialize_beamformers()
        self._initialize_entities()

    def _initialize_beamformers(self) -> None:
        cfg = self.config.beamforming
        for bs in self.network.base_stations:
            if not cfg.enabled:
                self.beamformers[bs.index] = UnityBeamformer()
                continue
            if cfg.initial_beam_index is None:
                rng = self.rngs.generator(f"beamforming:init:bs{bs.index}")
                initial_index = int(rng.integers(0, 2 ** cfg.log2_beams))
            else:
                initial_index = int(cfg.initial_beam_index)
            self.beamformers[bs.index] = DirectionalSectorBeamformer(cfg, initial_index)

    def _advance_beams(self) -> None:
        # One sector step per BS per logical tick.  Called at the end of each
        # tick so the configured/initial beam index is the one used at t=0.
        if not self.config.beamforming.enabled:
            return
        for bs in self.network.base_stations:
            self.beamformers[bs.index].advance()

    @staticmethod
    def _accepts_keyword(factory: Callable, keyword: str) -> bool:
        """Return whether a factory can accept a named keyword argument."""
        try:
            params = inspect.signature(factory).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            or (p.name == keyword and p.kind != inspect.Parameter.POSITIONAL_ONLY)
            for p in params
        )

    def _build_dynamics(
        self,
        factory: Callable,
        motion_config,
        *,
        center: np.ndarray,
    ) -> DynamicsModel:
        if not self._accepts_keyword(factory, "region"):
            raise ValueError(
                "a custom dynamics factory must accept a 'region' keyword: this "
                "simulator is toroidal-only and a factory that cannot receive the "
                "RegionConfig would silently build non-torus-aware dynamics"
            )
        kwargs = {"center": center, "region": self.config.network.region}
        return factory(motion_config, self.config.time_step_s, **kwargs)

    def _build_observation(
        self,
        observation_config,
        state_dim: int,
        sensor_position: np.ndarray,
    ):
        kwargs = {}
        if self._accepts_keyword(self.observation_factory, "region"):
            kwargs["region"] = self.config.network.region
        return self.observation_factory(
            observation_config, state_dim, sensor_position, **kwargs
        )

    def _project_filter_state(self, state: np.ndarray) -> np.ndarray:
        return wrap_state(state, self.config.network.region)

    def _initialize_entities(self) -> None:
        for ue in self.network.ues:
            rng = self.rngs.generator(f"mobility:init:{ue.id}")
            ue.state = initial_state(ue.position, self.config.ue_motion, rng)
            bs = self.network.serving_bs(ue)
            self.ue_dynamics[ue.id] = self._build_dynamics(
                self.ue_dynamics_factory, self.config.ue_motion, center=bs.position
            )
            self.queues[ue.id] = LindleyQueue(
                arrival_rate=self.config.communication.arrival_rate,
                service_scale=self.config.communication.service_scale,
            )

        for so in self.network.sensing_objects:
            rng = self.rngs.generator(f"mobility:init:{so.id}")
            so.state = initial_state(so.position, self.config.so_motion, rng)
            bs = self.network.serving_bs(so)
            dynamics = self._build_dynamics(
                self.so_dynamics_factory, self.config.so_motion, center=bs.position
            )
            self.so_dynamics[so.id] = dynamics
            init_rng = self.rngs.generator(f"filter:init:{so.id}")
            estimate = so.state.copy()
            if self.config.filtering.initial_state_error_std > 0:
                estimate += init_rng.normal(
                    0.0,
                    self.config.filtering.initial_state_error_std,
                    size=estimate.shape,
                )
            covariance = self.config.filtering.initial_covariance_scale * np.eye(estimate.size)
            initial_observation = self._build_observation(
                self.config.filtering.observation, so.state.size, bs.position
            )
            filter_kind = self._resolved_filter_kind(initial_observation.is_linear)
            filter_cls = KalmanFilter if filter_kind == "kf" else ExtendedKalmanFilter
            self.filters[so.id] = filter_cls(
                estimate,
                covariance,
                state_projector=self._project_filter_state,
            )

    def _resolved_filter_kind(self, observation_is_linear: bool | None = None) -> str:
        requested = self.config.filtering.kind
        if observation_is_linear is None:
            observation_is_linear = self.config.filtering.observation.kind == "linear"
        if requested == "auto":
            return "kf" if observation_is_linear else "ekf"
        if requested == "kf" and not observation_is_linear:
            raise ValueError("KF was selected with a nonlinear observation model; use EKF or auto")
        return requested

    def _move_entity(self, entity: NetworkEntity, dynamics: DynamicsModel) -> None:
        rng = self.rngs.generator(f"mobility:{entity.id}")
        assert entity.state is not None
        entity.state = dynamics.sample_next(entity.state, rng)

        # This convention is used throughout: in toroidal mode the
        # physical spatial state itself lives in the fundamental domain.  This
        # keeps state, channel position, handover geometry and filter geometry
        # synchronized instead of maintaining incompatible wrapped/unwrapped
        # copies.  Non-spatial state coordinates (e.g. velocity) are untouched.
        entity.state = wrap_state(entity.state, self.config.network.region)
        entity.position = entity.state[:2].copy()

        if self.config.handover_enabled:
            old_bs = entity.serving_bs_index
            self.network.reassign_to_nearest_bs(
                entity, self.config.network, margin=self.config.handover_margin_m
            )

            if entity.serving_bs_index != old_bs:
                bs = self.network.serving_bs(entity)
                motion_cfg = (
                    self.config.ue_motion
                    if entity.kind == "ue"
                    else self.config.so_motion
                )
                factory = (
                    self.ue_dynamics_factory
                    if entity.kind == "ue"
                    else self.so_dynamics_factory
                )

                new_dynamics = self._build_dynamics(
                    factory, motion_cfg, center=bs.position
                )

                if entity.kind == "ue":
                    self.ue_dynamics[entity.id] = new_dynamics
                else:
                    self.so_dynamics[entity.id] = new_dynamics

    #def _move_entity(self, entity: NetworkEntity, dynamics: DynamicsModel) -> None:
    #    rng = self.rngs.generator(f"mobility:{entity.id}")
    #    assert entity.state is not None
    #    entity.state = dynamics.sample_next(entity.state, rng)
    #    entity.state[:2] = wrap_position(entity.state[:2], self.config.network.region)
    #    entity.position = entity.state[:2].copy()
    #    if self.config.handover_enabled:
    #        old_bs = entity.serving_bs_index
    #        self.network.reassign_to_nearest_bs(entity, self.config.network)
    #        if entity.serving_bs_index != old_bs:
    #            bs = self.network.serving_bs(entity)
    #            motion_cfg = self.config.ue_motion if entity.kind == "ue" else self.config.so_motion
    #            factory = self.ue_dynamics_factory if entity.kind == "ue" else self.so_dynamics_factory
    #            new_dynamics = factory(motion_cfg, self.config.time_step_s, center=bs.position)
    #            if entity.kind == "ue":
    #                self.ue_dynamics[entity.id] = new_dynamics
    #            else:
    #                self.so_dynamics[entity.id] = new_dynamics

    def _physical_link_gain(self, a: np.ndarray, b: np.ndarray, stream_name: str) -> float:
        """One configured physical-channel realization, before beamforming."""
        d = distance(a, b, self.config.network.region)
        return self.channel.power_gain(d, self.rngs.generator(stream_name))

    def _beam_gain(self, tx_bs_index: int, target_position: np.ndarray) -> float:
        bs = self.network.base_stations[tx_bs_index]
        beam_target = np.asarray(target_position, dtype=float)[:2]
        if self.config.network.region.distance_model == "toroidal":
            # Feed the beamformer the nearest periodic image of the target so
            # its angular direction uses the same minimum-image link as the
            # path-loss/channel calculation.
            beam_target = bs.position + toroidal_displacement(
                beam_target, bs.position, self.config.network.region
            )
        return self.beamformers[tx_bs_index].gain_linear(bs.position, beam_target)

    def _communication_link_gain(
        self,
        tx_bs_index: int,
        target_position: np.ndarray,
        stream_name: str,
    ) -> float:
        """Configured physical gain followed by the scalar sector-beam gain."""
        bs = self.network.base_stations[tx_bs_index]
        physical = self._physical_link_gain(bs.position, target_position, stream_name)
        return physical * self._beam_gain(tx_bs_index, target_position)

    def _sensing_signal_gain(self, so: NetworkEntity) -> float:
        """Serving sensing gain under the selected two-way convention.

        ``square_law`` squares the one-way propagation gain directly,
        ``(H d^-alpha)^2``.  ``radar_equation`` interprets ``radar_cross_section``
        as a physical RCS in m^2 and applies the monostatic normalization
        ``4*pi*sigma/lambda^2``.  ``auto`` selects the square law for the
        exponential channel and radar-equation for RT.

        Antenna directivity: the one-way link carries a single sector-beam
        factor; the two-way (monostatic) echo BS -> object -> BS traverses the
        same serving-BS sector antenna on both transmit and receive, so the beam
        gain enters squared.  ``beam == 1`` when beamforming is disabled, so this
        only affects runs with an active directional beam.
        """
        serving = self.network.serving_bs(so)
        physical = self._physical_link_gain(
            serving.position, so.position, "channel:sensing:signal"
        )
        beam = self._beam_gain(serving.index, so.position)

        if not self.config.channel.sensing_two_way:
            return physical * beam

        gain_model = self.config.channel.sensing_gain_model
        if gain_model == "auto":
            gain_model = (
                "square_law"
                if str(self.config.channel.model).strip().lower() == "exponential"
                else "radar_equation"
            )

        if gain_model == "square_law":
            return physical**2 * beam**2

        c = self.config.channel.speed_of_light_mps
        f_c = self.config.channel.carrier_frequency_hz
        wavelength = c / f_c
        sigma = self.config.channel.radar_cross_section
        radar_correction = 4.0 * np.pi * sigma / wavelength**2
        return physical**2 * radar_correction * beam**2

    def _communication_sinr(self, ue: NetworkEntity) -> tuple[float, float, float]:
        serving = self.network.serving_bs(ue)
        signal_gain =  self._communication_link_gain(serving.index, ue.position, "channel:communication:signal")
        signal = self.tx_power_w * signal_gain
        interference = 0.0
        for bs in self.network.base_stations:
            if bs.index == serving.index:
                continue
            gain = self._communication_link_gain(
                bs.index, ue.position, "channel:communication:interference"
            )
            interference += self.tx_power_w * gain
        denominator = interference + self.noise_power_w
        sinr = signal / max(denominator, self.config.channel.epsilon)
        return float(sinr), float(interference), float(signal_gain)

    def _sensing_interference_by_bs(self) -> np.ndarray:
        values = np.zeros(len(self.network.base_stations), dtype=float)
        for serving in self.network.base_stations:
            interference = 0.0
            for bs in self.network.base_stations:
                if bs.index == serving.index:
                    continue
                # Sensing interference uses the same one-way transmitted signal
                # model toward the serving BS, not the radar-return gain.
                gain = self._communication_link_gain(
                    bs.index, serving.position, "channel:sensing:interference"
                )
                interference += self.tx_power_w * gain
            values[serving.index] = interference
        return values

    def _sensing_sinr(self, so: NetworkEntity, interference_w: float) -> tuple[float, float]:
        sensing_gain = self._sensing_signal_gain(so)
        signal = self.tx_power_w * sensing_gain
        denominator = interference_w + self.noise_power_w
        return (
            float(signal / max(denominator, self.config.channel.epsilon)),
            float(sensing_gain),
        )

    def run(self) -> LargeScaleSimulationResult:
        t_max = self.config.horizon
        n_bs = len(self.network.base_stations)
        queue_workloads = {ue.id: np.zeros(t_max + 1) for ue in self.network.ues}
        comm_sinr = {ue.id: np.full(t_max, np.nan) for ue in self.network.ues}
        comm_interf = {ue.id: np.full(t_max, np.nan) for ue in self.network.ues}
        comm_eff_gain = {ue.id: np.full(t_max, np.nan) for ue in self.network.ues}
        sensing_sinr = {so.id: np.full(t_max, np.nan) for so in self.network.sensing_objects}
        sensing_eff_gain = {so.id: np.full(t_max, np.nan) for so in self.network.sensing_objects}
        cov_norms = {so.id: np.zeros(t_max + 1, dtype=float) for so in self.network.sensing_objects}
        cov_traces = {so.id: np.zeros(t_max + 1, dtype=float) for so in self.network.sensing_objects}
        ue_trajectories = {ue.id: np.zeros((t_max + 1, 2), dtype=float) for ue in self.network.ues}
        so_trajectories = {so.id: np.zeros((t_max + 1, 2), dtype=float) for so in self.network.sensing_objects}
        for ue in self.network.ues:
            ue_trajectories[ue.id][0] = ue.position
        for so in self.network.sensing_objects:
            covariance = self.filters[so.id].covariance
            cov_norms[so.id][0] = float(np.linalg.norm(covariance, ord=2))
            cov_traces[so.id][0] = float(np.trace(covariance))
            so_trajectories[so.id][0] = so.position

        sensing_interf_history = np.full((t_max, n_bs), np.nan, dtype=float)
        phase_names: list[str] = []
        communication_active = np.zeros(t_max, dtype=bool)
        sensing_active = np.zeros(t_max, dtype=bool)
        beam_indices = np.full((t_max, n_bs), -1, dtype=int)

        # Serving BS of each entity at every tick (its value right after that
        # tick's move / handover).  With handover_enabled this is time-varying,
        # so per-BS metric aggregation must use it rather than the final
        # serving_bs_index.  ``*_serving_initial`` holds the pre-loop association
        # that the t=0 covariance / zero-workload samples belong to.
        ue_serving_initial = {ue.id: ue.serving_bs_index for ue in self.network.ues}
        so_serving_initial = {so.id: so.serving_bs_index for so in self.network.sensing_objects}
        ue_serving_history = {ue.id: np.empty(t_max, dtype=int) for ue in self.network.ues}
        so_serving_history = {so.id: np.empty(t_max, dtype=int) for so in self.network.sensing_objects}

        for t in range(t_max):
            # Logical time only: schedule and beam rotation are deterministic.
            state = self.scheduler.state_at(t)
            phase_names.append(state.phase_name)
            communication_active[t] = state.communication_active
            sensing_active[t] = state.sensing_active
            # Record the beam that is active *during* this tick (rotated at the
            # end of the previous tick); the configured/initial index is the one
            # used at t=0.
            if self.config.beamforming.enabled:
                for bs in self.network.base_stations:
                    beam_indices[t, bs.index] = self.beamformers[bs.index].current_beam_index

            # Snapshot each SO's dynamics *before* the move: it is the process
            # model that governs the t-1 -> t transition.  If this tick triggers
            # a handover the model is rebuilt (re-centred on the new serving BS)
            # inside _move_entity, but the filter's predict for tick t must use
            # this pre-handover model to match the transition that occurred.
            so_filter_dynamics: dict[str, DynamicsModel] = {}
            for ue in self.network.ues:
                self._move_entity(ue, self.ue_dynamics[ue.id])
                ue_trajectories[ue.id][t + 1] = ue.position
                ue_serving_history[ue.id][t] = ue.serving_bs_index
            for so in self.network.sensing_objects:
                so_filter_dynamics[so.id] = self.so_dynamics[so.id]
                self._move_entity(so, self.so_dynamics[so.id])
                so_trajectories[so.id][t + 1] = so.position
                so_serving_history[so.id][t] = so.serving_bs_index

            for ue in self.network.ues:
                q_rng = self.rngs.generator(f"queue:{ue.id}")
                if state.communication_active:
                    sinr, interference, effective_gain = self._communication_sinr(ue)
                    comm_sinr[ue.id][t] = sinr
                    comm_interf[ue.id][t] = interference
                    comm_eff_gain[ue.id][t] = effective_gain
                    queue_workloads[ue.id][t + 1] = self.queues[ue.id].update(
                        sinr, q_rng, service_enabled=True
                    )
                else:
                    queue_workloads[ue.id][t + 1] = self.queues[ue.id].update(
                        0.0, q_rng, service_enabled=False
                    )

            if state.sensing_active:
                sensing_interference = self._sensing_interference_by_bs()
                sensing_interf_history[t] = sensing_interference
            else:
                sensing_interference = np.full(n_bs, np.nan, dtype=float)

            for so in self.network.sensing_objects:
                bs = self.network.serving_bs(so)
                if state.sensing_active:
                    sinr, effective_gain = self._sensing_sinr(
                        so, sensing_interference[bs.index]
                    )
                    sensing_sinr[so.id][t] = sinr
                    sensing_eff_gain[so.id][t] = effective_gain
                    obs_cfg = self.config.filtering.observation
                    observation_model = self._build_observation(
                        obs_cfg, so.state.size, bs.position
                    )
                    r = measurement_covariance(obs_cfg, sinr)
                    measurement_rng = self.rngs.generator(f"measurement:{so.id}")
                    measurement = observation_model.observe(so.state) + measurement_rng.multivariate_normal(
                        np.zeros(r.shape[0]), r
                    )
                    _, covariance = self.filters[so.id].step(
                        so_filter_dynamics[so.id], observation_model, measurement, r
                    )
                else:
                    _, covariance = self.filters[so.id].predict_only(
                        so_filter_dynamics[so.id]
                    )
                cov_norms[so.id][t + 1] = float(np.linalg.norm(covariance, ord=2))
                cov_traces[so.id][t + 1] = float(np.trace(covariance))

            # Rotate every BS beam once per tick, after this tick's work, so the
            # configured initial_beam_index is the beam actually used at t=0.
            self._advance_beams()

        bs_metrics = self._aggregate_bs_metrics(
            queue_workloads,
            comm_sinr,
            comm_interf,
            sensing_sinr,
            cov_norms,
            cov_traces,
            sensing_interf_history,
            ue_serving_history,
            so_serving_history,
            ue_serving_initial,
            so_serving_initial,
        )
        association = self._association_metrics(bs_metrics)
        return LargeScaleSimulationResult(
            mode="non_cooperative",
            network=self.network,
            queue_workloads=queue_workloads,
            communication_sinr=comm_sinr,
            sensing_sinr=sensing_sinr,
            covariance_norms=cov_norms,
            covariance_traces=cov_traces,
            ue_trajectories=ue_trajectories,
            so_trajectories=so_trajectories,
            simulation_times=np.arange(t_max + 1, dtype=float) * self.config.time_step_s,
            region=self.config.network.region,
            bs_metrics=bs_metrics,
            association=association,
            tdd_phase_names=tuple(phase_names),
            communication_active=communication_active,
            sensing_active=sensing_active,
            beam_indices=beam_indices,
            communication_effective_gains=comm_eff_gain,
            sensing_effective_gains=sensing_eff_gain,
            metadata={
                "master_seed": self.config.master_seed,
                "filter_kind": (
                    "kf" if isinstance(next(iter(self.filters.values())), KalmanFilter) else "ekf"
                ) if self.filters else None,
                "channel_model": self.config.channel.model,
                "beamforming_enabled": self.config.beamforming.enabled,
                "beamforming_model": (
                    self.config.beamforming.model if self.config.beamforming.enabled else "unity"
                ),
                "tdd_enabled": self.config.tdd.enabled,
                "tdd_frame_length_slots": self.scheduler.frame_length_slots,
            },
        )

    @staticmethod
    def _safe_nanmean(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else float("nan")

    def _aggregate_bs_metrics(
        self,
        queue_workloads,
        comm_sinr,
        comm_interf,
        sensing_sinr,
        cov_norms,
        cov_traces,
        sensing_interf_history,
        ue_serving_history,
        so_serving_history,
        ue_serving_initial,
        so_serving_initial,
    ) -> dict[str, np.ndarray]:
        """Per-BS means with per-tick serving attribution.

        Every per-tick sample is credited to the BS that served the entity on
        that tick (``*_serving_history``); the ``t=0`` covariance and the
        zero-workload sample are credited to the pre-loop association
        (``*_serving_initial``).  With ``handover_enabled=False`` each entity
        keeps a single serving BS for the whole horizon, so this reduces to the
        previous static end-of-run grouping (identical up to summation order).
        """
        n_bs = len(self.network.base_stations)
        t_max = self.config.horizon

        keys = ("w", "xi_u", "sinr_com", "sigma", "sigma_trace", "sinr_sen")
        sums = {k: np.zeros(n_bs) for k in keys}
        counts = {k: np.zeros(n_bs) for k in keys}

        def add(key: str, bs_index: int, value: float) -> None:
            if np.isfinite(value):
                sums[key][bs_index] += value
                counts[key][bs_index] += 1.0

        for ue in self.network.ues:
            hist = ue_serving_history[ue.id]
            qw = queue_workloads[ue.id]
            ci = comm_interf[ue.id]
            cs = comm_sinr[ue.id]
            add("w", int(ue_serving_initial[ue.id]), qw[0])
            for t in range(t_max):
                b = int(hist[t])
                add("w", b, qw[t + 1])
                add("xi_u", b, ci[t])
                add("sinr_com", b, cs[t])

        for so in self.network.sensing_objects:
            hist = so_serving_history[so.id]
            cn = cov_norms[so.id]
            ct = cov_traces[so.id]
            ss = sensing_sinr[so.id]
            add("sigma", int(so_serving_initial[so.id]), cn[0])
            add("sigma_trace", int(so_serving_initial[so.id]), ct[0])
            for t in range(t_max):
                b = int(hist[t])
                add("sigma", b, cn[t + 1])
                add("sigma_trace", b, ct[t + 1])
                add("sinr_sen", b, ss[t])

        def finalize(key: str) -> np.ndarray:
            out = np.full(n_bs, np.nan)
            nonzero = counts[key] > 0
            out[nonzero] = sums[key][nonzero] / counts[key][nonzero]
            return out

        bar_xi_s = np.array(
            [self._safe_nanmean(sensing_interf_history[:, i]) for i in range(n_bs)]
        )

        return {
            "bar_workload": finalize("w"),
            "bar_covariance_norm": finalize("sigma"),
            "bar_covariance_trace": finalize("sigma_trace"),
            "bar_communication_interference": finalize("xi_u"),
            "bar_sensing_interference": bar_xi_s,
            "bar_communication_sinr": finalize("sinr_com"),
            "bar_sensing_sinr": finalize("sinr_sen"),
        }

    @staticmethod
    def _association_metrics(metrics: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
        pairs = {
            "interference": (
                metrics["bar_sensing_interference"],
                metrics["bar_communication_interference"],
            ),
            "sinr": (metrics["bar_communication_sinr"], metrics["bar_sensing_sinr"]),
            # Keep the summary metric identical to the scatter plot and to the
            # theoretical quantity of interest: Tr(Sigma), not ||Sigma||_2.
            "filter_queue": (metrics["bar_covariance_trace"], metrics["bar_workload"]),
        }
        return {
            name: {
                "association_ratio": association_ratio(x, y),
                "pearson": pearson_correlation(x, y),
            }
            for name, (x, y) in pairs.items()
        }


class JCASSimulator:
    """Single façade dispatching to mathematically distinct simulation strategies."""

    def __init__(
        self,
        config: SimulationConfig,
        *,
        ue_dynamics_factory: Callable = make_dynamics,
        so_dynamics_factory: Callable = make_dynamics,
        observation_factory: Callable = make_observation_model,
        channel: PhysicalChannel | None = None,
    ):
        config.validate()
        self.config = config
        self.ue_dynamics_factory = ue_dynamics_factory
        self.so_dynamics_factory = so_dynamics_factory
        self.observation_factory = observation_factory
        self.channel = channel

    def run(self) -> LargeScaleSimulationResult | NonCaptiveSimulationResult:
        if self.config.operation_mode == "non_cooperative":
            return LargeScaleJCASSimulator(
                self.config,
                ue_dynamics_factory=self.ue_dynamics_factory,
                so_dynamics_factory=self.so_dynamics_factory,
                observation_factory=self.observation_factory,
                channel=self.channel,
            ).run()
        if self.config.operation_mode == "non_captive":
            rng = RNGManager(self.config.master_seed).generator("non_captive:model")
            return run_supplied_non_captive_model(self.config.non_captive, rng)
        raise ValueError(f"unsupported operation mode: {self.config.operation_mode}")
