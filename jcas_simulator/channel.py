"""Configurable physical-channel models and registry.

Every channel model consumed by the large-scale simulator implements the same
minimal interface::

    power_gain(los_distance_m, rng) -> non-negative linear power gain

The simulation loop therefore remains independent of the concrete propagation
law.  Built-in models are ``rt`` (the ray-tracing-derived model, fitted to the
supplied UOULU D4.3 measurement campaign) and ``exponential`` (Rayleigh fading
with power-law path loss).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np

from .config import ChannelConfig


@runtime_checkable
class PhysicalChannel(Protocol):
    """Minimal interface required by the simulator's propagation layer."""

    def power_gain(self, los_distance_m: float, rng: np.random.Generator) -> float:
        """Return one non-negative linear power gain for a link distance."""
        ...


class RTChannel:
    """Synthetic multipath power-gain model fitted to UOULU ray tracing.

    The D4.3 mean vector and covariance matrix are retained exactly.  The
    fitted bivariate Gaussian is used for synthetic (excess delay, excess
    power gain) pairs.  Because physical excess delay is non-negative while
    an ordinary Gaussian has support on the whole real line, the default
    sampler rejects a *whole joint pair* whenever its excess-delay component
    is negative and redraws that pair from the same fitted Gaussian, keeping
    the generated propagation paths in the physical support implied by the
    D4.3 definition ``r = r_LOS + c * tau_ex``.
    """

    def __init__(self, config: ChannelConfig):
        self.config = config
        self._mean = np.array([config.mean_delay_s, config.mean_gain_db], dtype=float)
        self._covariance = np.array(
            [
                [config.covariance_delay_delay, config.covariance_delay_gain],
                [config.covariance_delay_gain, config.covariance_gain_gain],
            ],
            dtype=float,
        )

    def path_count(self, los_distance_m: float) -> int:
        """Rounded fitted number of paths."""
        raw = self.config.path_count_prefactor * np.exp(
            -self.config.path_count_decay_per_m * max(float(los_distance_m), 0.0)
        )
        return max(int(self.config.min_paths), int(np.rint(raw)))

    def _sample_joint_paths(
        self,
        n_paths: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw D4.3 ``(excess delay, excess gain dB)`` pairs.

        If ``rt_reject_negative_delay`` is enabled, every sampled pair with a
        negative excess delay is discarded in its entirety and redrawn.  This
        is rejection sampling from the fitted joint Gaussian conditioned on
        non-negative excess delay; importantly, the gain paired with a rejected
        delay is rejected too, so the fitted delay/gain dependence is not
        broken by component-wise clipping.
        """
        n_paths = int(n_paths)
        if n_paths <= 0:
            return np.empty((0, 2), dtype=float)

        samples = rng.multivariate_normal(self._mean, self._covariance, size=n_paths)
        if not self.config.rt_reject_negative_delay:
            return samples

        negative = samples[:, 0] < 0.0
        rounds = 0
        while np.any(negative):
            rounds += 1
            if rounds > self.config.rt_max_rejection_rounds:
                raise RuntimeError(
                    "RT excess-delay rejection sampling did not converge; "
                    "check the configured D4.3 delay mean/covariance."
                )
            samples[negative] = rng.multivariate_normal(
                self._mean,
                self._covariance,
                size=int(np.count_nonzero(negative)),
            )
            negative = samples[:, 0] < 0.0

        return samples

    def combine_paths(
        self,
        los_distance_m: float,
        delays_s: np.ndarray,
        gains_db: np.ndarray,
        phases_rad: np.ndarray,
    ) -> float:
        """Evaluate the D4.3 channel equation for deterministic path inputs.

        ``delays_s`` is interpreted as the excess delay tau_ex.  No logarithm,
        clipping, or other transform is applied here.  Stochastic non-negative
        support is enforced upstream by ``_sample_joint_paths``.
        """
        delays = np.asarray(delays_s, dtype=float).reshape(-1)
        gains_db = np.asarray(gains_db, dtype=float).reshape(-1)
        phases_rad = np.asarray(phases_rad, dtype=float).reshape(-1)
        if not (len(delays) == len(gains_db) == len(phases_rad)):
            raise ValueError("delay, gain, and phase arrays must have the same length")
        if len(delays) == 0:
            return 0.0

        r_los = max(float(los_distance_m), 0.0)
        r_total = r_los + delays * self.config.speed_of_light_mps
        if np.any(~np.isfinite(r_total)) or np.any(r_total <= 0.0):
            raise ValueError(
                "RT path distance must be strictly positive: "
                "r = r_LOS + c * tau_ex. Enable rt_reject_negative_delay "
                "for synthetic D4.3 sampling."
            )

        gains_linear = np.power(10.0, gains_db / 10.0)
        fspl = (
            self.config.speed_of_light_mps
            / (4.0 * np.pi * r_total * self.config.carrier_frequency_hz)
        ) ** 2
        phase_factors = np.exp(-1j * phases_rad)
        h = np.sum(np.sqrt(fspl * gains_linear) * phase_factors)
        return float(np.abs(h) ** 2)

    def power_gain(self, los_distance_m: float, rng: np.random.Generator) -> float:
        """Draw one channel power gain for a LOS separation in metres."""
        r_los = max(float(los_distance_m), 0.0)
        n_paths = self.path_count(r_los)
        if n_paths == 0:
            return 0.0

        samples = self._sample_joint_paths(n_paths, rng)
        phases_rad = 2.0 * np.pi * rng.random(n_paths)
        return self.combine_paths(
            r_los,
            samples[:, 0],
            samples[:, 1],
            phases_rad,
        )


def dbm_to_watts(dbm: float) -> float:
    return float(10.0 ** ((dbm - 30.0) / 10.0))


def noise_power_watts(config: ChannelConfig) -> float:
    return dbm_to_watts(config.noise_psd_dbm_per_hz) * config.bandwidth_hz


class ExponentialPowerLawChannel:
    """Rayleigh-fading channel: exponential power gain times regularized power-law loss.

    The returned gain is

        H * max(d_min, d)^(-alpha),   H ~ Exp(mean)

    An exponentially distributed power gain ``H`` is exactly Rayleigh fading;
    this is the same one-way channel law used by the original simulator. The
    existing sensing code remains responsible for squaring this one-way gain
    when ``sensing_two_way=True``; this preserves that sensing convention
    without duplicating sensing logic inside the channel class.
    """

    def __init__(self, config: ChannelConfig):
        self.config = config

    def power_gain(self, los_distance_m: float, rng: np.random.Generator) -> float:
        d = max(
            float(self.config.exponential_min_distance_m),
            float(los_distance_m),
        )
        fading = float(rng.exponential(scale=self.config.exponential_fading_mean))
        return fading * d ** (-self.config.exponential_path_loss_exponent)


ChannelFactory = Callable[[ChannelConfig], PhysicalChannel]
_CHANNEL_FACTORIES: dict[str, ChannelFactory] = {}


def register_channel_model(
    name: str,
    factory: ChannelFactory,
    *,
    overwrite: bool = False,
) -> None:
    """Register a config-selectable physical channel model.

    Example::

        register_channel_model("my_channel", lambda cfg: MyChannel(cfg))
        config = replace(config, channel=replace(config.channel, model="my_channel"))

    The model only needs a ``power_gain(distance, rng)`` method.  Direct object
    injection through ``JCASSimulator(..., channel=my_channel)`` remains
    available and does not require registration.
    """

    key = str(name).strip().lower()
    if not key:
        raise ValueError("channel model name must be non-empty")
    if key in _CHANNEL_FACTORIES and not overwrite:
        raise ValueError(f"channel model {name!r} is already registered")
    _CHANNEL_FACTORIES[key] = factory


def available_channel_models() -> tuple[str, ...]:
    """Return the currently registered config-selectable channel names."""

    return tuple(sorted(_CHANNEL_FACTORIES))


def make_channel(config: ChannelConfig) -> PhysicalChannel:
    """Construct the configured physical channel model."""

    key = str(config.model).strip().lower()
    try:
        channel = _CHANNEL_FACTORIES[key](config)
    except KeyError as exc:
        choices = ", ".join(available_channel_models()) or "<none>"
        raise ValueError(
            f"unknown channel model {config.model!r}; registered models: {choices}"
        ) from exc
    if not isinstance(channel, PhysicalChannel):
        raise TypeError(
            f"channel factory {key!r} returned an object without "
            "power_gain(distance, rng)"
        )
    return channel


# Built-ins.  Registration happens here so adding another model does not
# require changing the simulator orchestration.
register_channel_model("rt", RTChannel)
register_channel_model("exponential", ExponentialPowerLawChannel)
