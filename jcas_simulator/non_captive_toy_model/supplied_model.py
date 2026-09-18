"""Refactored wrapper around the supplied non-captive JCAS tracking model.

This module preserves the equations and RNG draw ordering of the supplied
reference implementation while using stable linear solves/Joseph covariance
updates.  The supplied model is a JCAS-vs-sensing-only tracking experiment;
it does not contain RTChannel, sector beamforming, an explicit TDD frame, or
a multi-BS coherent combining model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..config import NonCaptiveToyModelConfig


@dataclass
class NonCaptiveToyModelSimulationResult:
    mode: str
    true_state: np.ndarray
    jcas_estimate: np.ndarray
    sensing_only_estimate: np.ndarray
    jcas_covariance: np.ndarray
    sensing_only_covariance: np.ndarray
    relative_state: np.ndarray
    selected_bs_position: np.ndarray
    position_error_jcas: np.ndarray
    position_error_sensing_only: np.ndarray
    gain_error_jcas: np.ndarray
    smoothed_position_error_jcas: np.ndarray
    smoothed_position_error_sensing_only: np.ndarray
    percent_error: np.ndarray
    jcas_snr: np.ndarray
    sensing_only_snr: np.ndarray
    measurement_snr_true: np.ndarray
    simulation_times: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)


def moving_mean(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return x.copy()
    window = max(1, int(window))
    before = window // 2
    after = window - before - 1
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - before)
        hi = min(n, i + after + 1)
        out[i] = float(np.mean(x[lo:hi]))
    return out


def run_supplied_non_captive_toy_model(
    config: NonCaptiveToyModelConfig,
    rng: np.random.Generator,
) -> NonCaptiveToyModelSimulationResult:
    """Run the supplied JCAS-vs-sensing-only tracking experiment.

    Source semantics preserved:
    * both estimators — the joint (J) filter and the position-only
      sensing-only baseline (S) — select their own serving BS every tick
      from their own a-priori (predicted) position estimate; the true
      position is unobserved and is never used for association by either
      branch, so the two selections generally differ whenever J and S have
      diverged onto opposite sides of a cell boundary;
    * each branch's actual-measurement noise is scaled by the SNR at that
      branch's own true-vs-selected-BS relative state, and each branch's
      filter ``R`` is scaled by the SNR at that branch's own
      predicted-vs-selected-BS relative state;
    * the baseline's gain channel is never corrected: the second row of its
      Kalman gain ``K_bl`` is forced to zero, so its gain estimate simply
      decays as ``gain_rho`` towards the unconditional mean and the
      gain-covariance entry of ``sensing_only_covariance`` never shrinks,
      only grows through process noise;
    * process/measurement random draws occur in the same order and count as
      the source: 2 draws for the process noise, 2 for the JCAS measurement,
      1 for the baseline position measurement.

    The covariance update below uses the numerically-stable Joseph form,
    ``P = (I-K) P (I-K)^T + K R K^T``, in place of the source's
    ``P = (I-K) P``; for the optimal Kalman gain computed here the two are
    algebraically identical, the Joseph form is only more robust to
    floating-point round-off.
    """
    t_max = int(config.horizon)
    if t_max < 2:
        raise ValueError("non-captive horizon must be at least 2")

    bs_positions = config.delta * t_max * np.asarray(
        config.bs_relative_positions, dtype=float
    )
    z_matrix = np.array([[1.0, 0.0], [0.0, config.gain_rho]], dtype=float)
    deterministic_input = np.array([config.delta, 0.0], dtype=float)
    sigma_x = np.diag(
        [config.position_process_variance, config.gain_process_variance]
    )

    def dist(p: float) -> float:
        return float(np.sqrt(p * p + config.distance_epsilon))

    def snr(relative_state: np.ndarray) -> float:
        return max(
            dist(float(relative_state[0])) ** (-config.path_loss_exponent)
            * dist(float(relative_state[1])),
            1.0e-12,
        )

    def nearest_bs(position: float) -> float:
        return float(bs_positions[int(np.argmin(np.abs(position - bs_positions)))])

    x = np.zeros((2, t_max))
    x[:, 0] = [
        0.0,
        np.sqrt(config.gain_process_variance) * rng.standard_normal(),
    ]
    x_post = np.zeros_like(x)
    x_post[:, 0] = x[:, 0]
    x_post_bl = np.zeros_like(x)
    x_post_bl[:, 0] = x[:, 0]
    x_prior_bl = np.zeros_like(x)
    relative = np.zeros_like(x)
    selected_bs = np.zeros(t_max)
    p_jcas = np.zeros((2, 2, t_max))
    p_bl = np.zeros((2, 2, t_max))
    jcas_snr = np.full(t_max, np.nan, dtype=float)
    sensing_only_snr = np.full(t_max, np.nan, dtype=float)
    measurement_snr_true = np.full(t_max, np.nan, dtype=float)
    identity = np.eye(2)

    for t in range(1, t_max):
        # Same two scalar draws/order as the source implementation.
        process_noise = np.array(
            [
                np.sqrt(config.position_process_variance) * rng.standard_normal(),
                np.sqrt(config.gain_process_variance) * rng.standard_normal(),
            ]
        )
        x[:, t] = z_matrix @ x[:, t - 1] + deterministic_input + process_noise

        # A-priori (predicted) state/covariance for both branches, computed
        # before BS association: each branch's serving BS is selected from
        # its OWN a-priori position estimate -- the true position is
        # unobserved and is never used for association, by either branch.
        x_prior = z_matrix @ x_post[:, t - 1] + deterministic_input
        p_prior = z_matrix @ p_jcas[:, :, t - 1] @ z_matrix.T + sigma_x
        x_prior_bl[:, t] = z_matrix @ x_post_bl[:, t - 1] + deterministic_input
        p_prior_bl = z_matrix @ p_bl[:, :, t - 1] @ z_matrix.T + sigma_x

        # Two independent nearest-BS associations (generally different BSs).
        selected_bs[t] = nearest_bs(x_prior[0])
        baseline_bs = nearest_bs(x_prior_bl[0, t])

        # Relative states, each branch strictly against its OWN selected BS.
        relative[:, t] = x[:, t] - np.array([selected_bs[t], 0.0])
        prior_relative = x_prior - np.array([selected_bs[t], 0.0])
        relative_bl = x[:, t] - np.array([baseline_bs, 0.0])
        prior_relative_bl = x_prior_bl[:, t] - np.array([baseline_bs, 0.0])

        # Actual measurements use each branch's TRUE relative-state SNR.
        measurement_snr_true[t] = snr(relative[:, t])
        y = x[:, t] + measurement_snr_true[t] ** (-0.5) * rng.standard_normal(2)

        snr_true_baseline = snr(relative_bl)
        baseline_position_measurement = (
            x[0, t] + snr_true_baseline ** (-0.5) * rng.standard_normal()
        )
        # The gain channel is never observed by the baseline: its entry in
        # y_bl below is a placeholder, made irrelevant by zeroing the gain
        # row of K_bl (see below), rather than by matching the innovation.
        y_bl = np.array([baseline_position_measurement, 0.0])

        # Filter R uses each branch's PREDICTED relative-state SNR.
        jcas_snr[t] = snr(prior_relative)
        r = np.eye(2) / jcas_snr[t]
        s = p_prior + r
        k = np.linalg.solve(s, p_prior).T
        innovation = y - x_prior
        x_post[:, t] = x_prior + k @ innovation
        c = identity - k
        p_jcas[:, :, t] = c @ p_prior @ c.T + k @ r @ k.T

        sensing_only_snr[t] = snr(prior_relative_bl)
        r_bl = np.eye(2) / sensing_only_snr[t]
        s_bl = p_prior_bl + r_bl
        k_bl = np.linalg.solve(s_bl, p_prior_bl).T
        # The gain channel is not observed by the baseline: its Kalman-gain
        # row is truncated to zero, so the gain estimate simply predicts
        # forward (gain_rho * previous estimate) and its covariance entry
        # never shrinks, only grows through process noise.
        k_bl[1, :] = 0.0
        innovation_bl = y_bl - x_prior_bl[:, t]
        x_post_bl[:, t] = x_prior_bl[:, t] + k_bl @ innovation_bl
        c_bl = identity - k_bl
        p_bl[:, :, t] = c_bl @ p_prior_bl @ c_bl.T + k_bl @ r_bl @ k_bl.T

    error_jcas = np.abs(x_post[0] - x[0])
    error_bl = np.abs(x_post_bl[0] - x[0])
    gain_error = np.abs(x_post[1] - x[1])
    smooth_jcas = moving_mean(error_jcas, config.smoothing_window)
    smooth_bl = moving_mean(error_bl, config.smoothing_window)
    with np.errstate(divide="ignore", invalid="ignore"):
        percent = np.where(
            smooth_bl > 0,
            (smooth_jcas - smooth_bl) / smooth_bl * 100.0,
            np.nan,
        )

    return NonCaptiveToyModelSimulationResult(
        mode="non_captive_toy_model",
        true_state=x,
        jcas_estimate=x_post,
        sensing_only_estimate=x_post_bl,
        jcas_covariance=p_jcas,
        sensing_only_covariance=p_bl,
        relative_state=relative,
        selected_bs_position=selected_bs,
        position_error_jcas=error_jcas,
        position_error_sensing_only=error_bl,
        gain_error_jcas=gain_error,
        smoothed_position_error_jcas=smooth_jcas,
        smoothed_position_error_sensing_only=smooth_bl,
        percent_error=percent,
        jcas_snr=jcas_snr,
        sensing_only_snr=sensing_only_snr,
        measurement_snr_true=measurement_snr_true,
        simulation_times=np.arange(t_max, dtype=float) * config.delta,
        metadata={
            "rt_channel_used": False,
            "beamforming_defined_in_source": False,
            "tdd_defined_in_source": False,
            "delta": config.delta,
            "horizon": t_max,
        },
    )
