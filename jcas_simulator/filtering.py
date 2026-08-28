"""Parametric Kalman and Extended Kalman filtering."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
import numpy as np

from .config import RegionConfig
from .geometry import relative_state, toroidal_displacement
from .mobility import DynamicsModel


def wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


class ObservationModel(ABC):
    is_linear: bool = False

    @abstractmethod
    def observe(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def innovation(self, measurement: np.ndarray, predicted_measurement: np.ndarray) -> np.ndarray:
        return np.asarray(measurement, dtype=float) - np.asarray(predicted_measurement, dtype=float)


class CallableObservation(ObservationModel):
    """User-supplied observation function and Jacobian.

    ``innovation_fn`` can be supplied for wrapped/circular measurements or any
    other non-Euclidean residual convention.
    """

    def __init__(self, observe_fn, jacobian_fn, innovation_fn=None, *, is_linear: bool = False):
        self._observe_fn = observe_fn
        self._jacobian_fn = jacobian_fn
        self._innovation_fn = innovation_fn
        self.is_linear = bool(is_linear)

    def observe(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(self._observe_fn(np.asarray(state, dtype=float)), dtype=float)

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(self._jacobian_fn(np.asarray(state, dtype=float)), dtype=float)

    def innovation(self, measurement: np.ndarray, predicted_measurement: np.ndarray) -> np.ndarray:
        if self._innovation_fn is None:
            return super().innovation(measurement, predicted_measurement)
        return np.asarray(
            self._innovation_fn(
                np.asarray(measurement, dtype=float),
                np.asarray(predicted_measurement, dtype=float),
            ),
            dtype=float,
        )


class LinearObservation(ObservationModel):
    """Linear observation of state relative to an optional sensor reference.

    On a torus the first two relative coordinates use minimum-image
    displacement, matching ``H @ disp(x-bs)``.  The Jacobian
    remains the constant matrix almost everywhere on the torus.
    """

    is_linear = True

    def __init__(
        self,
        matrix: np.ndarray,
        reference: np.ndarray | None = None,
        region: RegionConfig | None = None,
    ):
        self.matrix = np.asarray(matrix, dtype=float)
        if self.matrix.ndim != 2:
            raise ValueError("linear observation matrix must be 2-D")
        self.reference = (
            np.zeros(self.matrix.shape[1], dtype=float)
            if reference is None
            else np.asarray(reference, dtype=float)
        )
        if self.reference.shape != (self.matrix.shape[1],):
            raise ValueError("linear observation reference has incompatible shape")
        self.region = region

    def observe(self, state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        if self.region is not None and self.region.distance_model == "toroidal":
            rel = relative_state(x, self.reference, self.region)
        else:
            rel = x - self.reference
        return self.matrix @ rel

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        return self.matrix.copy()


class RangeBearingObservation(ObservationModel):
    """2-D range/bearing, optionally with radial velocity.

    In toroidal mode range and bearing are computed from the minimum-image
    sensor-to-target displacement, so the observation and propagation layers
    use exactly the same physical relative geometry.
    """

    def __init__(
        self,
        sensor_position: np.ndarray,
        state_dim: int,
        include_range_rate: bool = False,
        region: RegionConfig | None = None,
    ):
        if state_dim not in (2, 4):
            raise ValueError("range/bearing observation supports state_dim 2 or 4")
        if include_range_rate and state_dim < 4:
            raise ValueError("range-rate observation requires a 4-D [x,y,vx,vy] state")
        self.sensor_position = np.asarray(sensor_position, dtype=float)[:2]
        self.state_dim = state_dim
        self.include_range_rate = include_range_rate
        self.region = region

    def _displacement(self, state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        if self.region is not None and self.region.distance_model == "toroidal":
            return toroidal_displacement(x[:2], self.sensor_position, self.region)
        return x[:2] - self.sensor_position

    def observe(self, state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        displacement = self._displacement(x)
        r = max(float(np.linalg.norm(displacement)), 1.0e-12)
        bearing = float(np.arctan2(displacement[1], displacement[0]))
        if not self.include_range_rate:
            return np.array([r, bearing], dtype=float)
        velocity = x[2:4]
        range_rate = float(displacement @ velocity / r)
        return np.array([r, bearing, range_rate], dtype=float)

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        d = self._displacement(x)
        r2 = max(float(d @ d), 1.0e-12)
        r = np.sqrt(r2)
        rows = 3 if self.include_range_rate else 2
        h = np.zeros((rows, self.state_dim), dtype=float)
        h[0, :2] = d / r
        h[1, 0] = -d[1] / r2
        h[1, 1] = d[0] / r2
        if self.include_range_rate:
            v = x[2:4]
            alpha = float(d @ v)
            h[2, :2] = v / r - alpha * d / (r ** 3)
            h[2, 2:4] = d / r
        return h

    def innovation(self, measurement: np.ndarray, predicted_measurement: np.ndarray) -> np.ndarray:
        residual = super().innovation(measurement, predicted_measurement)
        residual[1] = wrap_angle(residual[1])
        return residual


class BaseGaussianFilter(ABC):
    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        *,
        state_projector: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self._state_projector = state_projector
        state = np.asarray(initial_state, dtype=float).copy()
        self.state = self._project_state(state)
        self.covariance = np.asarray(initial_covariance, dtype=float).copy()
        if self.covariance.shape != (self.state.size, self.state.size):
            raise ValueError("initial covariance has incompatible shape")
        self.estimated_states: list[np.ndarray] = [self.state.copy()]
        self.covariance_states: list[np.ndarray] = [self.covariance.copy()]

    def _project_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        if self._state_projector is None:
            return state.copy()
        return np.asarray(self._state_projector(state), dtype=float)

    def predict(self, dynamics: DynamicsModel) -> tuple[np.ndarray, np.ndarray]:
        f = dynamics.jacobian(self.state)
        predicted_state = self._project_state(dynamics.transition(self.state))
        predicted_covariance = f @ self.covariance @ f.T + dynamics.process_covariance()
        predicted_covariance = 0.5 * (predicted_covariance + predicted_covariance.T)
        return predicted_state, predicted_covariance

    def predict_only(self, dynamics: DynamicsModel) -> tuple[np.ndarray, np.ndarray]:
        """Advance the filter without a measurement update."""
        predicted_state, predicted_covariance = self.predict(dynamics)
        self.state = predicted_state
        self.covariance = predicted_covariance
        self.estimated_states.append(self.state.copy())
        self.covariance_states.append(self.covariance.copy())
        return self.state.copy(), self.covariance.copy()

    @abstractmethod
    def step(
        self,
        dynamics: DynamicsModel,
        observation: ObservationModel,
        measurement: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _update_from_prediction(
        self,
        predicted_state: np.ndarray,
        predicted_covariance: np.ndarray,
        observation: ObservationModel,
        measurement: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        h = observation.jacobian(predicted_state)
        predicted_measurement = observation.observe(predicted_state)
        innovation = observation.innovation(measurement, predicted_measurement)
        r = np.asarray(measurement_covariance, dtype=float)
        s = h @ predicted_covariance @ h.T + r

        # K = P H^T S^{-1}, using a solve rather than an explicit inverse.
        gain = np.linalg.solve(s, h @ predicted_covariance).T
        updated_state = self._project_state(predicted_state + gain @ innovation)

        # Joseph form: numerically robust and PSD-preserving up to roundoff.
        identity = np.eye(predicted_covariance.shape[0])
        correction = identity - gain @ h
        updated_covariance = (
            correction @ predicted_covariance @ correction.T + gain @ r @ gain.T
        )
        updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)

        self.state = updated_state
        self.covariance = updated_covariance
        self.estimated_states.append(self.state.copy())
        self.covariance_states.append(self.covariance.copy())
        return self.state.copy(), self.covariance.copy()


class KalmanFilter(BaseGaussianFilter):
    """Standard KF for locally linear dynamics and a linear observation.

    The toroidal projection is nonlinear only at the fundamental-domain cut;
    away from that measure-zero cut the local Jacobian is the ordinary linear
    Jacobian.
    """

    def step(self, dynamics, observation, measurement, measurement_covariance):
        if not dynamics.is_linear:
            raise ValueError("KalmanFilter requires linear dynamics")
        if not observation.is_linear:
            raise ValueError("KalmanFilter requires a linear observation model")
        x_pred, p_pred = self.predict(dynamics)
        return self._update_from_prediction(x_pred, p_pred, observation, measurement, measurement_covariance)


class ExtendedKalmanFilter(BaseGaussianFilter):
    """EKF supporting linear or nonlinear observations/dynamics."""

    def step(self, dynamics, observation, measurement, measurement_covariance):
        x_pred, p_pred = self.predict(dynamics)
        return self._update_from_prediction(x_pred, p_pred, observation, measurement, measurement_covariance)
