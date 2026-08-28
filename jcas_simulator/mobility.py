"""Parametric UE/SO state and mobility models."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from .config import MotionConfig, RegionConfig
from .geometry import relative_state, wrap_state


class DynamicsModel(ABC):
    """State-transition interface shared by simulation and filtering."""

    state_dim: int
    is_linear: bool = True

    @abstractmethod
    def transition(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def process_covariance(self) -> np.ndarray:
        raise NotImplementedError

    def sample_next(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mean = self.transition(state)
        q = self.process_covariance()
        if np.allclose(q, 0.0):
            return mean
        return rng.multivariate_normal(mean=mean, cov=q)


class StaticDynamics(DynamicsModel):
    def __init__(self, state_dim: int, region: RegionConfig | None = None):
        self.state_dim = state_dim
        self._q = np.zeros((state_dim, state_dim), dtype=float)
        self.region = region

    def transition(self, state: np.ndarray) -> np.ndarray:
        out = np.asarray(state, dtype=float).copy()
        if self.region is not None:
            out = wrap_state(out, self.region)
        return out

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        return np.eye(self.state_dim)

    def process_covariance(self) -> np.ndarray:
        return self._q.copy()


class LinearGaussianDynamics(DynamicsModel):
    def __init__(
        self,
        matrix: np.ndarray,
        covariance: np.ndarray,
        offset: np.ndarray | None = None,
        region: RegionConfig | None = None,
    ):
        matrix = np.asarray(matrix, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("transition matrix must be square")
        if covariance.shape != matrix.shape:
            raise ValueError("process covariance must match transition matrix")
        self.matrix = matrix
        self.covariance = covariance
        self.state_dim = matrix.shape[0]
        self.offset = np.zeros(self.state_dim) if offset is None else np.asarray(offset, dtype=float)
        if self.offset.shape != (self.state_dim,):
            raise ValueError("offset has incompatible shape")
        self.region = region

    def transition(self, state: np.ndarray) -> np.ndarray:
        out = self.matrix @ np.asarray(state, dtype=float) + self.offset
        if self.region is not None:
            out = wrap_state(out, self.region)
        return out

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        return self.matrix.copy()

    def process_covariance(self) -> np.ndarray:
        return self.covariance.copy()


class GaussMarkovDynamics(LinearGaussianDynamics):
    """Stable AR(1) dynamics, optionally centred on a serving BS.

    In toroidal mode, a BS-centred AR(1) acts on the minimum-image displacement
    from the serving BS:

        x' = wrap(c + rho * disp(x-c)).

    This avoids a boundary crossing being interpreted as a displacement of
    almost one full simulation-window width.
    """

    def __init__(
        self,
        config: MotionConfig,
        center: np.ndarray | None = None,
        region: RegionConfig | None = None,
    ):
        d = int(config.state_dim)
        a = config.rho * np.eye(d)
        q = (config.process_noise_std ** 2) * np.eye(d)
        c = np.zeros(d, dtype=float)
        self.center_on_serving_bs = bool(config.center_on_serving_bs and center is not None)
        toroidal = region is not None and region.distance_model == "toroidal"
        if self.center_on_serving_bs:
            c[:2] = np.asarray(center, dtype=float)[:2]
        elif toroidal:
            # On a torus an AR(1) contraction has to act on the minimum-image
            # displacement from a *fixed* reference point.  Without a serving BS
            # the only translation-invariant choice is the fundamental-domain
            # centre.  Falling back to ``wrap(A x)`` on the raw wrapped
            # coordinate would make the transition depend on which periodic
            # image of the state is supplied (e.g. x and x - width map to
            # different successors), which is not a valid dynamics on the torus.
            c[:2] = (region.center_x, region.center_y)
        self.center = c
        # In toroidal mode the transition is always evaluated in the local
        # minimum-image frame around ``self.center`` (see ``transition``).  The
        # Euclidean affine ``offset`` below is only consulted off the torus.
        self._toroidal_centered = bool(toroidal)
        # Euclidean affine form: x' = c + A(x-c) = A x + (I-A)c.
        offset = (np.eye(d) - a) @ c
        super().__init__(matrix=a, covariance=q, offset=offset, region=region)

    def transition(self, state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        if self._toroidal_centered:
            # Build the local lift of x around the serving BS.  Spatial
            # coordinates use minimum-image displacement; non-spatial state
            # coordinates retain their ordinary relative value.
            local = relative_state(x, self.center, self.region)
            out = self.center + self.matrix @ local
            return wrap_state(out, self.region)
        return super().transition(x)


class ConstantSpeedDynamics(LinearGaussianDynamics):
    def __init__(self, config: MotionConfig, dt: float, region: RegionConfig | None = None):
        if config.state_dim != 4:
            raise ValueError("constant_speed requires state_dim=4")
        a = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, config.rho, 0.0],
                [0.0, 0.0, 0.0, config.rho],
            ],
            dtype=float,
        )
        q = (config.process_noise_std ** 2) * np.eye(4)
        super().__init__(matrix=a, covariance=q, region=region)


def make_dynamics(
    config: MotionConfig,
    dt: float,
    center: np.ndarray | None = None,
    region: RegionConfig | None = None,
) -> DynamicsModel:
    if config.kind == "static":
        return StaticDynamics(config.state_dim, region=region)
    if config.kind == "gauss_markov":
        return GaussMarkovDynamics(config, center=center, region=region)
    if config.kind == "constant_speed":
        return ConstantSpeedDynamics(config, dt=dt, region=region)
    raise ValueError(f"unsupported motion model: {config.kind}")


def initial_state(
    position: np.ndarray,
    config: MotionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    state = np.zeros(config.state_dim, dtype=float)
    state[:2] = np.asarray(position, dtype=float)[:2]
    if config.state_dim == 4:
        speed = rng.uniform(config.initial_speed_min_mps, config.initial_speed_max_mps)
        angle = rng.uniform(-np.pi, np.pi)
        state[2:] = speed * np.array([np.cos(angle), np.sin(angle)])
    return state
