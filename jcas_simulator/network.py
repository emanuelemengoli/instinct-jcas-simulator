"""Stochastic network generation: BS point process -> Voronoi -> UEs/SOs."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from .config import NetworkConfig, PopulationConfig, RegionConfig
from .geometry import bounded_voronoi_cells, distance, toroidal_displacement, wrap_position
from .rng import RNGManager


@dataclass
class BaseStation:
    index: int
    id: str
    position: np.ndarray
    cell: BaseGeometry


@dataclass
class NetworkEntity:
    index: int
    id: str
    kind: str
    position: np.ndarray
    serving_bs_index: int
    state: np.ndarray | None = None


@dataclass
class Network:
    base_stations: list[BaseStation]
    ues: list[NetworkEntity]
    sensing_objects: list[NetworkEntity]

    def serving_bs(self, entity: NetworkEntity) -> BaseStation:
        return self.base_stations[entity.serving_bs_index]

    def reassign_to_nearest_bs(
        self, entity: NetworkEntity, config: NetworkConfig, margin: float = 0.0
    ) -> None:
        """Reassign ``entity`` to its nearest BS on the torus, with hysteresis.

        The serving BS only changes when some other BS is closer than the
        current serving BS by strictly more than ``margin`` metres.  ``margin=0``
        is plain instantaneous nearest-BS handover.
        """
        current = entity.serving_bs_index
        best = min(
            range(len(self.base_stations)),
            key=lambda i: distance(entity.position, self.base_stations[i].position, config.region),
        )
        if best == current:
            return
        d_current = distance(
            entity.position, self.base_stations[current].position, config.region
        )
        d_best = distance(
            entity.position, self.base_stations[best].position, config.region
        )
        if d_best < d_current - margin:
            entity.serving_bs_index = best


def _draw_bs_count(config: NetworkConfig, rng: np.random.Generator) -> int:
    pp = config.base_stations
    if pp.fixed_count is not None:
        count = int(pp.fixed_count)
    else:
        count = int(rng.poisson(pp.intensity_per_m2 * config.region.area))
    return max(pp.minimum_count, count)


def _draw_population_count(pop: PopulationConfig, rng: np.random.Generator) -> int:
    if pop.count_model == "fixed":
        if pop.fixed_per_cell is None:
            raise ValueError("fixed_per_cell must be provided when count_model='fixed'")
        count = int(pop.fixed_per_cell)
    elif pop.count_model == "poisson":
        count = int(rng.poisson(pop.mean_per_cell))
    else:
        raise ValueError(f"unsupported population count model: {pop.count_model}")
    return max(pop.minimum_per_cell, count)


def _polygon_components(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [poly for poly in geometry.geoms if poly.area > 0]
    raise TypeError(f"expected Polygon or MultiPolygon, got {type(geometry).__name__}")


def _sample_uniform_polygonal(
    geometry: BaseGeometry,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample uniformly by first choosing a component proportional to area."""
    components = _polygon_components(geometry)
    areas = np.asarray([poly.area for poly in components], dtype=float)
    probs = areas / areas.sum()
    component = components[int(rng.choice(len(components), p=probs))]
    min_x, min_y, max_x, max_y = component.bounds
    while True:
        p = rng.uniform([min_x, min_y], [max_x, max_y])
        if component.covers(Point(float(p[0]), float(p[1]))):
            return p




def _cell_span(
    geometry: BaseGeometry,
    bs_position: np.ndarray,
    region: RegionConfig,
) -> np.ndarray:
    """Return a local x/y span for a cell, respecting periodic boundaries."""
    if region.distance_model != "toroidal":
        min_x, min_y, max_x, max_y = geometry.bounds
        return np.array([max_x - min_x, max_y - min_y], dtype=float)

    coords = []
    for component in _polygon_components(geometry):
        xy = np.asarray(component.exterior.coords, dtype=float)
        coords.extend(
            toroidal_displacement(point, bs_position, region) for point in xy
        )
    local = np.asarray(coords, dtype=float)
    span = np.ptp(local, axis=0) if len(local) else np.zeros(2, dtype=float)
    # Degenerate zero span is possible only in pathological numerical cases;
    # use an area-based fallback rather than a zero Gaussian standard deviation.
    fallback = np.sqrt(max(float(geometry.area), 1e-12))
    return np.where(span > 1e-12, span, fallback)
def _sample_in_polygon(
    polygon: BaseGeometry,
    count: int,
    rng: np.random.Generator,
    population: PopulationConfig,
    bs_position: np.ndarray,
    region: RegionConfig,
) -> np.ndarray:
    """Sample entities from a Euclidean or periodic Voronoi cell.

    Periodic cells may be disconnected in the central-window representation.
    Uniform samples are therefore drawn component-wise proportional to area.
    Gaussian samples are generated around the BS and wrapped before membership
    testing, so a cell crossing an edge is treated as one toroidal cell.
    """
    if count <= 0:
        return np.empty((0, 2), dtype=float)

    local_span = _cell_span(polygon, bs_position, region)
    samples: list[np.ndarray] = []
    attempts = 0
    max_attempts = max(50_000, 10_000 * count)
    while len(samples) < count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("failed to sample inside Voronoi cell")

        if population.placement == "uniform":
            p = _sample_uniform_polygonal(polygon, rng)
        elif population.placement == "gaussian_around_bs":
            # Preserve the previous scale convention.  On a torus the draw is
            # wrapped before membership testing, which correctly samples
            # across periodic boundaries.
            scale = population.placement_std_fraction * local_span
            p = rng.normal(np.asarray(bs_position, dtype=float)[:2], scale)
            if region.distance_model == "toroidal":
                p = wrap_position(p, region)
        else:
            raise ValueError(f"unsupported placement model: {population.placement}")

        if polygon.covers(Point(float(p[0]), float(p[1]))):
            samples.append(np.asarray(p, dtype=float))
    return np.asarray(samples, dtype=float)


def generate_network(config: NetworkConfig, rngs: RNGManager) -> Network:
    """Generate the intended stochastic network model.

    1. BSs are sampled from a homogeneous PPP (or a fixed-count conditional
       version for controlled experiments).
    2. Their bounded Voronoi tessellation is constructed.
    3. Each cell receives a random UE and SO count according to its configured
       population model.
    4. Entities are placed uniformly inside their own cell.
    """
    if config.region.distance_model != "toroidal":
        raise ValueError(
            "generate_network requires region.distance_model='toroidal'; the "
            "euclidean distance model is not supported"
        )
    rng_bs = rngs.generator("network:base_stations")
    rng_ue = rngs.generator("network:ue_population")
    rng_so = rngs.generator("network:so_population")

    n_bs = _draw_bs_count(config, rng_bs)
    x_min, y_min, x_max, y_max = config.region.bounds
    positions = rng_bs.uniform([x_min, y_min], [x_max, y_max], size=(n_bs, 2))
    cells = bounded_voronoi_cells(positions, config.region)
    base_stations = [
        BaseStation(index=i, id=f"bs_{i}", position=positions[i].copy(), cell=cells[i])
        for i in range(n_bs)
    ]

    ues: list[NetworkEntity] = []
    sos: list[NetworkEntity] = []
    ue_index = 0
    so_index = 0

    for bs in base_stations:
        n_ues = _draw_population_count(config.ue_population, rng_ue)
        n_sos = _draw_population_count(config.so_population, rng_so)
        ue_positions = _sample_in_polygon(
            bs.cell, n_ues, rng_ue, config.ue_population, bs.position, config.region
        )
        so_positions = _sample_in_polygon(
            bs.cell, n_sos, rng_so, config.so_population, bs.position, config.region
        )

        for p in ue_positions:
            ues.append(
                NetworkEntity(
                    index=ue_index,
                    id=f"ue_{ue_index}",
                    kind="ue",
                    position=p.copy(),
                    serving_bs_index=bs.index,
                )
            )
            ue_index += 1
        for p in so_positions:
            sos.append(
                NetworkEntity(
                    index=so_index,
                    id=f"so_{so_index}",
                    kind="so",
                    position=p.copy(),
                    serving_bs_index=bs.index,
                )
            )
            so_index += 1

    return Network(base_stations=base_stations, ues=ues, sensing_objects=sos)
