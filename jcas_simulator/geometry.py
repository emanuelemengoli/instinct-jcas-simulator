"""Geometry helpers for rectangular flat-torus simulations."""

from __future__ import annotations

import itertools
import numpy as np
from scipy.spatial import QhullError, Voronoi
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .config import RegionConfig


def toroidal_displacement(a: np.ndarray, b: np.ndarray, region: RegionConfig) -> np.ndarray:
    """Return the minimum-image displacement ``a - b`` on the rectangular torus.

    Only the first two coordinates are spatial.  The returned array therefore
    always has shape ``(2,)`` and lies in
    ``[-width/2,width/2) x [-height/2,height/2)``.
    """
    d = np.asarray(a, dtype=float)[:2] - np.asarray(b, dtype=float)[:2]
    d = d.copy()
    d[0] = (d[0] + region.width / 2.0) % region.width - region.width / 2.0
    d[1] = (d[1] + region.height / 2.0) % region.height - region.height / 2.0
    return d


def relative_state(state: np.ndarray, reference: np.ndarray, region: RegionConfig) -> np.ndarray:
    """Return ``state-reference`` using minimum-image spatial coordinates.

    This is the state-space analogue of ``disp(x-bs)``:
    position coordinates use the shortest toroidal displacement while any
    non-spatial coordinates (e.g. velocity) use ordinary subtraction.
    """
    state = np.asarray(state, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if state.shape != reference.shape:
        raise ValueError("state and reference must have the same shape")
    out = state - reference
    if region.distance_model == "toroidal":
        out = out.copy()
        out[:2] = toroidal_displacement(state[:2], reference[:2], region)
    return out


def distance(a: np.ndarray, b: np.ndarray, region: RegionConfig) -> float:
    """Minimum-image distance on the rectangular flat torus defined by ``region``."""
    if region.distance_model == "toroidal":
        d = toroidal_displacement(a, b, region)
    else:
        d = np.asarray(a, dtype=float)[:2] - np.asarray(b, dtype=float)[:2]
    return float(np.linalg.norm(d))


def wrap_position(position: np.ndarray, region: RegionConfig) -> np.ndarray:
    """Wrap a 2-D position into the configured rectangular fundamental domain."""
    x_min, y_min, x_max, y_max = region.bounds
    out = np.asarray(position, dtype=float).copy()
    out[0] = (out[0] - x_min) % (x_max - x_min) + x_min
    out[1] = (out[1] - y_min) % (y_max - y_min) + y_min
    return out


def wrap_state(state: np.ndarray, region: RegionConfig) -> np.ndarray:
    """Project only the spatial coordinates of a state onto the torus.

    Velocity or other non-spatial state coordinates are never wrapped.
    """
    out = np.asarray(state, dtype=float).copy()
    if region.distance_model == "toroidal":
        out[:2] = wrap_position(out[:2], region)
    return out


def _polygonal_only(geometry: BaseGeometry) -> BaseGeometry:
    """Discard zero-area artefacts after unions/intersections."""
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if isinstance(geometry, GeometryCollection):
        pieces = [g for g in geometry.geoms if isinstance(g, (Polygon, MultiPolygon)) and g.area > 0]
        return unary_union(pieces) if pieces else Polygon()
    return Polygon()


def periodic_voronoi_cells(points: np.ndarray, region: RegionConfig) -> list[BaseGeometry]:
    """Voronoi cells on a rectangular flat torus, represented in one window.

    The construction follows the standard periodic-image method.  BSs are
    replicated on a 5x5 lattice.  Ordinary Euclidean Voronoi cells are then
    computed for the tiled sites, and the cells of the central 3x3 images are
    intersected with the fundamental rectangle and unioned by original BS.

    The outer support ring (shifts +/-2) guarantees that every central-3x3
    image cell is bounded.  A toroidal cell can cross a rectangle boundary, so
    its central-window representation may legitimately be a ``MultiPolygon``.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("points must have shape (n, 2) with n >= 1")

    points = np.vstack([wrap_position(p, region) for p in points])
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("periodic Voronoi sites must be distinct")

    shift_pairs = list(itertools.product(range(-2, 3), repeat=2))
    shifts = np.array(
        [[kx * region.width, ky * region.height] for kx, ky in shift_pairs],
        dtype=float,
    )
    tiled_points = (points[None, :, :] + shifts[:, None, :]).reshape(-1, 2)

    try:
        vor = Voronoi(tiled_points)
    except QhullError as exc:
        raise RuntimeError("periodic Voronoi construction failed") from exc

    x_min, y_min, x_max, y_max = region.bounds
    window = Polygon(
        [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    )
    n = len(points)
    cells: list[BaseGeometry] = []

    for i in range(n):
        pieces: list[BaseGeometry] = []
        for shift_index, (kx, ky) in enumerate(shift_pairs):
            if abs(kx) > 1 or abs(ky) > 1:
                continue
            point_index = shift_index * n + i
            region_vertices = vor.regions[vor.point_region[point_index]]
            if not region_vertices:
                continue
            if -1 in region_vertices:
                # With the outer +/-2 support ring this should never happen for
                # a central-3x3 image.  Treat it as a construction failure, not
                # as an empty cell, because silently dropping it creates gaps.
                raise RuntimeError(
                    "periodic Voronoi central image has an unbounded region"
                )
            polygon = Polygon(vor.vertices[region_vertices]).buffer(0)
            if polygon.is_empty:
                continue
            clipped = _polygonal_only(polygon.intersection(window).buffer(0))
            if not clipped.is_empty and clipped.area > 1e-12:
                pieces.append(clipped)

        if not pieces:
            raise RuntimeError("periodic Voronoi construction produced an empty cell")
        cell = _polygonal_only(unary_union(pieces).buffer(0))
        if cell.is_empty or cell.area <= 0:
            raise RuntimeError("invalid periodic Voronoi cell")
        cells.append(cell)

    # Strong consistency checks: cells must tile exactly one fundamental
    # window, and their total area must equal the window area (no overlaps).
    union = unary_union(cells)
    missing_area = window.difference(union).area
    excess_area = union.difference(window).area
    overlap_area = sum(cell.area for cell in cells) - union.area
    tol = max(1e-9 * region.area, 1e-8)
    if missing_area > tol or excess_area > tol or overlap_area > tol:
        raise RuntimeError(
            "periodic Voronoi cells do not form a partition of the simulation window: "
            f"missing_area={missing_area}, excess_area={excess_area}, "
            f"overlap_area={overlap_area}"
        )
    return cells


def bounded_voronoi_cells(points: np.ndarray, region: RegionConfig) -> list[BaseGeometry]:
    """Voronoi cells for the configured network on the rectangular flat torus.

    Kept as the public entry point (``distance_model`` is toroidal-only); it
    delegates to :func:`periodic_voronoi_cells`.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("points must have shape (n, 2) with n >= 1")
    if region.distance_model != "toroidal":
        raise ValueError("bounded_voronoi_cells only supports distance_model='toroidal'")
    return periodic_voronoi_cells(points, region)
