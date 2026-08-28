"""Plots of the stored stochastic-network realization and its Voronoi cells."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from ..simulator import LargeScaleSimulationResult
from ._common import finish_figure


def _plot_stored_cells(ax, result: LargeScaleSimulationResult, *, alpha: float = 0.75) -> None:
    """Draw the exact stored Euclidean or periodic Voronoi cells.

    A periodic cell may appear as several polygons on opposite sides of the
    fundamental rectangle; draw every component rather than inventing an
    artificial line across the window.  All components of one cell are drawn
    in the same color, since they are periodic images of a single toroidal
    Voronoi region and not separate cells: without this, each wrapped piece
    would advance matplotlib's color cycle on its own and a single cell would
    misleadingly appear as several differently colored ones.
    """
    for bs in result.network.base_stations:
        cell = bs.cell
        polygons = [cell] if isinstance(cell, Polygon) else list(cell.geoms)
        if not polygons:
            continue
        first_xy = np.asarray(polygons[0].exterior.coords, dtype=float)
        line = ax.plot(first_xy[:, 0], first_xy[:, 1], linewidth=0.9, alpha=alpha)[0]
        cell_color = line.get_color()
        for polygon in polygons[1:]:
            xy = np.asarray(polygon.exterior.coords, dtype=float)
            ax.plot(xy[:, 0], xy[:, 1], linewidth=0.9, alpha=alpha, color=cell_color)


def _entity_positions(result: LargeScaleSimulationResult, kind: str, time_index: int) -> np.ndarray:
    trajectories = result.ue_trajectories if kind == "ue" else result.so_trajectories
    if not trajectories:
        return np.empty((0, 2), dtype=float)
    n_time = len(next(iter(trajectories.values())))
    index = time_index if time_index >= 0 else n_time + time_index
    if not 0 <= index < n_time:
        raise IndexError(f"time_index={time_index} is outside [0, {n_time - 1}]")
    return np.vstack([trajectories[key][index, :2] for key in sorted(trajectories)])


def plot_voronoi_network(
    result: LargeScaleSimulationResult,
    *,
    time_index: int = 0,
    ax=None,
    title: str = "JCAS network realization and Voronoi tessellation",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot the actual network realization stored in ``result``.

    No network geometry is regenerated. ``time_index=0`` reproduces the
    initial UE/SO placement used to start the simulation; other indices show
    the stored mobile-entity positions at that logical simulation time while
    retaining the same stored BS/Voronoi realization.
    """
    if not isinstance(result, LargeScaleSimulationResult):
        raise ValueError("Voronoi/network geometry is only defined for large-scale results")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    _plot_stored_cells(ax, result)
    bs_positions = np.vstack([bs.position for bs in result.network.base_stations])
    ue_positions = _entity_positions(result, "ue", time_index)
    so_positions = _entity_positions(result, "so", time_index)

    ax.scatter(bs_positions[:, 0], bs_positions[:, 1], marker="^", s=70, label="BS")
    if len(ue_positions):
        ax.scatter(ue_positions[:, 0], ue_positions[:, 1], marker="o", s=28, label="UE")
    if len(so_positions):
        ax.scatter(so_positions[:, 0], so_positions[:, 1], marker="s", s=28, label="SO")

    x_min, y_min, x_max, y_max = result.region.bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    finish_figure(fig, save_path=save_path, show=show)
    return fig, ax
