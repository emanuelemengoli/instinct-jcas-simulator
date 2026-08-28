"""Combined UE/SO trajectory visualization using trajectories retained by the simulator."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from ..simulator import LargeScaleSimulationResult
from ._common import finish_figure
from .network_plots import _plot_stored_cells


def _selected_ids(
    trajectories: dict[str, np.ndarray],
    requested: Sequence[str] | None,
    maximum: int | None,
) -> list[str]:
    if requested is None:
        ids = sorted(trajectories)
    else:
        ids = list(requested)
        missing = [entity_id for entity_id in ids if entity_id not in trajectories]
        if missing:
            raise KeyError(f"unknown entity ids: {missing}")
    if maximum is not None:
        if maximum < 0:
            raise ValueError("maximum entity count must be non-negative")
        ids = ids[:maximum]
    return ids


def _trajectory_segments(
    trajectory: np.ndarray,
    result: LargeScaleSimulationResult,
) -> list[np.ndarray]:
    """Split a wrapped trajectory so no artificial across-domain chord is drawn."""
    trajectory = np.asarray(trajectory, dtype=float)
    if len(trajectory) <= 1 or result.region.distance_model != "toroidal":
        return [trajectory]
    jumps = (
        (np.abs(np.diff(trajectory[:, 0])) > result.region.width / 2.0)
        | (np.abs(np.diff(trajectory[:, 1])) > result.region.height / 2.0)
    )
    cut_after = np.flatnonzero(jumps) + 1
    return [segment for segment in np.split(trajectory, cut_after) if len(segment)]


def plot_entity_trajectories(
    result: LargeScaleSimulationResult,
    *,
    ue_ids: Sequence[str] | None = None,
    so_ids: Sequence[str] | None = None,
    max_ues: int | None = None,
    max_sos: int | None = None,
    time_slice: slice | None = None,
    time_stride: int = 1,
    show_voronoi: bool = True,
    ax=None,
    title: str = "UEs, SOs trajectories",
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot UE and SO trajectories together from the stored simulation history."""
    if not isinstance(result, LargeScaleSimulationResult):
        raise ValueError("2-D UE/SO trajectories are only defined for large-scale results")
    if time_stride <= 0:
        raise ValueError("time_stride must be positive")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    if show_voronoi:
        _plot_stored_cells(ax, result, alpha=0.35)

    bs_positions = np.vstack([bs.position for bs in result.network.base_stations])
    ax.scatter(bs_positions[:, 0], bs_positions[:, 1], marker="^", s=70, label="BS")

    ue_keys = _selected_ids(result.ue_trajectories, ue_ids, max_ues)
    so_keys = _selected_ids(result.so_trajectories, so_ids, max_sos)

    def select_time(traj: np.ndarray) -> np.ndarray:
        selected = traj if time_slice is None else traj[time_slice]
        return np.asarray(selected[::time_stride, :2], dtype=float)

    ue_label_used = so_label_used = False
    ue_start_used = ue_end_used = so_start_used = so_end_used = False

    for entity_id in ue_keys:
        traj = select_time(result.ue_trajectories[entity_id])
        if len(traj) == 0:
            continue
        segments = _trajectory_segments(traj, result)
        first_line = ax.plot(
            segments[0][:, 0], segments[0][:, 1],
            linestyle="-", linewidth=1.1, alpha=0.75,
            label="UE trajectory" if not ue_label_used else None,
        )[0]
        entity_color = first_line.get_color()
        for segment in segments[1:]:
            ax.plot(segment[:, 0], segment[:, 1], linestyle="-", linewidth=1.1,
                    alpha=0.75, color=entity_color)
        ue_label_used = True
        ax.scatter(traj[0, 0], traj[0, 1], marker="o", s=30, color=entity_color,
                   label="UE start" if not ue_start_used else None)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="x", s=38, color=entity_color,
                   label="UE end" if not ue_end_used else None)
        ue_start_used = ue_end_used = True

    for entity_id in so_keys:
        traj = select_time(result.so_trajectories[entity_id])
        if len(traj) == 0:
            continue
        segments = _trajectory_segments(traj, result)
        first_line = ax.plot(
            segments[0][:, 0], segments[0][:, 1],
            linestyle="--", linewidth=1.1, alpha=0.75,
            label="SO trajectory" if not so_label_used else None,
        )[0]
        entity_color = first_line.get_color()
        for segment in segments[1:]:
            ax.plot(segment[:, 0], segment[:, 1], linestyle="--", linewidth=1.1,
                    alpha=0.75, color=entity_color)
        so_label_used = True
        ax.scatter(traj[0, 0], traj[0, 1], marker="s", s=30, color=entity_color,
                   label="SO start" if not so_start_used else None)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="+", s=45, color=entity_color,
                   label="SO end" if not so_end_used else None)
        so_start_used = so_end_used = True

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


def animate_entity_trajectories(
    result: LargeScaleSimulationResult,
    *,
    ue_ids: Sequence[str] | None = None,
    so_ids: Sequence[str] | None = None,
    max_ues: int | None = None,
    max_sos: int | None = None,
    time_slice: slice | None = None,
    time_stride: int = 1,
    max_frames: int | None = 200,
    trail_length: int = 15,
    fps: int = 8,
    show_voronoi: bool = True,
    title: str = "UEs, SOs trajectories",
    save_path: str | Path | None = None,
) -> bytes:
    """Animate stored UE/SO positions over time and encode the result as a GIF.

    Unlike ``plot_entity_trajectories`` (one static figure with full paths
    drawn as lines), this renders one frame per retained time step, with a
    short fading trail behind each entity's current position, and returns
    the encoded GIF as ``bytes`` (also writing it to ``save_path`` when
    given). No simulation, RNG, or metric computation is performed here;
    only stored trajectories are consumed.

    ``time_stride`` gives explicit control over frame spacing, as in
    ``plot_entity_trajectories``. When left at its default and the number of
    stored time steps exceeds ``max_frames``, the stride is instead chosen
    deterministically so the animation has at most ``max_frames`` frames
    (long horizons would otherwise produce a slow-to-render, very large GIF).
    """
    if not isinstance(result, LargeScaleSimulationResult):
        raise ValueError("2-D UE/SO trajectories are only defined for large-scale results")
    if time_stride <= 0:
        raise ValueError("time_stride must be positive")

    ue_keys = _selected_ids(result.ue_trajectories, ue_ids, max_ues)
    so_keys = _selected_ids(result.so_trajectories, so_ids, max_sos)

    def select_time(traj: np.ndarray) -> np.ndarray:
        selected = traj if time_slice is None else traj[time_slice]
        return np.asarray(selected[:, :2], dtype=float)

    ue_paths = {k: select_time(result.ue_trajectories[k]) for k in ue_keys}
    so_paths = {k: select_time(result.so_trajectories[k]) for k in so_keys}
    all_paths = [*ue_paths.values(), *so_paths.values()]
    if not all_paths or max(len(p) for p in all_paths) == 0:
        raise ValueError("no UE/SO trajectory samples to animate")
    n_time_full = max(len(p) for p in all_paths)

    stride = time_stride
    if stride == 1 and max_frames is not None and n_time_full > max_frames:
        stride = max(1, int(np.ceil(n_time_full / max_frames)))
    frame_indices = np.arange(0, n_time_full, stride)
    if frame_indices[-1] != n_time_full - 1:
        frame_indices = np.append(frame_indices, n_time_full - 1)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    if show_voronoi:
        _plot_stored_cells(ax, result, alpha=0.35)

    bs_positions = np.vstack([bs.position for bs in result.network.base_stations])
    ax.scatter(bs_positions[:, 0], bs_positions[:, 1], marker="^", s=70, label="BS", zorder=3)

    trail_scatter = ax.scatter([], [], s=10, color="tab:gray", alpha=0.35, zorder=2)
    ue_scatter = ax.scatter(
        [], [], marker="o", s=32, color="tab:orange", label="UE" if ue_paths else None, zorder=4
    )
    so_scatter = ax.scatter(
        [], [], marker="s", s=32, color="tab:green", label="SO" if so_paths else None, zorder=4
    )
    time_label = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7, ec="none"),
    )

    x_min, y_min, x_max, y_max = result.region.bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    if ue_paths or so_paths:
        ax.legend(loc="best")
    fig.tight_layout()

    def positions_at(paths: dict[str, np.ndarray], idx: int) -> np.ndarray:
        pts = [arr[min(idx, len(arr) - 1)] for arr in paths.values() if len(arr)]
        return np.asarray(pts) if pts else np.empty((0, 2))

    def trail_at(paths: dict[str, np.ndarray], idx: int) -> np.ndarray:
        segments = []
        for arr in paths.values():
            if len(arr) == 0:
                continue
            end = min(idx, len(arr) - 1)
            start = max(0, end - trail_length)
            segments.append(arr[start:end])
        return np.concatenate(segments, axis=0) if segments else np.empty((0, 2))

    def update(frame_pos: int):
        idx = int(frame_indices[frame_pos])
        ue_pts = positions_at(ue_paths, idx)
        so_pts = positions_at(so_paths, idx)
        trail_parts = [t for t in (trail_at(ue_paths, idx), trail_at(so_paths, idx)) if len(t)]
        trail_pts = np.concatenate(trail_parts, axis=0) if trail_parts else np.empty((0, 2))
        if len(ue_pts):
            ue_scatter.set_offsets(ue_pts)
        if len(so_pts):
            so_scatter.set_offsets(so_pts)
        trail_scatter.set_offsets(trail_pts)
        time_label.set_text(f"t = {idx}")
        return [ue_scatter, so_scatter, trail_scatter, time_label]

    anim = FuncAnimation(fig, update, frames=len(frame_indices), blit=False)

    tmp_path: Path | None = None
    try:
        if save_path is not None:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            out_path = tmp_path
        anim.save(out_path, writer=PillowWriter(fps=fps))
        data = out_path.read_bytes()
    finally:
        plt.close(fig)
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return data
