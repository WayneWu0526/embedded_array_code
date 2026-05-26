#!/usr/bin/env python3
"""Plot ground-truth poses as 3D scatter points colored by localization error."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Polygon
from matplotlib.legend_handler import HandlerPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize

from plot_cycle_pose import (
    CyclePoseData,
    Pose3D,
    average_quaternions_xyzw,
    load_result_pose_data,
    parse_pose,
    prepare_matplotlib,
    quaternion_xyzw_to_rotation_matrix,
    set_axes_equal,
    set_sparse_ticks,
)


DEFAULT_RESULT_DIRS = [
    Path("src/sensor_data_collection/result_merged_all"),
]
DEFAULT_SAVE_PATH = Path(
    "src/sensor_data_collection/plot/merged_ground_truth_error_scatter/"
    "merged_ground_truth_error_scatter.png"
)
DEFAULT_BATCH_SAVE_DIR = Path(
    "src/sensor_data_collection/plot/merged_ground_truth_error_scatter/"
    "sensor_method_xy_xz"
)
DEFAULT_MAX_POSITION_ERROR_MM = 50.0
DEFAULT_POSITION_ERROR_RANGE_MM = (5.0, 50.0)
DEFAULT_ORIENTATION_ERROR_RANGE_RAD = (np.deg2rad(0.5), 0.5)
DEFAULT_ERROR_SOURCE_KEY = "localization_cci_sensors_1346791012"
DEFAULT_ORIENTATION_ERROR_SOURCE_KEY = "localization"
POSITION_ERROR_COLORBAR_TICKS_MM = [10.0, 20.0, 30.0, 40.0, 50.0]
ORIENTATION_ERROR_COLORBAR_TICKS_RAD = [0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_PROJECTION_COLORBAR_GAP = 0.030
DEFAULT_PROJECTION_COLORBAR_PAIR_GAP = 0.125
DEFAULT_PROJECTION_COLORBAR_WIDTH = 0.016
DEFAULT_PROJECTION_COLORBAR_HEIGHT = 0.9
SENSOR_METHOD_ERROR_SOURCES = [
    ("sensors_12_cvt", "localization"),
    ("sensors_12_cci", "localization_cci"),
    ("sensors_1346791012_cvt", "localization_sensors_1346791012"),
    ("sensors_1346791012_cci", "localization_cci_sensors_1346791012"),
    ("sensors_2312_cvt", "localization_sensors_2312"),
    ("sensors_2312_cci", "localization_cci_sensors_2312"),
]


@dataclass(frozen=True)
class GroundTruthErrorEntry:
    label: str
    cycle_index: int
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    error_mm: Optional[float]
    orientation_error_rad: Optional[float]
    selected_method: Optional[str]
    source_poses: tuple[Pose3D, ...]


def _sequence_index(cycle_data: CyclePoseData, fallback_idx: int) -> int:
    return cycle_data.cycle_id if cycle_data.cycle_id is not None else fallback_idx


def _localization_position_error_mm(
    cycle_data: CyclePoseData,
    localization_key: str,
) -> Optional[float]:
    localization = cycle_data.raw.get(localization_key, {})
    raw_error = localization.get("position_error")
    if raw_error is not None:
        error_mm = float(raw_error) * 1000.0
    else:
        estimated_pose = parse_pose(f"{localization_key}_estimate", localization.get("pose"))
        if cycle_data.ground_truth_pose is None or estimated_pose is None:
            return None
        error_mm = (
            np.linalg.norm(estimated_pose.position - cycle_data.ground_truth_pose.position)
            * 1000.0
        )

    return error_mm if np.isfinite(error_mm) else None


def _localization_orientation_error_rad(
    cycle_data: CyclePoseData,
    localization_key: str,
) -> Optional[float]:
    localization = cycle_data.raw.get(localization_key, {})
    raw_error = localization.get("orientation_error")
    if raw_error is not None:
        error_rad = float(raw_error)
    else:
        estimated_pose = parse_pose(f"{localization_key}_estimate", localization.get("pose"))
        if cycle_data.ground_truth_pose is None or estimated_pose is None:
            return None
        dot = abs(
            float(
                np.dot(
                    cycle_data.ground_truth_pose.quaternion_xyzw,
                    estimated_pose.quaternion_xyzw,
                )
            )
        )
        error_rad = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))

    return error_rad if np.isfinite(error_rad) else None


def _best_localization_errors(
    cycle_data: CyclePoseData,
    position_error_source_key: str = DEFAULT_ERROR_SOURCE_KEY,
    orientation_error_source_key: str = DEFAULT_ORIENTATION_ERROR_SOURCE_KEY,
    selected_method_label: Optional[str] = None,
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    if (
        position_error_source_key not in cycle_data.raw
        or orientation_error_source_key not in cycle_data.raw
    ):
        return None, None, None

    position_error_mm = _localization_position_error_mm(
        cycle_data,
        position_error_source_key,
    )
    if position_error_mm is None:
        return None, None, None
    orientation_error_rad = _localization_orientation_error_rad(
        cycle_data,
        orientation_error_source_key,
    )
    if selected_method_label is None:
        selected_method_label = f"{position_error_source_key} ep, {orientation_error_source_key} eR"
    return position_error_mm, orientation_error_rad, selected_method_label


def _load_ground_truth_error_entries(
    result_dir: Path,
    label: Optional[str] = None,
    position_error_source_key: str = DEFAULT_ERROR_SOURCE_KEY,
    orientation_error_source_key: str = DEFAULT_ORIENTATION_ERROR_SOURCE_KEY,
    selected_method_label: Optional[str] = None,
) -> list[GroundTruthErrorEntry]:
    cycle_data_list = load_result_pose_data(result_dir)
    entries: list[GroundTruthErrorEntry] = []
    entry_label = label if label is not None else result_dir.name

    for fallback_idx, cycle_data in enumerate(cycle_data_list):
        if cycle_data.ground_truth_pose is None:
            continue
        position_error_mm, orientation_error_rad, selected_method = _best_localization_errors(
            cycle_data,
            position_error_source_key=position_error_source_key,
            orientation_error_source_key=orientation_error_source_key,
            selected_method_label=selected_method_label,
        )

        entries.append(
            GroundTruthErrorEntry(
                label=entry_label,
                cycle_index=_sequence_index(cycle_data, fallback_idx),
                position=cycle_data.ground_truth_pose.position,
                quaternion_xyzw=cycle_data.ground_truth_pose.quaternion_xyzw,
                error_mm=position_error_mm,
                orientation_error_rad=orientation_error_rad,
                selected_method=selected_method,
                source_poses=tuple(cycle_data.source_poses[:3]),
            )
        )

    return sorted(entries, key=lambda item: item.cycle_index)


def _error_color_map() -> Any:
    return LinearSegmentedColormap.from_list(
        "green_orange_red",
        ["#1a9850", "#fdae61", "#d73027"],
    )


def _error_norm(
    entries: Sequence[GroundTruthErrorEntry],
    error_range_mm: tuple[float, float] = DEFAULT_POSITION_ERROR_RANGE_MM,
) -> Normalize:
    valid_errors = np.array(
        [entry.error_mm for entry in entries if entry.error_mm is not None],
        dtype=float,
    )
    if len(valid_errors) == 0:
        raise ValueError("No finite localization position errors found.")
    return Normalize(vmin=error_range_mm[0], vmax=error_range_mm[1], clip=True)


def _orientation_error_color_map() -> Any:
    return _error_color_map()


def _orientation_error_norm(
    entries: Sequence[GroundTruthErrorEntry],
    error_range_rad: tuple[float, float] = DEFAULT_ORIENTATION_ERROR_RANGE_RAD,
) -> Normalize:
    valid_errors = np.array(
        [
            entry.orientation_error_rad
            for entry in entries
            if entry.orientation_error_rad is not None
        ],
        dtype=float,
    )
    if len(valid_errors) == 0:
        raise ValueError("No finite localization orientation errors found.")
    return Normalize(vmin=error_range_rad[0], vmax=error_range_rad[1], clip=True)


def _entries_by_label(
    entries: Sequence[GroundTruthErrorEntry],
) -> list[tuple[str, list[GroundTruthErrorEntry]]]:
    grouped: dict[str, list[GroundTruthErrorEntry]] = {}
    ordered_labels: list[str] = []
    for entry in entries:
        if entry.label not in grouped:
            grouped[entry.label] = []
            ordered_labels.append(entry.label)
        grouped[entry.label].append(entry)
    return [(label, grouped[label]) for label in ordered_labels]


def _entry_arrays(
    entries: Sequence[GroundTruthErrorEntry],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.array([entry.position for entry in entries], dtype=float)
    errors = np.array(
        [np.nan if entry.error_mm is None else entry.error_mm for entry in entries],
        dtype=float,
    )
    cycle_indices = np.array([entry.cycle_index for entry in entries], dtype=int)
    return points, errors, cycle_indices


def _dataset_marker(label_idx: int) -> str:
    return "o"


class HandlerArrow(HandlerPatch):
    def create_artists(
        self,
        legend: Any,
        orig_handle: Any,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Any,
    ) -> list[Any]:
        center = ydescent + height / 2.0
        arrow = FancyArrowPatch(
            (xdescent, center),
            (xdescent + width, center),
            arrowstyle="-|>",
            mutation_scale=fontsize * 0.9,
            linewidth=1.35,
            color="0.35",
            transform=trans,
        )
        return [arrow]


def _gt_x_axis(entry: GroundTruthErrorEntry) -> np.ndarray:
    return quaternion_xyzw_to_rotation_matrix(entry.quaternion_xyzw)[:, 0]


def _average_source_poses(entries: Sequence[GroundTruthErrorEntry]) -> list[Pose3D]:
    source_position_groups: list[list[np.ndarray]] = [[], [], []]
    source_quaternion_groups: list[list[np.ndarray]] = [[], [], []]
    for entry in entries:
        for source_idx, source_pose in enumerate(entry.source_poses[:3]):
            source_position_groups[source_idx].append(source_pose.position)
            source_quaternion_groups[source_idx].append(source_pose.quaternion_xyzw)

    averaged_poses = []
    for source_idx, source_positions in enumerate(source_position_groups):
        if not source_positions:
            continue
        averaged_poses.append(
            Pose3D(
                name=f"M_{source_idx + 1}",
                position=np.mean(np.array(source_positions, dtype=float), axis=0),
                quaternion_xyzw=average_quaternions_xyzw(
                    source_quaternion_groups[source_idx]
                ),
            )
        )

    return averaged_poses


def _auto_arrow_length(points: np.ndarray, scale: float = 0.070) -> float:
    span = float(np.max(np.ptp(points, axis=0)))
    return max(0.006, span * scale)


def plot_ground_truth_error_scatter_3d(
    entries: Sequence[GroundTruthErrorEntry],
    save_path: Optional[Path] = None,
    show: bool = True,
    marker_size: float = 130.0,
    arrow_length: Optional[float] = None,
    view: str = "default",
    dpi: int = 220,
) -> None:
    if not entries:
        raise ValueError("No ground_truth_pose entries found.")

    points, _, _ = _entry_arrays(entries)
    color_map = _error_color_map()
    color_norm = _error_norm(entries)
    arrow_color_map = _orientation_error_color_map()
    arrow_color_norm = _orientation_error_norm(entries)
    arrow_len = arrow_length if arrow_length is not None else _auto_arrow_length(points)

    plt = prepare_matplotlib(show, use_tex=True, font_size=16.0)
    fig = plt.figure(figsize=(17.6 / 2.54, 12.6 / 2.54))
    ax = fig.add_subplot(111, projection="3d")

    for label_idx, (label, label_entries) in enumerate(_entries_by_label(entries)):
        label_points, label_errors, _ = _entry_arrays(label_entries)
        marker = _dataset_marker(label_idx)
        finite_mask = np.isfinite(label_errors)
        if np.any(finite_mask):
            ax.scatter(
                label_points[finite_mask, 0],
                label_points[finite_mask, 1],
                label_points[finite_mask, 2],
                c=label_errors[finite_mask],
                cmap=color_map,
                norm=color_norm,
                s=marker_size,
                marker=marker,
                edgecolors="0.18",
                linewidths=0.45,
                alpha=0.95,
                depthshade=True,
            )

        if np.any(~finite_mask):
            ax.scatter(
                label_points[~finite_mask, 0],
                label_points[~finite_mask, 1],
                label_points[~finite_mask, 2],
                color="0.55",
                s=marker_size,
                marker=marker,
                edgecolors="0.18",
                linewidths=0.45,
                alpha=0.85,
                depthshade=True,
            )

    for entry in entries:
        point = entry.position
        direction = _gt_x_axis(entry)
        arrow_color = (
            arrow_color_map(arrow_color_norm(entry.orientation_error_rad))
            if entry.orientation_error_rad is not None
            else "0.45"
        )
        ax.quiver(
            point[0],
            point[1],
            point[2],
            direction[0] * arrow_len,
            direction[1] * arrow_len,
            direction[2] * arrow_len,
            color=arrow_color,
            arrow_length_ratio=0.28,
            linewidth=1.45,
            alpha=0.95,
        )

    scalar_mappable = plt.cm.ScalarMappable(norm=color_norm, cmap=color_map)
    scalar_mappable.set_array(
        np.array([entry.error_mm for entry in entries if entry.error_mm is not None])
    )
    colorbar = fig.colorbar(scalar_mappable, ax=ax, shrink=0.48, pad=0.06)
    colorbar.set_label(r"$e_{\bm{p}}$ [mm]")
    colorbar.set_ticks(POSITION_ERROR_COLORBAR_TICKS_MM)
    colorbar.ax.tick_params(labelsize=16)
    arrow_scalar_mappable = plt.cm.ScalarMappable(
        norm=arrow_color_norm,
        cmap=arrow_color_map,
    )
    arrow_scalar_mappable.set_array(
        np.array(
            [
                entry.orientation_error_rad
                for entry in entries
                if entry.orientation_error_rad is not None
            ]
        )
    )
    arrow_colorbar = fig.colorbar(
        arrow_scalar_mappable,
        ax=ax,
        shrink=0.48,
        pad=0.16,
    )
    arrow_colorbar.set_label(r"$e_{\bm{R}}$ [rad]")
    arrow_colorbar.set_ticks(ORIENTATION_ERROR_COLORBAR_TICKS_RAD)
    arrow_colorbar.ax.tick_params(labelsize=16)

    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.set_zlabel(r"$z$ [m]")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.zaxis.set_tick_params(labelsize=16)
    ax.grid(True, alpha=0.30)
    if view == "xy":
        ax.view_init(elev=90, azim=-90)
    elif view == "front":
        ax.view_init(elev=18, azim=-72)
    set_axes_equal(ax)
    set_sparse_ticks(ax, max_ticks=5)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_ground_truth_error_scatter_xy(
    entries: Sequence[GroundTruthErrorEntry],
    save_path: Optional[Path] = None,
    show: bool = True,
    marker_size: float = 95.0,
    arrow_length: Optional[float] = None,
    dpi: int = 220,
) -> None:
    if not entries:
        raise ValueError("No ground_truth_pose entries found.")

    color_map = _error_color_map()
    color_norm = _error_norm(entries)
    arrow_color_map = _orientation_error_color_map()
    arrow_color_norm = _orientation_error_norm(entries)
    all_points, _, _ = _entry_arrays(entries)
    arrow_len = arrow_length if arrow_length is not None else _auto_arrow_length(all_points)

    plt = prepare_matplotlib(show, use_tex=True, font_size=16.0)
    fig, ax = plt.subplots(figsize=(17.6 / 2.54, 12.6 / 2.54))

    for label_idx, (label, label_entries) in enumerate(_entries_by_label(entries)):
        points, errors, _ = _entry_arrays(label_entries)
        marker = _dataset_marker(label_idx)
        finite_mask = np.isfinite(errors)
        if np.any(finite_mask):
            ax.scatter(
                points[finite_mask, 0],
                points[finite_mask, 1],
                c=errors[finite_mask],
                cmap=color_map,
                norm=color_norm,
                s=marker_size,
                marker=marker,
                edgecolors="0.18",
                linewidths=0.45,
                alpha=0.95,
            )

        if np.any(~finite_mask):
            ax.scatter(
                points[~finite_mask, 0],
                points[~finite_mask, 1],
                color="0.55",
                s=marker_size,
                marker=marker,
                edgecolors="0.18",
                linewidths=0.45,
                alpha=0.85,
            )

        for entry in label_entries:
            direction_xy = _gt_x_axis(entry)[:2]
            direction_norm = np.linalg.norm(direction_xy)
            if direction_norm < 1e-9:
                continue
            direction_xy = direction_xy / direction_norm
            arrow_color = (
                arrow_color_map(arrow_color_norm(entry.orientation_error_rad))
                if entry.orientation_error_rad is not None
                else "0.45"
            )
            ax.arrow(
                entry.position[0],
                entry.position[1],
                direction_xy[0] * arrow_len,
                direction_xy[1] * arrow_len,
                color=arrow_color,
                width=arrow_len * 0.045,
                head_width=arrow_len * 0.22,
                head_length=arrow_len * 0.25,
                length_includes_head=True,
                alpha=0.95,
                zorder=4,
            )

    scalar_mappable = plt.cm.ScalarMappable(norm=color_norm, cmap=color_map)
    scalar_mappable.set_array(
        np.array([entry.error_mm for entry in entries if entry.error_mm is not None])
    )
    colorbar = fig.colorbar(scalar_mappable, ax=ax, shrink=0.58, pad=0.03)
    colorbar.set_label(r"$e_{\bm{p}}$ [mm]")
    colorbar.set_ticks(POSITION_ERROR_COLORBAR_TICKS_MM)
    colorbar.ax.tick_params(labelsize=16)
    arrow_scalar_mappable = plt.cm.ScalarMappable(
        norm=arrow_color_norm,
        cmap=arrow_color_map,
    )
    arrow_scalar_mappable.set_array(
        np.array(
            [
                entry.orientation_error_rad
                for entry in entries
                if entry.orientation_error_rad is not None
            ]
        )
    )
    arrow_colorbar = fig.colorbar(arrow_scalar_mappable, ax=ax, shrink=0.58, pad=0.12)
    arrow_colorbar.set_label(r"$e_{\bm{R}}$ [rad]")
    arrow_colorbar.set_ticks(ORIENTATION_ERROR_COLORBAR_TICKS_RAD)
    arrow_colorbar.ax.tick_params(labelsize=16)

    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.grid(True, alpha=0.30)
    ax.set_aspect("equal", adjustable="box")
    set_sparse_ticks(ax, max_ticks=5)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_projected_entries(
    ax: Any,
    entries: Sequence[GroundTruthErrorEntry],
    axes: tuple[int, int],
    unit_scale: float,
    color_map: Any,
    color_norm: Normalize,
    arrow_color_map: Any,
    arrow_color_norm: Normalize,
    marker_size: float,
    arrow_length: float,
    min_projection_norm: float = 1e-4,
    normalize_arrow_projection: bool = True,
) -> None:
    for _, label_entries in _entries_by_label(entries):
        points, errors, _ = _entry_arrays(label_entries)
        projected_points = points[:, list(axes)] * unit_scale
        finite_mask = np.isfinite(errors)
        if np.any(finite_mask):
            ax.scatter(
                projected_points[finite_mask, 0],
                projected_points[finite_mask, 1],
                c=errors[finite_mask],
                cmap=color_map,
                norm=color_norm,
                s=marker_size,
                marker="o",
                edgecolors="0.18",
                linewidths=0.45,
                alpha=0.95,
            )

        if np.any(~finite_mask):
            ax.scatter(
                projected_points[~finite_mask, 0],
                projected_points[~finite_mask, 1],
                color="0.55",
                s=marker_size,
                marker="o",
                edgecolors="0.18",
                linewidths=0.45,
                alpha=0.85,
            )

        for entry in label_entries:
            direction = _gt_x_axis(entry)[list(axes)]
            direction_norm = np.linalg.norm(direction)
            if direction_norm < min_projection_norm:
                continue
            if normalize_arrow_projection:
                direction = direction / direction_norm
            arrow_color = (
                arrow_color_map(arrow_color_norm(entry.orientation_error_rad))
                if entry.orientation_error_rad is not None
                else "0.45"
            )
            origin = entry.position[list(axes)] * unit_scale
            endpoint = origin + direction * arrow_length
            ax.annotate(
                "",
                xy=(endpoint[0], endpoint[1]),
                xytext=(origin[0], origin[1]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": arrow_color,
                    "lw": 1.35,
                    "mutation_scale": 9.0,
                    "shrinkA": 0.0,
                    "shrinkB": 0.0,
                },
                zorder=4,
            )


def _projected_arrow_points(
    entries: Sequence[GroundTruthErrorEntry],
    axes: tuple[int, int],
    unit_scale: float,
    arrow_length: float,
    min_projection_norm: float = 1e-4,
    normalize_arrow_projection: bool = True,
) -> np.ndarray:
    arrow_points = []
    for entry in entries:
        direction = _gt_x_axis(entry)[list(axes)]
        direction_norm = np.linalg.norm(direction)
        if direction_norm < min_projection_norm:
            continue
        if normalize_arrow_projection:
            direction = direction / direction_norm
        origin = entry.position[list(axes)] * unit_scale
        endpoint = origin + direction * arrow_length
        arrow_points.extend([origin, endpoint])
    return np.array(arrow_points, dtype=float)


def _expand_limits_to_include_points(
    limits: tuple[float, float],
    values: np.ndarray,
    pad_fraction: float = 0.04,
) -> tuple[float, float]:
    if len(values) == 0:
        return limits

    lower = min(float(limits[0]), float(np.min(values)))
    upper = max(float(limits[1]), float(np.max(values)))
    span = max(upper - lower, 1e-6)
    padding = span * pad_fraction
    return lower - padding, upper + padding


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    unique_points = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique_points) <= 1:
        return np.array(unique_points, dtype=float)

    def cross(
        origin: tuple[float, float],
        point_a: tuple[float, float],
        point_b: tuple[float, float],
    ) -> float:
        return (
            (point_a[0] - origin[0]) * (point_b[1] - origin[1])
            - (point_a[1] - origin[1]) * (point_b[0] - origin[0])
        )

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    return np.array(lower[:-1] + upper[:-1], dtype=float)


def _cylinder_projection_hull_mm(
    pose: Pose3D,
    axes: tuple[int, int] = (0, 1),
    radius_m: float = 0.028,
    length_m: float = 0.250,
    num_theta: int = 72,
    num_length: int = 24,
) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, num_theta, endpoint=False)
    axial = np.linspace(-length_m / 2.0, length_m / 2.0, num_length)
    theta_grid, axial_grid = np.meshgrid(theta, axial)
    local_points = np.stack(
        [
            radius_m * np.cos(theta_grid),
            radius_m * np.sin(theta_grid),
            axial_grid,
        ],
        axis=-1,
    ).reshape(-1, 3)

    rotation = quaternion_xyzw_to_rotation_matrix(pose.quaternion_xyzw)
    world_points = pose.position + local_points @ rotation.T
    projected_points_mm = world_points[:, list(axes)] * 1000.0
    return _convex_hull_2d(projected_points_mm)


def _source_projection_points_mm(
    source_poses: Sequence[Pose3D],
    axes: tuple[int, int] = (0, 1),
) -> np.ndarray:
    projection_points = []
    for source_pose in source_poses:
        hull = _cylinder_projection_hull_mm(source_pose, axes=axes)
        if len(hull) > 0:
            projection_points.append(hull)
    if not projection_points:
        return np.empty((0, 2), dtype=float)
    return np.vstack(projection_points)


def _parse_limit_pair(raw_values: Optional[Sequence[float]]) -> Optional[tuple[float, float]]:
    if raw_values is None:
        return None
    lower, upper = float(raw_values[0]), float(raw_values[1])
    if lower >= upper:
        raise ValueError("Axis limits must be passed as LOWER UPPER with LOWER < UPPER.")
    return lower, upper


def _draw_electromagnet_projections(
    ax: Any,
    source_poses: Sequence[Pose3D],
    axes: tuple[int, int] = (0, 1),
) -> None:
    if not source_poses:
        return

    face_colors = ("0.30", "0.45", "0.60")
    for source_idx, source_pose in enumerate(source_poses, start=1):
        hull = _cylinder_projection_hull_mm(source_pose, axes=axes)
        if len(hull) < 3:
            continue
        polygon = Polygon(
            hull,
            closed=True,
            facecolor=face_colors[(source_idx - 1) % len(face_colors)],
            edgecolor="0.12",
            linewidth=1.0,
            alpha=0.22,
            zorder=1,
        )
        ax.add_patch(polygon)

        source_position = source_pose.position[list(axes)] * 1000.0
        ax.scatter(
            source_position[0],
            source_position[1],
            marker="x",
            s=38,
            color="0.12",
            linewidths=1.0,
            zorder=6,
        )


def plot_ground_truth_error_scatter_projection_pair(
    entries: Sequence[GroundTruthErrorEntry],
    save_path: Optional[Path] = None,
    show: bool = True,
    marker_size: float = 95.0,
    arrow_length: Optional[float] = None,
    colorbar_gap: float = DEFAULT_PROJECTION_COLORBAR_GAP,
    colorbar_pair_gap: float = DEFAULT_PROJECTION_COLORBAR_PAIR_GAP,
    colorbar_width: float = DEFAULT_PROJECTION_COLORBAR_WIDTH,
    colorbar_height: float = DEFAULT_PROJECTION_COLORBAR_HEIGHT,
    xy_xlim: Optional[tuple[float, float]] = None,
    xy_ylim: Optional[tuple[float, float]] = None,
    xz_zlim: Optional[tuple[float, float]] = None,
    projection_mode: str = "xy_xz",
    dpi: int = 220,
) -> None:
    if not entries:
        raise ValueError("No ground_truth_pose entries found.")

    color_map = _error_color_map()
    color_norm = _error_norm(entries)
    arrow_color_map = _orientation_error_color_map()
    arrow_color_norm = _orientation_error_norm(entries)
    all_points, _, _ = _entry_arrays(entries)
    average_source_poses = _average_source_poses(entries)
    unit_scale = 1000.0
    all_points_mm = all_points * unit_scale
    if projection_mode == "yx_yz":
        horizontal_axis = 1
        top_axes = (1, 0)
        bottom_axes = (1, 2)
        top_ylabel = r"$x$ [mm]"
        bottom_xlabel = r"$y$ [mm]"
        bottom_ylabel = r"$z$ [mm]"
    else:
        horizontal_axis = 0
        top_axes = (0, 1)
        bottom_axes = (0, 2)
        top_ylabel = r"$y$ [mm]"
        bottom_xlabel = r"$x$ [mm]"
        bottom_ylabel = r"$z$ [mm]"

    horizontal_span_mm = max(float(np.ptp(all_points_mm[:, horizontal_axis])), 1e-6)
    arrow_len_xy = (
        arrow_length * unit_scale
        if arrow_length is not None
        else max(8.0, horizontal_span_mm * 0.045)
    )
    arrow_len_xz = arrow_length * unit_scale if arrow_length is not None else arrow_len_xy

    plt = prepare_matplotlib(show, use_tex=True, font_size=16.0)
    fig = plt.figure(figsize=(17.6 / 2.54, 10.0 / 2.54))
    plot_right = 0.70
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(3.0, 1.25),
        left=0.12,
        right=plot_right,
        bottom=0.16,
        top=0.97,
        hspace=0.12,
    )
    ax_xy = fig.add_subplot(grid[0, 0])
    ax_xz = fig.add_subplot(grid[1, 0], sharex=ax_xy)
    axes = (ax_xy, ax_xz)

    colorbar_bottom = 0.5 - colorbar_height / 2.0
    position_colorbar_left = plot_right + colorbar_gap
    orientation_colorbar_left = (
        position_colorbar_left + colorbar_width + colorbar_pair_gap
    )
    position_colorbar_ax = fig.add_axes(
        [
            position_colorbar_left,
            colorbar_bottom,
            colorbar_width,
            colorbar_height,
        ]
    )
    orientation_colorbar_ax = fig.add_axes(
        [
            orientation_colorbar_left,
            colorbar_bottom,
            colorbar_width,
            colorbar_height,
        ]
    )

    def padded_limits(values: np.ndarray, pad_fraction: float = 0.08) -> tuple[float, float]:
        lower = float(np.min(values))
        upper = float(np.max(values))
        span = max(upper - lower, 1e-6)
        padding = span * pad_fraction
        return lower - padding, upper + padding

    def rounded_pair(values: np.ndarray, base: float = 2.0) -> tuple[float, float]:
        lower = base * np.floor(float(np.min(values)) / base)
        upper = base * np.ceil(float(np.max(values)) / base)
        if np.isclose(lower, upper):
            upper = lower + base
        return lower, upper

    xy_arrow_points = _projected_arrow_points(
        entries,
        top_axes,
        unit_scale,
        arrow_len_xy,
        normalize_arrow_projection=True,
    )
    xz_arrow_points = _projected_arrow_points(
        entries,
        bottom_axes,
        unit_scale,
        arrow_len_xz,
        normalize_arrow_projection=False,
    )
    horizontal_values = [all_points_mm[:, horizontal_axis]]
    top_vertical_values = [all_points_mm[:, top_axes[1]]]
    z_values = [all_points_mm[:, 2]]
    if len(xy_arrow_points) > 0:
        horizontal_values.append(xy_arrow_points[:, 0])
        top_vertical_values.append(xy_arrow_points[:, 1])
    if len(xz_arrow_points) > 0:
        horizontal_values.append(xz_arrow_points[:, 0])
        z_values.append(xz_arrow_points[:, 1])

    horizontal_limits = padded_limits(
        np.concatenate(horizontal_values),
        pad_fraction=0.06,
    )
    top_vertical_limits = padded_limits(
        np.concatenate(top_vertical_values),
        pad_fraction=0.08,
    )
    z_limit_values = np.concatenate(z_values)
    z_ticks = rounded_pair(all_points_mm[:, 2], base=2.0)
    z_limits_from_arrows = padded_limits(z_limit_values, pad_fraction=0.08)
    z_limits = (
        min(z_ticks[0] - 0.8, z_limits_from_arrows[0]),
        max(z_ticks[1] + 0.8, z_limits_from_arrows[1]),
    )
    if xy_xlim is not None:
        horizontal_limits = xy_xlim
    if xy_ylim is not None:
        top_vertical_limits = xy_ylim
    if xz_zlim is not None:
        z_limits = xz_zlim

    projection_specs = [
        (axes[0], top_axes, bottom_xlabel, top_ylabel, arrow_len_xy),
        (axes[1], bottom_axes, bottom_xlabel, bottom_ylabel, arrow_len_xz),
    ]
    for ax, projection_axes, _, ylabel, projection_arrow_len in projection_specs:
        _plot_projected_entries(
            ax,
            entries,
            projection_axes,
            unit_scale,
            color_map,
            color_norm,
            arrow_color_map,
            arrow_color_norm,
            marker_size,
            projection_arrow_len,
            normalize_arrow_projection=(projection_axes == (0, 1)),
        )
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.grid(True, alpha=0.30)
        set_sparse_ticks(ax, max_ticks=5)
        ax.set_xlim(horizontal_limits)
    ax_xy.set_ylim(top_vertical_limits)
    ax_xz.set_ylim(z_limits)
    ax_xz.set_yticks(list(z_ticks))
    x_ticks = ax_xz.get_xticks()
    y_ticks = ax_xy.get_yticks()
    ax_xy.set_xlim(horizontal_limits)
    ax_xz.set_xlim(horizontal_limits)
    ax_xy.set_ylim(top_vertical_limits)
    ax_xz.set_xticks(x_ticks)
    ax_xy.set_xticks(x_ticks)
    ax_xy.set_yticks(y_ticks)
    _draw_electromagnet_projections(ax_xy, average_source_poses, axes=top_axes)
    ax_xy.tick_params(axis="x", labelbottom=False)
    axes[1].set_xlabel(bottom_xlabel)

    scalar_mappable = plt.cm.ScalarMappable(norm=color_norm, cmap=color_map)
    scalar_mappable.set_array(
        np.array([entry.error_mm for entry in entries if entry.error_mm is not None])
    )
    colorbar = fig.colorbar(scalar_mappable, cax=position_colorbar_ax)
    colorbar.set_label(r"$e_{\bm{p}}$ [mm]")
    colorbar.set_ticks(POSITION_ERROR_COLORBAR_TICKS_MM)
    colorbar.ax.tick_params(labelsize=16)

    arrow_scalar_mappable = plt.cm.ScalarMappable(
        norm=arrow_color_norm,
        cmap=arrow_color_map,
    )
    arrow_scalar_mappable.set_array(
        np.array(
            [
                entry.orientation_error_rad
                for entry in entries
                if entry.orientation_error_rad is not None
            ]
        )
    )
    arrow_colorbar = fig.colorbar(arrow_scalar_mappable, cax=orientation_colorbar_ax)
    arrow_colorbar.set_label(r"$e_{\bm{R}}$ [rad]")
    arrow_colorbar.set_ticks(ORIENTATION_ERROR_COLORBAR_TICKS_RAD)
    arrow_colorbar.ax.tick_params(labelsize=16)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=7.5,
            markerfacecolor="0.75",
            markeredgecolor="0.18",
            label=r"Position",
        ),
        FancyArrowPatch(
            (0.0, 0.0),
            (0.8, 0.0),
            arrowstyle="-|>",
            mutation_scale=12.0,
            linewidth=1.35,
            color="0.35",
            label=r"Orientation",
        ),
    ]
    ax_xy.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.55,
        facecolor="white",
        edgecolor="0.65",
        handlelength=1.5,
        borderaxespad=0.2,
        handler_map={FancyArrowPatch: HandlerArrow()},
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot every ground_truth_pose in a result directory as a 3D scatter point, "
            "colored by localization position error."
        ),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        nargs="+",
        default=DEFAULT_RESULT_DIRS,
        help="Directory containing cycle_*.json files.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels matching --result-dir entries.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=DEFAULT_SAVE_PATH,
        help="Output image path.",
    )
    parser.add_argument(
        "--batch-save-dir",
        type=Path,
        default=DEFAULT_BATCH_SAVE_DIR,
        help="Output directory used by --batch-sensor-methods.",
    )
    parser.add_argument(
        "--plot",
        choices=("3d", "xy", "xy_xz", "yx_yz"),
        default="3d",
        help=(
            "Plot type. Use xy for a planar projection, xy_xz for x-based "
            "stacked projections, or yx_yz for y-based stacked projections."
        ),
    )
    parser.add_argument(
        "--position-error-source-key",
        default=DEFAULT_ERROR_SOURCE_KEY,
        help="Localization dictionary key used for position error coloring/filtering.",
    )
    parser.add_argument(
        "--orientation-error-source-key",
        default=DEFAULT_ORIENTATION_ERROR_SOURCE_KEY,
        help="Localization dictionary key used for orientation error arrow coloring.",
    )
    parser.add_argument(
        "--batch-sensor-methods",
        action="store_true",
        help=(
            "Generate six xy_xz plots for the available sensor/method sources: "
            "12-CVT, 12-CCI, 1346791012-CVT, 1346791012-CCI, 2312-CVT, 2312-CCI."
        ),
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=130.0,
        help="Matplotlib scatter marker size.",
    )
    parser.add_argument(
        "--arrow-length",
        type=float,
        default=None,
        help="Orientation arrow length in meters. If omitted, it is chosen from data span.",
    )
    parser.add_argument(
        "--max-position-error-mm",
        type=float,
        default=DEFAULT_MAX_POSITION_ERROR_MM,
        help="Discard samples with position error larger than this value in mm.",
    )
    parser.add_argument(
        "--colorbar-gap",
        type=float,
        default=DEFAULT_PROJECTION_COLORBAR_GAP,
        help=(
            "For --plot xy_xz, normalized figure gap between the projection axes "
            "and the position-error colorbar."
        ),
    )
    parser.add_argument(
        "--colorbar-pair-gap",
        type=float,
        default=DEFAULT_PROJECTION_COLORBAR_PAIR_GAP,
        help=(
            "For --plot xy_xz, normalized figure gap between the position-error "
            "and orientation-error colorbars."
        ),
    )
    parser.add_argument(
        "--colorbar-width",
        type=float,
        default=DEFAULT_PROJECTION_COLORBAR_WIDTH,
        help="For --plot xy_xz, normalized figure width of each colorbar.",
    )
    parser.add_argument(
        "--colorbar-height",
        type=float,
        default=DEFAULT_PROJECTION_COLORBAR_HEIGHT,
        help="For --plot xy_xz, normalized figure height of each colorbar.",
    )
    parser.add_argument(
        "--xy-xlim",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOWER", "UPPER"),
        help="For --plot xy_xz, x-axis limits in mm.",
    )
    parser.add_argument(
        "--xy-ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOWER", "UPPER"),
        help="For --plot xy_xz, y-axis limits in mm.",
    )
    parser.add_argument(
        "--xz-zlim",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOWER", "UPPER"),
        help="For --plot xy_xz, z-axis limits in mm.",
    )
    parser.add_argument(
        "--view",
        choices=("default", "xy", "front"),
        default="default",
        help="Camera view for the 3D plot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Saved image resolution.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.labels is not None and len(args.labels) != len(args.result_dir):
        raise ValueError("--labels must match the number of --result-dir entries.")

    def load_entries(
        position_error_source_key: str,
        orientation_error_source_key: str,
        selected_method_label: Optional[str] = None,
    ) -> list[GroundTruthErrorEntry]:
        entries: list[GroundTruthErrorEntry] = []
        for idx, result_dir in enumerate(args.result_dir):
            label = args.labels[idx] if args.labels is not None else result_dir.name
            entries.extend(
                _load_ground_truth_error_entries(
                    result_dir,
                    label=label,
                    position_error_source_key=position_error_source_key,
                    orientation_error_source_key=orientation_error_source_key,
                    selected_method_label=selected_method_label,
                )
            )
        return [
            entry
            for entry in entries
            if entry.error_mm is None or entry.error_mm <= args.max_position_error_mm
        ]

    if args.batch_sensor_methods:
        args.batch_save_dir.mkdir(parents=True, exist_ok=True)
        for output_stem, localization_key in SENSOR_METHOD_ERROR_SOURCES:
            batch_entries = load_entries(
                localization_key,
                localization_key,
                selected_method_label=output_stem,
            )
            plot_ground_truth_error_scatter_projection_pair(
                batch_entries,
                save_path=args.batch_save_dir / f"{output_stem}_xy_xz.png",
                show=False,
                marker_size=args.marker_size,
                arrow_length=args.arrow_length,
                colorbar_gap=args.colorbar_gap,
                colorbar_pair_gap=args.colorbar_pair_gap,
                colorbar_width=args.colorbar_width,
                colorbar_height=args.colorbar_height,
                xy_xlim=_parse_limit_pair(args.xy_xlim),
                xy_ylim=_parse_limit_pair(args.xy_ylim),
                xz_zlim=_parse_limit_pair(args.xz_zlim),
                dpi=args.dpi,
            )
        return

    entries = load_entries(
        args.position_error_source_key,
        args.orientation_error_source_key,
    )

    if args.plot == "xy":
        plot_ground_truth_error_scatter_xy(
            entries,
            save_path=args.save,
            show=not args.no_show,
            marker_size=args.marker_size,
            arrow_length=args.arrow_length,
            dpi=args.dpi,
        )
    elif args.plot in ("xy_xz", "yx_yz"):
        plot_ground_truth_error_scatter_projection_pair(
            entries,
            save_path=args.save,
            show=not args.no_show,
            marker_size=args.marker_size,
            arrow_length=args.arrow_length,
            colorbar_gap=args.colorbar_gap,
            colorbar_pair_gap=args.colorbar_pair_gap,
            colorbar_width=args.colorbar_width,
            colorbar_height=args.colorbar_height,
            xy_xlim=_parse_limit_pair(args.xy_xlim),
            xy_ylim=_parse_limit_pair(args.xy_ylim),
            xz_zlim=_parse_limit_pair(args.xz_zlim),
            projection_mode=args.plot,
            dpi=args.dpi,
        )
    else:
        plot_ground_truth_error_scatter_3d(
            entries,
            save_path=args.save,
            show=not args.no_show,
            marker_size=args.marker_size,
            arrow_length=args.arrow_length,
            view=args.view,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
