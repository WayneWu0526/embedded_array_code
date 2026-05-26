#!/usr/bin/env python3
"""Plot source, ground-truth, and estimated poses from cycle JSON files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Pose3D:
    name: str
    position: np.ndarray
    quaternion_xyzw: np.ndarray


@dataclass(frozen=True)
class CyclePoseData:
    cycle_path: Path
    cycle_id: Optional[int]
    mode: Optional[str]
    source_poses: List[Pose3D]
    ground_truth_pose: Optional[Pose3D]
    estimated_pose: Optional[Pose3D]
    raw: Dict[str, Any]


def _vector_from_xyz(data: Dict[str, Any], field_name: str) -> np.ndarray:
    try:
        return np.array([data["x"], data["y"], data["z"]], dtype=float)
    except KeyError as exc:
        raise ValueError(f"Missing {field_name}.{exc.args[0]}") from exc


def parse_pose(name: str, pose: Optional[Dict[str, Any]]) -> Optional[Pose3D]:
    """Parse pose dictionaries that use either rotation or orientation."""
    if pose is None:
        return None

    if "position" not in pose:
        raise ValueError(f"{name} pose is missing position")

    quat_data = pose.get("rotation", pose.get("orientation"))
    if quat_data is None:
        raise ValueError(f"{name} pose is missing rotation/orientation")

    position = _vector_from_xyz(pose["position"], f"{name}.position")
    quaternion = np.array(
        [quat_data["x"], quat_data["y"], quat_data["z"], quat_data["w"]],
        dtype=float,
    )
    norm = np.linalg.norm(quaternion)
    if norm == 0.0:
        raise ValueError(f"{name} quaternion has zero norm")

    return Pose3D(
        name=name,
        position=position,
        quaternion_xyzw=quaternion / norm,
    )


def load_cycle_pose_data(cycle_path: Path) -> CyclePoseData:
    with cycle_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    header = raw.get("header", {})
    slot_data = raw.get("slot_data", [])
    source_poses: List[Pose3D] = []

    for source_idx, slot in enumerate(slot_data[:3], start=1):
        parsed = parse_pose(f"source_{source_idx}", slot.get("pose"))
        if parsed is not None:
            source_poses.append(parsed)

    ground_truth_pose = parse_pose("ground_truth", raw.get("ground_truth_pose"))

    localization = raw.get("localization", {})
    estimated_pose = parse_pose("estimate", localization.get("pose"))

    return CyclePoseData(
        cycle_path=cycle_path,
        cycle_id=header.get("cycle_id"),
        mode=header.get("mode"),
        source_poses=source_poses,
        ground_truth_pose=ground_truth_pose,
        estimated_pose=estimated_pose,
        raw=raw,
    )


def load_result_pose_data(result_dir: Path) -> List[CyclePoseData]:
    cycle_paths = sorted(result_dir.glob("cycle_*.json"))
    if not cycle_paths:
        raise ValueError(f"No cycle_*.json files found in {result_dir}")

    return [load_cycle_pose_data(cycle_path) for cycle_path in cycle_paths]


def _sequence_index(cycle_data: CyclePoseData, fallback_idx: int) -> int:
    return cycle_data.cycle_id if cycle_data.cycle_id is not None else fallback_idx


def build_mode_error_series(
    loaded_results: Sequence[tuple[Path, Sequence[CyclePoseData]]],
    labels: Optional[Sequence[str]] = None,
) -> List[tuple[str, List[CyclePoseData]]]:
    if labels is not None and len(labels) != len(loaded_results):
        raise ValueError("--result-labels must match the number of --result-dir entries.")

    grouped: Dict[str, List[CyclePoseData]] = {}
    order: List[str] = []

    for result_idx, (result_dir, cycle_data_list) in enumerate(loaded_results):
        for cycle_data in cycle_data_list:
            if labels is not None:
                label = labels[result_idx]
            else:
                label = str(cycle_data.mode or result_dir.name)
            label = str(label).upper()
            if label not in grouped:
                grouped[label] = []
                order.append(label)
            grouped[label].append(cycle_data)

    preferred_order = [label for label in ("CVT", "CCI") if label in grouped]
    remaining_order = [label for label in order if label not in preferred_order]
    ordered_labels = preferred_order + remaining_order

    series = []
    for label in ordered_labels:
        data_list = grouped[label]
        sorted_data = [
            cycle_data
            for _, cycle_data in sorted(
                enumerate(data_list),
                key=lambda item: _sequence_index(item[1], item[0]),
            )
        ]
        series.append((label, sorted_data))

    return series


def average_quaternions_xyzw(quaternions: Sequence[np.ndarray]) -> np.ndarray:
    if not quaternions:
        raise ValueError("Cannot average an empty quaternion sequence")

    reference = quaternions[0]
    covariance = np.zeros((4, 4), dtype=float)
    for quaternion in quaternions:
        aligned = quaternion if np.dot(quaternion, reference) >= 0.0 else -quaternion
        covariance += np.outer(aligned, aligned)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    average = eigenvectors[:, np.argmax(eigenvalues)]
    if np.dot(average, reference) < 0.0:
        average = -average

    return average / np.linalg.norm(average)


def average_pose(name: str, poses: Sequence[Optional[Pose3D]]) -> Optional[Pose3D]:
    valid_poses = [pose for pose in poses if pose is not None]
    if not valid_poses:
        return None

    positions = np.array([pose.position for pose in valid_poses], dtype=float)
    quaternions = [pose.quaternion_xyzw for pose in valid_poses]
    return Pose3D(
        name=name,
        position=np.mean(positions, axis=0),
        quaternion_xyzw=average_quaternions_xyzw(quaternions),
    )


def quaternion_xyzw_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def set_axes_equal(ax: Any) -> None:
    """Use the same visual scale for x, y, and z axes."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    plot_radius = 0.5 * max(x_range, y_range, z_range, 1e-9)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))


def set_zoomed_axes_equal(
    ax: Any,
    center: np.ndarray,
    radius: float,
) -> None:
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))


def set_sparse_ticks(ax: Any, max_ticks: int = 4) -> None:
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks, prune=None))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks, prune=None))
    if hasattr(ax, "zaxis"):
        ax.zaxis.set_major_locator(MaxNLocator(nbins=max_ticks, prune=None))


def set_sequence_axis_ticks(ax: Any, step: int = 5) -> None:
    from math import ceil
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator

    max_sequence = 0.0
    for line in ax.get_lines():
        x_data = np.asarray(line.get_xdata(orig=False), dtype=float)
        if x_data.size:
            max_sequence = max(max_sequence, float(np.nanmax(x_data)))

    upper = max(step, int(ceil(max_sequence / step)) * step)
    ax.set_xlim(-0.35, upper + 0.35)
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))


def draw_pose_axes(
    ax: Any,
    pose: Pose3D,
    axis_length: float,
    marker: str,
    label_color: str,
    linestyle: str = "-",
    label: Optional[str] = None,
    alpha: float = 1.0,
    draw_text: bool = True,
    text: Optional[str] = None,
) -> None:
    rotation = quaternion_xyzw_to_rotation_matrix(pose.quaternion_xyzw)
    origin = pose.position

    ax.scatter(
        origin[0],
        origin[1],
        origin[2],
        marker=marker,
        s=55,
        color=label_color,
        label=label if label is not None else pose.name,
        alpha=alpha,
        depthshade=False,
    )
    if draw_text:
        ax.text(
            origin[0],
            origin[1],
            origin[2],
            f" {text if text is not None else pose.name}",
            color=label_color,
            alpha=alpha,
        )

    axis_colors = ("tab:red", "tab:green", "tab:blue")
    for axis_idx, axis_color in enumerate(axis_colors):
        direction = rotation[:, axis_idx] * axis_length
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            color=axis_color,
            linestyle=linestyle,
            arrow_length_ratio=0.18,
            linewidth=1.4,
            alpha=alpha,
        )


def _cylinder_surface(
    center: np.ndarray,
    rotation: np.ndarray,
    radius: float,
    length: float,
    num_theta: int = 48,
    num_length: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, num_theta)
    axial = np.linspace(-length / 2.0, length / 2.0, num_length)
    theta_grid, axial_grid = np.meshgrid(theta, axial)

    local_points = np.stack(
        [
            radius * np.cos(theta_grid),
            radius * np.sin(theta_grid),
            axial_grid,
        ],
        axis=-1,
    )
    world_points = center + local_points @ rotation.T
    return world_points[..., 0], world_points[..., 1], world_points[..., 2]


def _circle_points(
    center: np.ndarray,
    rotation: np.ndarray,
    radius: float,
    axial_offset: float,
    num_theta: int = 48,
) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, num_theta)
    local_points = np.stack(
        [
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.full_like(theta, axial_offset),
        ],
        axis=-1,
    )
    return center + local_points @ rotation.T


def draw_electromagnet(
    ax: Any,
    pose: Pose3D,
    core_diameter: float = 0.040,
    coil_outer_diameter: float = 0.056,
    core_length: float = 0.250,
    coil_length: float = 0.230,
    core_alpha: float = 1.0,
    coil_alpha: float = 1.0,
    line_alpha: float = 0.95,
) -> None:
    rotation = quaternion_xyzw_to_rotation_matrix(pose.quaternion_xyzw)
    center = pose.position

    core_radius = core_diameter / 2.0
    coil_outer_radius = coil_outer_diameter / 2.0

    core_x, core_y, core_z = _cylinder_surface(
        center=center,
        rotation=rotation,
        radius=core_radius,
        length=core_length,
    )
    ax.plot_surface(
        core_x,
        core_y,
        core_z,
        color="#5a5a5a",
        alpha=core_alpha,
        linewidth=0,
        antialiased=True,
        shade=True,
    )

    coil_x, coil_y, coil_z = _cylinder_surface(
        center=center,
        rotation=rotation,
        radius=coil_outer_radius,
        length=coil_length,
    )
    ax.plot_surface(
        coil_x,
        coil_y,
        coil_z,
        color="#b87333",
        alpha=coil_alpha,
        linewidth=0,
        antialiased=True,
        shade=True,
    )

    z_axis = rotation[:, 2]
    endpoint_a = center - z_axis * core_length / 2.0
    endpoint_b = center + z_axis * core_length / 2.0
    ax.plot(
        [endpoint_a[0], endpoint_b[0]],
        [endpoint_a[1], endpoint_b[1]],
        [endpoint_a[2], endpoint_b[2]],
        color="0.12",
        linewidth=0.8,
        alpha=line_alpha,
    )

    for axial_offset in (-coil_length / 2.0, coil_length / 2.0):
        ring = _circle_points(
            center=center,
            rotation=rotation,
            radius=coil_outer_radius,
            axial_offset=axial_offset,
        )
        ax.plot(
            ring[:, 0],
            ring[:, 1],
            ring[:, 2],
            color="#7a431b",
            linewidth=0.6,
            alpha=line_alpha,
        )


def prepare_matplotlib(show: bool, use_tex: bool = False, font_size: float = 16.0) -> Any:
    if not show:
        import matplotlib

        matplotlib.use("Agg", force=True)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  Registers 3D projection.
    import matplotlib.pyplot as plt

    if use_tex:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{bm}",
                "font.family": "serif",
                "font.size": font_size,
                "axes.titlesize": font_size,
                "axes.labelsize": font_size,
                "xtick.labelsize": font_size,
                "ytick.labelsize": font_size,
                "legend.fontsize": font_size,
            }
        )

    return plt


def plot_cycle_pose(
    cycle_data: CyclePoseData,
    axis_length: float = 0.05,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    plt = prepare_matplotlib(show)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    source_colors = ("tab:purple", "tab:orange", "tab:cyan")
    for idx, pose in enumerate(cycle_data.source_poses):
        draw_pose_axes(
            ax,
            pose,
            axis_length=axis_length,
            marker="^",
            label_color=source_colors[idx % len(source_colors)],
        )

    if cycle_data.ground_truth_pose is not None:
        draw_pose_axes(
            ax,
            cycle_data.ground_truth_pose,
            axis_length=axis_length,
            marker="o",
            label_color="tab:green",
        )

    if cycle_data.estimated_pose is not None:
        draw_pose_axes(
            ax,
            cycle_data.estimated_pose,
            axis_length=axis_length,
            marker="x",
            label_color="tab:red",
            linestyle="--",
        )

    title_bits = [cycle_data.cycle_path.name]
    if cycle_data.mode is not None:
        title_bits.append(f"mode={cycle_data.mode}")
    if cycle_data.cycle_id is not None:
        title_bits.append(f"cycle_id={cycle_data.cycle_id}")

    ax.set_title(" | ".join(title_bits))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.35)
    set_axes_equal(ax)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, facecolor="white", transparent=False)

    if show:
        plt.show()
    else:
        plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot source, ground-truth, and estimated poses from cycle JSON files.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--cycle",
        type=Path,
        help="Path to cycle_XXXX.json.",
    )
    input_group.add_argument(
        "--result-dir",
        type=Path,
        nargs="+",
        help=(
            "Directory or directories containing cycle_*.json files. "
            "The first directory is used for the 3D view; all directories are "
            "grouped by mode for the error plots."
        ),
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, only the interactive window is shown.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional output directory. The image name is derived from the cycle file name.",
    )
    parser.add_argument(
        "--result-labels",
        nargs="+",
        default=None,
        help=(
            "Optional labels for --result-dir entries, e.g. CVT CCI. "
            "Use this when the JSON header.mode does not distinguish datasets."
        ),
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.05,
        help="Length of each pose coordinate axis in meters.",
    )
    parser.add_argument(
        "--view",
        choices=("default", "xy", "xy2d"),
        default="default",
        help="Overlay camera view. Use xy for a top-down 3D view, or xy2d for a 2D projection.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window. Useful with --save on headless machines.",
    )
    return parser


def _positions(poses: Sequence[Optional[Pose3D]]) -> np.ndarray:
    return np.array([pose.position for pose in poses if pose is not None], dtype=float)


def _plot_trajectory(
    ax: Any,
    poses: Sequence[Optional[Pose3D]],
    color: str,
    label: str,
    linestyle: str = "-",
    marker: str = ".",
    linewidth: float = 1.8,
    markersize: float = 4.5,
) -> None:
    points = _positions(poses)
    if len(points) == 0:
        return

    ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markersize=markersize,
        label=label,
    )


def _plot_error_segments(
    ax: Any,
    ground_truth_poses: Sequence[Optional[Pose3D]],
    estimated_poses: Sequence[Optional[Pose3D]],
) -> None:
    for ground_truth_pose, estimated_pose in zip(ground_truth_poses, estimated_poses):
        if ground_truth_pose is None or estimated_pose is None:
            continue
        points = np.vstack([ground_truth_pose.position, estimated_pose.position])
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color="0.55",
            linestyle="-",
            linewidth=0.45,
            alpha=0.40,
        )


def _annotate_trajectory_endpoint(
    ax: Any,
    poses: Sequence[Optional[Pose3D]],
    text: str,
    color: str,
    offset: np.ndarray,
    use_first: bool = False,
) -> None:
    valid_poses = [pose for pose in poses if pose is not None]
    if not valid_poses:
        return

    pose = valid_poses[0] if use_first else valid_poses[-1]
    position = pose.position + offset
    ax.text(position[0], position[1], position[2], text, color=color)


def _middle_valid_pose(poses: Sequence[Optional[Pose3D]]) -> Optional[Pose3D]:
    valid_poses = [pose for pose in poses if pose is not None]
    if not valid_poses:
        return None
    return valid_poses[len(valid_poses) // 2]


def _position_errors_mm(
    ground_truth_poses: Sequence[Optional[Pose3D]],
    estimated_poses: Sequence[Optional[Pose3D]],
) -> tuple[np.ndarray, np.ndarray]:
    indices = []
    errors = []
    for idx, (ground_truth_pose, estimated_pose) in enumerate(
        zip(ground_truth_poses, estimated_poses)
    ):
        if ground_truth_pose is None or estimated_pose is None:
            continue
        indices.append(idx)
        errors.append(np.linalg.norm(estimated_pose.position - ground_truth_pose.position) * 1000.0)

    return np.array(indices, dtype=int), np.array(errors, dtype=float)


def _localization_position_errors_mm(
    cycle_data_list: Sequence[CyclePoseData],
    localization_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    indices = []
    errors = []

    for idx, data in enumerate(cycle_data_list):
        localization = data.raw.get(localization_key, {})
        raw_error = localization.get("position_error")
        if raw_error is not None:
            error = float(raw_error) * 1000.0
        else:
            localization_pose = parse_pose(
                f"{localization_key}_estimate",
                localization.get("pose"),
            )
            if data.ground_truth_pose is None or localization_pose is None:
                continue
            error = (
                np.linalg.norm(localization_pose.position - data.ground_truth_pose.position)
                * 1000.0
            )

        if np.isfinite(error):
            indices.append(_sequence_index(data, idx))
            errors.append(error)

    return np.array(indices, dtype=int), np.array(errors, dtype=float)


def _orientation_errors_rad(cycle_data_list: Sequence[CyclePoseData]) -> tuple[np.ndarray, np.ndarray]:
    indices = []
    errors = []

    for idx, data in enumerate(cycle_data_list):
        localization = data.raw.get("localization", {})
        raw_error = localization.get("orientation_error")
        if raw_error is not None:
            error = float(raw_error)
        elif data.ground_truth_pose is not None and data.estimated_pose is not None:
            dot = abs(
                float(
                    np.dot(
                        data.ground_truth_pose.quaternion_xyzw,
                        data.estimated_pose.quaternion_xyzw,
                    )
                )
            )
            error = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))
        else:
            continue

        if np.isfinite(error):
            indices.append(idx)
            errors.append(error)

    return np.array(indices, dtype=int), np.array(errors, dtype=float)


def _localization_orientation_errors_rad(
    cycle_data_list: Sequence[CyclePoseData],
    localization_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    indices = []
    errors = []

    for idx, data in enumerate(cycle_data_list):
        localization = data.raw.get(localization_key, {})
        raw_error = localization.get("orientation_error")
        if raw_error is not None:
            error = float(raw_error)
        else:
            localization_pose = parse_pose(
                f"{localization_key}_estimate",
                localization.get("pose"),
            )
            if data.ground_truth_pose is None or localization_pose is None:
                continue
            dot = abs(
                float(
                    np.dot(
                        data.ground_truth_pose.quaternion_xyzw,
                        localization_pose.quaternion_xyzw,
                    )
                )
            )
            error = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))

        if np.isfinite(error):
            indices.append(_sequence_index(data, idx))
            errors.append(error)

    return np.array(indices, dtype=int), np.array(errors, dtype=float)


def plot_position_error_curve(
    ax: Any,
    ground_truth_poses: Sequence[Optional[Pose3D]],
    estimated_poses: Sequence[Optional[Pose3D]],
    max_ticks: int = 4,
) -> bool:
    cycle_indices, errors_mm = _position_errors_mm(ground_truth_poses, estimated_poses)
    if len(errors_mm) == 0:
        return False

    ax.plot(
        cycle_indices,
        errors_mm,
        color="#d62728",
        marker="o",
        markersize=3.0,
        linewidth=1.2,
    )
    ax.set_xlabel(r"Sequence")
    ax.set_ylabel(r"$e_{\bm{p}}$ [mm]")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    set_sparse_ticks(ax, max_ticks=max_ticks)
    return True


def plot_orientation_error_curve(
    ax: Any,
    cycle_data_list: Sequence[CyclePoseData],
    max_ticks: int = 4,
) -> bool:
    cycle_indices, errors_rad = _orientation_errors_rad(cycle_data_list)
    if len(errors_rad) == 0:
        return False

    ax.plot(
        cycle_indices,
        errors_rad,
        color="#1f77b4",
        marker="s",
        markersize=3.0,
        linewidth=1.2,
    )
    ax.set_xlabel(r"Sequence")
    ax.set_ylabel(r"$e_{\bm{R}}$ [rad]")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    set_sparse_ticks(ax, max_ticks=max_ticks)
    return True


def _mode_plot_style(label: str) -> Dict[str, Any]:
    mode = label.upper()
    if mode == "CONFIG1-CVT":
        return {"color": "#b2182b", "marker": "o", "linestyle": "-"}
    if mode == "CONFIG1-CCI":
        return {"color": "#ef8a62", "marker": "o", "linestyle": "-"}
    if mode == "CONFIG2-CVT":
        return {"color": "#2166ac", "marker": "D", "linestyle": "-"}
    if mode == "CONFIG2-CCI":
        return {"color": "#67a9cf", "marker": "D", "linestyle": "-"}
    if mode == "CONFIG3-CVT":
        return {"color": "#1b7837", "marker": "^", "linestyle": "-"}
    if mode == "CONFIG3-CCI":
        return {"color": "#7fbf7b", "marker": "^", "linestyle": "-"}
    if mode == "CVT":
        return {"color": "#b2182b", "marker": "o", "linestyle": "-"}
    if mode == "CCI":
        return {"color": "#ef8a62", "marker": "s", "linestyle": "-"}
    return {"color": "0.35", "marker": "^", "linestyle": "--"}


def build_localization_error_series(
    cycle_data_list: Sequence[CyclePoseData],
) -> List[tuple[str, str, Sequence[CyclePoseData]]]:
    series: List[tuple[str, str, Sequence[CyclePoseData]]] = []

    if any("localization" in data.raw for data in cycle_data_list):
        series.append(("Config1-CVT", "localization", cycle_data_list))
    if any("localization_cci" in data.raw for data in cycle_data_list):
        series.append(("Config1-CCI", "localization_cci", cycle_data_list))
    if any("localization_sensors_1346791012" in data.raw for data in cycle_data_list):
        series.append(
            (
                "Config2-CVT",
                "localization_sensors_1346791012",
                cycle_data_list,
            )
        )
    if any("localization_cci_sensors_1346791012" in data.raw for data in cycle_data_list):
        series.append(
            (
                "Config2-CCI",
                "localization_cci_sensors_1346791012",
                cycle_data_list,
            )
        )
    if any("localization_sensors_2312" in data.raw for data in cycle_data_list):
        series.append(("Config3-CVT", "localization_sensors_2312", cycle_data_list))
    if any("localization_cci_sensors_2312" in data.raw for data in cycle_data_list):
        series.append(("Config3-CCI", "localization_cci_sensors_2312", cycle_data_list))

    return series


def plot_mode_position_error_curves(
    ax: Any,
    mode_series: Sequence[tuple[str, Sequence[CyclePoseData]]],
    max_ticks: int = 4,
) -> bool:
    plotted = False
    for label, cycle_data_list in mode_series:
        ground_truth_poses = [data.ground_truth_pose for data in cycle_data_list]
        estimated_poses = [data.estimated_pose for data in cycle_data_list]
        cycle_indices, errors_mm = _position_errors_mm(
            ground_truth_poses,
            estimated_poses,
        )
        if len(errors_mm) == 0:
            continue

        ax.plot(
            cycle_indices,
            errors_mm,
            label=label,
            markersize=3.0,
            linewidth=1.2,
            **_mode_plot_style(label),
        )
        plotted = True

    if not plotted:
        return False

    ax.set_xlabel(r"Sequence")
    ax.set_ylabel(r"$e_{\bm{p}}$ [mm]")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    set_sparse_ticks(ax, max_ticks=max_ticks)
    return True


def plot_localization_position_error_curves(
    ax: Any,
    localization_series: Sequence[tuple[str, str, Sequence[CyclePoseData]]],
    max_ticks: int = 4,
) -> bool:
    plotted = False
    for label, localization_key, cycle_data_list in localization_series:
        cycle_indices, errors_mm = _localization_position_errors_mm(
            cycle_data_list,
            localization_key,
        )
        if len(errors_mm) == 0:
            continue

        ax.plot(
            cycle_indices,
            errors_mm,
            label=label,
            markersize=3.0,
            linewidth=1.2,
            **_mode_plot_style(label),
        )
        plotted = True

    if not plotted:
        return False

    ax.set_xlabel(r"Sequence")
    ax.set_ylabel(r"$e_{\bm{p}}$ [mm]")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    set_sparse_ticks(ax, max_ticks=max_ticks)
    return True


def plot_mode_orientation_error_curves(
    ax: Any,
    mode_series: Sequence[tuple[str, Sequence[CyclePoseData]]],
    max_ticks: int = 4,
) -> bool:
    plotted = False
    for label, cycle_data_list in mode_series:
        cycle_indices, errors_rad = _orientation_errors_rad(cycle_data_list)
        if len(errors_rad) == 0:
            continue

        ax.plot(
            cycle_indices,
            errors_rad,
            label=label,
            markersize=3.0,
            linewidth=1.2,
            **_mode_plot_style(label),
        )
        plotted = True

    if not plotted:
        return False

    ax.set_xlabel(r"Sequence")
    ax.set_ylabel(r"$e_{\bm{R}}$ [rad]")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    set_sparse_ticks(ax, max_ticks=max_ticks)
    return True


def plot_localization_orientation_error_curves(
    ax: Any,
    localization_series: Sequence[tuple[str, str, Sequence[CyclePoseData]]],
    max_ticks: int = 4,
) -> bool:
    plotted = False
    for label, localization_key, cycle_data_list in localization_series:
        cycle_indices, errors_rad = _localization_orientation_errors_rad(
            cycle_data_list,
            localization_key,
        )
        if len(errors_rad) == 0:
            continue

        ax.plot(
            cycle_indices,
            errors_rad,
            label=label,
            markersize=3.0,
            linewidth=1.2,
            **_mode_plot_style(label),
        )
        plotted = True

    if not plotted:
        return False

    ax.set_xlabel(r"Sequence")
    ax.set_ylabel(r"$e_{\bm{R}}$ [rad]")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    set_sparse_ticks(ax, max_ticks=max_ticks)
    return True


def add_error_inset(
    fig: Any,
    ground_truth_poses: Sequence[Optional[Pose3D]],
    estimated_poses: Sequence[Optional[Pose3D]],
) -> None:
    inset = fig.add_axes([0.12, 0.70, 0.28, 0.20])
    if not plot_position_error_curve(
        inset,
        ground_truth_poses,
        estimated_poses,
        max_ticks=3,
    ):
        inset.remove()


def add_local_xy_zoom(
    ax: Any,
    ground_truth_poses: Sequence[Optional[Pose3D]],
    estimated_poses: Sequence[Optional[Pose3D]],
) -> None:
    ground_truth_points = _xy_points(ground_truth_poses)
    estimated_points = _xy_points(estimated_poses)
    if len(ground_truth_points) == 0 and len(estimated_points) == 0:
        return

    point_sets = [
        points for points in (ground_truth_points, estimated_points) if len(points) > 0
    ]
    all_points = np.vstack(point_sets)
    lower = np.min(all_points, axis=0)
    upper = np.max(all_points, axis=0)
    span = np.maximum(upper - lower, 1e-9)
    padding = max(0.006, 0.25 * float(np.max(span)))
    center = 0.5 * (lower + upper)
    radius = 0.5 * float(np.max(span)) + padding

    inset_bounds = [0.055, 0.47, 0.40, 0.40]
    parent_bounds = ax.get_position()
    inset = ax.figure.add_axes(
        [
            parent_bounds.x0 + inset_bounds[0] * parent_bounds.width,
            parent_bounds.y0 + inset_bounds[1] * parent_bounds.height,
            inset_bounds[2] * parent_bounds.width,
            inset_bounds[3] * parent_bounds.height,
        ],
        zorder=100,
    )
    inset.set_zorder(100)
    inset.set_axisbelow(False)
    inset.patch.set_visible(True)
    inset.patch.set_facecolor("white")
    inset.patch.set_edgecolor("black")
    inset.patch.set_alpha(1.0)
    inset.set_facecolor((1.0, 1.0, 1.0, 1.0))
    for spine in inset.spines.values():
        spine.set_alpha(1.0)
        spine.set_linewidth(0.9)
    _plot_xy_error_segments(inset, ground_truth_poses, estimated_poses)
    _plot_xy_trajectory(
        inset,
        ground_truth_poses,
        color="black",
        marker="o",
        linewidth=0.5,
        markersize=2.4,
        zorder=5,
    )
    _plot_xy_trajectory(
        inset,
        estimated_poses,
        color="#d62728",
        linestyle="--",
        marker="^",
        linewidth=0.5,
        markersize=2.6,
        zorder=6,
    )
    inset.set_xlim(center[0] - 0.8 * radius, center[0] + 0.8 * radius)
    inset.set_ylim(center[1] - 0.6 * radius, center[1] + 0.6 * radius)
    inset.set_aspect("equal", adjustable="box")
    inset.grid(True, alpha=0.25, linewidth=0.5)
    inset.tick_params(axis="both", labelsize=12, pad=1)
    set_sparse_ticks(inset, max_ticks=2)


def draw_electromagnet_xy_projection(
    ax: Any,
    pose: Pose3D,
    outer_diameter: float = 0.056,
    core_length: float = 0.250,
    coil_length: float = 0.230,
) -> None:
    from matplotlib.patches import Rectangle

    rotation = quaternion_xyzw_to_rotation_matrix(pose.quaternion_xyzw)
    center = pose.position[:2]
    direction = rotation[:2, 2]
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-9:
        return

    direction = direction / direction_norm
    normal = np.array([-direction[1], direction[0]])
    angle_deg = np.degrees(np.arctan2(direction[1], direction[0]))

    coil_center = center - direction * coil_length / 2.0 - normal * outer_diameter / 2.0
    coil = Rectangle(
        coil_center,
        coil_length,
        outer_diameter,
        angle=angle_deg,
        facecolor="#b87333",
        edgecolor="#7a431b",
        linewidth=0.8,
        alpha=1.0,
        zorder=1,
    )
    ax.add_patch(coil)

    core_center = center - direction * core_length / 2.0
    core_end = center + direction * core_length / 2.0
    ax.plot(
        [core_center[0], core_end[0]],
        [core_center[1], core_end[1]],
        color="#4a4a4a",
        linewidth=2.4,
        solid_capstyle="round",
        zorder=2,
    )


def _xy_points(poses: Sequence[Optional[Pose3D]]) -> np.ndarray:
    return np.array([pose.position[:2] for pose in poses if pose is not None], dtype=float)


def _plot_xy_trajectory(
    ax: Any,
    poses: Sequence[Optional[Pose3D]],
    color: str,
    linestyle: str = "-",
    marker: str = ".",
    linewidth: float = 1.5,
    markersize: float = 3.0,
    zorder: int = 4,
) -> None:
    points = _xy_points(poses)
    if len(points) == 0:
        return
    ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        linestyle=linestyle,
        marker=marker,
        linewidth=linewidth,
        markersize=markersize,
        zorder=zorder,
    )


def _plot_xy_error_segments(
    ax: Any,
    ground_truth_poses: Sequence[Optional[Pose3D]],
    estimated_poses: Sequence[Optional[Pose3D]],
) -> None:
    for ground_truth_pose, estimated_pose in zip(ground_truth_poses, estimated_poses):
        if ground_truth_pose is None or estimated_pose is None:
            continue
        points = np.vstack([ground_truth_pose.position[:2], estimated_pose.position[:2]])
        ax.plot(
            points[:, 0],
            points[:, 1],
            color="0.55",
            linewidth=0.55,
            alpha=0.65,
            zorder=3,
        )


def plot_result_pose_overlay_xy2d(
    cycle_data_list: Sequence[CyclePoseData],
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    if not cycle_data_list:
        raise ValueError("cycle_data_list is empty")

    plt = prepare_matplotlib(show, use_tex=True, font_size=16.0)
    fig, ax = plt.subplots(figsize=(17.6 / 2.54, 12.6 / 2.54))

    source_names = ("Source 1", "Source 2", "Source 3")
    for source_idx, source_name in enumerate(source_names):
        source_sequence = [
            data.source_poses[source_idx]
            if len(data.source_poses) > source_idx
            else None
            for data in cycle_data_list
        ]
        source_pose = average_pose(source_name, source_sequence)
        if source_pose is not None:
            draw_electromagnet_xy_projection(ax, source_pose)

    ground_truth_poses = [data.ground_truth_pose for data in cycle_data_list]
    estimated_poses = [data.estimated_pose for data in cycle_data_list]
    _plot_xy_error_segments(ax, ground_truth_poses, estimated_poses)
    _plot_xy_trajectory(
        ax,
        ground_truth_poses,
        color="black",
        marker="o",
        linewidth=1.5,
        markersize=2.8,
        zorder=5,
    )
    _plot_xy_trajectory(
        ax,
        estimated_poses,
        color="#d62728",
        linestyle="--",
        marker="^",
        linewidth=1.3,
        markersize=3.2,
        zorder=6,
    )

    ground_truth_points = _xy_points(ground_truth_poses)
    estimated_points = _xy_points(estimated_poses)
    if len(ground_truth_points) > 0:
        ax.text(
            ground_truth_points[0, 0] - 0.035,
            ground_truth_points[0, 1] + 0.020,
            r"GT",
            color="black",
        )
    if len(estimated_points) > 0:
        ax.text(
            estimated_points[-1, 0] + 0.018,
            estimated_points[-1, 1] - 0.012,
            r"Estimate",
            color="#d62728",
        )

    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.grid(True, alpha=0.30)
    ax.set_aspect("equal", adjustable="box")
    set_sparse_ticks(ax, max_ticks=4)
    add_error_inset(fig, ground_truth_poses, estimated_poses)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.96)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, facecolor="white", transparent=False)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_result_pose_overlay(
    cycle_data_list: Sequence[CyclePoseData],
    mode_error_series: Optional[Sequence[tuple[str, Sequence[CyclePoseData]]]] = None,
    axis_length: float = 0.035,
    save_path: Optional[Path] = None,
    show: bool = True,
    view: str = "default",
) -> None:
    if not cycle_data_list:
        raise ValueError("cycle_data_list is empty")

    plt = prepare_matplotlib(show, use_tex=True, font_size=16.0)
    fig = plt.figure(figsize=(17.6 / 2.54, 10.0 / 2.54))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(2.25, 1.55),
        height_ratios=(1.0, 1.0),
        left=0.025,
        right=0.985,
        bottom=0.14,
        top=0.86,
        wspace=0.34,
        hspace=0.30,
    )
    ax = fig.add_subplot(grid[:, 0], projection="3d")
    position_error_ax = fig.add_subplot(grid[0, 1])
    orientation_error_ax = fig.add_subplot(grid[1, 1], sharex=position_error_ax)

    source_names = ("Source 1", "Source 2", "Source 3")
    ground_truth_poses = [data.ground_truth_pose for data in cycle_data_list]
    estimated_poses = [data.estimated_pose for data in cycle_data_list]
    source_poses_for_view: List[Pose3D] = []
    for source_idx, source_name in enumerate(source_names):
        source_sequence = [
            data.source_poses[source_idx]
            if len(data.source_poses) > source_idx
            else None
            for data in cycle_data_list
        ]
        source_pose = average_pose(source_name, source_sequence)
        if source_pose is not None:
            source_poses_for_view.append(source_pose)

    view_points = [pose.position for pose in source_poses_for_view]
    view_points.extend(
        pose.position
        for pose in ground_truth_poses + estimated_poses
        if pose is not None
    )
    zoom_center = None
    zoom_radius = 0.168
    if view_points:
        stacked_view_points = np.vstack(view_points)
        zoom_center = np.mean(stacked_view_points, axis=0) + np.array([-0.150, 0.300, 0.0])

    for source_pose in source_poses_for_view:
        draw_electromagnet(
            ax,
            source_pose,
            core_diameter=0.040,
            coil_outer_diameter=0.056,
            core_length=0.250,
            coil_length=0.230,
            core_alpha=0.42,
            coil_alpha=0.34,
            line_alpha=0.50,
        )
        draw_pose_axes(
            ax,
            source_pose,
            axis_length=0.080,
            marker=".",
            label_color="0.25",
            label="_nolegend_",
            alpha=0.72,
            draw_text=False,
        )

    localization_error_series = build_localization_error_series(cycle_data_list)
    if mode_error_series is None:
        if localization_error_series:
            mode_error_series = [
                (label, series_cycle_data_list)
                for label, _, series_cycle_data_list in localization_error_series
            ]
        else:
            mode_label = str(cycle_data_list[0].mode or "Result").upper()
            mode_error_series = [(mode_label, cycle_data_list)]

    _plot_error_segments(ax, ground_truth_poses, estimated_poses)

    _plot_trajectory(
        ax,
        ground_truth_poses,
        color="black",
        label=r"GT",
        marker="o",
        linewidth=0.5,
        markersize=3.2,
    )
    _plot_trajectory(
        ax,
        estimated_poses,
        color="#d62728",
        label=r"Estimate",
        linestyle="--",
        marker="^",
        linewidth=0.5,
        markersize=3.4,
    )
    ground_truth_frame_pose = _middle_valid_pose(ground_truth_poses)
    if ground_truth_frame_pose is not None:
        draw_pose_axes(
            ax,
            ground_truth_frame_pose,
            axis_length=0.026,
            marker=".",
            label_color="black",
            label="_nolegend_",
            alpha=0.85,
            draw_text=False,
        )

    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.set_zlabel(r"$z$ [m]")
    ax.xaxis.labelpad = 14
    ax.yaxis.labelpad = 14
    ax.zaxis.labelpad = 4
    if view == "xy":
        ax.view_init(elev=90, azim=-90)
    ax.grid(True, alpha=0.35)
    if zoom_center is not None:
        set_zoomed_axes_equal(ax, zoom_center, zoom_radius)
    else:
        set_axes_equal(ax)
    set_sparse_ticks(ax, max_ticks=3)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.96, 1.04),
        frameon=True,
        handlelength=2.4,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        fontsize=12,
    )
    ax.text2D(0.02, 0.94, r"(a)", transform=ax.transAxes)
    add_local_xy_zoom(ax, ground_truth_poses, estimated_poses)

    if localization_error_series:
        position_plotted = plot_localization_position_error_curves(
            position_error_ax,
            localization_error_series,
            max_ticks=5,
        )
    else:
        position_plotted = plot_mode_position_error_curves(
            position_error_ax,
            mode_error_series,
            max_ticks=5,
        )

    if position_plotted:
        if len(position_error_ax.get_lines()) > 1:
            position_error_ax.legend(
                loc="lower right",
                bbox_to_anchor=(1.0, 1.02),
                ncol=3,
                frameon=False,
                handlelength=0.9,
                labelspacing=0.10,
                handletextpad=0.25,
                borderaxespad=0.2,
                columnspacing=0.35,
                fontsize=12,
            )
        position_error_ax.text(
            -0.30,
            1.04,
            r"(b)",
            transform=position_error_ax.transAxes,
            clip_on=False,
            va="top",
        )
        position_error_ax.set_xlabel("")
        position_error_ax.tick_params(axis="x", labelbottom=False)
    else:
        position_error_ax.set_visible(False)

    if localization_error_series:
        orientation_plotted = plot_localization_orientation_error_curves(
            orientation_error_ax,
            localization_error_series,
            max_ticks=5,
        )
    else:
        orientation_plotted = plot_mode_orientation_error_curves(
            orientation_error_ax,
            mode_error_series,
            max_ticks=5,
        )

    if orientation_plotted:
        orientation_error_ax.text(
            -0.30,
            1.04,
            r"(c)",
            transform=orientation_error_ax.transAxes,
            clip_on=False,
            va="top",
        )
    else:
        orientation_error_ax.set_visible(False)

    for error_ax in (position_error_ax, orientation_error_ax):
        if error_ax.get_visible():
            set_sequence_axis_ticks(error_ax, step=5)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.save is not None and args.save_dir is not None:
        raise ValueError("Use either --save or --save-dir, not both.")
    if args.result_labels is not None and args.result_dir is None:
        raise ValueError("--result-labels can only be used with --result-dir.")
    if (
        args.result_labels is not None
        and args.result_dir is not None
        and len(args.result_labels) != len(args.result_dir)
    ):
        raise ValueError("--result-labels must match the number of --result-dir entries.")

    save_path = args.save
    if args.cycle is not None and args.save_dir is not None:
        save_path = args.save_dir / f"{args.cycle.stem}_pose.png"
    elif args.result_dir is not None and args.save_dir is not None:
        save_path = args.save_dir / f"{args.result_dir[0].name}_pose_overlay.png"

    if args.cycle is not None:
        cycle_data = load_cycle_pose_data(args.cycle)
        plot_cycle_pose(
            cycle_data,
            axis_length=args.axis_length,
            save_path=save_path,
            show=not args.no_show,
        )
    else:
        loaded_results = [
            (result_dir, load_result_pose_data(result_dir))
            for result_dir in args.result_dir
        ]
        cycle_data_list = list(loaded_results[0][1])
        if args.view == "xy2d":
            plot_result_pose_overlay_xy2d(
                cycle_data_list,
                save_path=save_path,
                show=not args.no_show,
            )
        else:
            plot_result_pose_overlay(
                cycle_data_list,
                mode_error_series=build_mode_error_series(
                    loaded_results,
                    labels=args.result_labels,
                ),
                axis_length=args.axis_length,
                save_path=save_path,
                show=not args.no_show,
                view=args.view,
            )


if __name__ == "__main__":
    main()
