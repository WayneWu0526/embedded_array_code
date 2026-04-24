#!/usr/bin/env python3
"""
Noise analysis: random ground-truth pose perturbation study.

Generates N random (R, p) ground truths near the original,
injects noise into B_meas at different levels, runs MaPS_Estimator,
and plots position/orientation error vs noise level with error bars.
Supports sweeping magnetic moment magnitude across multiple runs.

Usage:
    python noise_analysis_rp.py
    python noise_analysis_rp.py --magnitudes 100 200 300
"""

import sys
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from offline_utils import json_to_request
from maps_estimator import MaPS_Estimator
from localization_service_node import (
    build_D_cal,
    quaternion_to_rotation_matrix,
    compute_rotation_error,
    load_configuration,
    GS_TO_TESLA,
)
from mag_dipole_model import mag_dipole_model


def sample_random_rotation():
    """Sample a random rotation matrix uniformly from SO(3) using quaternion method."""
    q = np.random.randn(4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - w*z),         2*(x*z + w*y)],
        [    2*(x*y + w*z),     1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
        [    2*(x*z - w*y),         2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
    ])
    return R


def sample_random_position(center, radius=0.05):
    """Sample a random point uniformly within a sphere of given radius."""
    direction = np.random.randn(3)
    direction = direction / np.linalg.norm(direction)
    r = radius * (np.random.rand() ** (1.0 / 3.0))
    return center + r * direction


def generate_synthetic_B_meas_for_pose(p_sensor_array, R_sensor_array, D_LIST,
                                        gs_to_tesla,
                                        sources, sensor_ids, noise_level=0.0):
    """
    Generate synthetic B_meas for a given sensor array pose.

    Args:
        p_sensor_array: 3D position of sensor array center (global frame)
        R_sensor_array: 3x3 rotation matrix of sensor array orientation
        D_LIST: (12, 3) array of sensor positions relative to array center
        gs_to_tesla: conversion factor from Gs to Tesla
        sources: list of dicts with 'p_Ci' (3,) and 'm_Ci' (3,)
        sensor_ids: list of active sensor IDs (1-12)
        noise_level: noise standard deviation in Gs

    Returns:
        B_meas_cell: list of 3 x N matrices (one per source/slot)
        b_local_norm: float, |b_local| in Gs at array center (no offset)
    """
    B_meas_cell = []
    b_local_norms = []  # per-source clean field norms at center

    for src in sources:
        p_Ci = src['p_Ci']
        m_Ci = src['m_Ci']

        # Clean field at center (no offset)
        b_global_center, _ = mag_dipole_model(p_sensor_array, m_Ci, p_Ci, order=1)
        b_sensor_center = R_sensor_array.T @ b_global_center
        b_local_norms.append(np.linalg.norm(b_sensor_center))

        B_meas = np.zeros((3, len(sensor_ids)))
        for col_idx, sid in enumerate(sensor_ids):
            sensor_idx = sid - 1
            d_j = D_LIST[sensor_idx]
            p_sensor_global = p_sensor_array + R_sensor_array @ d_j

            b_global, _ = mag_dipole_model(p_sensor_global, m_Ci, p_Ci, order=1)
            b_sensor = R_sensor_array.T @ b_global

            b_sensor_gs = b_sensor / gs_to_tesla
            if noise_level > 0:
                b_sensor_gs = b_sensor_gs + noise_level * np.random.randn(3)

            B_meas[:, col_idx] = b_sensor_gs

        B_meas_cell.append(B_meas)

    b_local_norm = np.linalg.norm(b_local_norms)  # RSS across sources, in Gs
    return B_meas_cell, b_local_norm


def build_sources_from_json(json_path, moment_magnitude):
    """
    Build source list from cycle JSON with uniform moment magnitude for all slots.
    """
    req = json_to_request(json_path)
    slot_dict = {slot.slot: slot for slot in req.slot_data}

    sources = []
    for slot_idx in [0, 1, 2]:
        if slot_idx not in slot_dict:
            continue
        slot = slot_dict[slot_idx]
        p_Ci = np.array([slot.pose.position.x, slot.pose.position.y, slot.pose.position.z])
        quat = np.array([slot.pose.orientation.w, slot.pose.orientation.x,
                         slot.pose.orientation.y, slot.pose.orientation.z])
        z_axis = np.array([
            2 * (quat[1] * quat[3] + quat[0] * quat[2]),
            2 * (quat[2] * quat[3] - quat[0] * quat[1]),
            1 - 2 * (quat[1]**2 + quat[2]**2)
        ])
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-10)
        m_Ci = moment_magnitude * z_axis
        sources.append({'p_Ci': p_Ci, 'm_Ci': m_Ci})

    return sources, req


def run_noise_analysis_for_magnitude(json_path, moment_magnitude, D_LIST, gs_to_tesla,
                                      num_samples=100,
                                      radius=0.05, noise_levels=None, sensor_ids=None,
                                      rng_seed=42):
    """
    Run noise analysis for a single moment magnitude value.

    Args:
        D_LIST: (12, 3) sensor offset array
        gs_to_tesla: conversion from Gs to Tesla

    Returns:
        dict: stats per noise level
    """
    if noise_levels is None:
        noise_levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    # Build sources with given magnitude
    sources, req = build_sources_from_json(json_path, moment_magnitude)

    if sensor_ids is None:
        sensor_ids = list(req.sensor_ids)

    # Extract ground truth pose
    gt = req.ground_truth_pose
    p_gt_original = np.array([gt.position.x, gt.position.y, gt.position.z])
    R_gt_original = quaternion_to_rotation_matrix(np.array([
        gt.orientation.w, gt.orientation.x, gt.orientation.y, gt.orientation.z
    ]))

    M = len(sources)
    D_cal = build_D_cal(sensor_ids)

    # Pre-generate random ground truths (reproducible per magnitude to keep fair comparison)
    rng = np.random.default_rng(rng_seed)
    random_poses = []
    for i in range(num_samples):
        p_rand = sample_random_position(p_gt_original, radius)
        R_rand = sample_random_rotation()
        random_poses.append((p_rand, R_rand))

    # Results storage
    results = {nl: {'pos': [], 'ori': []} for nl in noise_levels}

    for nl_idx, noise_level in enumerate(noise_levels):
        for sample_idx, (p_gt, R_gt) in enumerate(random_poses):
            B_meas_cell, b_local_norm = generate_synthetic_B_meas_for_pose(
                p_gt, R_gt, D_LIST, gs_to_tesla, sources, sensor_ids, noise_level=noise_level
            )

            try:
                R_est, p_est, details = MaPS_Estimator(D_cal, sources, B_meas_cell)
            except Exception:
                results[noise_level]['pos'].append(np.nan)
                results[noise_level]['ori'].append(np.nan)
                continue

            pos_error = np.linalg.norm(p_est - p_gt)
            ori_error = compute_rotation_error(R_est, R_gt)

            results[noise_level]['pos'].append(pos_error)
            results[noise_level]['ori'].append(ori_error)

        valid_pos = [e for e in results[noise_level]['pos'] if not np.isnan(e)]
        if valid_pos:
            valid_ori = [e for e in results[noise_level]['ori'] if not np.isnan(e)]
            print(f"  m={moment_magnitude:.0f}, noise={noise_level:.1e}: "
                  f"{len(valid_pos)}/{num_samples} ok, "
                  f"pos={np.mean(valid_pos)*1000:.4f}mm, ori={np.degrees(np.mean(valid_ori)):.4f}deg")

    # Compute statistics
    stats = {}
    for nl in noise_levels:
        pos_arr = np.array([e for e in results[nl]['pos'] if not np.isnan(e)])
        ori_arr = np.array([e for e in results[nl]['ori'] if not np.isnan(e)])
        stats[nl] = {
            'pos_mean': np.mean(pos_arr) if len(pos_arr) > 0 else np.nan,
            'pos_std': np.std(pos_arr) if len(pos_arr) > 0 else np.nan,
            'ori_mean': np.mean(ori_arr) if len(ori_arr) > 0 else np.nan,
            'ori_std': np.std(ori_arr) if len(ori_arr) > 0 else np.nan,
            'n_valid': len(pos_arr),
        }

    return stats


def run_multi_magnitude_analysis(json_path, magnitudes, num_samples=100, radius=0.05,
                                  noise_levels=None, sensor_ids=None, rng_seed=42,
                                  output_path=None):
    """
    Run noise analysis across multiple moment magnitudes and plot all results.

    Args:
        json_path: path to cycle JSON file
        magnitudes: list of moment magnitude values (all slots use same value)
        num_samples: number of random ground-truth poses per magnitude
        radius: sphere radius in meters
        noise_levels: list of noise levels in Gs
        sensor_ids: list of sensor IDs to use
        rng_seed: random seed for reproducible ground-truth generation
        output_path: path to save the plot

    Returns:
        dict: {magnitude: stats}
    """
    if noise_levels is None:
        noise_levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    # Load configuration directly without ROS
    from sensor_array_config import get_config
    config = get_config('QMC6309')
    D_LIST_RAW = np.array(config.hardware.d_list)  # shape (12, 3)
    GS_TO_TESLA_VAL = config.gs_to_si

    # Patch into localization_service_node so build_D_cal works
    import localization_service_node as lsn
    lsn.D_LIST = D_LIST_RAW
    lsn.GS_TO_TESLA = GS_TO_TESLA_VAL

    # Normalize magnitudes to [0, 1] for brightness (larger = darker)
    m_min = min(magnitudes)
    m_max = max(magnitudes)

    all_stats = {}

    print(f"Running multi-magnitude analysis: {len(magnitudes)} magnitudes x "
          f"{len(noise_levels)} noise levels x {num_samples} samples...")

    for mag in magnitudes:
        print(f"\n=== Moment magnitude = {mag} A·m² ===")
        stats = run_noise_analysis_for_magnitude(
            json_path=json_path,
            moment_magnitude=mag,
            D_LIST=D_LIST_RAW,
            gs_to_tesla=GS_TO_TESLA_VAL,
            num_samples=num_samples,
            radius=radius,
            noise_levels=noise_levels,
            sensor_ids=sensor_ids,
            rng_seed=rng_seed
        )
        all_stats[mag] = stats

    # Plot — academic style
    # Figure size: 2-column, ~17.8cm wide, ~12cm tall
    fig, axes = plt.subplots(1, 2, figsize=(17.8 / 2.54, 12.0 / 2.54))

    # Academic rendering — Computer Modern style
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['axes.formatter.use_mathtext'] = True

    def _set_x_ticks(ax, noise_levels):
        """Format x tick labels as 10^{-5}, 10^{-4}, ... with mathtext."""
        labels = [rf'$10^{{{int(np.log10(v)):.0f}}}$' for v in noise_levels]
        ax.set_xticks(noise_levels)
        ax.set_xticklabels(labels)
        ax.yaxis.get_offset_text().set_visible(False)

    def _set_y_ticks(ax):
        """Set y tick labels as 10^{-2}, 10^{-1}, etc. with mathtext."""
        ylims = ax.get_ylim()
        from matplotlib.ticker import LogLocator
        loc = LogLocator(base=10, numticks=8)
        ticks = loc.tick_values(max(ylims[0], 1e-6), ylims[1])
        ticks = [t for t in ticks if ylims[0] * 0.9 <= t <= ylims[1] * 1.1]
        ticks = sorted(set(ticks))[:8]
        ax.set_yticks(ticks)
        ax.set_yticklabels([rf'$10^{{{int(np.log10(abs(t))):.0f}}}$' for t in ticks])

    # Normalize for brightness
    t_values = [(m - m_min) / (m_max - m_min) if m_max != m_min else 0.5 for m in magnitudes]

    # Position error — blue tones (left subplot)
    ax = axes[0]
    for mag, t in zip(magnitudes, t_values):
        stats = all_stats[mag]
        pos_means = [stats[nl]['pos_mean'] * 1000 for nl in noise_levels]
        pos_stds = [stats[nl]['pos_std'] * 1000 for nl in noise_levels]
        # Blue: darker for larger magnitude (t -> darker)
        base_color = np.array([0.0, 0.3, 1.0])  # blue base
        alpha = 0.3 + 0.7 * t
        color = tuple(base_color * alpha)
        lbl = rf'$m = {mag:.0f}$'
        # Connecting line
        ax.plot(noise_levels, pos_means, '-', color=color, linewidth=1.0,
                 solid_capstyle='round')
        # Markers + error bars
        ax.errorbar(noise_levels, pos_means, yerr=pos_stds,
                    fmt='o', capsize=4, color=color, label=lbl,
                    markersize=4, markeredgewidth=0.5,
                    markeredgecolor=color, elinewidth=1.0)

    ax.set_xlabel(r'$\mathrm{Noise}$ $\mathrm{[Gs]}$', fontsize=16)
    ax.set_ylabel(r'$\mathrm{Position\ Error}$ $\mathrm{[mm]}$', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=12, loc='upper left', frameon=True, fancybox=False, edgecolor='black')
    ax.tick_params(labelsize=14)
    _set_x_ticks(ax, noise_levels)
    _set_y_ticks(ax)

    # Orientation error — orange tones (right subplot)
    ax = axes[1]
    for mag, t in zip(magnitudes, t_values):
        stats = all_stats[mag]
        ori_means = [np.degrees(stats[nl]['ori_mean']) for nl in noise_levels]
        ori_stds = [np.degrees(stats[nl]['ori_std']) for nl in noise_levels]
        # Orange: darker for larger magnitude (t -> darker)
        base_color = np.array([1.0, 0.4, 0.0])  # orange base
        alpha = 0.3 + 0.7 * t
        color = tuple(base_color * alpha)
        lbl = rf'$m = {mag:.0f}$'
        # Connecting line
        ax.plot(noise_levels, ori_means, '-', color=color, linewidth=1.0,
                 solid_capstyle='round')
        # Markers + error bars
        ax.errorbar(noise_levels, ori_means, yerr=ori_stds,
                    fmt='o', capsize=4, color=color, label=lbl,
                    markersize=4, markeredgewidth=0.5,
                    markeredgecolor=color, elinewidth=1.0)

    ax.set_xlabel(r'$\mathrm{Noise}$ $\mathrm{[Gs]}$', fontsize=16)
    ax.set_ylabel(r'$\mathrm{Orientation\ Error}$ $[^{\circ}]$', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=12, loc='upper left', frameon=True, fancybox=False, edgecolor='black')
    ax.tick_params(labelsize=14)
    _set_x_ticks(ax, noise_levels)
    _set_y_ticks(ax)

    plt.tight_layout(pad=0.5)

    out_path = output_path
    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(json_path)),
            f'noise_analysis_rp_multi_m_cycle_{0:04d}.png'
        )
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {out_path}")
    plt.close()

    return all_stats


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Noise analysis with random ground-truth pose perturbation and magnitude sweep.'
    )
    parser.add_argument('json_path', nargs='?', default=None,
                        help='Path to cycle JSON file')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of random ground-truth samples per magnitude (default 100)')
    parser.add_argument('--radius', type=float, default=0.05,
                        help='Sphere radius in meters (default 0.05 = 5cm)')
    parser.add_argument('--magnitudes', type=str, default=None,
                        help='Comma-separated moment magnitudes (A·m²), default "50,100,...,500"')
    parser.add_argument('--output', type=str, default=None,
                        help='Output plot path')
    parser.add_argument('--sensors', type=str, default=None,
                        help='Comma-separated sensor IDs (e.g. "1,3,4,6,7,9,10,12")')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default 42)')

    args = parser.parse_args()

    default_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'sensor_data_collection', 'result', 'cycle_0000.json'
    )
    default_path = os.path.normpath(default_path)

    json_path = args.json_path if args.json_path else default_path
    if args.json_path is None:
        print(f"No path provided, using default: {json_path}")

    sensor_ids = None
    if args.sensors is not None:
        sensor_ids = [int(x.strip()) for x in args.sensors.split(',')]

    # Default magnitudes: 50, 250, 500
    if args.magnitudes is not None:
        magnitudes = [float(x.strip()) for x in args.magnitudes.split(',')]
    else:
        magnitudes = [50, 250, 500]

    noise_levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    all_stats = run_multi_magnitude_analysis(
        json_path,
        magnitudes=magnitudes,
        num_samples=args.samples,
        radius=args.radius,
        noise_levels=noise_levels,
        sensor_ids=sensor_ids,
        rng_seed=args.seed,
        output_path=args.output
    )

    # Print summary table
    print("\n" + "=" * 100)
    print("Summary: Position Error (mm) — rows=magnitudes, cols=noise levels")
    print("=" * 100)
    hdr = f"{'m':>6}" + "".join([f"{nl:>10.0e}" for nl in noise_levels])
    print(hdr)
    print("-" * (6 + 10 * len(noise_levels)))
    for mag in magnitudes:
        row = f"{mag:>6.0f}"
        for nl in noise_levels:
            row += f"{all_stats[mag][nl]['pos_mean']*1000:>10.4f}"
        print(row)

    print()
    print("=" * 100)
    print("Summary: Orientation Error (deg) — rows=magnitudes, cols=noise levels")
    print("=" * 100)
    print(hdr)
    print("-" * (6 + 10 * len(noise_levels)))
    for mag in magnitudes:
        row = f"{mag:>6.0f}"
        for nl in noise_levels:
            row += f"{np.degrees(all_stats[mag][nl]['ori_mean']):>10.4f}"
        print(row)


if __name__ == '__main__':
    main()
