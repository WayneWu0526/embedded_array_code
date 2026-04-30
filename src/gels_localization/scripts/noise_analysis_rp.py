#!/usr/bin/env python3
"""
Noise analysis: offset bias study.

Studies the effect of per-sensor offset bias on localization error.
For each moment magnitude, a fixed offset bias is randomly sampled from
[offset_levels], applied uniformly to ALL 100 random poses, with fixed
noise (0.01 Gs). Reports position/orientation error vs delta_o / |B| ratio.

Usage:
    python noise_analysis_rp.py
    python noise_analysis_rp.py --magnitudes 50 250 500
"""

import sys
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


# Fixed noise level (Gs)
FIXED_NOISE_LEVEL = 0.01

# Offset sweep: 0 to 0.5 with step 0.05
OFFSET_LEVELS = [round(x * 0.05, 2) for x in range(11)]  # 0.0, 0.05, ..., 0.5


def sample_random_rotation():
    """Sample a random rotation matrix uniformly from SO(3)."""
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
    """Sample a random point uniformly within a sphere."""
    direction = np.random.randn(3)
    direction = direction / np.linalg.norm(direction)
    r = radius * (np.random.rand() ** (1.0 / 3.0))
    return center + r * direction


def sample_sensor_offset_uniform(rng, n_sensors, delta_o):
    """
    Sample a per-sensor, per-component offset vector uniformly from [-delta_o, delta_o].

    Returns:
        offset: (n_sensors, 3) array in Gs
        actual_delta_o: max - min across all values in the offset array
    """
    offset = rng.uniform(-delta_o, delta_o, size=(n_sensors, 3))
    actual_delta_o = offset.max() - offset.min()
    return offset, actual_delta_o


def generate_synthetic_B_meas_for_pose(p_sensor_array, R_sensor_array, D_LIST,
                                        gs_to_tesla,
                                        sources, sensor_ids,
                                        noise_level=0.0, offset=None):
    """
    Generate synthetic B_meas for a given sensor array pose.

    Args:
        offset: (n_sensors, 3) array of per-sensor bias to add (Gs). If None, no offset.
    """
    B_meas_cell = []
    b_local_norms = []

    for src in sources:
        p_Ci = src['p_Ci']
        m_Ci = src['m_Ci']

        # Clean field at center, converted to Gs
        b_global_center, _ = mag_dipole_model(p_sensor_array, m_Ci, p_Ci, order=1)
        b_sensor_center = R_sensor_array.T @ b_global_center
        b_sensor_center_gs = b_sensor_center / gs_to_tesla
        b_local_norms.append(np.linalg.norm(b_sensor_center_gs))

        B_meas = np.zeros((3, len(sensor_ids)))
        for col_idx, sid in enumerate(sensor_ids):
            sensor_idx = sid - 1
            d_j = D_LIST[sensor_idx]
            p_sensor_global = p_sensor_array + R_sensor_array @ d_j

            b_global, _ = mag_dipole_model(p_sensor_global, m_Ci, p_Ci, order=1)
            b_sensor = R_sensor_array.T @ b_global
            b_sensor_gs = b_sensor / gs_to_tesla

            # Add fixed noise
            if noise_level > 0:
                b_sensor_gs = b_sensor_gs + noise_level * np.random.randn(3)

            # Add offset bias
            if offset is not None:
                b_sensor_gs = b_sensor_gs + offset[sensor_idx]

            B_meas[:, col_idx] = b_sensor_gs

        B_meas_cell.append(B_meas)

    b_local_norm = np.linalg.norm(b_local_norms)
    return B_meas_cell, b_local_norm


def build_sources_from_json(json_path, moment_magnitude):
    """Build source list from cycle JSON."""
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


def run_offset_analysis_for_magnitude(json_path, moment_magnitude, D_LIST, gs_to_tesla,
                                      num_samples=100, radius=0.05,
                                      sensor_ids=None, rng_seed=42):
    """
    Run offset bias analysis for a single moment magnitude.

    For each delta_o level in OFFSET_LEVELS:
        - Sample a random per-sensor per-component offset from uniform[-delta_o, delta_o]
        - Record actual_delta_o = max(offset) - min(offset)
        - Apply the SAME offset to ALL num_samples random poses
        - Fixed noise 0.01 Gs per pose

    Returns:
        list of (delta_o_over_B, pos_error, ori_error) tuples
    """
    sources, req = build_sources_from_json(json_path, moment_magnitude)
    if sensor_ids is None:
        sensor_ids = list(req.sensor_ids)

    n_sensors = len(sensor_ids)

    gt = req.ground_truth_pose
    p_gt_original = np.array([gt.position.x, gt.position.y, gt.position.z])
    R_gt_original = quaternion_to_rotation_matrix(np.array([
        gt.orientation.w, gt.orientation.x, gt.orientation.y, gt.orientation.z
    ]))

    D_cal = build_D_cal(sensor_ids)

    rng = np.random.default_rng(rng_seed)

    # Pre-generate num_samples random poses
    random_poses = []
    for i in range(num_samples):
        p_rand = sample_random_position(p_gt_original, radius)
        R_rand = sample_random_rotation()
        random_poses.append((p_rand, R_rand))

    # Pre-compute b_local_norm for each pose (no offset, no noise) for ratio denominator
    b_local_norms = []
    for p_gt, R_gt in random_poses:
        _, b_norm = generate_synthetic_B_meas_for_pose(
            p_gt, R_gt, D_LIST, gs_to_tesla, sources, sensor_ids,
            noise_level=0.0, offset=None
        )
        b_local_norms.append(b_norm)

    all_results = []

    for nominal_delta_o in OFFSET_LEVELS:
        # Sample ONE random offset configuration for this delta_o level
        off_rng = np.random.default_rng(rng_seed + int(nominal_delta_o * 1000))
        offset, actual_delta_o = sample_sensor_offset_uniform(off_rng, n_sensors, nominal_delta_o)

        for sample_idx, ((p_gt, R_gt), b_norm) in enumerate(zip(random_poses, b_local_norms)):
            B_meas_cell, _ = generate_synthetic_B_meas_for_pose(
                p_gt, R_gt, D_LIST, gs_to_tesla, sources, sensor_ids,
                noise_level=FIXED_NOISE_LEVEL, offset=offset
            )

            # Use actual_delta_o for ratio
            ratio = actual_delta_o / b_norm if b_norm > 0 else 0.0

            try:
                R_est, p_est, details = MaPS_Estimator(D_cal, sources, B_meas_cell)
            except Exception:
                continue

            pos_error = np.linalg.norm(p_est - p_gt)
            ori_error = compute_rotation_error(R_est, R_gt)
            all_results.append((ratio, pos_error, ori_error))

    return all_results


def run_multi_magnitude_analysis(json_path, magnitudes, num_samples=100, radius=0.05,
                                sensor_ids=None, rng_seed=42,
                                output_path=None):
    """
    Collect all (delta_o/|B|, error) pairs across magnitudes, then plot scatter + linear fit.
    """
    from sensor_array_config import get_config
    config = get_config('QMC6309')
    D_LIST_RAW = np.array(config.hardware.d_list)
    GS_TO_TESLA_VAL = config.gs_to_si

    import localization_service_node as lsn
    lsn.D_LIST = D_LIST_RAW
    lsn.GS_TO_TESLA = GS_TO_TESLA_VAL

    all_ratio = []
    all_pos_err = []
    all_ori_err = []

    for mag in magnitudes:
        print(f"\n=== Moment magnitude = {mag} A·m² ===")
        results = run_offset_analysis_for_magnitude(
            json_path=json_path,
            moment_magnitude=mag,
            D_LIST=D_LIST_RAW,
            gs_to_tesla=GS_TO_TESLA_VAL,
            num_samples=num_samples,
            radius=radius,
            sensor_ids=sensor_ids,
            rng_seed=rng_seed
        )
        for ratio, pos_err, ori_err in results:
            all_ratio.append(ratio)
            all_pos_err.append(pos_err)
            all_ori_err.append(ori_err)

        print(f"  Collected {len(results)} samples, ratio range: "
              f"[{min(r[0] for r in results):.4f}, {max(r[0] for r in results):.4f}]")

    all_ratio = np.array(all_ratio, dtype=float)
    all_pos_err = np.array(all_pos_err, dtype=float)
    all_ori_err = np.array(all_ori_err, dtype=float)

    # Filter valid (finite, non-nan, non-negative ratio, positive error) points
    valid_ratio = np.isfinite(all_ratio) & (all_ratio >= 0)
    valid_pos = valid_ratio & np.isfinite(all_pos_err) & (all_pos_err > 0)
    valid_ori = valid_ratio & np.isfinite(all_ori_err) & (all_ori_err > 0)

    ratio_v_pos = all_ratio[valid_pos]
    pos_v = all_pos_err[valid_pos] * 1000  # m -> mm

    ratio_v_ori = all_ratio[valid_ori]
    ori_v = np.degrees(all_ori_err[valid_ori])

    # Linear fit on linear scale: error = k * (delta_o / |B|) + b
    coeffs_pos = np.polyfit(ratio_v_pos, pos_v, 1)
    k_pos, b_pos = coeffs_pos[0], coeffs_pos[1]

    # For orientation: force line through the minimum point (ratio_min, e_R_min)
    # then fit slope k using remaining data
    idx_min_ori = np.argmin(ori_v)
    ratio_min_ori = ratio_v_ori[idx_min_ori]
    e_R_min = ori_v[idx_min_ori]
    mask_remaining = np.ones(len(ratio_v_ori), dtype=bool)
    mask_remaining[idx_min_ori] = False
    k_ori = np.polyfit(ratio_v_ori[mask_remaining], ori_v[mask_remaining] - e_R_min, 1)[0]
    b_ori = e_R_min - k_ori * ratio_min_ori

    fontsize = 16
    # Plot
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath,amsfonts,amssymb,amsthm,mathrsfs,mathtools}\usepackage{bm}\usepackage{dutchcal}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
    })

    fig, axes = plt.subplots(1, 2, figsize=(19.8 / 2.54, 8 / 2.54))

    from matplotlib.ticker import ScalarFormatter

    for ax in axes:
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    # ---- Position error subplot ----
    ax = axes[0]
    ax.scatter(ratio_v_pos, pos_v, alpha=0.3, s=10, color='black', edgecolors='none')

    # Linear fit line - start from actual min ratio (no extrapolation to 0)
    ratio_fit_pos = np.linspace(ratio_v_pos.min(), ratio_v_pos.max() * 1.05, 200)
    pos_fit = k_pos * ratio_fit_pos + b_pos
    ln_fit, = ax.plot(ratio_fit_pos, pos_fit, 'k-', linewidth=1.5)

    # Add vertical and horizontal lines for key delta_o levels
    key_ratios = [0.0, 0.00597, 0.03845]
    colors = ['#2ca02c', '#d62728', '#1f77b4']  # Green (Ideal), Red (Calibrated), Blue (Uncalibrated)
    labels = ['Ideal', 'Cal.', 'Raw']

    key_handles = []
    for r, c, l in zip(key_ratios, colors, labels):
        err_val = k_pos * r + b_pos
        ax.vlines(x=r, ymin=-150, ymax=err_val, color=c, linestyle='-', linewidth=1.5)
        ax.hlines(y=err_val, xmin=-1, xmax=r, color=c, linestyle='-', linewidth=1.5)
        ln, = ax.plot([], [], color=c, linewidth=1.5, label=l)
        key_handles.append(ln)

    # Force LaTeX rendering for ticks
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('$%g$'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('$%g$'))

    ax.set_xlim(left=-0.0133, right=0.08)
    ax.set_ylim(bottom=-30, top=180)

    ax.set_xlabel(r'$\Delta o / \|\bar{\bm{\mathcal{b}}}\|$')
    ax.set_ylabel(r'$e_{\bm{p}}\ \mathrm{[mm]}$')
    ax.grid(True, alpha=0.2, linestyle=':')
    # Legend order: Fit, Samples, Ideal, Cal., Raw
    ln_samples, = ax.plot([], [], color='black', linewidth=0, marker='o', markersize=4, label='Samples')
    ax.legend([ln_fit, ln_samples] + key_handles,
              ['Fit', 'Samples', 'Ideal', 'Cal.', 'Raw'],
              fontsize=12, loc='lower right')

    print(f"Position linear fit: error = {k_pos:.4f} * (Δ_o/|B|) + {b_pos:.4f}")

    # ---- Orientation error subplot ----
    ax = axes[1]
    ax.scatter(ratio_v_ori, ori_v, alpha=0.3, s=10, color='black', edgecolors='none')

    ratio_fit_ori = np.linspace(ratio_v_ori.min(), ratio_v_ori.max() * 1.05, 200)
    ori_fit = k_ori * ratio_fit_ori + b_ori
    ln_fit_ori, = ax.plot(ratio_fit_ori, ori_fit, 'k-', linewidth=1.5)

    ori_handles = []
    for r, c, l in zip(key_ratios, colors, labels):
        err_val = k_ori * r + b_ori
        ax.vlines(x=r, ymin=-150, ymax=err_val, color=c, linestyle='-', linewidth=1.5, alpha=0.8)
        ax.hlines(y=err_val, xmin=-1, xmax=r, color=c, linestyle='-', linewidth=1.5, alpha=0.8)
        ln, = ax.plot([], [], color=c, linewidth=1.5, label=l)
        ori_handles.append(ln)

    # Force LaTeX rendering for ticks
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('$%g$'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('$%g$'))

    ax.set_xlim(left=-0.0077, right=0.07)
    ax.set_ylim(bottom=-20, top=180)

    ax.set_xlabel(r'$\Delta o / \|\bar{\bm{\mathcal{b}}}\|$')
    ax.set_ylabel(r'$e_{\bm{R}}\ \mathrm{[^\circ]}$')
    ax.grid(True, alpha=0.2, linestyle=':')
    ln_samples_ori, = ax.plot([], [], color='black', linewidth=0, marker='o', markersize=4, label='Samples')
    ax.legend([ln_fit_ori, ln_samples_ori] + ori_handles,
              ['Fit', 'Samples', 'Ideal', 'Cal.', 'Raw'],
              fontsize=12, loc='upper left')

    print(f"Orientation linear fit: error = {k_ori:.4f} * (Δ_o/|B|) + {b_ori:.4f}")

    plt.tight_layout(pad=0.5)

    out_path = output_path
    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(json_path)),
            'noise_analysis_rp_snr.pdf'
        )
    plt.savefig(out_path, dpi=300)
    print(f"\nPlot saved to: {out_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Noise analysis: offset bias study with linear fit.'
    )
    parser.add_argument('json_path', nargs='?', default=None,
                        help='Path to cycle JSON file')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of random ground-truth samples per magnitude (default 100)')
    parser.add_argument('--radius', type=float, default=0.05,
                        help='Sphere radius in meters (default 0.05 = 5cm)')
    parser.add_argument('--magnitudes', type=str, default=None,
                        help='Comma-separated moment magnitudes (A·m²), default "50,250,500"')
    parser.add_argument('--output', '-o', type=str, default=None,
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

    # Fixed magnitude = 50 A·m², single fixed pose (num_samples=1)
    magnitudes = [50.0]
    num_samples_fixed = 20

    run_multi_magnitude_analysis(
        json_path,
        magnitudes=magnitudes,
        num_samples=num_samples_fixed,
        radius=args.radius,
        sensor_ids=sensor_ids,
        rng_seed=args.seed,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
