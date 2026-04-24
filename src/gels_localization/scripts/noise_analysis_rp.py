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

        # Clean field at center (no offset), converted to Gs
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
        list of (SNR, pos_error, ori_error) tuples for ALL samples
    """
    if noise_levels is None:
        noise_levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    sources, req = build_sources_from_json(json_path, moment_magnitude)
    if sensor_ids is None:
        sensor_ids = list(req.sensor_ids)

    gt = req.ground_truth_pose
    p_gt_original = np.array([gt.position.x, gt.position.y, gt.position.z])
    R_gt_original = quaternion_to_rotation_matrix(np.array([
        gt.orientation.w, gt.orientation.x, gt.orientation.y, gt.orientation.z
    ]))

    M = len(sources)
    D_cal = build_D_cal(sensor_ids)

    rng = np.random.default_rng(rng_seed)
    random_poses = []
    for i in range(num_samples):
        p_rand = sample_random_position(p_gt_original, radius)
        R_rand = sample_random_rotation()
        random_poses.append((p_rand, R_rand))

    all_results = []  # (SNR, pos_error, ori_error)

    for nl_idx, noise_level in enumerate(noise_levels):
        for sample_idx, (p_gt, R_gt) in enumerate(random_poses):
            B_meas_cell, b_local_norm = generate_synthetic_B_meas_for_pose(
                p_gt, R_gt, D_LIST, gs_to_tesla, sources, sensor_ids, noise_level=noise_level
            )

            SNR = b_local_norm / noise_level if noise_level > 0 else np.inf

            try:
                R_est, p_est, details = MaPS_Estimator(D_cal, sources, B_meas_cell)
            except Exception:
                continue

            pos_error = np.linalg.norm(p_est - p_gt)
            ori_error = compute_rotation_error(R_est, R_gt)
            all_results.append((SNR, pos_error, ori_error))

    return all_results


def run_multi_magnitude_analysis(json_path, magnitudes, num_samples=100, radius=0.05,
                                  noise_levels=None, sensor_ids=None, rng_seed=42,
                                  output_path=None):
    """
    Collect all (SNR, error) pairs across all magnitudes, then plot scatter + fit.
    """
    if noise_levels is None:
        noise_levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    from sensor_array_config import get_config
    config = get_config('QMC6309')
    D_LIST_RAW = np.array(config.hardware.d_list)
    GS_TO_TESLA_VAL = config.gs_to_si

    import localization_service_node as lsn
    lsn.D_LIST = D_LIST_RAW
    lsn.GS_TO_TESLA = GS_TO_TESLA_VAL

    all_snr = []
    all_pos_err = []
    all_ori_err = []

    for mag in magnitudes:
        print(f"\n=== Moment magnitude = {mag} A·m² ===")
        results = run_noise_analysis_for_magnitude(
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
        for snr, pos_err, ori_err in results:
            all_snr.append(snr)
            all_pos_err.append(pos_err)
            all_ori_err.append(ori_err)

    all_snr = np.array(all_snr)
    all_pos_err = np.array(all_pos_err)
    all_ori_err = np.array(all_ori_err)

    # Filter valid (finite SNR <= 1e5, non-nan, positive) points
    valid_snr = np.isfinite(all_snr) & (all_snr <= 1e5)
    valid_pos = valid_snr & ~(np.isnan(all_pos_err) | (all_pos_err <= 0))
    valid_ori = valid_snr & ~(np.isnan(all_ori_err) | (all_ori_err <= 0))

    # Plot — academic style
    fig, axes = plt.subplots(1, 2, figsize=(17.8 / 2.54, 12.0 / 2.54))

    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'font.size': 16,
    })

    from matplotlib.ticker import LogLocator, FuncFormatter

    # ---- Position error subplot ----
    ax = axes[0]
    snr_v = all_snr[valid_pos]
    pos_v = all_pos_err[valid_pos] * 1000  # convert to mm

    ax.scatter(snr_v, pos_v, alpha=0.3, s=10, color='steelblue', label='Fit')

    # Power-law fit on log-log
    log_snr = np.log10(snr_v)
    log_pos = np.log10(pos_v)
    coeffs = np.polyfit(log_snr, log_pos, 1)
    b_pos, log_a_pos = coeffs[0], coeffs[1]
    a_pos = 10**log_a_pos

    # Fit line
    snr_fit_pos = np.logspace(np.log10(snr_v.min()), np.log10(snr_v.max()), 200)
    pos_fit = a_pos * snr_fit_pos**b_pos
    ax.plot(snr_fit_pos, pos_fit, 'k-', linewidth=1.5, label='Samples')

    ax.set_xlabel(r'$\mathrm{SNR}$', fontsize=16)
    ax.set_ylabel(r'$\mathrm{Position\ Error}$ $\mathrm{[mm]}$', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([1e1, 1e2, 1e3, 1e4, 1e5])
    ax.set_xticklabels([r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: rf'$10^{{{int(np.log10(v)):d}}}$' if v > 0 else '')
    )
    ax.grid(True, alpha=0.3, which='both')
    ax.legend([r'$\mathrm{Samples}$', r'$\mathrm{Fit}$'], fontsize=11, loc='upper right')
    ax.tick_params(labelsize=14)

    print(f"Position fit: error = {a_pos:.4e} * SNR^({b_pos:.4f})")

    # ---- Orientation error subplot ----
    ax = axes[1]
    snr_v = all_snr[valid_ori]
    ori_v = np.degrees(all_ori_err[valid_ori])

    ax.scatter(snr_v, ori_v, alpha=0.3, s=10, color='darkorange', label='Fit')

    log_snr = np.log10(snr_v)
    log_ori = np.log10(ori_v)
    coeffs = np.polyfit(log_snr, log_ori, 1)
    b_ori, log_a_ori = coeffs[0], coeffs[1]
    a_ori = 10**log_a_ori

    snr_fit_ori = np.logspace(np.log10(snr_v.min()), np.log10(snr_v.max()), 200)
    ori_fit = a_ori * snr_fit_ori**b_ori
    ax.plot(snr_fit_ori, ori_fit, 'k-', linewidth=1.5, label='Samples')

    ax.set_xlabel(r'$\mathrm{SNR}$', fontsize=16)
    ax.set_ylabel(r'$\mathrm{Orientation\ Error}$ $[^{\circ}]$', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([1e1, 1e2, 1e3, 1e4, 1e5])
    ax.set_xticklabels([r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: rf'$10^{{{int(np.log10(v)):d}}}$' if v > 0 else '')
    )
    ax.grid(True, alpha=0.3, which='both')
    ax.legend([r'$\mathrm{Samples}$', r'$\mathrm{Fit}$'], fontsize=11, loc='upper right')
    ax.tick_params(labelsize=14)

    print(f"Orientation fit: error = {a_ori:.4e} * SNR^({b_ori:.4f})")

    plt.tight_layout(pad=0.5)

    out_path = output_path
    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(json_path)),
            f'noise_analysis_rp_snr.png'
        )
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {out_path}")
    plt.close()


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

    run_multi_magnitude_analysis(
        json_path,
        magnitudes=magnitudes,
        num_samples=args.samples,
        radius=args.radius,
        noise_levels=noise_levels,
        sensor_ids=sensor_ids,
        rng_seed=args.seed,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
