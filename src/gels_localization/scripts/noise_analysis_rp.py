#!/usr/bin/env python3
"""
Noise analysis: offset bias study with ray-extension sampling.

Studies the effect of per-sensor offset bias on localization error using
order-3 dipole model. For each base sample, extends along the ray from pbar
(mean of p_Ci) through the sample, generating step_mm-interval points up to
extension_mm beyond the base. Saves ep, eR, and distance-to-pbar as CSV.

Usage:
    python noise_analysis_rp.py
    python noise_analysis_rp.py --extension 400 --step 10
"""

import sys
import os

import numpy as np

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
from plot_noise_analysis import plot_errors


# Fixed noise level (Gs)
FIXED_NOISE_LEVEL = 0.005

# Offset sweep: single level at 0 (no offset bias, only base noise 0.01 Gs)
OFFSET_LEVELS = [0.0]


def compute_pbar(sources):
    """Compute pbar = mean(p_Ci) across all sources (centroid of source positions)."""
    p_Ck = np.column_stack([src['p_Ci'] for src in sources])
    pbar = np.mean(p_Ck, axis=1)
    return pbar


def extend_pose_along_ray(pbar, p_base, extension_mm=400, step_mm=10):
    """
    For a base sample at p_base, compute the direction from pbar to p_base,
    then generate extended points along the ray AWAY from pbar.

    For d = |p_base - pbar|, extended points are at distances:
        d + step_mm, d + 2*step_mm, ..., d + extension_mm

    Args:
        pbar: centroid of source positions (3,)
        p_base: base sample position (3,)
        extension_mm: maximum extension beyond base sample distance (mm)
        step_mm: step size between extended points (mm)

    Returns:
        list of extended positions (3,) in meters
    """
    direction = p_base - pbar
    d = np.linalg.norm(direction)
    if d < 1e-10:
        return []  # p_base at pbar, no defined direction

    direction_normalized = direction / d

    extended = []
    # step_mm, 2*step_mm, ..., extension_mm
    for L_mm in range(step_mm, extension_mm + step_mm, step_mm):
        L_m = L_mm / 1000.0  # convert mm to m
        p_extended = pbar + (d + L_m) * direction_normalized
        extended.append(p_extended)

    return extended


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
                                        noise_level=0.0, offset=None,
                                        order=3):
    """
    Generate synthetic B_meas for a given sensor array pose.

    Args:
        offset: (n_sensors, 3) array of per-sensor bias to add (Gs). If None, no offset.
        order: dipole model order (1, 3, or 5)
    """
    B_meas_cell = []
    b_local_norms = []

    for src in sources:
        p_Ci = src['p_Ci']
        m_Ci = src['m_Ci']

        # Clean field at center, converted to Gs
        b_global_center, _ = mag_dipole_model(p_sensor_array, m_Ci, p_Ci, order=order)
        b_sensor_center = R_sensor_array.T @ b_global_center
        b_sensor_center_gs = b_sensor_center / gs_to_tesla
        b_local_norms.append(np.linalg.norm(b_sensor_center_gs))

        B_meas = np.zeros((3, len(sensor_ids)))
        for col_idx, sid in enumerate(sensor_ids):
            sensor_idx = sid - 1
            d_j = D_LIST[sensor_idx]
            p_sensor_global = p_sensor_array + R_sensor_array @ d_j

            b_global, _ = mag_dipole_model(p_sensor_global, m_Ci, p_Ci, order=order)
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
                                      sensor_ids=None, rng_seed=42,
                                      extension_mm=400, step_mm=10):
    """
    Run offset bias analysis for a single moment magnitude.

    For each delta_o level in OFFSET_LEVELS:
        - Sample a random per-sensor per-component offset from uniform[-delta_o, delta_o]
        - Record actual_delta_o = max(offset) - min(offset)
        - Apply the SAME offset to ALL base samples (extended along rays from pbar)
        - Fixed noise 0.01 Gs per pose

    Each base sample is extended along the ray from pbar (mean of p_Ci) through
    the base sample, at step_mm intervals up to extension_mm beyond the base.

    Returns:
        list of (delta_o_over_B, pos_error, ori_error) tuples
    """
    sources, req = build_sources_from_json(json_path, moment_magnitude)
    if sensor_ids is None:
        sensor_ids = list(req.sensor_ids)

    # Compute pbar = centroid of all source positions
    pbar = compute_pbar(sources)

    n_sensors = len(sensor_ids)

    gt = req.ground_truth_pose
    p_gt_original = np.array([gt.position.x, gt.position.y, gt.position.z])
    R_gt_original = quaternion_to_rotation_matrix(np.array([
        gt.orientation.w, gt.orientation.x, gt.orientation.y, gt.orientation.z
    ]))

    D_cal = build_D_cal(sensor_ids)

    rng = np.random.default_rng(rng_seed)

    # Use ground truth pose as the single base pose (no random sampling)
    base_poses = [(p_gt_original, R_gt_original)]

    # Compute b_local_norm at base pose
    _, b_norm_base = generate_synthetic_B_meas_for_pose(
        p_gt_original, R_gt_original, D_LIST, gs_to_tesla, sources, sensor_ids,
        noise_level=0.0, offset=None, order=3
    )
    d_base = np.linalg.norm(p_gt_original - pbar)
    d_to_sources = [np.linalg.norm(p_gt_original - src['p_Ci']) * 1000 for src in sources]
    print(f"  Base pose: d_pbar = {d_base*1000:.2f} mm, B_norm = {b_norm_base:.6f} Gs")
    print(f"  Distances to sources: {d_to_sources[0]:.2f}, {d_to_sources[1]:.2f}, {d_to_sources[2]:.2f} mm")

    # Pre-compute b_local_norm for each EXTENDED pose (no offset, no noise)
    # Structure: list of (p_extended, R_base, d_from_pbar) tuples
    extended_poses = []
    b_local_norms = []
    for p_base, R_base in base_poses:
        d_base = np.linalg.norm(p_base - pbar)
        p_extended_list = extend_pose_along_ray(pbar, p_base, extension_mm, step_mm)
        for p_ext in p_extended_list:
            d_from_pbar = np.linalg.norm(p_ext - pbar)
            extended_poses.append((p_ext, R_base, d_from_pbar))
            _, b_norm = generate_synthetic_B_meas_for_pose(
                p_ext, R_base, D_LIST, gs_to_tesla, sources, sensor_ids,
                noise_level=0.0, offset=None, order=3
            )
            b_local_norms.append(b_norm)

    all_results = []

    for nominal_delta_o in OFFSET_LEVELS:
        # Sample ONE random offset configuration for this delta_o level
        off_rng = np.random.default_rng(rng_seed + int(nominal_delta_o * 1000))
        offset, actual_delta_o = sample_sensor_offset_uniform(off_rng, n_sensors, nominal_delta_o)

        for (p_gt, R_gt, d_from_pbar), b_norm in zip(extended_poses, b_local_norms):
            B_meas_cell, _ = generate_synthetic_B_meas_for_pose(
                p_gt, R_gt, D_LIST, gs_to_tesla, sources, sensor_ids,
                noise_level=FIXED_NOISE_LEVEL, offset=offset, order=3
            )

            # Use actual_delta_o for ratio
            ratio = actual_delta_o / b_norm if b_norm > 0 else 0.0

            try:
                R_est, p_est, _ = MaPS_Estimator(D_cal, sources, B_meas_cell)
            except Exception:
                continue

            pos_error = np.linalg.norm(p_est - p_gt)
            ori_error = compute_rotation_error(R_est, R_gt)
            # pos_error [m], ori_error [rad], d_from_pbar [m], b_norm [Gs]
            all_results.append((ratio, pos_error, ori_error, d_from_pbar, b_norm))

    return all_results


def run_multi_magnitude_analysis(json_path, magnitudes, num_samples=100, radius=0.05,
                                sensor_ids=None, rng_seed=42,
                                output_path=None,
                                extension_mm=400, step_mm=10):
    """
    Run error analysis for multiple moment magnitudes using order=3 dipole model.
    Each base sample is extended along the ray from pbar through the base sample,
    at step_mm intervals up to extension_mm beyond the base sample distance.

    Saves a CSV with three columns: ep [mm], eR [deg], distance [mm].

    Args:
        output_path: path to save CSV file. If None, saves to
            <json_dir>/noise_analysis_order3.csv
    """
    from sensor_array_config import get_config
    config = get_config('QMC6309')
    D_LIST_RAW = np.array(config.hardware.d_list)
    GS_TO_TESLA_VAL = config.gs_to_si

    import localization_service_node as lsn
    lsn.D_LIST = D_LIST_RAW
    lsn.GS_TO_TESLA = GS_TO_TESLA_VAL

    all_pos_err = []   # position error in meters
    all_ori_err = []   # orientation error in radians
    all_distance = []  # distance from sample to pbar in meters
    all_B_norm = []    # magnetic field strength in Gs

    for mag in magnitudes:
        print(f"\n=== Moment magnitude = {mag} A·m² (order=3) ===")
        results = run_offset_analysis_for_magnitude(
            json_path=json_path,
            moment_magnitude=mag,
            D_LIST=D_LIST_RAW,
            gs_to_tesla=GS_TO_TESLA_VAL,
            num_samples=num_samples,
            radius=radius,
            sensor_ids=sensor_ids,
            rng_seed=rng_seed,
            extension_mm=extension_mm,
            step_mm=step_mm
        )
        for ratio, pos_err, ori_err, d_from_pbar, b_norm in results:
            all_pos_err.append(pos_err)
            all_ori_err.append(ori_err)
            all_distance.append(d_from_pbar)
            all_B_norm.append(b_norm)

        print(f"  Collected {len(results)} samples")

    # Convert to numpy and filter invalid entries
    all_pos_err = np.array(all_pos_err, dtype=float)
    all_ori_err = np.array(all_ori_err, dtype=float)
    all_distance = np.array(all_distance, dtype=float)

    valid = (np.isfinite(all_pos_err) & np.isfinite(all_ori_err) &
             np.isfinite(all_distance) & (all_pos_err >= 0) & (all_ori_err >= 0))

    ep_mm = all_pos_err[valid] * 1000.0        # m -> mm
    eR_deg = np.degrees(all_ori_err[valid])     # rad -> deg
    dist_mm = all_distance[valid] * 1000.0     # m -> mm
    B_norm = np.array(all_B_norm, dtype=float)[valid]  # Gs

    # Save as CSV: ep [mm], eR [deg], distance [mm], B_norm [Gs]
    out_path = output_path
    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(json_path)),
            'noise_analysis_order1.csv'
        )

    csv_data = np.column_stack([ep_mm, eR_deg, dist_mm, B_norm])
    header = 'ep [mm],eR [deg],distance [mm],B_norm [Gs]'
    np.savetxt(out_path, csv_data, delimiter=',', header=header, comments='',
               fmt='%.6f')
    print(f"\nCSV saved to: {out_path}")
    print(f"  Total samples: {len(ep_mm)}")
    print(f"  ep range:   [{ep_mm.min():.3f}, {ep_mm.max():.3f}] mm")
    print(f"  eR range:   [{eR_deg.min():.3f}, {eR_deg.max():.3f}] deg")
    print(f"  dist range: [{dist_mm.min():.3f}, {dist_mm.max():.3f}] mm")
    print(f"  B_norm range: [{B_norm.min():.6f}, {B_norm.max():.6f}] Gs")
    print(f"\n  Noise floor (FIXED_NOISE_LEVEL): {FIXED_NOISE_LEVEL} Gs")
    # Find where |B| drops below 10x noise (SNR ~ 10) and below noise itself
    idx_below_noise = np.where(B_norm < FIXED_NOISE_LEVEL)[0]
    idx_below_10x = np.where(B_norm < 10 * FIXED_NOISE_LEVEL)[0]
    if len(idx_below_10x) > 0:
        dist_at_10x = dist_mm[idx_below_10x[0]]
        print(f"  |B| < 10*noise first at dist = {dist_at_10x:.1f} mm (SNR ~ 10)")
    else:
        print(f"  |B| never drops below 10*noise in the sampled range")
    if len(idx_below_noise) > 0:
        dist_at_noise = dist_mm[idx_below_noise[0]]
        print(f"  |B| < noise first at dist = {dist_at_noise:.1f} mm (SNR ~ 1)")
    else:
        print(f"  |B| never drops below noise in the sampled range")

    # Plot ep and eR vs distance
    plot_errors(ep_mm, eR_deg, dist_mm, B_norm, out_path)


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
                        help='Output CSV path (default: <json_dir>/noise_analysis_order3.csv)')
    parser.add_argument('--sensors', type=str, default=None,
                        help='Comma-separated sensor IDs (e.g. "1,3,4,6,7,9,10,12")')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default 42)')
    parser.add_argument('--extension', type=int, default=400,
                        help='Extension distance beyond base sample in mm (default 400)')
    parser.add_argument('--step', type=int, default=10,
                        help='Step size for ray extension in mm (default 10)')

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

    # Magnitude set to 20 A·m²
    magnitudes = [20.0]
    num_samples_fixed = 1

    run_multi_magnitude_analysis(
        json_path,
        magnitudes=magnitudes,
        num_samples=num_samples_fixed,
        radius=args.radius,
        sensor_ids=sensor_ids,
        rng_seed=args.seed,
        output_path=args.output,
        extension_mm=args.extension,
        step_mm=args.step
    )


if __name__ == '__main__':
    main()
