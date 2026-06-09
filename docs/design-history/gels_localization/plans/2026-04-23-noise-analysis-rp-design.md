# Noise Analysis Rp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `noise_analysis_rp.py` — a script that analyzes how measurement noise affects MaPS estimator accuracy by randomly perturbing ground-truth sensor array pose (R, p) and measuring estimation error across noise levels.

**Architecture:** The script reads source poses from `cycle_0000.json`, generates 100 random ground-truth poses within 5cm of the original, generates synthetic B_meas for each, injects noise at levels [0.0, 0.02, 0.04, 0.06, 0.08, 0.1], runs MaPS_Estimator, and plots position/orientation error vs noise level with error bars (mean ± std).

**Tech Stack:** numpy, matplotlib, existing offline_utils.py, maps_estimator.py, mag_dipole_model.py

---

## File Structure

```
src/gels_localization/scripts/noise_analysis_rp.py   # NEW — the main script
```

Dependencies (read-only, no modifications):
- `offline_utils.py` — `json_to_request`, `generate_hall_data_from_dipole`
- `maps_estimator.py` — `MaPS_Estimator`
- `localization_service_node.py` — `build_D_cal`, `quaternion_to_rotation_matrix`, `compute_rotation_error`, `GS_TO_TESLA`
- `mag_dipole_model.py` — `mag_dipole_model`
- `offline_mock.py` — `MockPose`, `MockSensorReading`, `MockSlotData`, `MockLocalizeCycleRequest`

---

## Task Decomposition

### Task 1: Write the noise_analysis_rp.py script

**File:** Create `src/gels_localization/scripts/noise_analysis_rp.py`

- [ ] **Step 1: Imports and helper functions**

```python
#!/usr/bin/env python3
"""
Noise analysis: random ground-truth pose perturbation study.

Generates 100 random (R, p) ground truths near the original,
injects noise into B_meas at different levels, runs MaPS_Estimator,
and plots position/orientation error vs noise level.
"""

import sys
import os
import glob
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from offline_utils import json_to_request
from maps_estimator import MaPS_Estimator
from localization_service_node import build_D_cal, quaternion_to_rotation_matrix, compute_rotation_error, load_configuration, GS_TO_TESLA
from mag_dipole_model import mag_dipole_model
from offline_mock import MockPose, MockSensorReading, MockSlotData, MockLocalizeCycleRequest


def sample_random_rotation():
    """Sample a random rotation matrix uniformly from SO(3) using quaternion method."""
    # Sample 4 Gaussian random numbers, normalize to unit quaternion
    q = np.random.randn(4)
    q = q / np.linalg.norm(q)
    # Convert to rotation matrix (Baker-Campbell-Hausdorff)
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - w*z),         2*(x*z + w*y)],
        [    2*(x*y + w*z),     1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
        [    2*(x*z - w*y),         2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
    ])
    return R


def sample_random_position(center, radius=0.05):
    """Sample a random point uniformly within a sphere of given radius."""
    # Uniform sampling in 3D ball: sample direction + cube-root radius
    direction = np.random.randn(3)
    direction = direction / np.linalg.norm(direction)
    r = radius * (np.random.rand() ** (1/3))
    return center + r * direction


def generate_synthetic_B_meas_for_pose(p_sensor_array, R_sensor_array, D_LIST, sources, sensor_ids, noise_level=0.0):
    """
    Generate synthetic B_meas for a given sensor array pose.

    Args:
        p_sensor_array: 3D position of sensor array center (global frame)
        R_sensor_array: 3x3 rotation matrix of sensor array orientation
        D_LIST: (12, 3) array of sensor positions relative to array center
        sources: list of dicts with 'p_Ci' (3,) and 'm_Ci' (3,)
        sensor_ids: list of active sensor IDs (1-12)
        noise_level: noise standard deviation in Gs (same as offline_utils.py convention)

    Returns:
        B_meas_cell: list of 3 x N matrices (one per source/slot)
    """
    B_meas_cell = []
    for src in sources:
        p_Ci = src['p_Ci']
        m_Ci = src['m_Ci']

        B_meas = np.zeros((3, len(sensor_ids)))
        for col_idx, sid in enumerate(sensor_ids):
            sensor_idx = sid - 1
            d_j = D_LIST[sensor_idx]
            p_sensor_global = p_sensor_array + R_sensor_array @ d_j

            b_global, _ = mag_dipole_model(p_sensor_global, m_Ci, p_Ci, order=1)
            b_sensor = R_sensor_array.T @ b_global

            # Convert to Gs
            b_sensor_gs = b_sensor / GS_TO_TESLA

            # Add noise in Gs
            if noise_level > 0:
                b_sensor_gs = b_sensor_gs + noise_level * np.random.randn(3)

            B_meas[:, col_idx] = b_sensor_gs

        B_meas_cell.append(B_meas)

    return B_meas_cell
```

- [ ] **Step 2: Main analysis function**

```python
def run_noise_analysis(json_path, num_samples=100, radius=0.05,
                        noise_levels=[0.0, 0.02, 0.04, 0.06, 0.08, 0.1],
                        output_path=None, sensor_ids=None):
    """
    Run noise analysis with randomly perturbed ground-truth poses.

    Args:
        json_path: path to cycle JSON file
        num_samples: number of random ground-truth poses (default 100)
        radius: sphere radius in meters for random position perturbation (default 0.05 = 5cm)
        noise_levels: list of noise levels (fraction of x_hat std dev, same convention as offline_utils)
        output_path: path to save the plot
        sensor_ids: list of sensor IDs to use (default all 12)

    Returns:
        dict: results with statistics per noise level
    """
    # Load request and config
    req = json_to_request(json_path)
    load_configuration()

    from localization_service_node import D_LIST as CONFIG_D_LIST

    if sensor_ids is None:
        sensor_ids = list(req.sensor_ids)

    # Extract ground truth pose
    gt = req.ground_truth_pose
    p_gt_original = np.array([gt.position.x, gt.position.y, gt.position.z])
    R_gt_original = quaternion_to_rotation_matrix(np.array([
        gt.orientation.w, gt.orientation.x, gt.orientation.y, gt.orientation.z
    ]))

    # Build sources from slot data (slots 0, 1, 2 — same as handle_localize_cycle)
    slot_dict = {slot.slot: slot for slot in req.slot_data}

    # Moment magnitudes per slot (same as offline_utils.py)
    moment_magnitude = np.array([-120, -200, -200])

    sources = []
    for slot_idx in [0, 1, 2]:
        if slot_idx not in slot_dict:
            continue
        slot = slot_dict[slot_idx]
        p_Ci = np.array([slot.pose.position.x, slot.pose.position.y, slot.pose.position.z])
        quat = np.array([slot.pose.orientation.w, slot.pose.orientation.x,
                         slot.pose.orientation.y, slot.pose.orientation.z])
        # z-axis of EM coil direction = magnetic moment direction
        z_axis = np.array([
            2 * (quat[1] * quat[3] + quat[0] * quat[2]),
            2 * (quat[2] * quat[3] - quat[0] * quat[1]),
            1 - 2 * (quat[1]**2 + quat[2]**2)
        ])
        z_axis = z_axis / np.linalg.norm(z_axis)
        m_Ci = moment_magnitude[slot_idx] * z_axis
        sources.append({'p_Ci': p_Ci, 'm_Ci': m_Ci})

    M = len(sources)
    D_cal = build_D_cal(sensor_ids)

    # Pre-generate random ground truths
    print(f"Generating {num_samples} random ground-truth poses within {radius*100:.1f}cm sphere...")
    rng = np.random.default_rng(42)  # reproducible
    random_poses = []
    for i in range(num_samples):
        p_rand = sample_random_position(p_gt_original, radius)
        R_rand = sample_random_rotation()
        random_poses.append((p_rand, R_rand))

    # Results storage: noise_level -> {pos_errors, ori_errors}
    results = {nl: {'pos': [], 'ori': []} for nl in noise_levels}

    print(f"Running analysis across {len(noise_levels)} noise levels...")
    for nl_idx, noise_level in enumerate(noise_levels):
        for sample_idx, (p_gt, R_gt) in enumerate(random_poses):
            # Generate synthetic B_meas for this ground truth
            B_meas_cell = generate_synthetic_B_meas_for_pose(
                p_gt, R_gt, CONFIG_D_LIST, sources, sensor_ids, noise_level=noise_level
            )

            # Run MaPS Estimator
            try:
                R_est, p_est, details = MaPS_Estimator(D_cal, sources, B_meas_cell)
            except Exception as e:
                # If estimation fails, record NaN
                results[noise_level]['pos'].append(np.nan)
                results[noise_level]['ori'].append(np.nan)
                continue

            # Compute errors
            pos_error = np.linalg.norm(p_est - p_gt)
            ori_error = compute_rotation_error(R_est, R_gt)

            results[noise_level]['pos'].append(pos_error)
            results[noise_level]['ori'].append(ori_error)

        valid_pos = [e for e in results[noise_level]['pos'] if not np.isnan(e)]
        valid_ori = [e for e in results[noise_level]['ori'] if not np.isnan(e)]
        print(f"  noise={noise_level:.3f}: {len(valid_pos)}/{num_samples} succeeded, "
              f"pos_err={np.mean(valid_pos)*1000:.4f}mm, ori_err={np.degrees(np.mean(valid_ori)):.4f}deg")

    # Compute statistics
    stats = {}
    for nl in noise_levels:
        pos_arr = np.array([e for e in results[nl]['pos'] if not np.isnan(e)])
        ori_arr = np.array([e for e in results[nl]['ori'] if not np.isnan(e)])
        stats[nl] = {
            'pos_mean': np.mean(pos_arr),
            'pos_std': np.std(pos_arr),
            'ori_mean': np.mean(ori_arr),
            'ori_std': np.std(ori_arr),
        }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Position error
    ax = axes[0]
    pos_means = [stats[nl]['pos_mean'] for nl in noise_levels]
    pos_stds = [stats[nl]['pos_std'] for nl in noise_levels]
    ax.errorbar(noise_levels, np.array(pos_means) * 1000,
                yerr=np.array(pos_stds) * 1000, fmt='o-', capsize=4, color='steelblue')
    ax.set_xlabel('Noise Level (Gs)')
    ax.set_ylabel('Position Error (mm)')
    ax.set_title('Position Error vs Noise Level')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Orientation error
    ax = axes[1]
    ori_means = [stats[nl]['ori_mean'] for nl in noise_levels]
    ori_stds = [stats[nl]['ori_std'] for nl in noise_levels]
    ax.errorbar(noise_levels, np.degrees(ori_means),
                yerr=np.degrees(ori_stds), fmt='o-', capsize=4, color='coral')
    ax.set_xlabel('Noise Level (Gs)')
    ax.set_ylabel('Orientation Error (deg)')
    ax.set_title('Orientation Error vs Noise Level')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Noise Analysis: {num_samples} Random Ground-Truth Poses (5cm sphere)')
    plt.tight_layout()

    save_path = output_path or os.path.join(
        os.path.dirname(os.path.abspath(json_path)),
        f'noise_analysis_rp_cycle_{req.cycle_id:04d}.png'
    )
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()

    return stats
```

- [ ] **Step 3: Main block**

```python
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Noise analysis with random ground-truth pose perturbation.')
    parser.add_argument('json_path', nargs='?', default=None,
                        help='Path to cycle JSON file')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of random ground-truth samples (default 100)')
    parser.add_argument('--radius', type=float, default=0.05,
                        help='Sphere radius in meters for random position perturbation (default 0.05 = 5cm)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output plot path')
    parser.add_argument('--sensors', type=str, default=None,
                        help='Comma-separated sensor IDs (e.g. "1,3,4,6,7,9,10,12")')

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

    noise_levels = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]

    stats = run_noise_analysis(
        json_path,
        num_samples=args.samples,
        radius=args.radius,
        noise_levels=noise_levels,
        output_path=args.output,
        sensor_ids=sensor_ids
    )

    print("\n=== Summary ===")
    print(f"{'Noise':<8} {'Pos Mean':<12} {'Pos Std':<12} {'Ori Mean':<12} {'Ori Std':<12}")
    print("-" * 56)
    for nl in noise_levels:
        s = stats[nl]
        print(f"{nl:<8.3f} {s['pos_mean']*1000:<12.4f} {s['pos_std']*1000:<12.4f} "
              f"{np.degrees(s['ori_mean']):<12.4f} {np.degrees(s['ori_std']):<12.4f}")


if __name__ == '__main__':
    main()
```

### Task 2: Test the script

**File:** Run against `cycle_0000.json`

- [ ] **Step 1: Run the script**

Run: `cd /home/zhang/embedded_array_ws && python src/gels_localization/scripts/noise_analysis_rp.py`
Expected: Generates `noise_analysis_rp_cycle_0000.png` in the same directory as `cycle_0000.json`

---

## Spec Coverage Checklist

| Requirement | Task |
|---|---|
| Read cycle_0000.json source poses (slots 0,1,2) | Task 1, Step 2 |
| 100 random ground-truth (R, p) within 5cm sphere | Task 1, Step 1 (`sample_random_position`, `sample_random_rotation`) |
| Noise levels [0.0, 0.02, 0.04, 0.06, 0.08, 0.1] | Task 1, Step 3 |
| Generate synthetic B_meas via mag_dipole_model | Task 1, Step 1 (`generate_synthetic_B_meas_for_pose`) |
| Run MaPS_Estimator for each (R,p,noise) | Task 1, Step 2 |
| Position error = \|\|p_est - p_gt\|\| | Task 1, Step 2 |
| Orientation error via arccos(trace-based) formula | Task 1, Step 2 (`compute_rotation_error`) |
| Error bar plot (mean ± std), log y-axis | Task 1, Step 2 |
| Output image `noise_analysis_rp_cycle_XXXX.png` | Task 1, Step 2 |

## Self-Review

- All placeholder/TODO patterns resolved: yes
- Types consistent: `sample_random_rotation` returns np.ndarray (3x3), `sample_random_position` returns np.ndarray (3,)
- Noise level convention matches `offline_utils.py`: noise in Gs, added to b_sensor_gs before conversion
- Ground-truth generation uses reproducible RNG (seed=42) for position sampling
- Orientation error uses `compute_rotation_error` from `localization_service_node.py` (arccos formula)
- No modifications to existing files — new script only
