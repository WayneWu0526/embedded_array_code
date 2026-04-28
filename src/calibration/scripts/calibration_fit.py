#!/usr/bin/env python3
"""
Calibration fitting without magnitude normalization.

流程:
1. 对每组 (channel, voltage) 的所有行，计算 b_ref
2. 拼接所有 channel-voltage 数据
3. 对每个传感器拟合 D*b_corr + e = b_ref
4. 保存结果并评估
"""

import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/zhang/embedded_array_ws/src')
from calibration.lib.center_field_estimator import CenterFieldEstimator


def process_channel_voltage(csv_path):
    """Process a single channel-voltage CSV file.

    Returns:
        b_raw: (N, 36) raw sensor data
        b_ref: (N, 3) center field estimate (no normalization)
    """
    df = pd.read_csv(csv_path)
    b_raw = df.values.astype(np.float64)  # (N, 36)
    N = b_raw.shape[0]

    # Compute b_ref for each row
    est = CenterFieldEstimator()
    b_ref = est.estimate_batch(b_raw)  # (N, 3)

    return b_raw, b_ref


def solve_per_sensor(b_corr_all, b_ref_all):
    """Solve D*b_corr + e = b_ref for each sensor using lstsq."""
    N_total, N_sensor, _ = b_corr_all.shape
    results = {}

    for sid_idx in range(N_sensor):
        b_corr_i = b_corr_all[:, sid_idx, :]  # (N_total, 3)
        b_ref_i = b_ref_all                    # (N_total, 3)

        # Build A matrix: each row is [b_corr^T ⊗ I_3, I_3]
        A = np.zeros((N_total * 3, 12))
        b_vec = np.zeros(N_total * 3)

        for n in range(N_total):
            b_c = b_corr_i[n]   # (3,)
            b_r = b_ref_i[n]    # (3,)
            for k in range(3):
                row = n * 3 + k
                # D structure: D[0,:] multiplies b_corr[0], etc.
                # b_ref[k] = D[k,0]*b_c[0] + D[k,1]*b_c[1] + D[k,2]*b_c[2] + e[k]
                A[row, 0:3] = b_c[0] * np.eye(3)[k]
                A[row, 3:6] = b_c[1] * np.eye(3)[k]
                A[row, 6:9] = b_c[2] * np.eye(3)[k]
                A[row, 9:12] = np.eye(3)[k]
                b_vec[row] = b_r[k]

        # Solve least squares
        x, res, rank, s = np.linalg.lstsq(A, b_vec, rcond=None)

        D = x[0:9].reshape(3, 3, order='F')
        e = x[9:12].reshape(3, 1)

        # Compute RMSE
        b_pred = A @ x
        residuals = b_pred - b_vec
        rmse = np.sqrt(np.mean(residuals**2))

        results[sid_idx + 1] = {
            'D': D.tolist(),
            'e': e.tolist(),
            'rmse': float(rmse),
        }
        print(f"Sensor {sid_idx + 1}: RMSE = {rmse:.6f}, Rank = {rank}")

    return results


def evaluate_per_file(results, base_dir, channels, voltages, est):
    """Evaluate calibration quality per file."""
    for ch in channels:
        ch_dir = Path(base_dir) / ch
        for v in voltages:
            csv_path = ch_dir / f'manual_record_{v}V.csv'
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            b_raw = df.values.astype(np.float64)
            N = b_raw.shape[0]

            b_ref = est.estimate_batch(b_raw)

            b_raw_rs = b_raw.reshape(-1, 12, 3)
            b_corr = np.zeros_like(b_raw_rs)
            for n in range(N):
                filtered = est._filter_to_selected_sensors(b_raw_rs[n])
                b_corr[n] = est.apply_r_corr(filtered)

            rmses = []
            for sid in range(1, 13):
                D = np.array(results[sid]['D'])
                e = np.array(results[sid]['e']).flatten()
                b_pred = (D @ b_corr[:, sid-1, :].T).T + e
                diff = b_pred - b_ref
                rmse = np.sqrt(np.mean(diff**2))
                rmses.append(rmse)

            mag = np.linalg.norm(b_ref, axis=1).mean()
            print(f"  {ch}/{v}V: N={N}, avg_mag={mag:.2f}, "
                  f"mean_RMSE={np.mean(rmses):.4f}, max_RMSE={np.max(rmses):.4f}")


def main():
    base_dir = '/home/zhang/embedded_array_ws/src/sensor_data_collection/data'
    cleaned_dir = Path(base_dir) / 'cleaned_aggressive'
    channels = ['manual_x', 'manual_y', 'manual_z']
    voltages = ['1', '2', '3', '4', '5']

    # Step 1: Process each channel-voltage (truncated + hampel cleaned data)
    print("Step 1: Processing each channel-voltage (no normalization)...")
    all_b_raw = []
    all_b_ref = []

    for ch in channels:
        for v in voltages:
            csv_path = cleaned_dir / f'{ch}_{v}V_cleaned.csv'
            if not csv_path.exists():
                print(f"Warning: {csv_path} not found, skipping")
                continue
            b_raw, b_ref = process_channel_voltage(csv_path)
            all_b_raw.append(b_raw)
            all_b_ref.append(b_ref)
            avg_mag = np.linalg.norm(b_ref, axis=1).mean()
            print(f"  {ch}/{v}V: {b_raw.shape[0]} rows, avg_mag = {avg_mag:.6f}")

    # Step 2: Concatenate
    print("\nStep 2: Concatenating data...")
    b_raw_all = np.concatenate(all_b_raw, axis=0)
    b_ref_all = np.concatenate(all_b_ref, axis=0)
    print(f"  Total rows: {b_raw_all.shape[0]}")

    # Reshape and apply R_CORR
    b_raw_all = b_raw_all.reshape(-1, 12, 3)

    print("\nStep 3: Applying R_CORR...")
    est = CenterFieldEstimator()
    b_corr_all = np.zeros_like(b_raw_all)
    for n in range(b_raw_all.shape[0]):
        filtered = est._filter_to_selected_sensors(b_raw_all[n])
        b_corr_all[n] = est.apply_r_corr(filtered)
    print(f"  b_corr_all shape: {b_corr_all.shape}")

    # Step 4: Solve D*b_corr + e = b_ref for each sensor
    print("\nStep 4: Solving D*b_corr + e = b_ref for each sensor...")
    results = solve_per_sensor(b_corr_all, b_ref_all)

    # Save to JSON (convert int keys to str for JSON compatibility)
    output_path = Path(base_dir) / 'calibration_results_aggressive.json'
    with open(output_path, 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Step 5: Evaluate per file
    print("\nStep 5: Per-file RMSE evaluation...")
    evaluate_per_file(results, base_dir, channels, voltages, est)

    # Print summary
    print("\n=== Calibration Summary ===")
    rmses = [results[k]['rmse'] for k in results]
    print(f"Average Training RMSE: {np.mean(rmses):.6f}")
    print(f"Max RMSE: {np.max(rmses):.6f} (Sensor {max(results, key=lambda k: results[k]['rmse'])}")
    print(f"Min RMSE: {np.min(rmses):.6f} (Sensor {min(results, key=lambda k: results[k]['rmse'])})")


if __name__ == '__main__':
    main()
