#!/usr/bin/env python3
"""
Calibration fitting variant: normalize b_ref magnitude per (channel, voltage).

For each config:
  - b_ref_norm[n] = b_ref[n] * (mean |b_ref| / |b_ref[n]|)
  - Fit D*b_corr + e = b_ref_norm
  - Evaluate against both b_ref_norm (training fit) and b_ref (fair comparison)

Comparison output: residual_std_normalized.csv vs residual_std_table.csv
"""

import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/zhang/embedded_array_ws/src')
from calibration.lib.center_field_estimator import CenterFieldEstimator


def solve_per_sensor(b_corr_all, b_ref_all):
    N_total, N_sensor, _ = b_corr_all.shape
    results = {}
    for sid_idx in range(N_sensor):
        b_corr_i = b_corr_all[:, sid_idx, :]
        b_ref_i = b_ref_all
        A = np.zeros((N_total * 3, 12))
        b_vec = np.zeros(N_total * 3)
        for n in range(N_total):
            b_c = b_corr_i[n]
            b_r = b_ref_i[n]
            for k in range(3):
                row = n * 3 + k
                A[row, 0:3] = b_c[0] * np.eye(3)[k]
                A[row, 3:6] = b_c[1] * np.eye(3)[k]
                A[row, 6:9] = b_c[2] * np.eye(3)[k]
                A[row, 9:12] = np.eye(3)[k]
                b_vec[row] = b_r[k]
        x, res, rank, s = np.linalg.lstsq(A, b_vec, rcond=None)
        D = x[0:9].reshape(3, 3, order='F')
        e = x[9:12].reshape(3, 1)
        b_pred = A @ x
        residuals = b_pred - b_vec
        rmse = np.sqrt(np.mean(residuals**2))
        results[sid_idx + 1] = {'D': D.tolist(), 'e': e.tolist(), 'rmse': float(rmse)}
    return results


def delta_o_per_row(b_raw_rs, est, D_arr, e_arr, b_ref_eval):
    """Compute Delta_o for each row using D,e for prediction and b_ref_eval for comparison."""
    N = b_raw_rs.shape[0]
    Delta_o = np.zeros(N)
    for n in range(N):
        filtered = est._filter_to_selected_sensors(b_raw_rs[n])
        b_corr_n = est.apply_r_corr(filtered)
        o_n = np.zeros((12, 3))
        for s in range(12):
            o_n[s] = D_arr[s] @ b_corr_n[s, :] + e_arr[s] - b_ref_eval[n]
        o_bar_n = np.mean(o_n, axis=0)
        Delta_o[n] = np.sqrt(np.mean(np.sum((o_n - o_bar_n) ** 2, axis=1)))
    return Delta_o


def delta_o_pre(b_raw_rs, est):
    N = b_raw_rs.shape[0]
    Delta_o = np.zeros(N)
    for n in range(N):
        b_ref_n = est.estimate_from_row(b_raw_rs[n])
        filtered = est._filter_to_selected_sensors(b_raw_rs[n])
        b_corr_n = est.apply_r_corr(filtered)
        o_n = b_corr_n - b_ref_n
        o_bar_n = np.mean(o_n, axis=0)
        Delta_o[n] = np.sqrt(np.mean(np.sum((o_n - o_bar_n) ** 2, axis=1)))
    return Delta_o


def main():
    base_dir = '/home/zhang/embedded_array_ws/src/sensor_data_collection/data'
    cleaned_dir = Path(base_dir) / 'cleaned_aggressive'
    channels = ['manual_x', 'manual_y', 'manual_z']
    voltages = ['1', '2', '3', '4', '5']
    est = CenterFieldEstimator()

    # ── Collect data per config ───────────────────────────────────────────────
    configs = {}
    for ch in channels:
        for v in voltages:
            csv_path = cleaned_dir / f'{ch}_{v}V_cleaned.csv'
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            b_raw = df.values.astype(np.float64)
            N = b_raw.shape[0]
            b_raw_rs = b_raw.reshape(-1, 12, 3)
            b_ref = est.estimate_batch(b_raw)
            mag = np.linalg.norm(b_ref, axis=1)
            mean_mag = mag.mean()
            b_ref_norm = b_ref * (mean_mag / mag[:, None])
            configs[(ch, v)] = {
                'b_raw_rs': b_raw_rs, 'b_ref': b_ref,
                'b_ref_norm': b_ref_norm, 'mean_mag': mean_mag, 'N': N,
            }
            print(f"  {ch}/{v}V: N={N}, mean_mag={mean_mag:.4f}")

    # ── Concatenate ───────────────────────────────────────────────────────────
    all_b_raw_rs   = np.concatenate([v['b_raw_rs']    for v in configs.values()], axis=0)
    all_b_ref      = np.concatenate([v['b_ref']       for v in configs.values()], axis=0)
    all_b_ref_norm = np.concatenate([v['b_ref_norm']  for v in configs.values()], axis=0)

    # Apply R_CORR
    N_total = all_b_raw_rs.shape[0]
    b_corr_all = np.zeros_like(all_b_raw_rs)
    for n in range(N_total):
        filtered = est._filter_to_selected_sensors(all_b_raw_rs[n])
        b_corr_all[n] = est.apply_r_corr(filtered)

    # ── Fit with NORMALIZED b_ref ─────────────────────────────────────────────
    print("\n=== Fitting with normalized b_ref ===")
    results_norm = solve_per_sensor(b_corr_all, all_b_ref_norm)
    D_arr_norm = np.array([results_norm[s]['D'] for s in range(1, 13)])
    e_arr_norm = np.array([results_norm[s]['e'] for s in range(1, 13)]).squeeze()

    # ── Fit with ORIGINAL b_ref (for comparison baseline) ─────────────────────
    print("=== Fitting with original b_ref ===")
    results_orig = solve_per_sensor(b_corr_all, all_b_ref)
    D_arr_orig = np.array([results_orig[s]['D'] for s in range(1, 13)])
    e_arr_orig = np.array([results_orig[s]['e'] for s in range(1, 13)]).squeeze()

    # ── Per-config evaluation ──────────────────────────────────────────────────
    print("\n=== Per-config results ===")
    rows = []
    for ch in channels:
        for v in voltages:
            if (ch, v) not in configs:
                continue
            cfg = configs[(ch, v)]
            b_raw_rs = cfg['b_raw_rs']
            b_ref_cfg = cfg['b_ref']
            b_ref_norm_cfg = cfg['b_ref_norm']
            N = cfg['N']

            # Pre (no calibration)
            d_pre = delta_o_pre(b_raw_rs, est)

            # Post with original calibration (trained on original b_ref)
            d_post_orig = delta_o_per_row(b_raw_rs, est, D_arr_orig, e_arr_orig, b_ref_cfg)

            # Post with normalized calibration evaluated against ORIGINAL b_ref (fair comparison)
            d_post_norm_fair = delta_o_per_row(b_raw_rs, est, D_arr_norm, e_arr_norm, b_ref_cfg)

            # Post with normalized calibration evaluated against NORMALIZED b_ref (training metric)
            d_post_norm_train = delta_o_per_row(b_raw_rs, est, D_arr_norm, e_arr_norm, b_ref_norm_cfg)

            rows.append({
                'channel': ch,
                'voltage': int(v),
                'n_rows': N,
                'mean_mag': cfg['mean_mag'],
                'pre_mean': np.mean(d_pre),
                'post_orig_mean': np.mean(d_post_orig),
                'post_norm_fair': np.mean(d_post_norm_fair),
                'post_norm_train': np.mean(d_post_norm_train),
                'pre_std': np.std(d_pre),
                'post_orig_std': np.std(d_post_orig),
                'post_norm_fair_std': np.std(d_post_norm_fair),
                'post_norm_train_std': np.std(d_post_norm_train),
            })

            print(f"{ch}/{v}V  pre={np.mean(d_pre):.4f}  "
                  f"orig={np.mean(d_post_orig):.4f}  "
                  f"norm_fair={np.mean(d_post_norm_fair):.4f}  "
                  f"norm_train={np.mean(d_post_norm_train):.4f}")

    df = pd.DataFrame(rows)

    # Save comparison CSV
    out_path = Path(__file__).parent.parent / 'data' / 'residual_std_normalized.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Summary
    print("\n=== Overall mean Delta_o ===")
    print(f"  pre:           {df['pre_mean'].mean():.6f}")
    print(f"  post (orig):   {df['post_orig_mean'].mean():.6f}")
    print(f"  post (norm fair, vs orig b_ref): {df['post_norm_fair'].mean():.6f}")
    print(f"  post (norm train, vs norm b_ref): {df['post_norm_train'].mean():.6f}")

    # Save normalized calibration JSON
    cal_out = Path(base_dir) / 'calibration_results_normalized.json'
    with open(cal_out, 'w') as f:
        json.dump({str(k): v for k, v in results_norm.items()}, f, indent=2)
    print(f"Calibration JSON saved to {cal_out}")


if __name__ == '__main__':
    main()
