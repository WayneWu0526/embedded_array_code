#!/usr/bin/env python3
"""
Concatenate all cleaned_aggressive CSV files into a single long CSV.
Order: channels X→Y→Z, voltages 5→4→3→2→1.

Extra columns per row:
  - b_ref_x, b_ref_y, b_ref_z  (3, center field estimate)
  - Delta_o_pre   (std of o_i pre-calibration)
  - Delta_o_post  (std of o_i post-calibration)
"""

import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '/home/zhang/embedded_array_ws/src')
from calibration.lib.center_field_estimator import CenterFieldEstimator


def compute_row_stats(b_raw_rs, est, D_arr, e_arr):
    """Compute per-row stats for one file.

    Args:
        b_raw_rs: (N, 12, 3) raw sensor data
        est: CenterFieldEstimator
        D_arr: (12, 3, 3) calibration matrices
        e_arr: (12, 3) calibration offsets

    Returns:
        b_ref_arr: (N, 3) center field per row
        delta_pre:  (N,) Delta_o pre-calibration
        delta_post: (N,) Delta_o post-calibration
    """
    N = b_raw_rs.shape[0]

    b_ref_arr = np.zeros((N, 3))
    delta_pre = np.zeros(N)
    delta_post = np.zeros(N)

    for n in range(N):
        # b_ref from raw data (R_CORR applied inside)
        b_ref_n = est.estimate_from_row(b_raw_rs[n])
        b_ref_arr[n] = b_ref_n

        # b_corr
        filtered = est._filter_to_selected_sensors(b_raw_rs[n])
        b_corr_n = est.apply_r_corr(filtered)

        # Pre-cal: o = b_corr - b_ref
        o_pre = b_corr_n - b_ref_n
        o_bar_pre = np.mean(o_pre, axis=0)
        delta_pre[n] = np.sqrt(np.mean(np.sum((o_pre - o_bar_pre) ** 2, axis=1)))

        # Post-cal: o = (D*b_corr + e) - b_ref
        o_post = np.zeros((12, 3))
        for s in range(12):
            o_post[s] = D_arr[s] @ b_corr_n[s, :] + e_arr[s] - b_ref_n
        o_bar_post = np.mean(o_post, axis=0)
        delta_post[n] = np.sqrt(np.mean(np.sum((o_post - o_bar_post) ** 2, axis=1)))

    return b_ref_arr, delta_pre, delta_post


def main():
    base_dir = '/home/zhang/embedded_array_ws/src/sensor_data_collection/data'
    cleaned_dir = Path(base_dir) / 'cleaned_aggressive'
    cal_json = Path(base_dir) / 'calibration_results_aggressive.json'
    out_dir = Path(__file__).parent.parent / 'data'
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cal_json) as f:
        cal_data = json.load(f)

    est = CenterFieldEstimator()

    D_arr = np.zeros((12, 3, 3))
    e_arr = np.zeros((12, 3))
    for sid in range(1, 13):
        D_arr[sid - 1] = np.array(cal_data[str(sid)]['D'])
        e_arr[sid - 1] = np.array(cal_data[str(sid)]['e']).flatten()

    channels = ['manual_x', 'manual_y', 'manual_z']
    voltages = ['5', '4', '3', '2', '1']   # 5→1

    all_dfs_pre = []
    all_dfs_post = []

    for ch in channels:
        for v in voltages:
            csv_path = cleaned_dir / f'{ch}_{v}V_cleaned.csv'
            if not csv_path.exists():
                print(f'Warning: {csv_path} not found, skipping')
                continue

            df = pd.read_csv(csv_path)
            b_raw = df.values.astype(np.float64)
            b_raw_rs = b_raw.reshape(-1, 12, 3)
            N = b_raw.shape[0]

            b_ref_arr, delta_pre, delta_post = compute_row_stats(b_raw_rs, est, D_arr, e_arr)

            for tag, delta_arr, all_dfs in [('pre', delta_pre, all_dfs_pre), ('post', delta_post, all_dfs_post)]:
                df_out = df.copy()
                df_out['b_ref_x'] = b_ref_arr[:, 0]
                df_out['b_ref_y'] = b_ref_arr[:, 1]
                df_out['b_ref_z'] = b_ref_arr[:, 2]
                df_out['Delta_o'] = delta_arr
                df_out['channel'] = ch
                df_out['voltage'] = int(v)
                df_out['source_file'] = f'{ch}_{v}V_cleaned.csv'
                all_dfs.append(df_out)

            print(f'  {ch}/{v}V: {N} rows')

    df_pre = pd.concat(all_dfs_pre, ignore_index=True)
    df_post = pd.concat(all_dfs_post, ignore_index=True)

    out_pre = out_dir / 'residual_b_corr.csv'
    out_post = out_dir / 'residual_b_calibrated.csv'

    df_pre.to_csv(out_pre, index=False)
    df_post.to_csv(out_post, index=False)
    print(f'\nSaved: {out_pre}  shape={df_pre.shape}')
    print(f'Saved: {out_post}  shape={df_post.shape}')


if __name__ == '__main__':
    main()
