#!/usr/bin/env python3
"""
Affine model calibration for sensor array.

Per-sensor model:  b_corrected = D_i @ b_raw + e_i

For each CSV file:
  - Compute b_ref via CenterFieldEstimator (all sensors in the selected array contribute)
  - b_ref_norm[n] = b_ref[n] * (mean |b_ref| / |b_ref[n]|)  [per-CSV normalization]
  - Fit D_i @ b_raw + e_i = b_ref_norm  per sensor
  - Output: affine_model_params.json with D_i, e_i per sensor

Pipeline: b_raw(orientation-aligned, Gs) -> D_i @ b_raw + e_i -> b_corrected
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import argparse

from calibration import CenterFieldEstimator
from sensor_array_config import get_array_config, list_array_configs


def solve_per_sensor(b_corr_all, b_ref_all):
    N_total, N_sensor, _ = b_corr_all.shape
    results = {}
    for sid_idx in range(N_sensor):
        b_corr_i = b_corr_all[:, sid_idx, :]
        b_ref_i = b_ref_all
        A = np.zeros((N_total * 3, 12))  # 9 affine matrix terms + 3 bias terms
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
    n_sensors = est.n_sensors
    Delta_o = np.zeros(N)
    for n in range(N):
        filtered = est._filter_to_selected_sensors(b_raw_rs[n])
        b_corr_n = filtered
        o_n = np.zeros((n_sensors, 3))
        for s in range(n_sensors):
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
        b_corr_n = filtered
        o_n = b_corr_n - b_ref_n
        o_bar_n = np.mean(o_n, axis=0)
        Delta_o[n] = np.sqrt(np.mean(np.sum((o_n - o_bar_n) ** 2, axis=1)))
    return Delta_o


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Fit per-sensor affine calibration from manual_record_*.csv files.",
    )
    parser.add_argument(
        "--array-config",
        default=None,
        help="Array config name under sensor_array_config/config/arrays. If omitted, prompt interactively.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "data" / "manual_calibration",
        help="Directory containing manual_record_*.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output affine_model_params.json path. Defaults to the selected array config bundle.",
    )
    return parser


def select_array_config(array_config_arg):
    if array_config_arg:
        return array_config_arg

    array_configs = list_array_configs()
    if not array_configs:
        raise RuntimeError("No array configs found under sensor_array_config/config/arrays")

    print("\nSelect array config to calibrate:")
    for idx, name in enumerate(array_configs, start=1):
        print(f"  {idx}. {name}")

    while True:
        choice = input(f"Array config [1-{len(array_configs)}]: ").strip()
        try:
            idx = int(choice)
        except ValueError:
            print("Please enter a number.")
            continue
        if 1 <= idx <= len(array_configs):
            return array_configs[idx - 1]
        print(f"Please enter a number between 1 and {len(array_configs)}.")


def main():
    args = build_arg_parser().parse_args()
    base_dir = args.data_dir
    array_config_name = select_array_config(args.array_config)
    array_config = get_array_config(array_config_name)
    n_sensors = int(array_config.manifest.n_sensors)
    est = CenterFieldEstimator(sensor_config=array_config)
    default_output = (
        Path(__file__).resolve().parents[2]
        / "sensor_array_config"
        / "config"
        / "arrays"
        / array_config_name
        / "affine_model_params.json"
    )
    output_path = args.output if args.output is not None else default_output

    # ── Collect data per CSV (each CSV normalized independently) ───────────────
    csv_files = sorted(base_dir.glob('manual_record_*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No manual_record_*.csv files found in {base_dir}")
    configs = {}
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        b_raw = df.values.astype(np.float64)
        expected_cols = n_sensors * 3
        if b_raw.shape[1] != expected_cols:
            raise ValueError(
                f"{csv_path} has {b_raw.shape[1]} columns, expected {expected_cols} "
                f"for array_config={array_config_name}"
            )
        N = b_raw.shape[0]
        b_raw_rs = b_raw.reshape(-1, n_sensors, 3)
        b_ref, b_corr = est.estimate_batch(b_raw)
        mag = np.linalg.norm(b_ref, axis=1)
        mean_mag = mag.mean()
        b_ref_norm = b_ref * (mean_mag / mag[:, None])
        configs[csv_path.name] = {
            'b_raw_rs': b_raw_rs, 'b_ref': b_ref,
            'b_ref_norm': b_ref_norm, 'b_corr': b_corr,
            'mean_mag': mean_mag, 'N': N,
        }
        print(f"  {csv_path.name}: N={N}, mean_mag={mean_mag:.4f}")

    # ── Concatenate ───────────────────────────────────────────────────────────
    all_b_raw_rs   = np.concatenate([v['b_raw_rs']    for v in configs.values()], axis=0)
    all_b_ref      = np.concatenate([v['b_ref']       for v in configs.values()], axis=0)
    all_b_ref_norm = np.concatenate([v['b_ref_norm']  for v in configs.values()], axis=0)
    all_b_corr     = np.concatenate([v['b_corr']      for v in configs.values()], axis=0)

    # Input CSVs are expected to contain raw Gs data after R_CORR orientation alignment.
    # No additional R_CORR step is applied during affine calibration.

    # ── Fit with NORMALIZED b_ref ─────────────────────────────────────────────
    print("\n=== Fitting affine model with normalized b_ref ===")
    results_norm = solve_per_sensor(all_b_corr, all_b_ref_norm)
    D_arr_norm = np.array([results_norm[s]['D'] for s in range(1, n_sensors + 1)])
    e_arr_norm = np.array([results_norm[s]['e'] for s in range(1, n_sensors + 1)]).squeeze()

    # ── Fit with ORIGINAL b_ref (for comparison baseline) ─────────────────────
    print("=== Fitting with original b_ref ===")
    results_orig = solve_per_sensor(all_b_corr, all_b_ref)
    D_arr_orig = np.array([results_orig[s]['D'] for s in range(1, n_sensors + 1)])
    e_arr_orig = np.array([results_orig[s]['e'] for s in range(1, n_sensors + 1)]).squeeze()

    # ── Per-CSV evaluation ────────────────────────────────────────────────────
    print("\n=== Per-CSV results ===")
    rows = []
    for csv_name, cfg in configs.items():
        b_raw_rs = cfg['b_raw_rs']
        b_ref_cfg = cfg['b_ref']
        b_ref_norm_cfg = cfg['b_ref_norm']
        N = cfg['N']

        # Pre (no calibration)
        d_pre = delta_o_pre(b_raw_rs, est)

        # Post with original calibration (trained on original b_ref)
        d_post_orig = delta_o_per_row(b_raw_rs, est, D_arr_orig, e_arr_orig, b_ref_cfg)

        # Post with affine calibration evaluated against ORIGINAL b_ref (fair comparison)
        d_post_norm_fair = delta_o_per_row(b_raw_rs, est, D_arr_norm, e_arr_norm, b_ref_cfg)

        # Post with affine calibration evaluated against NORMALIZED b_ref (training metric)
        d_post_norm_train = delta_o_per_row(b_raw_rs, est, D_arr_norm, e_arr_norm, b_ref_norm_cfg)

        rows.append({
            'csv': csv_name,
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

        print(f"{csv_name}  pre={np.mean(d_pre):.4f}  "
              f"orig={np.mean(d_post_orig):.4f}  "
              f"norm_fair={np.mean(d_post_norm_fair):.4f}  "
              f"norm_train={np.mean(d_post_norm_train):.4f}")

    df = pd.DataFrame(rows)

    # Save comparison CSV (optional diagnostic output)
    # out_path = Path(__file__).parent.parent / 'data' / 'residual_std_normalized.csv'
    # df.to_csv(out_path, index=False)
    # print(f"\nSaved to {out_path}")

    # Summary
    print("\n=== Overall mean Delta_o ===")
    print(f"  pre:           {df['pre_mean'].mean():.6f}")
    print(f"  post (orig):   {df['post_orig_mean'].mean():.6f}")
    print(f"  post (norm fair, vs orig b_ref): {df['post_norm_fair'].mean():.6f}")
    print(f"  post (norm train, vs norm b_ref): {df['post_norm_train'].mean():.6f}")

    # Save affine calibration JSON
    cal_out = output_path.expanduser()
    cal_out.parent.mkdir(parents=True, exist_ok=True)
    output = {"sensors": []}
    for sid, params in results_norm.items():
        output["sensors"].append({
            'sensor_id': int(sid),
            'D_i': params['D'],
            'e_i': list(np.array(params['e']).flatten())
        })
    with open(cal_out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Calibration JSON saved to {cal_out}")


if __name__ == '__main__':
    main()
