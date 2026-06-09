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

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from calibration import (
    CenterFieldEstimator,
    delta_o_per_row,
    delta_o_pre,
    load_manual_record_sets,
    result_arrays,
    solve_per_sensor,
    stack_record_sets,
    write_affine_model_params,
)
from sensor_array_config import get_array_config, list_array_configs


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
    records = load_manual_record_sets(base_dir, est, array_config_name)
    for csv_name, record in records.items():
        print(f"  {csv_name}: N={record.n_rows}, mean_mag={record.mean_mag:.4f}")

    # ── Concatenate ───────────────────────────────────────────────────────────
    stacked = stack_record_sets(records)
    all_b_corr = stacked["b_corr"]
    all_b_ref = stacked["b_ref"]
    all_b_ref_norm = stacked["b_ref_norm"]

    # Input CSVs are expected to contain raw Gs data after R_CORR orientation alignment.
    # No additional R_CORR step is applied during affine calibration.

    # ── Fit with NORMALIZED b_ref ─────────────────────────────────────────────
    print("\n=== Fitting affine model with normalized b_ref ===")
    results_norm = solve_per_sensor(all_b_corr, all_b_ref_norm)
    D_arr_norm, e_arr_norm = result_arrays(results_norm, n_sensors)

    # ── Fit with ORIGINAL b_ref (for comparison baseline) ─────────────────────
    print("=== Fitting with original b_ref ===")
    results_orig = solve_per_sensor(all_b_corr, all_b_ref)
    D_arr_orig, e_arr_orig = result_arrays(results_orig, n_sensors)

    # ── Per-CSV evaluation ────────────────────────────────────────────────────
    print("\n=== Per-CSV results ===")
    rows = []
    for csv_name, record in records.items():
        b_raw_rs = record.b_raw_rs
        b_ref_cfg = record.b_ref
        b_ref_norm_cfg = record.b_ref_norm
        n_rows = record.n_rows

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
            'n_rows': n_rows,
            'mean_mag': record.mean_mag,
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
    write_affine_model_params(cal_out, results_norm)
    print(f"Calibration JSON saved to {cal_out}")


if __name__ == '__main__':
    main()
