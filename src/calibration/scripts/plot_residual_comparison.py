#!/usr/bin/env python3
"""
Plot calibration residuals (o = b_corr - b_ref) before and after calibration.

For each (channel, voltage):
  For each row n:
    b_ref[n]   = center field estimate from that row
    o_i[n,s]  = b_corr[n,s,:] - b_ref[n,:]  (12 sensors, 3D each)
    o_bar[n]   = mean_s(o_i[n,s,:])           (3D vector)
    Delta_o[n] = sqrt(mean_s(|o_i[n,s,:] - o_bar[n]|^2))  (scalar)

Subplot: 3 rows (x,y,z) x 5 cols (V1-V5)
Pre-cal: lightgray curve, Post-cal: red curve.
"""

import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath,amsfonts,amssymb,amsthm,mathrsfs,mathtools}\usepackage{bm}\usepackage{dutchcal}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

sys.path.insert(0, '/home/zhang/embedded_array_ws/src')
from calibration.lib.center_field_estimator import CenterFieldEstimator


def compute_delta_o_per_row(b_raw, est, D_arr=None, e_arr=None):
    """Compute Delta_o for each row.

    Args:
        b_raw: (N, 36) raw sensor data
        est: CenterFieldEstimator
        D_arr: optional (12, 3, 3) calibration matrices
        e_arr: optional (12, 3) calibration offsets

    Returns:
        Delta_o: (N,) array of per-row residual std (scalar)
    """
    N_total = b_raw.shape[0]
    b_raw_rs = b_raw.reshape(-1, 12, 3)  # (N, 12, 3)

    # Apply R_CORR -> b_corr
    b_corr = np.zeros_like(b_raw_rs)
    for n in range(N_total):
        filtered = est._filter_to_selected_sensors(b_raw_rs[n])
        b_corr[n] = est.apply_r_corr(filtered)

    # Per-row: compute b_ref[n]
    Delta_o = np.zeros(N_total)
    for n in range(N_total):
        # b_ref from raw data (R_CORR applied inside estimate_from_row)
        b_ref_n = est.estimate_from_row(b_raw_rs[n])  # (3,)

        # Apply calibration if provided
        if D_arr is not None and e_arr is not None:
            o_n = np.zeros((12, 3))
            for s in range(12):
                o_n[s] = D_arr[s] @ b_corr[n, s, :] + e_arr[s] - b_ref_n
        else:
            o_n = b_corr[n] - b_ref_n  # (12, 3)

        # o_bar[n] = mean over 12 sensors
        o_bar_n = np.mean(o_n, axis=0)  # (3,)
        # Delta_o[n] = std over 12 sensors
        Delta_o[n] = np.sqrt(np.mean(np.sum((o_n - o_bar_n) ** 2, axis=1)))

    return Delta_o


def main():
    base_dir = '/home/zhang/embedded_array_ws/src/sensor_data_collection/data'
    cleaned_dir = Path(base_dir) / 'cleaned_aggressive'
    cal_json = Path(base_dir) / 'calibration_results_aggressive.json'

    with open(cal_json) as f:
        cal_data = json.load(f)

    channels = ['manual_x', 'manual_y', 'manual_z']
    voltages = ['1', '2', '3', '4', '5']

    est = CenterFieldEstimator()

    # Build per-sensor D and e arrays
    D_arr = np.zeros((12, 3, 3))
    e_arr = np.zeros((12, 3))
    for sid in range(1, 13):
        D_arr[sid - 1] = np.array(cal_data[str(sid)]['D'])
        e_arr[sid - 1] = np.array(cal_data[str(sid)]['e']).flatten()

    # Collect stats for CSV
    stats_rows = []

    fig, axes = plt.subplots(3, 5, figsize=(7.8, 12 / 2.54),
                            gridspec_kw={'top': 0.92, 'bottom': 0.1,
                                         'left': 0.12, 'right': 0.98})
    row_labels = ['Channel X', 'Channel Y', 'Channel Z']
    col_labels = ['5', '10', '15', '20', '25']

    for row_idx, ch in enumerate(channels):
        for col_idx, v in enumerate(voltages):
            ax = axes[row_idx, col_idx]
            csv_path = cleaned_dir / f'{ch}_{v}V_cleaned.csv'
            if not csv_path.exists():
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            df = pd.read_csv(csv_path)
            b_raw = df.values.astype(np.float64)
            N = b_raw.shape[0]

            delta_pre = compute_delta_o_per_row(b_raw, est, D_arr=None, e_arr=None)
            delta_post = compute_delta_o_per_row(b_raw, est, D_arr=D_arr, e_arr=e_arr)

            stats_rows.append({
                'channel': ch,
                'voltage': v,
                'n_rows': N,
                'pre_mean': np.mean(delta_pre),
                'pre_std': np.std(delta_pre),
                'post_mean': np.mean(delta_post),
                'post_std': np.std(delta_post),
            })

            x = np.linspace(0, 2 * np.pi, N)

            ax.fill_between(x, -delta_pre, delta_pre, alpha=0.3, color='gray')
            ax.plot(x,  delta_pre, color='gray', linewidth=0.5)
            ax.plot(x, -delta_pre, color='gray', linewidth=0.5)

            ax.fill_between(x, -delta_post, delta_post, alpha=0.3, color='red')
            ax.plot(x,  delta_post, color='red', linewidth=0.5)
            ax.plot(x, -delta_post, color='red', linewidth=0.5)

            ax.axhline(0, color='black', linewidth=0.3)
            ax.set_xlim(0, 2 * np.pi)
            ax.set_xticks([0, np.pi, 2 * np.pi])
            ax.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$'])

            if col_idx == 0:
                ax.set_ylabel(r'$\Delta o$ [Gs]')
            if row_idx == 2:
                ax.set_xlabel(r'$\theta$ [rad]')

        # Channel subtitle at top-left of each row
        axes[row_idx, 0].set_title(row_labels[row_idx], pad=2, loc='left', fontsize=8)

    # Column labels at the top
    for col_idx, label in enumerate(col_labels):
        axes[0, col_idx].annotate(label,
                                   xy=(0.5, 1.18), xycoords='axes fraction',
                                   ha='center', va='bottom', fontsize=8)

    plt.subplots_adjust(hspace=0.5, wspace=0.35)
    out_path = Path(__file__).parent.parent / 'plots' / 'residual_comparison.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=600)
    print(f'Saved to {out_path}')

    # Save stats CSV
    stats_df = pd.DataFrame(stats_rows)
    stats_path = Path(__file__).parent.parent / 'data' / 'residual_stats.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f'Saved stats to {stats_path}')


if __name__ == '__main__':
    main()
