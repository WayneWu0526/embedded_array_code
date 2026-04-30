#!/usr/bin/env python3
"""
Plot calibration residuals using normalized b_ref calibration.

Curves:
  Gray:  pre-calibration  (no calibration applied)
  Red:   post-calibration (original b_ref D,e)
  Blue:  post-calibration (normalized b_ref D,e)

Subplot: 3 rows (x,y,z) x 5 cols (V1-V5)
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


def compute_delta_o(b_raw_rs, est, D_arr, e_arr, b_ref_all, apply_cal):
    """Compute Delta_o per row.

    Args:
        apply_cal: if True use D,e; if False compute pre-calibration residuals
    """
    N = b_raw_rs.shape[0]
    Delta_o = np.zeros(N)
    for n in range(N):
        b_ref_n = b_ref_all[n]
        filtered = est._filter_to_selected_sensors(b_raw_rs[n])
        b_corr_n = est.apply_r_corr(filtered)
        if apply_cal:
            o_n = np.zeros((12, 3))
            for s in range(12):
                o_n[s] = D_arr[s] @ b_corr_n[s, :] + e_arr[s] - b_ref_n
        else:
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

    # Load original calibration
    with open(Path(base_dir) / 'calibration_results_aggressive.json') as f:
        cal_orig = json.load(f)
    D_orig = np.array([cal_orig[str(s)]['D'] for s in range(1, 13)])
    e_orig = np.array([cal_orig[str(s)]['e'] for s in range(1, 13)]).squeeze()

    # Load normalized calibration
    with open(Path(base_dir) / 'calibration_results_normalized.json') as f:
        cal_norm = json.load(f)
    D_norm = np.array([cal_norm[str(s)]['D'] for s in range(1, 13)])
    e_norm = np.array([cal_norm[str(s)]['e'] for s in range(1, 13)]).squeeze()

    # Plot
    fig, axes = plt.subplots(3, 5, figsize=(7.8, 12 / 2.54),
                             gridspec_kw={'top': 0.92, 'bottom': 0.1,
                                          'left': 0.12, 'right': 0.98})
    row_labels = ['Channel X', 'Channel Y', 'Channel Z']
    col_labels = ['5', '10', '15', '20', '25']

    stats_rows = []

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
            b_raw_rs = b_raw.reshape(-1, 12, 3)

            # b_ref and normalized version
            b_ref = est.estimate_batch(b_raw)
            mag = np.linalg.norm(b_ref, axis=1)
            mean_mag = mag.mean()
            b_ref_norm = b_ref * (mean_mag / mag[:, None])

            # Three Delta_o curves
            d_pre  = compute_delta_o(b_raw_rs, est, None, None, b_ref, apply_cal=False)
            d_orig = compute_delta_o(b_raw_rs, est, D_orig, e_orig, b_ref, apply_cal=True)
            d_norm = compute_delta_o(b_raw_rs, est, D_norm, e_norm, b_ref, apply_cal=True)

            x = np.linspace(0, 2 * np.pi, N)

            ax.fill_between(x, -d_pre, d_pre, alpha=0.3, color='gray')
            ax.plot(x,  d_pre, color='gray', linewidth=0.5)
            ax.plot(x, -d_pre, color='gray', linewidth=0.5)

            ax.fill_between(x, -d_orig, d_orig, alpha=0.3, color='red')
            ax.plot(x,  d_orig, color='red', linewidth=0.5)
            ax.plot(x, -d_orig, color='red', linewidth=0.5)

            ax.fill_between(x, -d_norm, d_norm, alpha=0.3, color='blue')
            ax.plot(x,  d_norm, color='blue', linewidth=0.5)
            ax.plot(x, -d_norm, color='blue', linewidth=0.5)

            ax.axhline(0, color='black', linewidth=0.3)
            ax.set_xlim(0, 2 * np.pi)
            ax.set_xticks([0, np.pi, 2 * np.pi])
            ax.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$'])

            if col_idx == 0:
                ax.set_ylabel(r'$\Delta o$ [Gs]')
            if row_idx == 2:
                ax.set_xlabel(r'$\theta$ [rad]')

            stats_rows.append({
                'channel': ch, 'voltage': int(v), 'n_rows': N,
                'mean_mag': mean_mag,
                'pre_mean': np.mean(d_pre),
                'post_orig_mean': np.mean(d_orig),
                'post_norm_mean': np.mean(d_norm),
                'pre_std': np.std(d_pre),
                'post_orig_std': np.std(d_orig),
                'post_norm_std': np.std(d_norm),
            })

        axes[row_idx, 0].set_title(row_labels[row_idx], pad=2, loc='left', fontsize=8)

    for col_idx, label in enumerate(col_labels):
        axes[0, col_idx].annotate(label,
                                   xy=(0.5, 1.18), xycoords='axes fraction',
                                   ha='center', va='bottom', fontsize=8)

    plt.subplots_adjust(hspace=0.5, wspace=0.35)

    # Gray = pre, Red = orig, Blue = norm
    fig.legend(['pre', 'orig', 'norm'],
               loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.98))

    out_path = Path(__file__).parent.parent / 'plots' / 'residual_comparison_normalized.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=600)
    print(f'Saved to {out_path}')

    # Save stats
    stats_df = pd.DataFrame(stats_rows)
    stats_path = Path(__file__).parent.parent / 'data' / 'residual_stats_normalized.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f'Stats saved to {stats_path}')

    print('\nOverall mean Delta_o:')
    print(f"  pre:       {stats_df['pre_mean'].mean():.6f}")
    print(f"  post_orig: {stats_df['post_orig_mean'].mean():.6f}")
    print(f"  post_norm: {stats_df['post_norm_mean'].mean():.6f}")


if __name__ == '__main__':
    main()
