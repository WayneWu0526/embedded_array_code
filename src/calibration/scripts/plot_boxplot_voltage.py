#!/usr/bin/env python3
"""
Boxplot of calibration effect: sensor-b_ref error and pseudo-gradient Frobenius norm
across all channels, grouped by voltage (1V to 5V).
Uses a grouped layout and log scale for clear visibility across different energy levels.

Usage:
    python plot_boxplot_voltage.py
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

# Enable LaTeX rendering similar to MATLAB's interpreter latex
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath,amsfonts,amssymb,amsthm,mathrsfs,mathtools}\usepackage{bm}\usepackage{dutchcal}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman"],
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

calib_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(calib_root))

from calibration.lib.center_field_estimator import CenterFieldEstimator
from sensor_array_config.base import get_config

def compute_pseudo_gradient(B_meas, P_mat, C_pinv, S_mat):
    B_tilde = B_meas @ P_mat
    h_vec = B_tilde.reshape(-1, order='F')
    x_hat = C_pinv @ h_vec
    X_hat = (S_mat @ x_hat).reshape((3, 3), order='F')
    return X_hat

def main():
    n_sensors = 12
    calib_json = calib_root / 'sensor_data_collection' / 'data' / 'calibration_results_aggressive.json'
    cleaned_dir = calib_root / 'sensor_data_collection' / 'data' / 'cleaned_aggressive'

    with open(calib_json) as f:
        calib_results = json.load(f)

    est = CenterFieldEstimator()

    # S_mat for symmetric traceless gradient
    S_mat = np.array([
        [1,  0, 0, 0, 0],[0,  1, 0, 0, 0],[0,  0, 1, 0, 0],[0,  1, 0, 0, 0],
        [0,  0, 0, 1, 0],[0,  0, 0, 0, 1],[0,  0, 1, 0, 0],[0,  0, 0, 0, 1],
        [-1, 0, 0, -1, 0],
    ], dtype=float)

    D_cal = est.full_d_list.T
    C_mat = np.kron(D_cal.T, np.eye(3)) @ S_mat
    C_pinv = np.linalg.pinv(C_mat)
    P_mat = np.eye(n_sensors) - (1.0/n_sensors) * np.outer(np.ones(n_sensors), np.ones(n_sensors))

    channels = ['manual_x', 'manual_y', 'manual_z']
    voltages = ['1V', '2V', '3V', '4V', '5V']
    voltages_tex = [r"5", r"10", r"15", r"20", r"25"]

    # Collect data per voltage
    all_err_raw = {v: [] for v in voltages}
    all_err_cal = {v: [] for v in voltages}
    all_fro_raw = {v: [] for v in voltages}
    all_fro_cal = {v: [] for v in voltages}

    for volt in voltages:
        for ch in channels:
            csv_path = cleaned_dir / f"{ch}_{volt}_cleaned.csv"
            if not csv_path.exists(): continue

            df = pd.read_csv(csv_path)
            b_raw = df.values.astype(np.float64)
            b_raw_rs = b_raw.reshape(-1, 12, 3)
            b_ref = est.estimate_batch(b_raw)

            for n in range(b_raw_rs.shape[0]):
                filtered = est._filter_to_selected_sensors(b_raw_rs[n])
                b_raw_rcorr_n = est.apply_r_corr(filtered)
                b_corr_n = np.zeros((12, 3))
                for sid in range(1, 13):
                    D_i = np.array(calib_results[str(sid)]['D'])
                    e_i = np.array(calib_results[str(sid)]['e']).flatten()
                    b_corr_n[sid-1] = D_i @ b_raw_rcorr_n[sid-1] + e_i
                
                for sid in range(n_sensors):
                    all_err_raw[volt].append(np.linalg.norm(b_raw_rcorr_n[sid] - b_ref[n]))
                    all_err_cal[volt].append(np.linalg.norm(b_corr_n[sid] - b_ref[n]))
                
                all_fro_raw[volt].append(np.linalg.norm(compute_pseudo_gradient(b_raw_rcorr_n.T, P_mat, C_pinv, S_mat), 'fro'))
                all_fro_cal[volt].append(np.linalg.norm(compute_pseudo_gradient(b_corr_n.T, P_mat, C_pinv, S_mat), 'fro'))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(19.8 / 2.54, 8 / 2.54))

    def draw_grouped(ax, raw_dict, cal_dict, title, ylabel):
        pos = 1
        h_raw, h_cal = None, None
        for v in voltages:
            b1 = ax.boxplot(raw_dict[v], positions=[pos], widths=0.6, patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor='white', color='black'), medianprops=dict(color='black'))
            b2 = ax.boxplot(cal_dict[v], positions=[pos+0.8], widths=0.6, patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor='#ffcccc', color='red'), medianprops=dict(color='darkred'))
            if h_raw is None: h_raw = b1["boxes"][0]
            if h_cal is None: h_cal = b2["boxes"][0]
            pos += 3
            
        ax.set_xticks(np.arange(1, len(voltages)*3, 3) + 0.4)
        ax.set_xticklabels(voltages_tex)
        # ax.set_title(title)
        ax.set_xlabel(r"$\|\bar{\bm{\mathcal{b}}}\|_2$ [Gs]")
        ax.set_ylabel(ylabel)
        # ax.set_yscale('log')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend([h_raw, h_cal], [r"Raw", r"Cal."], loc='upper left', fontsize=12)

    draw_grouped(axes[0], all_err_raw, all_err_cal, r"Sensor-to-Reference Error", r"$e_{\bm{\mathcal{b}}}$ [Gs]")
    draw_grouped(axes[1], all_fro_raw, all_fro_cal, r"Pseudo-gradient Frobenius Norm", r"$\|\delta{\hat{\bm{X}}}\|_F \text{ [Gs/m]}$")

    plt.tight_layout(pad=0.5)
    out_path = calib_root / 'calibration' / 'plots' / 'boxplot_voltage_combined.pdf'
    plt.savefig(out_path, dpi=300)
    print(f"Final plot saved to {out_path}")

if __name__ == '__main__':
    main()
