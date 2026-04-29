#!/usr/bin/env python3
"""
Plot calibration effect: sensor-b_ref error and pseudo-gradient Frobenius norm
for any channel/voltage combination.

Usage:
    python plot_stem_combined.py --channel manual_x --voltage 4V
    python plot_stem_combined.py -ch manual_z -v 1V
    python plot_stem_combined.py  # defaults: manual_x, 4V
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

calib_root = Path(__file__).parent.parent.parent  # .../src/calibration/scripts/ -> .../src/
sys.path.insert(0, str(calib_root))

from calibration.lib.center_field_estimator import CenterFieldEstimator
from sensor_array_config.base import get_config


def compute_pseudo_gradient(B_meas, P_mat, C_pinv, S_mat):
    """B_meas: (3, N), returns X_hat (3,3) symmetric traceless"""
    B_tilde = B_meas @ P_mat
    h_vec = B_tilde.reshape(-1, order='F')
    x_hat = C_pinv @ h_vec
    X_hat = (S_mat @ x_hat).reshape((3, 3), order='F')
    return X_hat


def main():
    parser = argparse.ArgumentParser(description='Plot calibration effect for any channel/voltage')
    parser.add_argument('--channel', '-ch', default='manual_x',
                        help='Channel name, e.g. manual_x, manual_y, manual_z (default: manual_x)')
    parser.add_argument('--voltage', '-v', default='4V',
                        help='Voltage, e.g. 1V, 2V, 3V, 4V, 5V (default: 4V)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output PNG path (default: stem_{channel}_{voltage}_combined.png)')
    args = parser.parse_args()

    channel = args.channel
    voltage = args.voltage

    sensor_config = get_config("QMC6309")
    n_sensors = 12

    # Paths: .../src/calibration/scripts/ -> .../src/
    calib_root = Path(__file__).parent.parent.parent
    calib_json = calib_root / 'sensor_data_collection' / 'data' / 'calibration_results_aggressive.json'
    cleaned_dir = calib_root / 'sensor_data_collection' / 'data' / 'cleaned_aggressive'

    with open(calib_json) as f:
        calib_results = json.load(f)

    est = CenterFieldEstimator()

    csv_path = cleaned_dir / f'{channel}_{voltage}_cleaned.csv'
    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    b_raw = df.values.astype(np.float64)
    N = b_raw.shape[0]
    b_raw_rs = b_raw.reshape(-1, 12, 3)
    print(f"Loaded {channel}/{voltage}: {N} rows, {n_sensors} sensors")

    # b_ref
    b_ref = est.estimate_batch(b_raw)

    # Calibrated: R_CORR + D_i/e_i
    b_corr = np.zeros_like(b_raw_rs)
    for n in range(N):
        filtered = est._filter_to_selected_sensors(b_raw_rs[n])
        b_rcorr = est.apply_r_corr(filtered)
        for sid in range(1, 13):
            D_i = np.array(calib_results[str(sid)]['D'])
            e_i = np.array(calib_results[str(sid)]['e']).flatten()
            b_corr[n, sid-1] = D_i @ b_rcorr[sid-1] + e_i

    # Raw+R_CORR (fair comparison)
    b_raw_rcorr = np.zeros_like(b_raw_rs)
    for n in range(N):
        filtered = est._filter_to_selected_sensors(b_raw_rs[n])
        b_raw_rcorr[n] = est.apply_r_corr(filtered)

    # ---- Per-row sensor-b_ref mean norm error ----
    errors_raw = np.zeros((N, n_sensors))
    errors_cal = np.zeros((N, n_sensors))
    for n in range(N):
        for sid in range(n_sensors):
            errors_raw[n, sid] = np.linalg.norm(b_raw_rcorr[n, sid] - b_ref[n])
            errors_cal[n, sid] = np.linalg.norm(b_corr[n, sid] - b_ref[n])

    mean_err_raw = np.mean(errors_raw, axis=1)
    mean_err_cal = np.mean(errors_cal, axis=1)

    # ---- Pseudo-gradient ----
    S_mat = np.array([
        [1,  0, 0, 0, 0],
        [0,  1, 0, 0, 0],
        [0,  0, 1, 0, 0],
        [0,  1, 0, 0, 0],
        [0,  0, 0, 1, 0],
        [0,  0, 0, 0, 1],
        [0,  0, 1, 0, 0],
        [0,  0, 0, 0, 1],
        [-1, 0, 0, -1, 0],
    ], dtype=float)

    D_cal = est.full_d_list.T  # (3, 12)
    C_mat = np.kron(D_cal.T, np.eye(3)) @ S_mat
    C_pinv = np.linalg.pinv(C_mat)

    N_sel = n_sensors
    ones_N = np.ones(N_sel)
    P_mat = np.eye(N_sel) - (1.0/N_sel) * np.outer(ones_N, ones_N)

    fro_raw = np.zeros(N)
    fro_cal = np.zeros(N)
    for n in range(N):
        X_raw = compute_pseudo_gradient(b_raw_rcorr[n].T, P_mat, C_pinv, S_mat)
        X_cal = compute_pseudo_gradient(b_corr[n].T, P_mat, C_pinv, S_mat)
        fro_raw[n] = np.linalg.norm(X_raw, 'fro')
        fro_cal[n] = np.linalg.norm(X_cal, 'fro')

    # ---- Print summary ----
    print(f"\nSensor-b_ref error:  Raw={np.mean(mean_err_raw):.4f}, Cal={np.mean(mean_err_cal):.4f}")
    print(f"Pseudo-gradient ||X||_F:  Raw={np.mean(fro_raw):.4f}, Cal={np.mean(fro_cal):.4f}")
    print(f"  Error reduction: {(np.mean(mean_err_raw)-np.mean(mean_err_cal))/np.mean(mean_err_raw)*100:.1f}%")
    print(f"  Gradient reduction: {(np.mean(fro_raw)-np.mean(fro_cal))/np.mean(fro_raw)*100:.1f}%")

    # ---- Subsample 25% ----
    indices = np.arange(0, N, 4)
    theta = indices / (N - 1) * 2 * np.pi

    # ---- Plot ----
    ch_label = channel.replace('manual_', '').upper()
    plt.rcParams['font.size'] = 8
    fig, axes = plt.subplots(1, 2, figsize=(8.9*2/2.54, 3*2/2.54))

    # Left: sensor-b_ref error
    ax = axes[0]
    (ml_raw, sl_raw, bl_raw), (ml_cal, sl_cal, bl_cal) = \
        ax.stem(theta, mean_err_raw[indices], linefmt='k-', markerfmt='ko', basefmt='k-',
                label='Raw data', use_line_collection=True), \
        ax.stem(theta, mean_err_cal[indices], linefmt='r-', markerfmt='ro', basefmt='r-',
                label='Cal. data', use_line_collection=True)
    plt.setp(ml_raw, markersize=3)
    plt.setp(ml_cal, markersize=3)
    plt.setp(sl_raw, linewidth=0.8, alpha=0.7)
    plt.setp(sl_cal, linewidth=0.8, alpha=0.7)
    plt.rcParams['text.usetex'] = True
    ax.set_xlabel(r'$\theta$ [rad]', fontsize=8)
    ax.set_ylabel(r'b [Gs]', fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

    # Right: pseudo-gradient
    ax = axes[1]
    (ml_raw, sl_raw, bl_raw), (ml_cal, sl_cal, bl_cal) = \
        ax.stem(theta, fro_raw[indices], linefmt='k-', markerfmt='ko', basefmt='k-',
                label='Raw data', use_line_collection=True), \
        ax.stem(theta, fro_cal[indices], linefmt='r-', markerfmt='ro', basefmt='r-',
                label='Cal. data', use_line_collection=True)
    plt.setp(ml_raw, markersize=3)
    plt.setp(ml_cal, markersize=3)
    plt.setp(sl_raw, linewidth=0.8, alpha=0.7)
    plt.setp(sl_cal, linewidth=0.8, alpha=0.7)
    plt.rcParams['text.usetex'] = True
    ax.set_xlabel(r'$\theta$ [rad]', fontsize=8)
    ax.set_ylabel(r'$\|\delta X\|_F$ [Gs/m]', fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

    plt.tight_layout(pad=0.3)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = calib_root / 'calibration' / 'plots' / f'stem_{channel}_{voltage}_combined.png'

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
