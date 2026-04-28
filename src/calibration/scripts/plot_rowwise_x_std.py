#!/usr/bin/env python3
"""
plot_rowwise_x_std.py - 绘制 R_CORR 变换后每行 x/y/z 分量的标准差

功能：
1. 读取 manual_{channel} 目录下所有 voltage (1V~5V) 的原始数据
2. 对每颗传感器应用 R_CORR 旋转变换
3. 计算每一行 12 个传感器 x/y/z 分量的 std
4. 绘制 3x5 布局：3行(X/Y/Z) x 5列(1V~5V)，横轴为行索引，纵轴为 ±std 竖线

使用方法:
    python plot_rowwise_x_std.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# R_CORR matrices from sensor_array_params.json (column-major order)
R_CORR_ENTRIES = [
    {"sensor_ids": [1, 2, 3], "matrix": [1, 0, 0, 0, 0, -1, 0, 1, 0]},
    {"sensor_ids": [4, 5, 6], "matrix": [1, 0, 0, 0, 0, 1, 0, -1, 0]},
    {"sensor_ids": [7, 8, 9], "matrix": [-1, 0, 0, 0, 0, 1, 0, 1, 0]},
    {"sensor_ids": [10, 11, 12], "matrix": [-1, 0, 0, 0, 0, -1, 0, -1, 0]},
]

VOLTAGES = [1, 2, 3, 4, 5]
CHANNEL = 'x'  # 可修改为 'y' 或 'z'


def build_r_corr_dict():
    """Build sensor_id -> R_CORR matrix dictionary."""
    r_corr_dict = {}
    for entry in R_CORR_ENTRIES:
        mat = np.array(entry["matrix"]).reshape(3, 3, order='F')
        for sid in entry["sensor_ids"]:
            r_corr_dict[sid] = mat
    return r_corr_dict


def load_manual_csv(csv_path, n_sensors=12):
    """Load manual calibration CSV and return (N, n_sensors, 3) array."""
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.startswith('sensor_')]
    data = df[cols].values
    n_samples = data.shape[0]
    reshaped = np.zeros((n_samples, n_sensors, 3))
    for i in range(n_sensors):
        reshaped[:, i, :] = data[:, i*3:(i+1)*3]
    return reshaped


def apply_r_corr_rotation(b_corr, r_corr):
    """Apply R_CORR rotation: b_rot = R_CORR @ b_corr.T"""
    return b_corr @ r_corr.T


def main():
    data_dir = Path(__file__).parent.parent.parent / 'sensor_data_collection' / 'data' / f'manual_{CHANNEL}'
    r_corr_dict = build_r_corr_dict()

    # Load all voltages
    all_data = {}
    for voltage in VOLTAGES:
        csv_path = data_dir / f'manual_record_{voltage}V.csv'
        if not csv_path.exists():
            print(f"[WARN] CSV not found: {csv_path}")
            continue
        raw_data = load_manual_csv(csv_path, n_sensors=12)
        # Apply R_CORR
        transformed = np.zeros_like(raw_data)
        for sid in range(1, 13):
            transformed[:, sid-1, :] = apply_r_corr_rotation(
                raw_data[:, sid-1, :], r_corr_dict[sid]
            )
        all_data[voltage] = transformed
        print(f"Loaded {voltage}V: {transformed.shape[0]} samples")

    # Compute std per row for each voltage
    std_data = {}  # voltage -> {axis: std_per_row}
    for voltage, transformed in all_data.items():
        std_data[voltage] = {
            'X': np.std(transformed[:, :, 0], axis=1),
            'Y': np.std(transformed[:, :, 1], axis=1),
            'Z': np.std(transformed[:, :, 2], axis=1),
        }

    # Print statistics
    for voltage in VOLTAGES:
        if voltage not in std_data:
            continue
        print(f"\n=== {voltage}V ===")
        for axis in ['X', 'Y', 'Z']:
            std = std_data[voltage][axis]
            print(f"  {axis}: mean={np.mean(std):.4f}, max={np.max(std):.4f}, min={np.min(std):.4f}")

    # Plot: 3 rows (X/Y/Z) x 5 columns (1V~5V)
    fig, axes = plt.subplots(3, 5, figsize=(25, 12), sharex=True)
    colors = {'X': 'blue', 'Y': 'green', 'Z': 'red'}

    for col, voltage in enumerate(VOLTAGES):
        if voltage not in std_data:
            continue
        for row, axis in enumerate(['X', 'Y', 'Z']):
            ax = axes[row, col]
            std = std_data[voltage][axis]
            n_samples = len(std)
            row_indices = np.arange(n_samples)

            ax.axhline(0, color='black', linewidth=0.5)
            ax.fill_between(row_indices, -std, std, alpha=0.3, color=colors[axis])
            ax.plot(row_indices, std, '-', color=colors[axis], linewidth=0.8)
            ax.plot(row_indices, -std, '-', color=colors[axis], linewidth=0.8)
            ax.set_ylabel(f'{axis}\nStd', fontsize=9)
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(f'{voltage}V', fontsize=12)
            if row == 2:
                ax.set_xlabel('Row Index', fontsize=10)

    # Overall title
    fig.suptitle(f'Row-wise XYZ Std After R_CORR (manual_{CHANNEL}, 1V~5V)', fontsize=14, y=1.01)

    plt.tight_layout()
    out_path = f'/home/zhang/embedded_array_ws/src/calibration/plots/rowwise_xyz_std_manual_{CHANNEL}_all_voltages.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {out_path}")


if __name__ == '__main__':
    main()
