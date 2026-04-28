#!/usr/bin/env python3
"""
remove_outliers.py - 检测并插值替换传感器数据中的孤立离群点

算法：
    1. 对每个 (sensor, axis) 的时间序列
    2. 用 backward-only 滚动窗口计算 median 和 MAD
    3. |val - median| > k * MAD 的点标记为 outlier
    4. 用线性插值替换 outlier

用法：
    python remove_outliers.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json


# ---------- 可调参数 ----------
K = 3.0          # MAD 阈值乘子
WIN = 11         # 滚动窗口大小（仅用历史数据，左侧窗口）
# ---------------------------


def rolling_mad_outliers(s, win, k):
    """
    检测 s 中的离群点，使用左侧-only 滚动 MAD。

    Returns:
        outlier: bool Series，与 s 等长
        rm:      rolling median Series
    """
    rm = s.rolling(win, min_periods=1).median()
    mad = (s - rm).abs().rolling(win, min_periods=1).median() * 1.4826
    outlier = (s - rm).abs() > k * mad
    return outlier, rm


def linear_interp(s, outlier):
    """
    对 outlier 位置用线性插值替换（基于最近左右好点）。
    返回新的 Series。
    """
    s_fixed = s.astype(float).copy()
    n = len(s)
    idxs = np.where(outlier)[0]

    for idx in idxs:
        # 找左侧最近好点
        prev = idx - 1
        while prev >= 0 and outlier.iloc[prev]:
            prev -= 1
        # 找右侧最近好点
        next_g = idx + 1
        while next_g < n and outlier.iloc[next_g]:
            next_g += 1

        if prev >= 0 and next_g < n:
            alpha = (idx - prev) / (next_g - prev)
            s_fixed.iloc[idx] = s.iloc[prev] + alpha * (s.iloc[next_g] - s.iloc[prev])
        elif prev >= 0:
            s_fixed.iloc[idx] = s.iloc[prev]
        elif next_g < n:
            s_fixed.iloc[idx] = s.iloc[next_g]

    return s_fixed


def process_csv(csv_path):
    """
    读取单个 CSV，检测离群点，插值替换，返回离群信息。
    返回 (n_outliers, outlier_dict)
    """
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.startswith('sensor_')]
    raw = df[cols].values.astype(float)
    n_samples, n_sensors = raw.shape[0], 12

    # reshape to (n_samples, n_sensors, 3)
    data = np.zeros((n_samples, n_sensors, 3))
    for i in range(n_sensors):
        data[:, i, :] = raw[:, i*3:(i+1)*3]

    outlier_report = {}   # {f'S{sid}_{ax}': [row1, row2, ...]}
    data_fixed = np.zeros_like(data)

    for sid in range(1, 13):
        si = sid - 1
        for ax, ax_name in enumerate(['X', 'Y', 'Z']):
            s = pd.Series(data[:, si, ax])

            # Pass 1: detect outliers
            outlier, rm = rolling_mad_outliers(s, WIN, K)
            n_out = int(outlier.sum())
            outlier_rows_1b = (np.where(outlier)[0] + 1).tolist()  # 1-based

            if n_out > 0:
                outlier_report[f'S{sid}_{ax_name}'] = outlier_rows_1b

            # Interpolate
            s_fixed = linear_interp(s, outlier)
            data_fixed[:, si, ax] = s_fixed.values

    return data_fixed, outlier_report


def save_clean_csv(data_fixed, original_csv_path, out_dir):
    """将清理后的数据写回 CSV，格式与原始一致。"""
    n_samples, n_sensors = data_fixed.shape[0], 12

    rows = []
    for i in range(n_samples):
        row = []
        for sid in range(1, 13):
            x = data_fixed[i, sid-1, 0]
            y = data_fixed[i, sid-1, 1]
            z = data_fixed[i, sid-1, 2]
            row.extend([f'{x:.6f}', f'{y:.6f}', f'{z:.6f}'])
        rows.append(row)

    header = []
    for sid in range(1, 13):
        header.extend([f'sensor_{sid}_x', f'sensor_{sid}_y', f'sensor_{sid}_z'])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / original_csv_path.name

    import csv as csv_lib
    with open(out_path, 'w', newline='') as f:
        writer = csv_lib.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    return out_path


def main():
    base_dir = Path(__file__).parent.parent.parent / 'sensor_data_collection' / 'data'
    out_base = Path(__file__).parent.parent / 'data' / 'cleaned'
    out_base.mkdir(parents=True, exist_ok=True)

    channels  = ['x', 'y', 'z']
    voltages  = [1, 2, 3, 4, 5]

    all_reports = {}

    for channel in channels:
        for voltage in voltages:
            csv_path = base_dir / f'manual_{channel}' / f'manual_record_{voltage}V.csv'
            if not csv_path.exists():
                continue

            print(f'Processing: {csv_path.name} ...')
            data_fixed, report = process_csv(csv_path)

            # Save cleaned CSV
            out_path = save_clean_csv(data_fixed, csv_path, out_base / f'manual_{channel}')

            # Collect report
            key = f'manual_{channel}_{voltage}V'
            all_reports[key] = {
                'csv': str(csv_path),
                'output': str(out_path),
                'outliers': report,
                'total_outliers': sum(len(v) for v in report.values())
            }

            # Print summary
            total = all_reports[key]['total_outliers']
            if total > 0:
                print(f'  → {total} outliers in {csv_path.name}:')
                for k, rows in report.items():
                    print(f'      {k}: {rows}')
            else:
                print(f'  → no outliers found')

    # Save JSON report
    report_path = out_base / 'outlier_report.json'
    with open(report_path, 'w') as f:
        json.dump(all_reports, f, indent=2)

    print(f'\nReport saved to: {report_path}')
    print(f'Cleaned CSVs saved to: {out_base}')


if __name__ == '__main__':
    main()
