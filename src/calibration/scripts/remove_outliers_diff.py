#!/usr/bin/env python3
"""
remove_outliers_diff.py - 基于行间差分的离群点检测与线性插值替换

算法：
    1. 对每个 (sensor, axis) 计算 row-to-row diff: d[i] = x[i+1] - x[i]
    2. 基于 diff 序列计算 threshold = median(|d|) + k * MAD(|d|)
    3. 标记 |d[i]| > threshold 的位置，确定异常行（取 |d| 较大者）
    4. 用最近左右好点线性插值替换异常值

用法：
    python remove_outliers_diff.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import csv as csv_lib


# ---------- 可调参数 ----------
K = 3.0    # diff 阈值乘子：threshold = median(|d|) + k * MAD(|d|)
# -----------------------------


def detect_and_fix_diff(series):
    """
    给定 1D array，检测行间 diff 异常点并线性插值替换。
    返回 (fixed, outlier_rows_1b)
    """
    x = series.astype(float)
    n = len(x)
    diffs = np.diff(x)                          # n-1 elements

    # Threshold on |diffs|
    ad = np.abs(diffs)
    median_ad = np.median(ad)
    mad = 1.4826 * np.median(ad)
    threshold = median_ad + K * mad

    # Find outlier diffs
    outlier_mask = ad > threshold               # (n-1,)
    outlier_idx = np.where(outlier_mask)[0]     # 0-based index into diffs

    # Convert diff index to row index (1-based)
    # For outlier diff d[i]: the spike is the point that is farther from its local context
    outlier_rows_1b = []
    for i in outlier_idx:
        # Context for x[i]   = [x[i-1], x[i+1]]
        # Context for x[i+1] = [x[i],   x[i+2]]
        ctx_i   = np.median([x[i-1], x[i+1]]) if (i > 0 and i+1 < n) else x[i]
        ctx_ip1 = np.median([x[i],   x[i+2]]) if (i+2 < n) else x[i+1]

        dev_i   = abs(x[i]   - ctx_i)
        dev_ip1 = abs(x[i+1] - ctx_ip1)

        if dev_i >= dev_ip1:
            outlier_rows_1b.append(i + 1)   # x[i+1] (1-based row i+1) has larger deviation → is the spike
        else:
            outlier_rows_1b.append(i + 2)   # x[i+2] (1-based row i+2) is the spike

    outlier_rows_1b = sorted(set(outlier_rows_1b))  # deduplicate, already 1-based

    # Linear interpolation replacement — batch mode for consecutive outliers
    x_fixed = x.copy()
    outlier_set = set(outlier_rows_1b)  # 1-based

    # Group consecutive outlier rows into regions
    i = 0
    while i < len(outlier_rows_1b):
        r_start = outlier_rows_1b[i]       # 1-based start of consecutive region
        r_end = r_start
        j = i + 1
        while j < len(outlier_rows_1b) and outlier_rows_1b[j] == r_end + 1:
            r_end = outlier_rows_1b[j]
            j += 1
        # region: [r_start, r_end] inclusive, 1-based
        # find nearest good point before and after the entire region
        prev_good_r = r_start - 2     # 0-based of the last known good before region
        while prev_good_r >= 0 and (prev_good_r + 1) in outlier_set:
            prev_good_r -= 1
        next_good_r = r_end           # 0-based of first known good after region
        while next_good_r < n and (next_good_r + 1) in outlier_set:
            next_good_r += 1

        for r in range(r_start, r_end + 1):
            idx = r - 1
            if prev_good_r >= 0 and next_good_r < n:
                alpha = (idx - prev_good_r) / (next_good_r - prev_good_r)
                x_fixed[idx] = x[prev_good_r] + alpha * (x[next_good_r] - x[prev_good_r])
            elif prev_good_r >= 0:
                x_fixed[idx] = x[prev_good_r]
            elif next_good_r < n:
                x_fixed[idx] = x[next_good_r]

        i = j

    return x_fixed, outlier_rows_1b


def process_csv(csv_path):
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.startswith('sensor_')]
    raw = df[cols].values.astype(float)
    n_samples, n_sensors = raw.shape[0], 12

    # reshape to (n_samples, n_sensors, 3)
    data = np.zeros((n_samples, n_sensors, 3))
    for i in range(n_sensors):
        data[:, i, :] = raw[:, i*3:(i+1)*3]

    outlier_report = {}
    data_fixed = np.zeros_like(data)

    for sid in range(1, 13):
        si = sid - 1
        for ax, ax_name in enumerate(['X', 'Y', 'Z']):
            s = data[:, si, ax]
            fixed, rows = detect_and_fix_diff(s)
            data_fixed[:, si, ax] = fixed
            if rows:
                outlier_report[f'S{sid}_{ax_name}'] = rows

    return data_fixed, outlier_report


def save_clean_csv(data_fixed, original_csv_path, out_dir):
    n_samples, n_sensors = data_fixed.shape[0], 12
    rows = []
    for i in range(n_samples):
        row = []
        for sid in range(1, 13):
            row.extend([
                f'{data_fixed[i, sid-1, 0]:.6f}',
                f'{data_fixed[i, sid-1, 1]:.6f}',
                f'{data_fixed[i, sid-1, 2]:.6f}',
            ])
        rows.append(row)

    header = []
    for sid in range(1, 13):
        header.extend([f'sensor_{sid}_x', f'sensor_{sid}_y', f'sensor_{sid}_z'])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / original_csv_path.name

    with open(out_path, 'w', newline='') as f:
        writer = csv_lib.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return out_path


def main():
    base_dir = Path(__file__).parent.parent.parent / 'sensor_data_collection' / 'data'
    out_base = Path(__file__).parent.parent / 'data' / 'cleaned_diff'
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

            out_path = save_clean_csv(data_fixed, csv_path, out_base / f'manual_{channel}')

            key = f'manual_{channel}_{voltage}V'
            total = sum(len(v) for v in report.values())
            # Convert int64 to native int for JSON serialization
            report_native = {k: [int(r) for r in v] for k, v in report.items()}
            all_reports[key] = {
                'csv': str(csv_path),
                'output': str(out_path),
                'outliers': report_native,
                'total_outliers': int(total),
            }

            if total > 0:
                print(f'  → {total} outliers in {csv_path.name}:')
                for k, rows in report.items():
                    print(f'      {k}: {rows}')
            else:
                print(f'  → no outliers found')

    report_path = out_base / 'outlier_report.json'
    with open(report_path, 'w') as f:
        json.dump(all_reports, f, indent=2)
    print(f'\nReport saved to: {report_path}')
    print(f'Cleaned CSVs saved to: {out_base}')


if __name__ == '__main__':
    main()