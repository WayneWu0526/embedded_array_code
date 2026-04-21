#!/usr/bin/env python3
"""
Compare consistency calibration results:
1. Using only positive polarity data (POS_X, POS_Y, POS_Z)
2. Using both positive and negative polarity data

Usage:
    python compare_polarity_consistency.py
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

from consistency_fit import (
    FieldCondition, load_csv_data_raw, compute_stable_mean,
    apply_ellipsoid_correction_to_data, apply_r_corr_rotation,
    fit_D_and_e, apply_consistency_correction, validate_consistency,
    batch_consistency_fit, load_csv_data
)
from sensor_array_config.base import get_config


def fit_D_and_e_positive_only(
    b_norm_dict: Dict[int, np.ndarray],
    n_sensors: int = 12,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    拟合缩放矩阵 D_i 和残余偏置 e_i（仅使用 positive 条件）

    仅使用: POS_X, POS_Y, POS_Z (不含 NEG)
    """
    # 仅使用 positive 条件
    positive_conditions = [FieldCondition.POS_X, FieldCondition.POS_Y, FieldCondition.POS_Z]

    # Step 1: 计算每颗传感器在每个条件下的增量
    delta = {}
    for sid in range(1, n_sensors + 1):
        delta[sid] = {}
        b_zero = b_norm_dict[FieldCondition.ZERO][sid - 1]
        for cond in positive_conditions:
            delta[sid][cond] = b_norm_dict[cond][sid - 1] - b_zero

    # Step 2: 计算阵列平均增量
    delta_bar = {}
    for cond in positive_conditions:
        delta_bar[cond] = np.mean([delta[sid][cond] for sid in range(1, n_sensors + 1)], axis=0)

    # 打印 positive 条件的平均增量
    print("\n  [DEBUG] Positive-only 条件下的阵列平均增量 delta_bar:")
    print(f"  {'Condition':<10} {'X':>12} {'Y':>12} {'Z':>12}")
    print(f"  {'-'*46}")
    for cond in positive_conditions:
        cond_str = {FieldCondition.POS_X: '+x', FieldCondition.POS_Y: '+y', FieldCondition.POS_Z: '+z'}[cond]
        print(f"  {cond_str:<10} {delta_bar[cond][0]:>12.6f} {delta_bar[cond][1]:>12.6f} {delta_bar[cond][2]:>12.6f}")

    # Step 3: 拟合 D_i（完整 3x3 矩阵，使用最小二乘）
    D_list = []
    for sid in range(1, n_sensors + 1):
        M = np.stack([delta[sid][c] for c in positive_conditions], axis=1)  # 3x3
        B = np.stack([delta_bar[c] for c in positive_conditions], axis=1)   # 3x3

        D_i, residuals, rank, s = np.linalg.lstsq(M.T, B.T, rcond=None)
        D_i = D_i.T
        D_list.append(D_i)

    # Step 4: 计算0场阵列平均向量
    b_bar_zero = np.mean([b_norm_dict[FieldCondition.ZERO][sid - 1] for sid in range(1, n_sensors + 1)], axis=0)

    # Step 5: 拟合 e_i
    e_list = []
    for sid in range(1, n_sensors + 1):
        b_zero = b_norm_dict[FieldCondition.ZERO][sid - 1]
        e_i = b_bar_zero - D_list[sid - 1] @ b_zero
        e_list.append(e_i)

    return D_list, e_list


def compare_validation(
    csv_dir: Path,
    D_list: List[np.ndarray],
    e_list: List[np.ndarray],
    D_list_pos: List[np.ndarray],
    e_list_pos: List[np.ndarray],
    intrinsic_params,
    r_corr: Dict[int, np.ndarray],
    n_sensors: int = 12,
):
    """
    对比两种方法在校正前、仅positive、bothpolarity 下的标准差
    """
    data_files = {
        'background': 'consistency_calib_background.csv',
        'ch1_positive': 'consistency_calib_ch1_positive.csv',
        'ch1_negative': 'consistency_calib_ch1_negative.csv',
        'ch2_positive': 'consistency_calib_ch2_positive.csv',
        'ch2_negative': 'consistency_calib_ch2_negative.csv',
        'ch3_positive': 'consistency_calib_ch3_positive.csv',
        'ch3_negative': 'consistency_calib_ch3_negative.csv',
    }

    channel_to_axis = {1: 'x', 2: 'y', 3: 'z'}
    axis_to_cond_pos = {'x': FieldCondition.POS_X, 'y': FieldCondition.POS_Y, 'z': FieldCondition.POS_Z}
    axis_to_cond_neg = {'x': FieldCondition.NEG_X, 'y': FieldCondition.NEG_Y, 'z': FieldCondition.NEG_Z}

    condition_map = {'background': FieldCondition.ZERO}
    for ch, axis in channel_to_axis.items():
        condition_map[f'ch{ch}_positive'] = axis_to_cond_pos[axis]
        condition_map[f'ch{ch}_negative'] = axis_to_cond_neg[axis]

    # 加载并处理数据
    b_norm_means = {}
    for name, filename in data_files.items():
        csv_path = Path(csv_dir) / filename
        if not csv_path.exists():
            continue

        sensors_raw = load_csv_data_raw(csv_path, n_sensors)

        # 椭球校正
        sensors_ellipsoid = np.zeros_like(sensors_raw)
        for sid in range(1, n_sensors + 1):
            o_i = np.array(intrinsic_params.params[sid].o_i)
            C_i = np.array(intrinsic_params.params[sid].C_i)
            sensors_ellipsoid[:, sid-1, :] = apply_ellipsoid_correction_to_data(
                sensors_raw[:, sid-1, :], o_i, C_i
            )

        # R_CORR 旋转变换
        sensors_rot = np.zeros_like(sensors_ellipsoid)
        for sid in range(1, n_sensors + 1):
            if sid in r_corr:
                sensors_rot[:, sid-1, :] = apply_r_corr_rotation(
                    sensors_ellipsoid[:, sid-1, :], r_corr[sid]
                )
            else:
                sensors_rot[:, sid-1, :] = sensors_ellipsoid[:, sid-1, :]

        b_norm_means[condition_map[name]] = compute_stable_mean(sensors_rot)

    conditions = [FieldCondition.ZERO] + FieldCondition.all_conditions()

    # 计算校正前后和两种方法的差异
    results = []
    for cond in conditions:
        before = b_norm_means[cond]
        after_both = np.stack([apply_consistency_correction(before[s-1], D_list[s-1], e_list[s-1])
                              for s in range(1, n_sensors + 1)])
        after_pos = np.stack([apply_consistency_correction(before[s-1], D_list_pos[s-1], e_list_pos[s-1])
                              for s in range(1, n_sensors + 1)])

        for axis, axis_name in enumerate(['X', 'Y', 'Z']):
            std_b = float(np.std(before[:, axis]))
            std_a_both = float(np.std(after_both[:, axis]))
            std_a_pos = float(np.std(after_pos[:, axis]))
            imp_both = float((std_b - std_a_both) / (std_b + 1e-10) * 100)
            imp_pos = float((std_b - std_a_pos) / (std_b + 1e-10) * 100)
            results.append({
                'condition': FieldCondition.to_string(cond),
                'axis': axis_name,
                'before': std_b,
                'after_positive_only': std_a_pos,
                'after_both_polarities': std_a_both,
                'imp_pos': imp_pos,
                'imp_both': imp_both,
                'diff': imp_both - imp_pos,  # 正值表示both更好
            })

    return results


def main():
    csv_dir = Path('/home/zhang/embedded_array_ws/src/calibration/data/consistency')
    sensor_config = get_config("QMC6309")
    n_sensors = sensor_config.manifest.n_sensors
    intrinsic_params = sensor_config.intrinsic

    # Build R_CORR dict
    r_corr = {}
    for entry in sensor_config.hardware.R_CORR:
        mat = np.array(entry.matrix).reshape(3, 3, order='F')
        for sid in entry.sensor_ids:
            r_corr[sid] = mat

    # 使用原有的一致性校正（正负数据都用）
    print("=" * 70)
    print("Running consistency calibration with BOTH polarities...")
    print("=" * 70)
    results_both, amp_factor_both = batch_consistency_fit(
        csv_dir=csv_dir,
        output_path=None,  # don't save
        sensor_config=sensor_config,
        auto_detect=True,
        intrinsic_params=intrinsic_params,
        r_corr=r_corr,
    )
    D_list_both = [np.array(r.D_i) for r in results_both]
    e_list_both = [np.array(r.e_i) for r in results_both]

    # 仅使用 positive 数据
    print("\n")
    print("=" * 70)
    print("Running consistency calibration with POSITIVE-ONLY data...")
    print("=" * 70)

    data_files = {
        'background': 'consistency_calib_background.csv',
        'ch1_positive': 'consistency_calib_ch1_positive.csv',
        'ch1_negative': 'consistency_calib_ch1_negative.csv',
        'ch2_positive': 'consistency_calib_ch2_positive.csv',
        'ch2_negative': 'consistency_calib_ch2_negative.csv',
        'ch3_positive': 'consistency_calib_ch3_positive.csv',
        'ch3_negative': 'consistency_calib_ch3_negative.csv',
    }

    channel_to_axis = {1: 'x', 2: 'y', 3: 'z'}
    axis_to_cond_pos = {'x': FieldCondition.POS_X, 'y': FieldCondition.POS_Y, 'z': FieldCondition.POS_Z}
    axis_to_cond_neg = {'x': FieldCondition.NEG_X, 'y': FieldCondition.NEG_Y, 'z': FieldCondition.NEG_Z}

    condition_map = {'background': FieldCondition.ZERO}
    for ch, axis in channel_to_axis.items():
        condition_map[f'ch{ch}_positive'] = axis_to_cond_pos[axis]
        condition_map[f'ch{ch}_negative'] = axis_to_cond_neg[axis]

    b_norm_means = {}
    for name, filename in data_files.items():
        csv_path = Path(csv_dir) / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        sensors_raw = load_csv_data_raw(csv_path, n_sensors)

        # 椭球校正 + R_CORR
        sensors_ellipsoid = np.zeros_like(sensors_raw)
        for sid in range(1, n_sensors + 1):
            o_i = np.array(intrinsic_params.params[sid].o_i)
            C_i = np.array(intrinsic_params.params[sid].C_i)
            sensors_ellipsoid[:, sid-1, :] = apply_ellipsoid_correction_to_data(
                sensors_raw[:, sid-1, :], o_i, C_i
            )

        sensors_rot = np.zeros_like(sensors_ellipsoid)
        for sid in range(1, n_sensors + 1):
            if sid in r_corr:
                sensors_rot[:, sid-1, :] = apply_r_corr_rotation(
                    sensors_ellipsoid[:, sid-1, :], r_corr[sid]
                )
            else:
                sensors_rot[:, sid-1, :] = sensors_ellipsoid[:, sid-1, :]

        b_norm_means[condition_map[name]] = compute_stable_mean(sensors_rot)

    D_list_pos, e_list_pos = fit_D_and_e_positive_only(b_norm_means, n_sensors)

    # 打印 D_i, e_i 对比
    print("\n")
    print("=" * 70)
    print("D_i, e_i 参数对比 (每行一个传感器)")
    print("=" * 70)
    print(f"  {'Sensor':<8} {'Method':<12} {'D_ix':<10} {'D_iy':<10} {'D_iz':<10} {'e_ix':<9} {'e_iy':<9} {'e_iz':<9}")
    print("  " + "-" * 80)
    for i in range(n_sensors):
        sid = i + 1
        D_b = np.array(D_list_both[i])
        e_b = np.array(e_list_both[i])
        D_p = np.array(D_list_pos[i])
        e_p = np.array(e_list_pos[i])
        print(f"  {sid:<8} {'BOTH':<12} {D_b[0,0]:<10.4f} {D_b[1,1]:<10.4f} {D_b[2,2]:<10.4f} "
              f"{e_b[0]:<+9.4f} {e_b[1]:<+9.4f} {e_b[2]:<+9.4f}")
        print(f"  {sid:<8} {'POS_ONLY':<12} {D_p[0,0]:<10.4f} {D_p[1,1]:<10.4f} {D_p[2,2]:<10.4f} "
              f"{e_p[0]:<+9.4f} {e_p[1]:<+9.4f} {e_p[2]:<+9.4f}")
        print()

    # 打印 D_i 矩阵完整对比
    print("\n")
    print("=" * 70)
    print("D_i 矩阵完整对比 (3x3)")
    print("=" * 70)
    for i in range(n_sensors):
        sid = i + 1
        D_b = np.array(D_list_both[i])
        D_p = np.array(D_list_pos[i])
        diff = D_b - D_p
        print(f"\n  Sensor {sid}:")
        print(f"    BOTH:     {D_b[0,:].round(4)}")
        print(f"             {D_b[1,:].round(4)}")
        print(f"             {D_b[2,:].round(4)}")
        print(f"    POS_ONLY: {D_p[0,:].round(4)}")
        print(f"             {D_p[1,:].round(4)}")
        print(f"             {D_p[2,:].round(4)}")
        print(f"    DIFF:     {diff[0,:].round(4)}")
        print(f"             {diff[1,:].round(4)}")
        print(f"             {diff[2,:].round(4)}")

    # 验证对比
    print("\n\n")
    print("=" * 70)
    print("验证对比: 校正前后标准差")
    print("=" * 70)
    validation_results = compare_validation(
        csv_dir, D_list_both, e_list_both, D_list_pos, e_list_pos,
        intrinsic_params, r_corr, n_sensors
    )

    print(f"\n  {'Condition':<8} {'Axis':<6} {'校正前':<12} {'PosOnly':<12} {'BothPol':<12} {'Imp Pos':<9} {'Imp Both':<10} {'Diff':<8}")
    print("  " + "-" * 90)

    improvements_pos = []
    improvements_both = []
    for r in validation_results:
        print(f"  {r['condition']:<8} {r['axis']:<6} {r['before']:<12.6f} {r['after_positive_only']:<12.6f} "
              f"{r['after_both_polarities']:<12.6f} {r['imp_pos']:>+8.1f}% {r['imp_both']:>+9.1f}% {r['diff']:>+7.1f}%")
        improvements_pos.append(r['imp_pos'])
        improvements_both.append(r['imp_both'])

    # 过滤极端异常值后计算均值
    improvements_pos_f = [x for x in improvements_pos if x > -50]
    improvements_both_f = [x for x in improvements_both if x > -50]

    print("\n  " + "-" * 90)
    print(f"\n  统计汇总 (排除 improvement < -50% 的异常值):")
    print(f"    {'Method':<20} {'Mean':<10} {'Median':<10} {'Max':<10} {'Min':<10}")
    print(f"    {'-'*60}")
    print(f"    {'Positive-Only':<20} {np.mean(improvements_pos_f):>+8.1f}% {np.median(improvements_pos_f):>+8.1f}% "
          f"{np.max(improvements_pos_f):>+8.1f}% {np.min(improvements_pos_f):>+8.1f}%")
    print(f"    {'Both Polarities':<20} {np.mean(improvements_both_f):>+8.1f}% {np.median(improvements_both_f):>+8.1f}% "
          f"{np.max(improvements_both_f):>+8.1f}% {np.min(improvements_both_f):>+8.1f}%")

    # 逐条件统计
    print(f"\n  按条件统计:")
    print(f"    {'Condition':<8} {'Mean Imp Pos':<14} {'Mean Imp Both':<14} {'Both - Pos':<10}")
    print(f"    {'-'*50}")
    for cond_str in ['0', '+x', '-x', '+y', '-y', '+z', '-z']:
        indices = [i for i, r in enumerate(validation_results) if r['condition'] == cond_str]
        if indices:
            mean_pos = np.mean([improvements_pos[i] for i in indices])
            mean_both = np.mean([improvements_both[i] for i in indices])
            diff = mean_both - mean_pos
            print(f"    {cond_str:<8} {mean_pos:>+12.1f}% {mean_both:>+12.1f}% {diff:>+8.1f}%")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()