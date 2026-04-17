#!/usr/bin/env python3
"""
consistency_fit.py - Phase 2: 一致性校准算法

功能：
1. 读取 consistency 数据 CSV（已过椭球+旋转校正）
2. 计算 D_i (缩放矩阵) 和 e_i (残余偏置)

使用方法:
    from calibration.lib.consistency_fit import consistency_fit, batch_consistency_fit

参考论文公式:
    b_i^{final} = D_i * b_i^{corr} + e_i           (Phase 2 一致性校正)
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from sensor_array_config.base import get_config, SensorArrayConfig, ConsistencyParamsSet, ConsistencyParams


# ============== 场条件枚举 ==============
class FieldCondition:
    ZERO = 0       # 0场 (background)
    POS_X = 1      # +Bx (CH1 positive)
    NEG_X = 2      # -Bx (CH1 negative)
    POS_Y = 3      # +By (CH2 positive)
    NEG_Y = 4      # -By (CH2 negative)
    POS_Z = 5      # +Bz (CH3 positive)
    NEG_Z = 6      # -Bz (CH3 negative)

    @staticmethod
    def all_conditions():
        return [FieldCondition.POS_X, FieldCondition.NEG_X,
                FieldCondition.POS_Y, FieldCondition.NEG_Y,
                FieldCondition.POS_Z, FieldCondition.NEG_Z]

    @staticmethod
    def to_string(cond):
        return {0: '0', 1: '+x', 2: '-x', 3: '+y', 4: '-y', 5: '+z', 6: '-z'}.get(cond, '?')


# ============== 数据类 ==============
@dataclass
class ConsistencyResult:
    """单颗传感器一致性校准结果"""
    sensor_id: int
    csv_file: str
    D_i: List[List[float]]  # 缩放矩阵 (3×3) - 对角矩阵
    e_i: List[float]        # 残余偏置 (3,)
    d_i: Dict[str, float]   # 对角元素 {'x': d_ix, 'y': d_iy, 'z': d_iz}
    fit_info: Dict

    def to_dict(self) -> dict:
        return asdict(self)


# ============== 数据加载 ==============
def load_csv_data(csv_path: Path, n_sensors: int = 12) -> np.ndarray:
    """
    加载 CSV 并提取传感器数据

    Args:
        csv_path: CSV 文件路径
        n_sensors: 传感器数量

    Returns:
        data: (N_samples, n_sensors, 3) 原始数据
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.startswith('sensor_')]
    data = df[cols].values
    n = data.shape[0]
    reshaped = np.zeros((n, n_sensors, 3))
    for i in range(n_sensors):
        reshaped[:, i, :] = data[:, i*3:(i+1)*3]
    return reshaped


def compute_stable_mean(data: np.ndarray, skip: int = 5) -> np.ndarray:
    """
    计算稳定段均值，跳过前后各 skip 个样本

    Args:
        data: (N, 12, 3) 数据
        skip: 跳过的样本数

    Returns:
        mean: (12, 3) 稳定段均值
    """
    n = data.shape[0]
    start = min(skip, n // 10)
    end = max(n - skip, n * 9 // 10)
    return data[start:end].mean(axis=0)


# ============== 核心算法 ==============
def fit_D_and_e(b_norm_dict: Dict[int, np.ndarray], n_sensors: int = 12) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    拟合缩放矩阵 D_i 和残余偏置 e_i

    Phase 2 核心算法：使 n_sensors 颗传感器在各个磁场方向上响应一致。

    原理：
    - 在某一磁场方向上，阵列平均增量为 delta_bar
    - 各传感器增量 delta[sid] 应与 delta_bar 成比例
    - 比例系数即为 D_i 的元素（D_i 为完整 3x3 矩阵，可校正交叉轴耦合）

    Args:
        b_norm_dict: {condition: (n_sensors, 3)} 椭球校正后的数据
        n_sensors: 传感器数量

    Returns:
        D_list: [D_1, ..., D_n], 每个 D_i 为 3x3 矩阵
        e_list: [e_1, ..., e_n], 每个 e_i ∈ ℝ³
    """
    conditions = FieldCondition.all_conditions()

    # Step 1: 计算每颗传感器在每个条件下的增量
    delta = {}
    for sid in range(1, n_sensors + 1):
        delta[sid] = {}
        b_zero = b_norm_dict[FieldCondition.ZERO][sid - 1]
        for cond in conditions:
            delta[sid][cond] = b_norm_dict[cond][sid - 1] - b_zero

    # Step 2: 计算阵列平均增量
    delta_bar = {}
    for cond in conditions:
        delta_bar[cond] = np.mean([delta[sid][cond] for sid in range(1, n_sensors + 1)], axis=0)

    # 打印 6 个场条件的平均增量
    print("\n  [DEBUG] 阵列平均增量 delta_bar:")
    print(f"  {'Condition':<10} {'X':>12} {'Y':>12} {'Z':>12}")
    print(f"  {'-'*46}")
    for cond in conditions:
        cond_str = FieldCondition.to_string(cond)
        print(f"  {cond_str:<10} {delta_bar[cond][0]:>12.6f} {delta_bar[cond][1]:>12.6f} {delta_bar[cond][2]:>12.6f}")

    # Step 3: 拟合 D_i（完整 3x3 矩阵，使用最小二乘）
    # 对于每个传感器，解 D_i @ delta_c = delta_bar[c] for all c
    # 这是一个线性最小二乘问题: D_i @ M = B, 其中 M 和 B 都是 3x6 矩阵
    D_list = []
    for sid in range(1, n_sensors + 1):
        # 构建矩阵 M = [delta[sid][c1], delta[sid][c2], ..., delta[sid][c6]] (3x6)
        # 构建矩阵 B = [delta_bar[c1], delta_bar[c2], ..., delta_bar[c6]] (3x6)
        M = np.stack([delta[sid][c] for c in conditions], axis=1)  # 3x6
        B = np.stack([delta_bar[c] for c in conditions], axis=1)   # 3x6

        # 解 D_i @ M = B，使用最小二乘
        # D_i = B @ M^T @ (M @ M^T)^-1 或使用 lstsq
        D_i, residuals, rank, s = np.linalg.lstsq(M.T, B.T, rcond=None)  # M.T: 6x3, B.T: 6x3, result: 3x3
        D_i = D_i.T  # 转置得到 3x3
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


def apply_consistency_correction(b_ellipsoid: np.ndarray, D_i: np.ndarray, e_i: np.ndarray) -> np.ndarray:
    """
    应用一致性校正公式

    Formula: b_final = D_i * b_ellipsoid + e_i

    Args:
        b_ellipsoid: (N, 3) 或 (3,) 椭球校正后数据
        D_i: (3, 3) 缩放矩阵（对角矩阵）
        e_i: (3,) 残余偏置

    Returns:
        b_final: (N, 3) 或 (3,) 校正后数据
    """
    b_ellipsoid = np.asarray(b_ellipsoid)
    D_i = np.asarray(D_i)
    e_i = np.asarray(e_i)

    return (b_ellipsoid @ D_i.T) + e_i


# ============== 主 API ==============
def consistency_fit(
    csv_dir: Path,
    data_files: Dict[str, str] = None,
    sensor_config: SensorArrayConfig = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
    """
    执行一致性校准（单次调用完成全部计算）

    注意: 输入数据应为已经过椭球校正和旋转校正的数据。
    不再自动加载椭球参数，需在数据采集阶段确保使用校正后的数据。

    Args:
        csv_dir: CSV 文件所在目录
        data_files: 可选，自定义数据文件映射
        sensor_config: 可选，传感器配置对象

    Returns:
        (D_list, e_list, fit_info)
    """
    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    n_sensors = sensor_config.manifest.n_sensors

    if data_files is None:
        data_files = {
            'background': 'consistency_calib_background.csv',
            'ch1_positive': 'consistency_calib_ch1_positive.csv',
            'ch1_negative': 'consistency_calib_ch1_negative.csv',
            'ch2_positive': 'consistency_calib_ch2_positive.csv',
            'ch2_negative': 'consistency_calib_ch2_negative.csv',
            'ch3_positive': 'consistency_calib_ch3_positive.csv',
            'ch3_negative': 'consistency_calib_ch3_negative.csv',
        }

    condition_map = {
        'background': FieldCondition.ZERO,
        'ch1_positive': FieldCondition.POS_Z,
        'ch1_negative': FieldCondition.NEG_Z,
        'ch2_positive': FieldCondition.POS_Y,
        'ch2_negative': FieldCondition.NEG_Y,
        'ch3_positive': FieldCondition.POS_X,
        'ch3_negative': FieldCondition.NEG_X,
    }

    # 加载并处理数据（数据已经是 ellipsoid + rotation 校正后的）
    b_norm_means = {}
    for name, filename in data_files.items():
        csv_path = Path(csv_dir) / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        sensors_raw = load_csv_data(csv_path, n_sensors)
        # 数据已经是 ellipsoid + rotation 校正后的，直接使用
        b_norm_means[condition_map[name]] = compute_stable_mean(sensors_raw)

    # 拟合 D_i 和 e_i
    D_list, e_list = fit_D_and_e(b_norm_means, n_sensors)

    fit_info = {
        'method': 'relative_consistency_fit',
        'n_sensors': n_sensors,
        'conditions': ['ZERO', 'POS_X', 'NEG_X', 'POS_Y', 'NEG_Y', 'POS_Z', 'NEG_Z'],
    }

    return D_list, e_list, fit_info


def batch_consistency_fit(
    csv_dir: Path,
    output_path: Path = None,
    sensor_config: SensorArrayConfig = None,
) -> List[ConsistencyResult]:
    """
    批量处理一致性校准，返回完整结果

    Args:
        csv_dir: CSV 文件所在目录
        output_path: 可选，输出 JSON 路径
        sensor_config: 可选，传感器配置对象

    Returns:
        List of ConsistencyResult for all n_sensors sensors
    """
    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    n_sensors = sensor_config.manifest.n_sensors

    D_list, e_list, fit_info = consistency_fit(csv_dir, data_files=None, sensor_config=sensor_config)

    results = []
    for i in range(n_sensors):
        result = ConsistencyResult(
            sensor_id=i + 1,
            csv_file=str(csv_dir),
            D_i=D_list[i].tolist(),
            e_i=e_list[i].tolist(),
            d_i={'x': float(D_list[i][0,0]), 'y': float(D_list[i][1,1]), 'z': float(D_list[i][2,2])},
            fit_info=fit_info
        )
        results.append(result)

    # 保存参数
    if output_path is not None:
        save_consistency_params(results, output_path, sensor_config)

    return results


def validate_consistency(
    csv_dir: Path,
    D_list: List[np.ndarray],
    e_list: List[np.ndarray],
    sensor_config: SensorArrayConfig = None,
) -> Dict:
    """
    验证一致性校正效果

    Args:
        csv_dir: CSV 文件所在目录
        D_list: D_i 矩阵列表
        e_list: e_i 向量列表
        sensor_config: 可选，传感器配置对象

    Returns:
        包含校正前后标准差对比的字典
    """
    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    n_sensors = sensor_config.manifest.n_sensors

    data_files = {
        'background': 'consistency_calib_background.csv',
        'ch1_positive': 'consistency_calib_ch1_positive.csv',
        'ch1_negative': 'consistency_calib_ch1_negative.csv',
        'ch2_positive': 'consistency_calib_ch2_positive.csv',
        'ch2_negative': 'consistency_calib_ch2_negative.csv',
        'ch3_positive': 'consistency_calib_ch3_positive.csv',
        'ch3_negative': 'consistency_calib_ch3_negative.csv',
    }

    condition_map = {
        'background': FieldCondition.ZERO,
        'ch1_positive': FieldCondition.POS_Z,
        'ch1_negative': FieldCondition.NEG_Z,
        'ch2_positive': FieldCondition.POS_Y,
        'ch2_negative': FieldCondition.NEG_Y,
        'ch3_positive': FieldCondition.POS_X,
        'ch3_negative': FieldCondition.NEG_X,
    }

    # 数据已经是 ellipsoid + rotation 校正后的，直接使用
    b_norm_means = {}
    for name, filename in data_files.items():
        csv_path = Path(csv_dir) / filename
        if not csv_path.exists():
            continue
        sensors_raw = load_csv_data(csv_path, n_sensors)
        b_norm_means[condition_map[name]] = compute_stable_mean(sensors_raw)

    conditions = [FieldCondition.ZERO] + FieldCondition.all_conditions()
    validation = {'conditions': [], 'axes': [], 'before': [], 'after': [], 'improvement_pct': []}

    for cond in conditions:
        before = b_norm_means[cond]
        after = np.stack([apply_consistency_correction(before[s-1], D_list[s-1], e_list[s-1])
                         for s in range(1, n_sensors + 1)])

        for axis, axis_name in enumerate(['X', 'Y', 'Z']):
            std_b = float(np.std(before[:, axis]))
            std_a = float(np.std(after[:, axis]))
            imp = float((std_b - std_a) / (std_b + 1e-10) * 100)
            validation['conditions'].append(FieldCondition.to_string(cond))
            validation['axes'].append(axis_name)
            validation['before'].append(std_b)
            validation['after'].append(std_a)
            validation['improvement_pct'].append(imp)

    return validation


def save_consistency_params(
    results: List[ConsistencyResult],
    output_path: Path,
    sensor_config: SensorArrayConfig = None,
):
    """保存一致性校准参数到 JSON 文件"""
    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    params_set = ConsistencyParamsSet(params={
        r.sensor_id: ConsistencyParams(D_i=r.D_i, e_i=r.e_i)
        for r in results
    })
    params_set.to_json(str(output_path))
    print(f"Consistency params saved to: {output_path}")


def load_consistency_params(
    params_path: Path,
    sensor_config: SensorArrayConfig = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    从 JSON 文件加载一致性校准参数

    Args:
        params_path: 参数文件路径
        sensor_config: 可选，传感器配置对象

    Returns:
        (D_dict, e_dict): sensor_id -> 矩阵/向量
    """
    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    params_set = ConsistencyParamsSet.from_json(str(params_path))
    D_dict = {sid: np.array(p.D_i) for sid, p in params_set.params.items()}
    e_dict = {sid: np.array(p.e_i) for sid, p in params_set.params.items()}
    return D_dict, e_dict


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Phase 2 Consistency Calibration - Fit D_i and e_i',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process default data directory
  python -m calibration.lib.consistency_fit /home/zhang/embedded_array_ws/src/calibration/data/consistency

  # Process with custom output
  python -m calibration.lib.consistency_fit ./data/consistency -o ./consistency_params.json

  # Validate existing params
  python -m calibration.lib.consistency_fit ./data/consistency --validate
        """
    )
    parser.add_argument('csv_dir', nargs='?', default=None,
                        help='Directory containing consistency CSV files')
    parser.add_argument('-o', '--output', dest='output_path',
                        help='Output JSON path (default: serial_processor/config/consistency_params.json)')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation after fitting')
    parser.add_argument('--no-save', action='store_true',
                        help='Only compute, do not save')

    args = parser.parse_args()

    # Default path
    if args.csv_dir is None:
        csv_dir = Path(__file__).parent.parent.parent / 'data' / 'consistency'
    else:
        csv_dir = Path(args.csv_dir)

    # Default output (only used when --no-save is NOT set)
    if args.output_path is None:
        output_path = Path(__file__).parent.parent.parent.parent / 'sensor_array_config' / 'config' / 'QMC6309' / 'consistency_params.json'
    else:
        output_path = Path(args.output_path)

    print("=" * 70)
    print("Phase 2: Consistency Calibration (CLI)")
    print("=" * 70)
    print(f"  CSV dir:   {csv_dir}")
    if args.no_save:
        print(f"  Output:    (disabled by --no-save)")
    else:
        print(f"  Output:    {output_path}")
    print()

    if not csv_dir.exists():
        print(f"[ERROR] CSV directory not found: {csv_dir}")
        sys.exit(1)

    # Determine output path (None if --no-save)
    save_path = None if args.no_save else output_path

    # Run consistency fit
    results = batch_consistency_fit(csv_dir, save_path, sensor_config=get_config("QMC6309"))

    # Print fitted parameters
    print("\n  Fitted parameters:")
    print("  " + "-" * 65)
    print(f"  {'Sensor':<8} {'D_ix':<10} {'D_iy':<10} {'D_iz':<10} {'e_ix':<9} {'e_iy':<9} {'e_iz':<9}")
    print("  " + "-" * 65)
    for i, r in enumerate(results):
        D = np.array(r.D_i)
        e = np.array(r.e_i)
        print(f"  {r.sensor_id:<8} {D[0,0]:<10.4f} {D[1,1]:<10.4f} {D[2,2]:<10.4f} "
              f"{e[0]:<+9.4f} {e[1]:<+9.4f} {e[2]:<+9.4f}")

    # Always run validation (unless --no-save but still show report)
    print("\n" + "=" * 70)
    print("Validation Report")
    print("=" * 70)
    D_list = [np.array(r.D_i) for r in results]
    e_list = [np.array(r.e_i) for r in results]
    validation = validate_consistency(csv_dir, D_list, e_list)

    print(f"\n  Dispersion (std of 12 sensors):")
    print("  " + "-" * 55)
    print(f"  {'Condition':<8} {'Axis':<6} {'校正前':<12} {'校正后':<12} {'改善'}")
    print("  " + "-" * 55)

    improvements = []
    for i in range(len(validation['conditions'])):
        cond = validation['conditions'][i]
        axis = validation['axes'][i]
        std_b = validation['before'][i]
        std_a = validation['after'][i]
        imp = validation['improvement_pct'][i]
        improvements.append(imp)
        print(f"  {cond:<8} {axis:<6} {std_b:<12.6f} {std_a:<12.6f} {imp:>+6.1f}%")

    # Summary statistics (exclude severe degradation outliers for mean calc)
    improvements_filtered = [x for x in improvements if x > -50]
    print("\n  Summary:")
    print("  " + "-" * 40)
    print(f"  {'Mean improvement:':<22} {np.mean(improvements_filtered):>+.1f}%")
    print(f"  {'Median improvement:':<22} {np.median(improvements_filtered):>+.1f}%")
    print(f"  {'Max improvement:':<22} {np.max(improvements_filtered):>+.1f}%")
    print(f"  {'Min improvement:':<22} {np.min(improvements_filtered):>+.1f}%")

    # Per-condition summary
    print("\n  Per-condition mean improvement:")
    print("  " + "-" * 40)
    conditions_list = ['0', '+x', '-x', '+y', '-y', '+z', '-z']
    for cond in conditions_list:
        indices = [i for i, c in enumerate(validation['conditions']) if c == cond]
        if indices:
            mean_imp = np.mean([improvements[i] for i in indices])
            bar = '=' * int(max(0, mean_imp) / 5)
            print(f"  {cond:<8} {mean_imp:>+6.1f}%  {bar}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
