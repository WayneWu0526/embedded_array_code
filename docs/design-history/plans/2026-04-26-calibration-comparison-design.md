# Calibration Comparison: R_CORR-only vs Full Calibration

## Goal

实现两个校准路径的逐行传感器一致性指标对比：
- **Path A**: `R_CORR × b_raw`（仅 R_CORR 旋转，不做任何椭球/增益校正）
- **Path B**: `R_CORR × s_i × (C_i × (b_raw - o_i))`（完整校准：椭球 + 增益 + R_CORR）

通过逐行计算的传感器间分量标准差（within-row sensor std）评价两者的优劣。

---

## 指标定义

对形状为 `(N, 12, 3)` 的数据（ N 行，12 传感器，x/y/z 三分量）：

**逐行指标**（对每一行）：
```
std_x(row) = std( all 12 sensors' x component at this row )
std_y(row) = std( all 12 sensors' y component at this row )
std_z(row) = std( all 12 sensors' z component at this row )
```

**跨行聚合**（对所有行/所有 channel/voltage）：
```
grand_mean_std    = mean( all std_x + std_y + std_z across all rows )
grand_max_std     = max( all std_x + std_y + std_z across all rows )
percentile_95_std = np.percentile( all std values, 95 )
per_channel_std   = { ch: mean std across rows for this channel }
per_voltage_std   = { voltage: mean std across rows for this voltage }
per_sensor_std    = { sensor_id: mean deviation of this sensor from row mean }
```

---

## 数据来源

| 数据 | 路径 |
|------|------|
| 手动标定原始数据 | `sensor_data_collection/data/manual_{x,y,z}/manual_record_{1-5}V.csv` |
| magnitude 参考 | `sensor_data_collection/data/magnitude.txt` |
| 椭球参数 (o_i, C_i) | `sensor_array_config/config/qmc6309/intrinsic_params.json` |
| R_CORR 矩阵 | `sensor_array_config/config/qmc6309/sensor_array_params.json` |
| 模值增益 s_i | 由 `compute_sensor_gains()` 已在 session 中计算并保存在 `compute_sensor_gains()` 的返回值中 |

---

## 文件变更

| 文件 | 变更 |
|------|------|
| `src/calibration/lib/consistency_fit.py` | 新增 `compute_rowwise_sensor_consistency_metric()`、`compare_calibration_methods()` |
| `src/calibration/scripts/consistency_calibration.py` | 新增 `--calibration-check` CLI 模式，调用 `compare_calibration_methods()` |

---

## 实现步骤

### Task 1: 实现 `compute_rowwise_sensor_consistency_metric()`

**文件**: `src/calibration/lib/consistency_fit.py`

**函数签名**:
```python
def compute_rowwise_sensor_consistency_metric(
    data: np.ndarray,          # (N, 12, 3)
    sensor_ids: List[int] = None
) -> Dict:
```

**实现逻辑**:
1. 断言 `data.shape[1] == 12`（12 传感器）
2. 对每一行（axis=0），对 x/y/z 三个分量分别计算 12 个传感器的 std
   - `std_x_per_row = np.std(data[:, :, 0], axis=1)` → shape (N,)
   - `std_y_per_row = np.std(data[:, :, 1], axis=1)` → shape (N,)
   - `std_z_per_row = np.std(data[:, :, 2], axis=1)` → shape (N,)
3. 拼接为 `all_stds = np.concatenate([std_x_per_row, std_y_per_row, std_z_per_row])`
4. 计算聚合指标：
   - `grand_mean_std = float(np.mean(all_stds))`
   - `grand_max_std = float(np.max(all_stds))`
   - `percentile_95_std = float(np.percentile(all_stds, 95))`
5. 返回完整字典

**返回字典结构**:
```python
{
    'grand_mean_std': float,
    'grand_max_std': float,
    'percentile_95_std': float,
    'std_x_per_row': np.ndarray (N,),    # 保留给调试用
    'std_y_per_row': np.ndarray (N,),
    'std_z_per_row': np.ndarray (N,),
    'n_rows': int,
}
```

- [ ] **Step 1**: 在 `consistency_fit.py` 末尾（`main()` 之前）添加 `compute_rowwise_sensor_consistency_metric()` 函数

```python
def compute_rowwise_sensor_consistency_metric(
    data: np.ndarray,
    sensor_ids: List[int] = None,
) -> Dict:
    """
    计算逐行传感器间分量标准差（within-row sensor std）

    对数据形状 (N, 12, 3)，对每一行的 12 个传感器计算 x/y/z 三个分量的 std。

    Args:
        data: (N, 12, 3) 数据
        sensor_ids: 可选，传感器 ID 列表（不用于计算，仅保留在返回中）

    Returns:
        包含聚合指标的字典
    """
    data = np.asarray(data)
    assert data.ndim == 3 and data.shape[1] == 12, \
        f"Expected (N, 12, 3), got {data.shape}"
    n_rows = data.shape[0]

    std_x_per_row = np.std(data[:, :, 0], axis=1)
    std_y_per_row = np.std(data[:, :, 1], axis=1)
    std_z_per_row = np.std(data[:, :, 2], axis=1)

    all_stds = np.concatenate([std_x_per_row, std_y_per_row, std_z_per_row])

    return {
        'grand_mean_std': float(np.mean(all_stds)),
        'grand_max_std': float(np.max(all_stds)),
        'percentile_95_std': float(np.percentile(all_stds, 95)),
        'std_x_per_row': std_x_per_row,
        'std_y_per_row': std_y_per_row,
        'std_z_per_row': std_z_per_row,
        'n_rows': n_rows,
    }
```

- [ ] **Step 2**: 运行单元测试验证

```python
# 在 python REPL 中验证
import numpy as np
from calibration.lib.consistency_fit import compute_rowwise_sensor_consistency_metric

# 构造理想情况：所有传感器完全一致
ideal = np.zeros((100, 12, 3))
result = compute_rowwise_sensor_consistency_metric(ideal)
assert result['grand_mean_std'] < 1e-10, f"Expected ~0, got {result['grand_mean_std']}"

# 构造随机情况
random_data = np.random.randn(100, 12, 3)
result = compute_rowwise_sensor_consistency_metric(random_data)
assert result['grand_mean_std'] > 0.0
assert 'n_rows' in result
print("All assertions passed")
```

- [ ] **Step 3**: 提交

```bash
git add src/calibration/lib/consistency_fit.py
git commit -m "feat(consistency): add compute_rowwise_sensor_consistency_metric"
```

---

### Task 2: 实现 `compare_calibration_methods()`

**文件**: `src/calibration/lib/consistency_fit.py`

**函数签名**:
```python
def compare_calibration_methods(
    data_dir: Path,             # .../sensor_data_collection/data
    magnitude_path: Path,       # .../magnitude.txt
    intrinsic_params: IntrinsicParamsSet,
    r_corr_dict: Dict[int, np.ndarray],   # {sensor_id: (3,3)}
    sensor_gains: Dict[int, Dict],         # {sensor_id: {'s_i': float}}
    sensor_config: SensorArrayConfig = None,
    logger=None,
) -> Dict:
```

**实现逻辑**:

**数据加载**：遍历 `manual_x/y/z` 目录，对每个 channel/voltage 组合用 `load_manual_calibration_data()` 加载原始数据，形状为 `(N, 12, 3)`。

**Path A — 仅 R_CORR**：
```python
b_a = np.zeros_like(b_raw)
for sid in 1..12:
    b_a[:, sid-1, :] = apply_r_corr_rotation(b_raw[:, sid-1, :], r_corr_dict[sid])
```
（即 `R_CORR × b_raw`，不对每个传感器单独校正）

**Path B — 完整校准**：
```python
b_b = np.zeros_like(b_raw)
for sid in 1..12:
    b_ellipsoid = apply_ellipsoid_correction_to_data(b_raw[:, sid-1, :], o_i, C_i)
    b_with_gain = sensor_gains[sid]['s_i'] * b_ellipsoid
    b_b[:, sid-1, :] = apply_r_corr_rotation(b_with_gain, r_corr_dict[sid])
```
（即 `R_CORR × s_i × (C_i × (b_raw - o_i))`）

**计算指标**：对 Path A 和 Path B 的结果分别调用 `compute_rowwise_sensor_consistency_metric()`。

**返回结构**:
```python
{
    'path_a': {
        'method': 'R_CORR × b_raw',
        'metric': compute_rowwise_sensor_consistency_metric result dict,
    },
    'path_b': {
        'method': 'R_CORR × s_i × (C_i × (b_raw - o_i))',
        'metric': compute_rowwise_sensor_consistency_metric result dict,
    },
    'per_channel_results': {
        'x': {'path_a': metric_dict, 'path_b': metric_dict},
        'y': {...},
        'z': {...},
    },
    'per_voltage_results': {
        5: {'path_a': metric_dict, 'path_b': metric_dict},
        ...
    },
}
```

- [ ] **Step 1**: 在 `consistency_fit.py` 中添加 `compare_calibration_methods()` 函数

```python
def compare_calibration_methods(
    data_dir: Path,
    magnitude_path: Path,
    intrinsic_params: IntrinsicParamsSet,
    r_corr_dict: Dict[int, np.ndarray],
    sensor_gains: Dict[int, Dict],
    sensor_config: SensorArrayConfig = None,
    logger=None,
) -> Dict:
    """
    对比两个校准路径的逐行传感器一致性指标

    Path A (baseline): R_CORR × b_raw
    Path B (full):     R_CORR × s_i × (C_i × (b_raw - o_i))

    Args:
        data_dir: .../sensor_data_collection/data 目录
        magnitude_path: magnitude.txt 路径
        intrinsic_params: 椭球参数 (o_i, C_i)
        r_corr_dict: {sensor_id: (3,3)} R_CORR 矩阵字典
        sensor_gains: {sensor_id: {'s_i': float}} 模值增益字典
        sensor_config: 传感器配置
        logger: 日志函数

    Returns:
        包含两个路径指标的完整对比结果字典
    """
    if logger is None:
        def logger(*args, **kwargs):
            print(*args, **kwargs)

    if sensor_config is None:
        sensor_config = get_config("QMC6309")
    n_sensors = sensor_config.manifest.n_sensors

    magnitude_data = parse_magnitude_txt(magnitude_path)
    channels = list(magnitude_data.keys())

    # 收集所有数据路径
    all_raw_data = []  # list of (channel, voltage, data) tuples
    for channel in channels:
        for voltage in VOLTAGE_ORDER:
            try:
                raw_data = load_manual_calibration_data(
                    data_dir, channel, voltage, n_sensors
                )
                all_raw_data.append((channel, voltage, raw_data))
            except FileNotFoundError:
                logger(f"[WARN] Missing: manual_{channel}/manual_record_{voltage}V.csv")
                continue

    # ---------- Path A: R_CORR × b_raw ----------
    all_path_a_concat = []
    # ---------- Path B: R_CORR × s_i × (C_i × (b_raw - o_i)) ----------
    all_path_b_concat = []
    # ---------- 按 channel 分组 ----------
    per_channel_a = {ch: [] for ch in channels}
    per_channel_b = {ch: [] for ch in channels}
    # ---------- 按 voltage 分组 ----------
    per_voltage_a = {v: [] for v in VOLTAGE_ORDER}
    per_voltage_b = {v: [] for v in VOLTAGE_ORDER}

    for channel, voltage, b_raw in all_raw_data:
        # Path A
        b_a = np.zeros_like(b_raw)
        for sid in range(1, n_sensors + 1):
            b_a[:, sid-1, :] = apply_r_corr_rotation(b_raw[:, sid-1, :], r_corr_dict[sid])

        # Path B
        b_b = np.zeros_like(b_raw)
        for sid in range(1, n_sensors + 1):
            o_i = np.array(intrinsic_params.params[sid].o_i)
            C_i = np.array(intrinsic_params.params[sid].C_i)
            s_i = sensor_gains[sid]['s_i']
            b_ellipsoid = apply_ellipsoid_correction_to_data(b_raw[:, sid-1, :], o_i, C_i)
            b_with_gain = s_i * b_ellipsoid
            b_b[:, sid-1, :] = apply_r_corr_rotation(b_with_gain, r_corr_dict[sid])

        # 收集用于全局指标
        all_path_a_concat.append(b_a)
        all_path_b_concat.append(b_b)
        per_channel_a[channel].append(b_a)
        per_channel_b[channel].append(b_b)
        per_voltage_a[voltage].append(b_a)
        per_voltage_b[voltage].append(b_b)

    # 合并所有数据
    all_path_a = np.concatenate(all_path_a_concat, axis=0)  # (N_total, 12, 3)
    all_path_b = np.concatenate(all_path_b_concat, axis=0)

    # 计算全局指标
    metric_a = compute_rowwise_sensor_consistency_metric(all_path_a)
    metric_b = compute_rowwise_sensor_consistency_metric(all_path_b)

    # 计算 per-channel 指标
    per_channel_results = {}
    for ch in channels:
        if per_channel_a[ch]:
            ca = np.concatenate(per_channel_a[ch], axis=0)
            cb = np.concatenate(per_channel_b[ch], axis=0)
            per_channel_results[ch] = {
                'path_a': compute_rowwise_sensor_consistency_metric(ca),
                'path_b': compute_rowwise_sensor_consistency_metric(cb),
            }

    # 计算 per-voltage 指标
    per_voltage_results = {}
    for v in VOLTAGE_ORDER:
        if per_voltage_a[v]:
            va = np.concatenate(per_voltage_a[v], axis=0)
            vb = np.concatenate(per_voltage_b[v], axis=0)
            per_voltage_results[v] = {
                'path_a': compute_rowwise_sensor_consistency_metric(va),
                'path_b': compute_rowwise_sensor_consistency_metric(vb),
            }

    result = {
        'path_a': {
            'method': 'R_CORR × b_raw',
            'metric': metric_a,
        },
        'path_b': {
            'method': 'R_CORR × s_i × (C_i × (b_raw - o_i))',
            'metric': metric_b,
        },
        'per_channel_results': per_channel_results,
        'per_voltage_results': per_voltage_results,
        'total_samples': all_path_a.shape[0],
    }

    # 打印摘要
    logger("\n" + "=" * 60)
    logger("Calibration Comparison Summary")
    logger("=" * 60)
    logger(f"  Total samples: {all_path_a.shape[0]}")
    logger("")
    logger(f"  {'Metric':<25} {'Path A (R_CORR)':>18} {'Path B (Full)':>18} {'Improvement':>12}")
    logger(f"  {'-' * 75}")

    for key in ['grand_mean_std', 'grand_max_std', 'percentile_95_std']:
        va = metric_a[key]
        vb = metric_b[key]
        imp = (va - vb) / (va + 1e-10) * 100
        logger(f"  {key:<25} {va:>18.6f} {vb:>18.6f} {imp:>+11.1f}%")

    logger(f"  {'-' * 75}")
    logger("  Per-channel:")
    for ch in channels:
        if ch in per_channel_results:
            ma = per_channel_results[ch]['path_a']['grand_mean_std']
            mb = per_channel_results[ch]['path_b']['grand_mean_std']
            imp = (ma - mb) / (ma + 1e-10) * 100
            logger(f"    {ch.upper()}: mean_std A={ma:.4f}, B={mb:.4f}, imp={imp:+.1f}%")

    logger("  Per-voltage:")
    for v in VOLTAGE_ORDER:
        if v in per_voltage_results:
            ma = per_voltage_results[v]['path_a']['grand_mean_std']
            mb = per_voltage_results[v]['path_b']['grand_mean_std']
            imp = (ma - mb) / (ma + 1e-10) * 100
            logger(f"    {v}V: mean_std A={ma:.4f}, B={mb:.4f}, imp={imp:+.1f}%")

    logger("=" * 60)

    return result
```

- [ ] **Step 2**: 验证函数可导入

```bash
cd /home/zhang/embedded_array_ws
python -c "from calibration.lib.consistency_fit import compare_calibration_methods; print('Import OK')"
```

- [ ] **Step 3**: 提交

```bash
git add src/calibration/lib/consistency_fit.py
git commit -m "feat(consistency): add compare_calibration_methods for path A vs B comparison"
```

---

### Task 3: 添加 `--calibration-check` CLI 模式

**文件**: `src/calibration/scripts/consistency_calibration.py`

在 `main()` 或 CLI argparse 部分添加 `--calibration-check` 选项，调用 `compare_calibration_methods()`。

需要加载：
1. `intrinsic_params` — `SensorArrayConfig.intrinsic`
2. `r_corr_dict` — `SensorArrayHardwareParams` 的 `R_CORR` 字段
3. `sensor_gains` — 上一步 `compute_sensor_gains()` 的返回值

**注意**: `sensor_gains` 需要先运行 `compute_sensor_gains()` 计算得到。本次 CLI 可以接受一个可选参数 `--sensor-gains` 来加载之前保存的 gains JSON，或者在 CLI 内部调用 `compute_sensor_gains()` 并将结果传入。

- [ ] **Step 1**: 在 argparse 中添加 `--calibration-check` 和 `--sensor-gains` 参数

```python
parser.add_argument('--calibration-check', action='store_true',
                    help='Compare R_CORR-only vs full calibration using rowwise sensor std')
parser.add_argument('--sensor-gains-json', type=str, default=None,
                    help='Path to sensor gains JSON (if not provided, compute internally)')
```

- [ ] **Step 2**: 添加 `run_calibration_check()` 函数

```python
def run_calibration_check(args, sensor_config):
    """运行校准路径对比"""
    from calibration.lib.consistency_fit import (
        compare_calibration_methods,
        compute_sensor_gains,
    )
    from sensor_array_config.base import SensorArrayHardwareParams

    data_dir = Path(args.data_dir)
    magnitude_path = data_dir / 'magnitude.txt'
    intrinsic_params = sensor_config.intrinsic

    # 构建 r_corr_dict
    r_corr_dict = {}
    for entry in sensor_config.hardware.R_CORR:
        mat = np.array(entry.matrix).reshape(3, 3, order='F')
        for sid in entry.sensor_ids:
            r_corr_dict[sid] = mat

    # 获取或计算 sensor_gains
    if args.sensor_gains_json:
        import json
        with open(args.sensor_gains_json) as f:
            gains_raw = json.load(f)
        sensor_gains = {
            int(sid): {'s_i': data['s_i']}
            for sid, data in gains_raw.items()
        }
    else:
        gains_result = compute_sensor_gains(
            data_dir=data_dir,
            magnitude_path=magnitude_path,
            sensor_config=sensor_config,
            intrinsic_params=intrinsic_params,
        )
        sensor_gains = {sid: {'s_i': gains_result[sid]['s_i']}
                        for sid in gains_result}

    # 执行对比
    result = compare_calibration_methods(
        data_dir=data_dir,
        magnitude_path=magnitude_path,
        intrinsic_params=intrinsic_params,
        r_corr_dict=r_corr_dict,
        sensor_gains=sensor_gains,
        sensor_config=sensor_config,
    )
    return result
```

- [ ] **Step 3**: 在 `main()` 中添加分支

```python
if args.calibration_check:
    run_calibration_check(args, sensor_config)
    sys.exit(0)
```

- [ ] **Step 4**: 验证 CLI 可运行

```bash
cd /home/zhang/embedded_array_ws
python -m calibration.scripts.consistency_calibration --calibration-check \
    --data-dir /home/zhang/embedded_array_ws/src/sensor_data_collection/data \
    2>&1 | head -60
```

预期：输出 Path A vs Path B 的 grand_mean_std 对比表格。

- [ ] **Step 5**: 提交

```bash
git add src/calibration/scripts/consistency_calibration.py
git commit -m "feat(consistency): add --calibration-check CLI for path A vs B comparison"
```

---

## 验证标准

1. `compute_rowwise_sensor_consistency_metric` 对全零输入返回 `grand_mean_std ≈ 0`
2. `--calibration-check` 能正常执行，输出包含 Path A 和 Path B 的 `grand_mean_std` 对比
3. Path B 的 `grand_mean_std` 应显著小于 Path A（预期改善 > 50%）
4. 代码无语法错误，函数签名与上述描述一致
