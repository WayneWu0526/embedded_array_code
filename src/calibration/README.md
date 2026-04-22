# calibration

## 概述

传感器阵列校准模块，分为两部分：

- **标定算法**（`lib/ellipsoid_fit/`、`lib/consistency_fit.py`）：根据采集数据拟合校准参数
- **正向校正**（`lib/forward/`）：在运行时应用校准参数，输出校正后数据

两部分通过 `sensor_array_config` 共享参数文件。

## 子模块

### 1. 椭球校准 (Ellipsoid Calibration)

将传感器原始测量从椭球分布"还原"为球形分布，校正 Hard iron offset 和 Soft iron 误差。

**启动:**
```bash
roslaunch calibration ellipsoid_calibration.launch
```

**数据处理:**
```bash
# 批量处理
python -m calibration.lib.ellipsoid_fit <csv_dir>
```

### 2. 一致性校准 (Consistency Calibration)

使 n 颗传感器在各个磁场方向上响应一致。

**启动:**
```bash
roslaunch calibration consistency_calibration.launch
```

**数据处理:**
```bash
# 批量处理
python -m calibration.lib.consistency_fit <csv_dir>
```

#### 校正流程

一致性校准接收 **raw 数据**（而非预处理后的 ellipsoid 数据），在校准节点内部应用椭球校正：

```
stm_uplink_raw (原始数据)
    ↓
CSV 保存
    ↓
consistency_fit: 对 raw 数据应用椭球校正 (o_i, C_i)
    ↓
计算放大系数 (amp_factor = ||b_corr|| / ||b_raw||)
    ↓
拟合一致性参数 (D_i, e_i)
    ↓
保存 consistency_params.json (包含 amp_factor)
```

#### 放大系数 (amp_factor)

椭球校正 (C_i 矩阵) 会对测量结果产生各向异性缩放。为恢复到原始测量量级，一致性校准输出 `amp_factor`：

- 计算方式：`amp_factor = mean(||b_corr||) / mean(||b_raw||)`（背景条件）
- 存储位置：`consistency_params.json["amp_factor"]`
- 用途：供 `serial_node_tdm.py` 在发布最终数据时除以 `amp_factor`，恢复到 raw 水平

#### 输出文件

| 文件 | 内容 |
|------|------|
| `consistency_calib_background.csv` | 背景数据（所有通道 OFF） |
| `consistency_calib_ch{N}_{polarity}.csv` | 各通道正/负极性数据 |
| `consistency_params.json` | D_i, e_i, amp_factor |

### 4. 正向校正 (Forward Correction)

运行时应用校准参数，将 raw 传感器数据转换为校正后数据。采用 Strategy 模式，可按需替换/禁用任意校正步骤。

**参数来源：**
- 椭球参数 → `sensor_array_config.intrinsic`
- 旋转参数 → `sensor_array_config.hardware`
- 一致性参数 → `sensor_array_config.consistency`

**校正流程（默认）：**

```
raw 数据
  │
  ▼
EllipsoidCorrection        公式: (b - o_i) @ C_i.T
  │
  ▼
RCorrRotation             公式: R_CORR @ b
  │
  ▼
ConsistencyCorrection     公式: (D_i @ b + e_i) / amp_factor
  │
  ▼
校正后数据
```

**使用方式：**

```python
from calibration.lib.forward import CalibrationProcessor

# 默认 pipeline（ellipsoid → r_corr → consistency）
calib = CalibrationProcessor(sensor_config)

# 丢弃 consistency（例如特定场景不需要）
from calibration.lib.forward import EllipsoidCorrection, RCorrRotation
steps = [EllipsoidCorrection(cfg), RCorrRotation(cfg)]
calib = CalibrationProcessor(cfg, steps=steps)

# 校正单次采样
cx, cy, cz = calib.apply(sensor_id=1, x=0.1, y=0.2, z=0.3)
```

**在 `serial_node_tdm.py` 中：**

```python
# 原来 ~120 行内联逻辑 → 替换为 3 行
from calibration.lib.forward import CalibrationProcessor

self._calib = CalibrationProcessor(self._sensor_config)

# run() 循环中
cx, cy, cz = self._calib.apply(s.id, s.x, s.y, s.z)
```

**可插拔性：**

| 操作 | 涉及文件 | TDM 代码 |
|------|---------|---------|
| 添加新校正步骤 | 新建 Step 类 + 指定 steps | 不改 |
| 删除旧校正步骤 | 改变 steps 参数 | 不改 |
| 修改某步骤算法 | 只改那个 Step 类 | 不改 |
| 改变执行顺序 | 调整 steps 列表 | 不改 |

---

### 5. 手持校准 (Handheld Calibration)

用于现场快速校准，无需机械臂设备。

**启动:**
```bash
roslaunch calibration handheld_calibration.launch
```

### 6. diana7 安全姿态保存

保存机械臂安全姿态，用于后续校准采样。

```bash
roslaunch calibration save_diana7_home.launch
```

## 文件结构

```
src/calibration/
├── launch/
│   ├── ellipsoid_calibration.launch   # 椭球校准采样
│   ├── consistency_calibration.launch   # 一致性校准
│   ├── handheld_calibration.launch     # 手持校准
│   └── save_diana7_home.launch         # 保存安全姿态
├── scripts/
│   ├── ellipsoid_calibration_sampler.py
│   ├── consistency_calibration.py
│   ├── handheld_calibration_sampler.py
│   └── save_diana7_home.py
├── lib/
│   ├── ellipsoid_fit/               # 椭球标定算法（拟合参数）
│   ├── consistency_fit.py           # 一致性标定算法（拟合参数）
│   └── forward/                     # 正向校正（应用参数）
│       ├── __init__.py
│       ├── base.py                  # CorrectionStep 抽象基类
│       ├── processor.py             # CalibrationProcessor 门面类
│       ├── ellipsoid.py             # 椭球校正 step
│       ├── r_corr.py                # R_CORR 旋转 step
│       └── consistency.py           # 一致性校正 step
└── config/
    └── diana7_home.yaml             # 机械臂安全姿态
```

## 校准流程

1. **椭球校准 (Phase 1)**: 采集半球方向磁场数据，校正单传感器固有误差
2. **一致性校准 (Phase 2)**: 采集多方向磁场数据，使阵列响应一致
3. **手持校准**: 现场快速校准（可选）

**正向校正**在 `lib/forward/` 中实现，供运行时使用（见第 4 节）。
