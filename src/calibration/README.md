# calibration

## 概述

传感器阵列校准模块，包含椭球校准、一致性校准和手持校准功能。

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

### 3. 手持校准 (Handheld Calibration)

用于现场快速校准，无需机械臂设备。

**启动:**
```bash
roslaunch calibration handheld_calibration.launch
```

### 4. diana7 安全姿态保存

保存机械臂安全姿态，用于后续校准采样。

```bash
roslaunch calibration save_diana7_home.launch
```

## 文件结构

```
src/calibration/
├── launch/
│   ├── ellipsoid_calibration.launch   # 椭球校准采样
│   ├── consistency_calibration.launch # 一致性校准
│   ├── handheld_calibration.launch    # 手持校准
│   └── save_diana7_home.launch        # 保存安全姿态
├── scripts/
│   ├── ellipsoid_calibration_sampler.py
│   ├── consistency_calibration.py
│   ├── handheld_calibration_sampler.py
│   └── save_diana7_home.py
├── lib/
│   ├── ellipsoid_fit/               # 椭球校正算法
│   └── consistency_fit.py           # 一致性校正算法
└── config/
    └── diana7_home.yaml             # 机械臂安全姿态
```

## 校准流程

1. **椭球校准 (Phase 1)**: 采集半球方向磁场数据，校正单传感器固有误差
2. **一致性校准 (Phase 2)**: 采集多方向磁场数据，使阵列响应一致
3. **手持校准**: 现场快速校准（可选）
