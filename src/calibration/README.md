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
