# Phase 1: 单颗传感器椭球校准 (Ellipsoid Fitting)

## 概述

本模块实现 QMC6309 磁传感器的 Phase 1 椭球校准，将传感器原始测量从椭球分布"还原"为球形分布。

---

## 椭球校正原理

### 问题来源

传感器原始测量不准确，产生两种误差：

| 误差类型 | 来源 | 性质 |
|---------|------|------|
| **Hard iron offset** (o_i) | 焊盘残磁、PCB铜层、封装磁性材料 | 固定偏置 |
| **Soft iron + Scale + Non-orthogonality** (A_i) | 三轴灵敏度不一致、轴不正交 | 非均匀缩放 |

### 数学模型

传感器原始输出满足：
```
b_raw = A × b_true + o_i
```

理想传感器在均匀磁场中应测得**完美球形**分布，实际测得的是**椭球**。

椭球方程：
```
(b_raw - o_i)^T × W × (b_raw - o_i) = 1
```

### 校正目标

找到一个变换 `C_i`，使得校正后接近球形：
```
b_corr = C_i × (b_raw - o_i)  ≈  球形
```

其中 `W = C_i^T × C_i`

### 是否会缩放测量结果？

**是的，会缩放。但这是必要的校正，不是失真。**

```
原始:  x=0.6, y=0.5, z=0.5  → |b| ≈ 0.9 (椭球)
校正后: x=0.5, y=0.5, z=0.5  → |b| ≈ 0.5 (球)
```

类比：把歪着的椭圆框拍正，是"还原真实形状"，不是失真。

### o_i 和 C_i 能否跨位置使用？

**可以。** 这两个系数是**传感器的固有属性**，不是环境函数。

- 地磁场方向/强度变化 → 正常测量变化，参数仍有效
- 附近有强磁性材料 → 可能需重新校准
- 温度大幅变化 → 参数可能漂移

---

## 算法详解

### Full Ellipsoid Fitting（本模块使用）

对于椭球方程：
```
Ax² + By² + Cz² + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz = 1
```

**步骤：**

1. 构建设计矩阵 (N × 9)
2. 最小二乘求解 9 个系数
3. 提取二次型矩阵 W 和 offset o_i
4. Cholesky 分解: `W = L × L^T`
5. 校正矩阵: `C_i = L^(-1)`

### 阈值判定

| eigenvalue_ratio | 类型 | 校正方法 |
|-------------------|------|---------|
| < 1.5 | 球形 | offset only |
| 1.5 ~ 3.0 | axis-aligned | 对角矩阵 C_i |
| ≥ 3.0 | full ellipsoid | 完整 3×3 矩阵 |

---

## 使用方法

### 1. 加载校准参数

```python
import json
import numpy as np

# 加载内参
with open('s1_ellipsoid_fit/result/intrinsic_params_150317.json') as f:
    params = json.load(f)
```

### 2. 应用校正

```python
# 对第 i 个传感器应用校正
i = 0  # sensor 1
o_i = np.array(params['sensors'][i]['o_i'])
C_i = np.array(params['sensors'][i]['C_i'])

# b_raw: N×3 原始数据
b_corr = (b_raw - o_i) @ C_i.T  # 等价于 C_i @ (b_raw - o_i)
```

### 3. 批量校正所有传感器

```python
for sensor in params['sensors']:
    sid = sensor['sensor_id']
    o_i = np.array(sensor['o_i'])
    C_i = np.array(sensor['C_i'])

    col_x = f'sensor_{sid}_x'
    col_y = f'sensor_{sid}_y'
    col_z = f'sensor_{sid}_z'

    b_raw = df[[col_x, col_y, col_z]].values
    b_corr = (b_raw - o_i) @ C_i.T
```

### 4. 命令行运行

```bash
cd code_calibration/s1_ellipsoid_fit/script
python s1_main.py
```

---

## 验证方法

```python
# 校正后验证
radius_std_after = np.std(np.linalg.norm(b_corr, axis=1))
radius_std_before = np.std(np.linalg.norm(b_raw - o_i, axis=1))
improvement = radius_std_before / radius_std_after

# 预期：improvement >> 1
# 良好校正：improvement > 10x
```

---

## 当前校正结果 (ellipsoid_calib_20260414_150317.csv)

| 传感器 | ratio | 改善比率 | radius: before → after |
|--------|-------|----------|------------------------|
| 1 | 1.57 | 16.2x | 0.102 → 0.0063 |
| 2 | 1.56 | 19.9x | 0.100 → 0.0050 |
| 3 | 1.59 | 14.9x | 0.101 → 0.0068 |
| ... | ... | ... | ... |
| 平均 | 1.56-1.62 | **19.0x** | 0.10 → 0.005 |

---

## 与 Phase 2-5 的关系

Phase 1 完成后，数据满足：
```
b_i^{corr} = C_i × (b_i^{raw} - o_i)
```

但 12 颗传感器仍不在同一坐标系，需要：

| Phase | 操作 | 公式 |
|-------|------|------|
| Phase 2 | 旋转对齐 | `b_i^{aligned} = R_i × b_i^{corr}` |
| Phase 3 | 残余偏置 | `b_i^{final} = b_i^{aligned} - δ_i` |
| Phase 4 | 几何位置 | 标定 `d_i` |
| Phase 5 | 梯度验证 | 验证 `X̂ ≈ X_gt` |

---

## 文件结构

```
s1_ellipsoid_fit/
├── script/
│   ├── ellipsoid_fit.py       # 主拟合函数（自动分类路由）
│   ├── axis_aligned_fit.py    # axis-aligned 简化算法
│   ├── full_ellipsoid_fit.py  # full ellipsoid 算法
│   ├── apply_calibration.py   # 应用校准工具
│   ├── s1_main.py            # 主入口脚本
│   └── __init__.py            # 模块初始化
├── plot/                      # 输出图像
│   └── calibration_*_sensor*.png
├── result/                    # 输出结果
│   ├── intrinsic_params_150317.json    # 内参（仅 o_i, C_i）
│   └── calibration_params_*.json       # 完整报告
└── README.md
```
