# Ellipsoid Fitting - 椭球校准

## 概述

QMC6309 磁传感器的椭球校准，将传感器原始测量从椭球分布"还原"为球形分布。

## 数学模型

传感器原始输出满足：
```
b_raw = A × b_true + o_i
```

理想传感器在均匀磁场中应测得**完美球形**分布，实际测得的是**椭球**。

椭球方程：
```
(b_raw - o_i)^T × W × (b_raw - o_i) = 1
```

校正目标：
```
b_corr = C_i × (b_raw - o_i)  ≈  球形
```

其中 `W = C_i^T × C_i`，`C_i` 是 3×3 校正矩阵，`o_i` 是 hard iron offset。

---

## 使用方法

### 1. Python 调用

```python
from calibration.lib.ellipsoid_fit import ellipsoid_fit, batch_ellipsoid_fit

# 单颗传感器校准
result = ellipsoid_fit(b_raw, sensor_id=1)
# result.o_i: offset 向量 (3,)
# result.C_i: 校正矩阵 (3×3)
# result.improvement_ratio: 改善倍数

# 批量校准
results = batch_ellipsoid_fit('data.csv', output_dir='report/')
```

### 2. 命令行

```bash
python -c "
from calibration.lib.ellipsoid_fit import batch_ellipsoid_fit
results = batch_ellipsoid_fit('data/raw_data.csv', output_dir='report/')
"
```

---

## 校准结果

### handheld_calib_20260415_175629.csv (500点)

| 传感器 | eigenvalue_ratio | 改善倍数 | radius: before → after |
|--------|-----------------|----------|------------------------|
| 1 | 1.38 | 14.0x | 0.0965 → 0.0069 |
| 2 | 1.40 | 21.5x | 0.0941 → 0.0044 |
| 3 | 1.37 | 14.7x | 0.0936 → 0.0064 |
| 4 | 1.39 | 17.9x | 0.0931 → 0.0052 |
| 5 | 1.39 | 17.5x | 0.0952 → 0.0054 |
| 6 | 1.39 | 18.8x | 0.0942 → 0.0050 |
| 7 | 1.44 | 17.0x | 0.0924 → 0.0054 |
| 8 | 1.42 | 15.0x | 0.0935 → 0.0062 |
| 9 | 1.44 | 15.6x | 0.0948 → 0.0061 |
| 10 | 1.43 | 17.2x | 0.0930 → 0.0054 |
| 11 | 1.43 | 16.1x | 0.0934 → 0.0058 |
| 12 | 1.41 | 16.0x | 0.0931 → 0.0058 |

---

## 拟合算法详解

### full_ellipsoid_fit — 代数最小二乘法

适用于 **eigenvalue_ratio > 4.0** 的传感器（椭球有明显倾斜，特征向量不沿坐标轴）。

**算法步骤：**

1. **构建设计矩阵**
   椭球标准方程的代数形式：
   ```
   Ax² + By² + Cz² + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz = 1
   ```
   对每个测量点 (x, y, z) 构成一行，构建 N×9 设计矩阵：
   ```
   A_design = [x², y², z², 2xy, 2xz, 2yz, 2x, 2y, 2z]
   ```

2. **最小二乘求解**
   ```
   A_design · [A B C D E F G H I]^T = 1
   ```
   通过 `np.linalg.lstsq` 求解 9 个系数。

3. **提取二次型矩阵 W**
   ```
   W = | A  D  E |
       | D  B  F |
       | E  F  C |
   linear = [G, H, I]
   ```

4. **计算 Hard Iron Offset**
   ```
   o_i = -W⁻¹ · linear
   ```

5. **计算校正矩阵 C_i**
   - 优先使用 **Cholesky 分解**：W = L·Lᵀ，则 C_i = L⁻¹
   - 若 W 非正定（数值误差），使用**特征值分解**：
     ```
     W = V·Λ·Vᵀ  →  C_i = V·Λ^{-1/2}·Vᵀ
     ```

**关键公式：**
```
校准后: b_corr = C_i · (b_raw - o_i)  ≈ 球形
```

---

### full_ellipsoid_fit_iterative — 迭代优化法

对噪声较大的数据更鲁棒，迭代交替估计 offset 和 W。

**算法步骤：**

1. 初始化 offset（默认使用数据均值）
2. 迭代循环（最多 100 次）：
   - 固定当前 offset，对中心化数据做椭球拟合，估计 W
   - 根据 W 更新 offset：o_new = -W⁻¹ · linear
   - 检查收敛：||o_new - o_i|| < 1e-6
3. 最终用收敛的 offset 估计校正矩阵 C_i

---

## API

### ellipsoid_fit(b_raw, sensor_id, csv_file='', use_iterative=False)

单颗传感器椭球拟合。

**参数:**
- `b_raw`: N×3 原始磁场数据
- `sensor_id`: 传感器编号 (1-12)
- `csv_file`: 原始数据文件名
- `use_iterative`: 是否使用迭代方法（更鲁棒但更慢）

**返回:** `CalibrationResult`

### batch_ellipsoid_fit(csv_path, output_dir=None)

批量校准 CSV 中所有 12 颗传感器。

### apply_calibration(b_raw, o_i, C_i)

应用椭球校准公式: `b_corr = C_i @ (b_raw - o_i)`

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
