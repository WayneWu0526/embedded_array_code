# result 文件夹说明

## 文件列表

### 原始数据（Cycle JSON）

| 文件 | 说明 |
|------|------|
| `cycle_0000.json` | 基础数据集，CVT 模式，12 传感器，Ground Truth 位置 (0.0942, 0.2381, 0.1745) m |
| `cycle_0001.json` | 第二个 cycle 数据 |
| `cycle_0000_standup.json` | standup 配置实验 |
| `cycle_0000_laydown.json` | laydown 配置实验 |
| `cycle_0000_5mm.json` | 5mm 位移实验 |
| `cycle_0000_2cm.json` / `cycle_0000_4cm.json` | 不同距离配置实验 |
| `cycle_0000_15Gs_raw.json` / `cycle_0000_25Gs_raw.json` | 不同磁场强度原始数据 |
| `cycle_0000_pm.json` / `cycle_0001_pm.json` / `cycle_0002_pm.json` | 永磁体（permanent magnet）实验 |

### 噪声分析结果

#### CSV 数据文件

| 文件 | 来源版本 | 偶极子阶数 | 说明 |
|------|---------|-----------|------|
| `noise_analysis_order3.csv` | 当前版 (47d1d6d) | order=3 | 射线延伸分析，magnitudes=20 A·m² |
| `noise_analysis_order_noise.csv` | 当前版 (47d1d6d) | order=3 | 同上，使用不同 cycle 或参数 |

**CSV 格式：**

| 列名 | 单位 | 含义 |
|------|------|------|
| `ep [mm]` | mm | 位置误差 \|\|p_est − p_gt\|\| |
| `eR [deg]` | deg | 旋转误差（基于旋转矩阵迹的 arccos 公式） |
| `distance [mm]` | mm | 测量点到源质心 pbar 的距离，非磁源到传感器的距离 |
| `B_norm [Gs]` | Gs | 传感器阵列中心的合成磁场范数 |

**distance 字段详解：**

distance 表示测量点到**源质心** pbar 的距离：

```
pbar = mean(p_Ci)  ← 三个电磁线圈源位置的质心
       ↑
       │ 射线方向
       ↑
p_base（cycle JSON 中的 Ground Truth 位置）

distance = ||p_extended − pbar|| = d_base + L_step
```

其中 d_base = ||p_base − pbar|| ≈ 60~77 mm（取决于具体 cycle），step=10mm，共 40 个采样点，延伸到 ~400~467 mm。

随着 distance 增加，B_norm 从 ~30 Gs 衰减到 <1 Gs，SNR 恶化导致定位误差急剧增大。

#### 可视化图表

| 文件 | 来源版本 | 说明 |
|------|---------|------|
| `noise_analysis_order3.pdf` | 当前版 | ep/eR vs distance，order-3 偶极子模型，log scale |
| `noise_analysis_order1.pdf` | 当前版 | 同上，order-1 偶极子模型 |
| `noise_analysis_rp_snr.png` / `.pdf` | 旧版 (64f9156) | SNR 噪声分析，Δo/|B| vs 误差，线性拟合 |
| `noise_analysis_rp_cycle_0000.png` | 旧版 | 多 moment magnitude 组合的 offset bias 分析 |
| `noise_analysis_rp_multi_m_cycle_0000.png` | 旧版 | 同上，更多 magnitude 组合 |

### 位姿恢复结果

| 文件 | 说明 |
|------|------|
| `pose_plot_cycle_0000.png` | 3D 可视化：真实数据 vs 合成数据的定位结果对比 |

---

## 噪声分析实验两种方法对比

### 方法一：射线延伸法（当前版本，commit 47d1d6d）

```
数据来源：noise_analysis_rp.py
核心逻辑：
  1. 从 cycle JSON 读取三个电磁线圈的源位姿（slot 0/1/2）
  2. 计算源质心 pbar = mean(p_Ci)
  3. 以 Ground Truth pose 为 base，沿 pbar→base 射线方向扩展
  4. 每 step=10mm 一个采样点，共 40 点，延伸 400mm
  5. 用 order-3 偶极子模型生成合成磁场
  6. 固定噪声 FIXED_NOISE_LEVEL=0.005 Gs，offset=0
  7. MaPS_Estimator 估计 pose，计算 ep, eR
输出：CSV（ep, eR, distance, B_norm）+ PDF（ep/eR vs distance）
研究问题：纯噪声 floor 下，距离（进而 B_norm 衰减）如何影响误差
```

### 方法二：随机姿态扰动 + Offset Bias 扫描（旧版，commit 64f9156）

```
数据来源：noise_analysis_rp.py（旧版代码）
核心逻辑：
  1. 以 GT 为中心，在 5cm 球内随机采样 100 个 (R, p)
  2. 对每个 offset 水平 delta_o ∈ [0.0, 0.05, ..., 0.5] Gs：
     - 采样随机 per-sensor offset ∈ [-delta_o, delta_o]^3
     - 在所有 100 个 pose 上施加相同 offset
     - 固定噪声 0.01 Gs
  3. 计算比值 delta_o / |B_bar|
  4. 线性拟合：ep = k * (delta_o/|B|) + b
输出：noise_analysis_rp_snr.pdf/png
研究问题：传感器校准残差（offset bias）相对磁场强度的比例对误差的影响
       本质是另一种 SNR 指标——offset 充当"噪声"，|B| 充当"信号"
```

---

## CSV 字段几何含义图解

```
                    三个磁源
                   × diana7 (slot 0)
                  × arm1 (slot 1)
                  × arm2 (slot 2)

              pbar = centroid of (p_Ci)

               ● ← p_base（GT 真实位置）
              ↗
    distance = ||p_base − pbar|| ≈ 60~77 mm（初始）
         ↗
    p_extended_1  distance + 10 mm
         ↗
    p_extended_2  distance + 20 mm
         ...
         ↗
    p_extended_40 distance + 400 mm
```

---

## 磁源参数

| 磁源 | Slot | 磁矩幅值 |
|------|------|---------|
| diana7 | 0 | -120 A·m² |
| arm1 | 1 | -200 A·m² |
| arm2 | 2 | -200 A·m² |

磁化方向：local -y 轴（从各 slot pose 的四元数提取 z 轴）

---

## cycle_0000.json 几何参数

```
Ground Truth 位置: (0.0942, 0.2381, 0.1745) m
源质心 pbar:       (0.0995, 0.2362, 0.2413) m
GT 到 pbar 距离:   ≈ 67.09 mm

Slot 0 (diana7): (0.0925, 0.0542, 0.1840) m
Slot 1 (arm1):   (0.2262, 0.3696, 0.2140) m
Slot 2 (arm2):   (-0.0201, 0.2847, 0.3260) m
```
