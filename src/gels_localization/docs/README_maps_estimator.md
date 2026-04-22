# MaPS Estimator Algorithm

## Overview

MaPS (Magnetic Pose Stimator) 实现论文中的 Algorithm 1，用于从磁场测量数据估计 6-DoF 位姿（旋转矩阵 R ∈ SO(3) 和位置向量 p ∈ ℝ³）。

**输入：**
- `D_cal`：传感器在传感器坐标系下的偏置，形状 3 × N
- `sources`：M 个磁源的列表，每个包含 `p_Ci`（源在标定坐标系中的位置，3维）和 `m_Ci`（源磁矩，3维）
- `B_meas_cell`：M 个测量单元格列表，每个是 3 × N 的磁场测量矩阵

**输出：**
- `R_est`：估计的旋转矩阵
- `p_est`：估计的位置向量（3维）

---

## 算法流程

### 第一步：预计算

#### 1.1 局部场估计的零空间投影（Eq. 5）

```python
Q_bar = null(D_cal)
g = Q_bar.T @ np.ones(N)
```

- 计算传感器偏置矩阵 `D_cal` 的零空间基 `Q_bar`
- 向量 `g` 用于后续局部场估计的投影

#### 1.2 成对差异计算（Eq. 8）

```python
Np = N * (N - 1) // 2
D_delta = np.zeros((3, Np))
# 两两相减：D_delta[:, idx] = D_cal[:, i] - D_cal[:, j]
```

- 构建 N 个传感器两两之间的位置差异矩阵 `D_delta`

#### 1.3 约束矩阵构建（Eq. 9-11）

```python
S_mat = np.array([
    [1,  0,  0,  0,  0],  # (1,1)
    [0,  1,  0,  0,  0],  # (2,1)
    [0,  0,  1,  0,  0],  # (3,1)
    [0,  1,  0,  0,  0],  # (1,2)
    [0,  0,  0,  1,  0],  # (2,2)
    [0,  0,  0,  0,  1],  # (3,2)
    [0,  0,  1,  0,  0],  # (1,3)
    [0,  0,  0,  0,  1],  # (2,3)
    [-1, 0,  0, -1,  0],  # (3,3)
])

C_mat = np.kron(D_delta.T, np.eye(3)) @ S_mat
```

- `S_mat` 是对称无迹梯度张量 X 的参数化矩阵
- `X = [x1 x2 x3; x2 x4 x5; x3 x5 -(x1+x4)]`
- 通过克罗内克积构建约束矩阵 `C_mat`

---

### 第二步：局部量估计

对每个测量单元格 i = 1, ..., M 进行：

#### 2.1 局部场估计（Eq. 5）

```python
if Q_bar.size == 0:
    b_hat_locals[:, i] = (B_meas @ np.ones(N)) / N
else:
    b_hat_locals[:, i] = (B_meas @ Q_bar @ g) / (np.linalg.norm(g)**2)
```

- 将测量场投影到传感器偏置的零空间，得到局部场估计 `b_hat`

#### 2.2 局部梯度张量估计（Eq. 8-11）

```python
B_delta = np.zeros((3, Np))
# 两两相减：B_delta[:, idx] = B_meas[:, ii] - B_meas[:, jj]

h_vec = B_delta.flatten(order='F')  # 列主序
C_pinv = np.linalg.pinv(C_mat)
x_param = C_pinv @ h_vec
X_hat = S_mat @ x_param
X_hat = X_hat.reshape((3, 3), order='F')
```

- 计算测量场的成对差异 `B_delta`
- 通过伪逆求解参数，得到对称无迹梯度张量 `X_hat`

#### 2.3 局部相对位移估计（Eq. 13）

```python
rho_hats[:, i] = compute_rho_hat_eigen(X_hat, b_hat_locals[:, i])
```

**特征值求解法（避免直接矩阵求逆）：**

```python
def compute_rho_hat_eigen(X, b_hat):
    # 计算特征值并排序
    evals = np.linalg.eigvalsh(X)
    evals_sorted = np.sort(evals)[::-1]
    lambda_max = evals_sorted[0]
    lambda_med = evals_sorted[1]
    lambda_min = evals_sorted[2]

    # rho_hat = 3 / (lambda_max * lambda_min) * (X + lambda_med * I) @ b_hat
    rho_hat = 3.0 / (lambda_max * lambda_min) * (X + lambda_med * np.eye(3)) @ b_hat
    return rho_hat
```

---

### 第三步：全局位姿恢复

使用标准刚性配准（Kabsch 算法）进行绝对方向估计。

#### 3.1 堆叠全局源位置

```python
p_Ck = np.column_stack([sources[i]['p_Ci'] for i in range(M)])  # shape: (3, M)
rho_k = rho_hats  # shape: (3, M)
```

#### 3.2 计算质心

```python
p_C_bar = np.mean(p_Ck, axis=1, keepdims=True)  # shape: (3, 1)
rho_bar = np.mean(rho_k, axis=1, keepdims=True)  # shape: (3, 1)
```

#### 3.3 中心化对应点

```python
p_C_tilde = p_Ck - p_C_bar  # shape: (3, M)
rho_tilde = rho_k - rho_bar  # shape: (3, M)
```

#### 3.4 Kabsch 求解

由于约束关系 `p_C_k = p - R ρ_k`，中心化后满足：
```
P_C_tilde = -R RHO_tilde
```

因此对 `(P_C_tilde, -RHO_tilde)` 求解 Kabsch 问题：

```python
R_est = kabsch_solver(p_C_tilde, -rho_tilde)
```

**Kabsch 算法实现：**

```python
def kabsch_solver(V, U):
    # 求解正交 Procrustes 问题：min_{R in SO(3)} ||V - R @ U||_F
    H = V @ U.T
    L, _, Wh = np.linalg.svd(H)

    d = np.sign(np.linalg.det(L @ Wh))
    if d == 0:
        d = 1.0

    D = np.diag([1.0, 1.0, d])
    R = L @ D @ Wh
    return R
```

#### 3.5 位置恢复

```python
# 每个源的位置估计
p_ests = np.zeros((3, M))
for i in range(M):
    p_ests[:, i] = sources[i]['p_Ci'] + R_est @ rho_hats[:, i]

# 最终位置（质心平均）
p_est = (p_C_bar + R_est @ rho_bar).reshape(3)
```

---

## 数学公式汇总

| 公式 | 描述 |
|------|------|
| Eq. 5 | 零空间投影：`b_hat = (B_meas @ Q_bar @ g) / ||g||²` |
| Eq. 8 | 成对差异：`D_delta, B_delta` |
| Eq. 9-11 | 梯度张量参数化约束矩阵 `C_mat` |
| Eq. 13 | 相对位移：`ρ_hat = -3 X⁻¹ b_hat`（特征值求解） |
| Eq. 18 | 位置平均 |

---

## 关键实现细节

1. **避免矩阵求逆**：使用特征值分解计算 `ρ_hat`，而非直接求 `X⁻¹`，提高数值稳定性
2. **列主序**：`h_vec = B_delta.flatten(order='F')` 确保与 MATLAB 实现一致
3. **伪逆求解**：`C_pinv @ h_vec` 比 `lstsq` 对秩亏矩阵更稳定
4. **Kabsch 修正**：通过添加对角矩阵 `D` 确保旋转矩阵行列式为 +1（属于 SO(3)）
