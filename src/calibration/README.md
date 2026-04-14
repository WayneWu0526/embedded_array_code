# calibration

## 组件：

1. diana机械臂
2. embedded array

## 所需要的节点：

1. ~/zlab_robots/zlab_robots_bringup/diana7，末端执行器装配magnetometer_array，无需启动任何可视化界面。
 - 用于控制diana7机械臂。
2. embedded_array_ws/sensor_data_collection，stm32_manual.launch，
 - 用于启动下位机并开启持续向上位机发送测量数据的手动模式
3. 主采样节点
    - 1. 完成和diana7机械臂的连接，注意：必须保证diana7保持当前状态一动不动。
    - 2. 完成和embedded_array_ws的连接，确保能够正确读取其发送的12个传感器的数据
    - 3. 旋转diana7的末端关节（joint7）到-180（注意关节限位，可能是-179度）
    - 4. 从负关节限位开始，订阅5次传感器的数据，并取平均。然后将所有传感器的读数和当前diana7最后一个关节的关节角度、存储到一个数据包中。
    - 5. 增加1度，直到关节来到正关节限位（可能是180度或者179度）
    - 6. 完成一次采样，退出程序，将数据包保存为csv格式。
4. 说明
    - 数据包中的磁场数据是以Gs单位，最大32.000，最小-32.000，所有保留到小数点后三位。
    - 传感器角度以1度为最小分辨率即可。
    - 理论上将获取360（或者少几个，取决于关节限制）行数据，每行包含关节角度、id=1-12的所有传感器的读数（每个三列）
    - csv名字中包含一个简短的时间记录。
5. 写一个实验顺序对照表。我们将按照时间顺序依次完成：30x, 30y, 30z, 25x, 25y, 25z, 直到5x, 5y, 5z，共计18组实验结果，按照时间顺序完成。
6. 重点：由于机械臂当前处于危险位姿，因此连接上diana7之后，应该不发送任何控制信号，甚至任何初始化信号。diana7只有第七关节被允许活动，所以需要额外小心和注意。

# 补充：
添加对信号发生器：fy8300的控制。我们的信号config采用signal_params_calib.yaml中定义的参数。
仿照data_collection_node.py中的代码，来实现对fy8300的连接和初始化。
具体的操作步骤：

单独启动CHANEL-3，等待其启动完毕，大约10s，然后开始机械臂的旋转和数据采集。
机械臂旋转完毕后，关闭CHANEL-3，等待其完全关闭，大约10s。
将机械臂旋转回初始位置，准备下一组实验，保存csv数据。

启动CHANEL-2，等待其启动完毕，大约10s，然后开始机械臂的旋转和数据采集。
机械臂旋转完毕后，关闭CHANEL-2，等待其完全关闭，大约10s。
将机械臂旋转回初始位置，准备下一组实验，保存csv数据

启动CHANEL-1，等待其启动完毕，大约10s，然后开始机械臂的旋转和数据采集。
机械臂旋转完毕后，关闭CHANEL-1，等待其完全关闭，大约10s。
将机械臂旋转回初始位置，准备下一组实验，保存csv数据

# 椭球矫正采样 (Ellipsoid Calibration Sampling)

## 概述

椭球矫正采样用于收集传感器阵列在空间不同方向上的磁场测量数据，用于后续的磁力计椭球校准（硬铁/软铁校正）。

## 硬件配置

- **diana7 机械臂**：末端安装 `magnetometer_array`
- **STM32 传感器板**：手动模式（无需 FY8300 信号发生器）
- **embedded_sensor_array**：绑定在机械臂末端

## 使用方法

### 步骤 1：保存安全姿态

1. 手动将 diana7 移动到安全姿态
2. 运行保存脚本：

```bash
roslaunch calibration save_diana7_home.launch
```

这会将当前关节角度保存到 `config/diana7_home.yaml`

### 步骤 2：运行椭球校准采样

```bash
roslaunch calibration ellipsoid_calibration.launch
```

脚本会自动：
1. 加载 `config/diana7_home.yaml`
2. 移动到记录的关节姿态
3. 开始半球采样

## 采样原理

1. 机械臂移动到预设的安全姿态（从 `diana7_home.yaml` 加载）
2. 保持 Joint7 = 0
3. 使用 **Fibonacci 半球分布（Z轴正向）** 生成均匀分布的姿态方向
4. 对每个姿态方向：
   - 通过 MoveIt 控制机械臂只改变末端姿态（位置保持不变）
   - 等待 0.5 秒稳定时间
   - 采集 10 次传感器数据并取平均
   - 保存到 CSV 文件

## 输出数据格式

CSV 文件包含 39 列：

| 列名 | 描述 |
|------|------|
| timestamp | ISO 格式时间戳 |
| pos_x, pos_y, pos_z | 末端执行器位置 (m) |
| qx, qy, qz, qw | 末端执行器姿态 (四元数) |
| sensor_1_x ~ sensor_12_z | 12 个传感器的 XYZ 磁场数据 (Gs) |

## 参数配置

| 参数 | 默认值 | 描述 |
|------|--------|------|
| num_poses | 10 | 姿态数量（测试用 10，后续可增加到 500） |
| num_samples | 10 | 每个姿态的采样次数 |
| settling_time | 0.5s | 移动后稳定时间 |
| speed_scaling | 0.1 | MoveIt 速度缩放因子 |
| output_dir | `calibration/data/` | CSV 输出目录 |

## 数据获取方式

传感器数据通过 `stm_uplink` topic 订阅获取（与 `calibration_joint_sweep.py` 一致）：

- Topic: `stm_uplink`
- Message type: `StmUplink`
- 数据来源: `serial_processor` 包的 `serial_node_tdm.py` 节点

每次采集时会等待接收到最新数据后进行平均。

## 文件结构

```
src/calibration/
├── launch/
│   ├── ellipsoid_calibration.launch  # 椭球校准采样启动文件
│   └── save_diana7_home.launch       # 保存安全姿态启动文件
├── scripts/
│   ├── ellipsoid_calibration_sampler.py  # 主采样脚本
│   └── save_diana7_home.py              # 保存关节姿态脚本
├── config/
│   └── diana7_home.yaml                 # 安全姿态配置（运行后生成）
└── data/
    └── ellipsoid_calib_YYYYMMDD_HHMMSS.csv  # 输出数据
```

## 算法说明

### Fibonacci 半球分布

使用黄金角螺旋（Golden Angle Spiral）在 Z轴正向半球上均匀分布采样点：

```
y = 1 - i/(n-1)  # 从 1 到 0，对应 phi 从 0 到 π/2
phi = arccos(y)
theta = golden_angle * i  # golden_angle = π * (3 - √5) ≈ 2.39996
```

半球分布确保 vz > 0（TCP Z轴指向上方），Joint7 锁定为 0。

### 姿态控制

- 通过 MoveIt 的 `set_pose_target()` 设置目标姿态
- 位置分量保持当前值不变
- 只改变姿态分量（通过四元数指定方向）
- 使用 `go(wait=True)` 等待运动完成

## 注意事项

1. **安全第一**：此脚本只改变机械臂姿态，不改变位置，但仍需注意关节限位
2. **无 FY8300**：椭球矫正模式不使用信号发生器，只采集环境磁场数据
3. **半球采样**：与 joint7 旋转采样不同，椭球矫正覆盖 Z轴正向半球方向
4. **Joint7 = 0**：采样过程中关节7锁定为0
5. **数据用途**：采集的数据用于后续计算传感器的椭球校准参数（3x3 C 矩阵和 b_bias 向量）
