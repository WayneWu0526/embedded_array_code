# Sensor Array Config

这个目录存放 `sensor_array_config` 读取的整板硬件配置。每个子目录代表一块具体硬件板，例如 `ak09973d`、`qmc6309`、`tmag3001`。

- `<name>/board.json`：整板配置入口，声明要加载哪些子配置文件。
- `<name>/magnetometer.json`：磁传感器型号、量程、单位换算、数量、分组和 frame。
- `<name>/array.json`：磁阵列几何位置 `d_list` 和安装方向修正 `R_CORR`。
- `<name>/affine.json`：每个磁传感器的 affine 标定参数 `D_i`、`e_i`。
- `<name>/imu.json`：IMU 缩放、frame、位置和轴变换。

代码入口保持不变：

```python
from sensor_array_config import get_hardware_config, list_hardware_configs

hw = get_hardware_config("ak09973d")
```

## 目录结构

每个硬件配置目录描述一块具体板子：

```text
config/ak09973d/
  board.json
  magnetometer.json
  array.json
  affine.json
  imu.json
```

运行时使用的名字就是目录名，所以：

```bash
hardware_config:=ak09973d
```

会加载：

```text
config/ak09973d/board.json
```

## `board.json`

整板配置入口文件。

| 字段 | 作用 |
| --- | --- |
| `name` | 硬件配置名，应与目录名一致。 |
| `firmware_protocol` | 该板子期望使用的固件串口协议。 |
| `calibration_status` | 人看的标定状态，例如 `placeholder`、`measured`。 |
| `notes` | 自由备注。 |
| `files` | 子配置文件名，loader 会按这里的引用组装完整硬件配置。 |

## `magnetometer.json`

磁传感器芯片和阵列数量相关信息。

| 字段 | 作用 |
| --- | --- |
| `name`, `manufacturer` | 磁传感器型号和厂商。 |
| `bit_width`, `range_gs`, `adu_to_gs`, `gs_to_tesla` | 从固件 raw/ADU 到 Gs/Tesla 的单位换算参数。 |
| `n_sensors` | 这块板上的磁传感器数量。 |
| `n_groups`, `sensors_per_group` | 逻辑分组信息，用于阵列布局和方向修正。 |
| `frame_id` | 磁阵列数据使用的 ROS frame。 |
| `calibration_status`, `notes` | 磁传感器相关的标定状态和备注。 |

## `array.json`

磁传感器阵列的物理布局和每个传感器的安装方向修正。

| 字段 | 作用 |
| --- | --- |
| `d_list` | 每个磁传感器相对阵列原点的位置，单位是米；每一行是一个 `[x, y, z]`。 |
| `R_CORR` | 方向修正配置。每个条目把一组 `sensor_ids` 映射到同一个 3x3 修正矩阵。矩阵按 column-major 顺序展开。 |

`R_CORR` 在 `serial_processor` 中应用，发生在发布 `stm_uplink_raw` 之前。因此：

```text
stm_uplink_raw = 已完成单位换算 + R_CORR 方向统一，但未做 affine 标定
stm_uplink     = stm_uplink_raw 再经过 affine 标定
```

## `affine.json`

每个磁传感器的 affine 标定参数。

| 字段 | 作用 |
| --- | --- |
| `sensors` | 标定条目列表，每个 sensor 一个条目。 |
| `sensor_id` | 1-based 传感器编号。 |
| `D_i` | 该传感器的 3x3 affine 矩阵。 |
| `e_i` | 该传感器的 3 维 affine 偏置。 |

在线修正公式是：

```text
b_cal = D_i @ b_raw + e_i
```

`calibration_affine_model.py` 默认会写回这个文件：

```text
config/<hardware_config>/affine.json
```

## `imu.json`

IMU 的缩放参数、frame、相对板子的位姿和轴变换。

| 字段 | 作用 |
| --- | --- |
| `name`, `imu_type` | IMU 配置名和芯片型号。 |
| `imu_frame_id` | IMU ROS 消息使用的 frame id。 |
| `publish_scaled_imu` | 是否发布缩放后的 `sensor_msgs/Imu`。 |
| `accel_range_g`, `gyro_range_dps` | 加速度计和陀螺仪量程。 |
| `accel_odr_hz`, `gyro_odr_hz` | 加速度计和陀螺仪输出频率。 |
| `accel_lsb_per_g`, `gyro_lsb_per_dps` | raw 数据到物理单位的缩放系数。 |
| `temp_lsb_per_c`, `temp_offset_c` | 温度换算参数。 |
| `position_m` | IMU 相对板子/阵列原点的位置，单位是米。 |
| `rotation_deg` | 人看的安装角度备注；运行时不使用它做轴变换。 |
| `axis_transform_matrix` | 运行时真正使用的 3x3 轴变换矩阵，会应用到 accel/gyro 向量。 |

`rotation_deg` 和 `axis_transform_matrix` 最好描述同一个物理安装关系，但当前运行时代码只使用 `axis_transform_matrix`。如果两者不一致，以 `axis_transform_matrix` 为准。
