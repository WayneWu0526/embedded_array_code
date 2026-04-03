# 传感器数据采集系统说明文档

## 硬件：

1. 上位机（ROS Noetic系统）
2. FY8300信号发生器
3. 传感器阵列，STM32开发板
4. ZED2i摄像头

## 硬件连接：

1. ZED2i -> USB -> 上位机
2. 传感器阵列 -> STM32开发板 -> USB -> 上位机
3. FY8300信号发生器 -> USB -> 上位机（自动模式需要）
4. FY8300信号发生器 -> TTL -> STM32开发板（自动模式需要）

## 采集和处理流程：

### 触发模式

系统支持两种触发模式，通过 `manual_trigger` 参数切换：

| 模式 | manual_trigger | 触发方式 | STM32 Mode | num_positions |
|------|---------------|---------|------------|--------------|
| 自动模式 | false | FY8300 TTL 信号 | 0x01 (CVT) / 0x02 (CCI) | 4 (CVT) / 3 (CCI) |
| 手动模式 | true | 用户按 Enter 键 | 0x00 | 可配置，默认 CVT=4, CCI=3 |

### 初始化：
**注意**：STM32的下行指令必须在ZED2i和FY8300（自动模式）都完成初始化后再下发，确保系统处于就绪状态。无需对FY8300进行置0操作。

1. 若launch文件中STM32串口参数为`value="/dev/ttyACM"`则系统会自动扫描所在串口`ttyACM0 or ttyACM1`
2. 上位机通过USB连接ZED2i摄像头，获取图像数据，以tf广播。等待TF可查询后再继续。
3. **手动模式跳过步骤 4 和 5**。自动模式继续步骤 4-5。
4. 上位机通过USB连接控制FY8300信号发生器，设置所需的信号参数（频率、幅度、波形等），使能输出。**注意**：FY8300使能后无反馈确认，需手动延迟约10秒等待其输出稳定。
5. 等待FY8300输出稳定后，上位机通过USB向STM32开发板发送下行命令包（12字节），包含模式、传感器Bitmap、Settling Time等参数。

### 采集：

**自动模式（Mode 0x01/0x02）**：
1. STM32自增cycle_id，初始化slot=0。
2. STM32接收FY8300的TTL信号，等待settling time后触发传感器阵列进行数据采集。
3. 传感器阵列采集数据，并通过USB将数据发送回上位机。
4. 上位机接收传感器数据，查询当前时刻tf中对应slot的位姿数据，与传感器数据协同打包。
5. 重复步骤2-4，直到收到cycle_end=1，完成当前cycle，调用GELS服务进行位姿估计。

**手动模式（Mode 0x00）**：
1. STM32 连续向上位机发送传感器数据（无需 FY8300 触发）。
2. 用户手动移动机械臂到新位置，按 Enter 键保存当前数据（多帧平均 + TF pose）。
3. 重复步骤 2，直到保存满 num_positions 个位置。
4. 调用 GELS 位姿估计算法服务。

## 数据处理：
1. 上位机每完成一个cycle，调用GELS位姿估计算法服务。

## 各数据格式：
**注意**:单片机在完全配置(12 sensors)时,最小采样和传输需要的时间为14.00ms,若配置文件中的`sampling_time`值小于1400,则自动计为1400(14ms)

### 上位机 → STM32 下行命令包（12 bytes）

| 字段 | 长度 | 说明 |
|------|------|------|
| Header | 2 bytes | 包头，固定 `0xAA55` |
| Version | 1 byte | 协议版本，当前 `0x01` |
| Mode | 1 byte | 采集模式，`0x00`=手动模式，`0x01`=恒压模式，`0x02`=恒流模式 |
| Sensor Bitmap | 2 bytes | bit0=传感器1，bit11=传感器12，如 `0x000F` 表示启用 1,2,3,4 |
| Settling Time | 2 bytes | 单位 0.01ms，范围 0~655.35ms（如 100ms = 10000） |
| Cycle Time | 4 bytes | 单位 0.01ms，范围 0~10000.00ms（如 1000ms = 100000） |
| Cycle Num | 1 bytes | 总cycle数目，若为0则一直测量，非零则测量到固定数目后传感器自动重置 |

**示例：** `AA55 01 01 000F 2710 000186A0`
- Header: `AA55`
- Version: `01`
- Mode: `01`（恒压）
- Sensors: `000F`（启用 1,2,3,4）
- Settling Time: `2710` = 10000 = 100.00ms
- Cycle Time: `000186A0` = 100000 = 1000.00ms

### STM32 → 上位机 回复包（3 bytes）

| 字段 | 长度 | 说明 |
|------|------|------|
| Header | 2 bytes | 包头，固定 `0xAA55` |
| Status | 1 byte | `0x00`=成功，`0x01`=参数错误，`0x02`=传感器编号无效，`0xFF`=未知错误 |

**示例：** `AA55 00`（成功）

### STM32 → 上位机 上行数据包

| 字段 | 长度 | 说明 |
|------|------|------|
| Header | 2 bytes | 包头，固定 `0xAA55` |
| Version | 1 byte | 协议版本，当前 `0x01` |
| cycle_id | 2 bytes | cycle 编号（STM32 自增） |
| slot | 1 byte | 当前 slot 序号（恒压: 0-3，恒流: 0-2） |
| Bitmap | 2 bytes | bit0=传感器1，bit11=传感器12，如 `0x000F` 表示启用 1,2,3,4 |
| Timestamp | 8 bytes | 采集时刻，单位微秒（μs），STM32 上电后计时 |
| sensor_data | N×7 bytes | 每传感器: SensorID(1 byte) + X(2 bytes) + Y(2 bytes) + Z(2 bytes)，只发启用的. 类型为整形,真值为原始数据乘以缩放系数: `32/32768` |
| cycle_end | 1 byte | `0x00`=不是最后一个 slot，`0x01`=当前 cycle 的最后一个 slot |
| slot_end| 2 bytes | 数据包结束标志位`0x0D0A`，即\r\n |

**示例：** bitmap=0x000F，恒压模式，slot=3（最后一个 slot）

```
AA55 01 0000 03 000F 00000000000000FA [01 x y z] [02 x y z] [03 x y z] [04 x y z] 01
```

- Header: `AA55`
- Version: `01`
- cycle_id: `0000`（第 0 个 cycle）
- slot: `03`（第 4 个 slot，0-based）
- Bitmap: `000F`（传感器 1,2,3,4）
- Timestamp: `00000000000000FA` = 250 μs（示例值）
- sensor_data: 传感器 1-4 的 xyz 数据
- cycle_end: `01`（这是当前 cycle 的最后一个 slot）

## PC 端 Cycle 组包格式

PC 端将每次上行数据包与 TF 查询结果组合，存入 JSON 文件供后续算法使用。

### Slot 与 Pose 对应关系（参考 frame: `lab_table`）

| slot | 恒压模式 (CVT) | 恒流模式 (CCI) | 手动模式 |
|------|----------------|----------------|----------|
| 0 | `diana7_em_tcp_filt` pose + sensor_data | `diana7_em_tcp_filt` pose + sensor_data | `manual_arm_frame` pose + sensor_data |
| 1 | `arm1_em_tcp_filt` pose + sensor_data | `arm1_em_tcp_filt` pose + sensor_data | `manual_arm_frame` pose + sensor_data |
| 2 | `arm2_em_tcp_filt` pose + sensor_data | `arm2_em_tcp_filt` pose + sensor_data | `manual_arm_frame` pose + sensor_data |
| 3 | sensor_data only | — | — |
| cycle_end | `sensor_array_filt` pose (ground truth) | `sensor_array_filt` pose (ground truth) | — |

### 手动模式 (Mode 0x00)

手动模式用于手动移动执行器（电磁铁或永磁铁）进行测试。执行器固定在机械臂末端，用户手动移动机械臂到 N 个不同位置，每个位置按 Enter 键保存数据。N 个位置都保存后，调用 GELS 位姿估计算法服务。

**配置参数**（在 `task_params.yaml` 中设置）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `manual_trigger` | false | 设为 true 启用手动模式 |
| `manual_arm_frame` | `diana7_em_tcp_filt` | 执行器所在的机械臂：`diana7_em_tcp_filt`、`arm1_em_tcp_filt`、`arm2_em_tcp_filt` |
| `num_positions` | 4 (CVT) / 3 (CCI) | 定位所需的最小位置数 |
| `num_frames_to_average` | 10 | 每个位置采集的帧数做平均 |

**特点**：
- STM32 连续向上位机发送传感器数据（无需 FY8300 触发）
- 用户按 Enter 键触发数据保存（每按一次代表当前 slot 完成）
- 每个位置采集 N 帧做平均
- CVT 模式按 4 次 Enter 完成一个 cycle，CCI 模式按 3 次

**下行命令包**：Mode=0x00，Settling Time 和 Cycle Time 设为 0。

**上行数据包**：与标准协议格式兼容，但 slot、cycle_end 固定为 0。

| 字段 | 长度 | 说明 |
|------|------|------|
| Header | 2 bytes | 包头，固定 `0xAA55` |
| Version | 1 byte | 协议版本，当前 `0x01` |
| cycle_id | 2 bytes | 固定 `0x0000` |
| slot | 1 byte | 固定 `0x00` |
| Bitmap | 2 bytes | bit0=传感器1，bit11=传感器12 |
| Timestamp | 8 bytes | 采集时刻，单位微秒（μs），STM32 上电后计时 |
| sensor_data | N×7 bytes | 每传感器: SensorID(1 byte) + X(2 bytes) + Y(2 bytes) + Z(2 bytes) |
| cycle_end | 1 byte | 固定 `0x00` |
| slot_end | 2 bytes | 数据包结束标志位 `0x0D0A` |

**示例：** bitmap=0x0FFF（12 传感器），手动模式

```
AA55 01 0000 00 0FFF 00000000000000FA [01 x y z] ... [12 x y z] 00
```

- Header: `AA55`
- Version: `01`
- cycle_id: `0000`（固定）
- slot: `00`（固定）
- Bitmap: `0FFF`（12 传感器）
- Timestamp: `00000000000000FA` = 250 μs（示例值）
- sensor_data: 传感器 1-12 的 xyz 数据
- cycle_end: `00`（固定）
- slot_end: `0D0A`

### JSON 文件格式

```json
{
  "header": {
    "cycle_id": 0,
    "mode": "CVT",
    "num_slots": 4
  },
  "stm_timestamp": 12345678,
  "pc_timestamp": 1234567890.123,
  "slot_data": [
    {
      "slot": 0,
      "pose": {
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
      },
      "sensor_data": [
        {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
        {"id": 2, "x": 0.0, "y": 0.0, "z": 0.0}
      ]
    }
  ],
  "ground_truth_pose": {
    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
  }
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| header.cycle_id | int | cycle 编号 |
| header.mode | string | `"CVT"`=恒压模式，`"CCI"`=恒流模式（手动模式也使用CVT/CCI） |
| header.num_slots | int | 本 cycle 的 slot 总数（CVT=4, CCI=3） |
| stm_timestamp | int | 本 cycle 最后一个 slot 的 STM32 时刻，单位微秒 |
| pc_timestamp | float | 本 cycle 最后一个 slot 的 PC 接收时刻，单位秒 |
| slot_data[].slot | int | slot 序号 |
| slot_data[].pose | object | 该 slot 对应的机械臂末端位姿（CVT slot=3 时此项不存在） |
| slot_data[].pose.position | object | xyz，单位米 |
| slot_data[].pose.rotation | object | 四元数 xyzw |
| slot_data[].sensor_data | array | 该 slot 的传感器采样数据 |
| slot_data[].sensor_data[].id | int | 传感器编号 1-12 |
| slot_data[].sensor_data[].x/y/z | float | 磁场三分量原始值 |
| ground_truth_pose | object | sensor_array_filt 相对于 lab_table 的位姿，作为真值参考 |

### 存储方式

每完成一个 cycle，将上述 JSON 写入 `output_dir/cycle_{cycle_id:04d}.json`。EKF 算法服务可按需读取这些文件进行位姿估计。


