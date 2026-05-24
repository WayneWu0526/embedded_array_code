# Serial Node TDM

TDM (Time Division Multiplexing) 串口节点，负责与 STM32 通信并发布 raw 与标定后的传感器数据。

## 话题

### 发布 (Publishers)

| Topic | Type | Description |
|-------|------|-------------|
| `stm_uplink` | StmUplink | 标定后的数据 (`stm_uplink_raw` + affine D_i/e_i) |
| `stm_uplink_raw` | StmUplink | raw 数据：STM32 ADU/bit -> Gs，并已完成 R_CORR 方位统一，未做 calibration |
| `stm_magnitude` | Float32MultiArray | 标定后的磁场模长 |
| `stm_magnitude_raw` | Float32MultiArray | raw 磁场模长 |

### 订阅 (Subscribers)

| Topic | Type | Description |
|-------|------|-------------|
| `stm_downlink` | StmDownlink | 下行命令 (模式、位图、时序参数) |

## 数据校正流程

```
b_stm (STM32 ADU/bit)
    ↓
1. 单位换算: b_gs = b_stm × adu_to_gs
    ↓
2. R_CORR 方位统一: b_raw = R_CORR[sid] × b_gs
    ↓
stm_uplink_raw 发布
    ↓
3. affine calibration: b_cal = D_i × b_raw + e_i
    ↓
stm_uplink 发布
```

## 参数文件

| 参数 | 来源 | 用途 |
|------|------|------|
| `adu_to_gs` | sensor manifest | STM32 ADU/bit 到 Gs 的单位换算 |
| `R_CORR` | `sensor_array_params.json` | 传感器安装方位统一，属于 raw 数据构造步骤 |
| `D_i`, `e_i` | `affine_model_params.json` | affine calibration |

## 启动

```bash
rosrun serial_processor serial_node_tdm.py _port:=/dev/ttyACM0 _sensor_type:=QMC6309
```

## 二进制协议

### 下行 (PC → STM32): 13 bytes
```
Header(2) + Version(1) + Mode(1) + Bitmap(2) + SettlingTime(2) + CycleTime(4) + CycleNum(1)
```

### 上行 (STM32 → PC): Variable
```
Header(2) + Version(1) + cycle_id(2) + slot(1) + Bitmap(2) + Timestamp(8) + sensor_data(N×7) + cycle_end(1)
```
