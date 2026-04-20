# Serial Node TDM

TDM (Time Division Multiplexing) 串口节点，负责与 STM32 通信并发布校正后的传感器数据。

## 话题

### 发布 (Publishers)

| Topic | Type | Description |
|-------|------|-------------|
| `stm_uplink` | StmUplink | 完全校正后的数据 (ellipsoid + R_CORR + consistency) |
| `stm_uplink_raw` | StmUplink | 原始数据 (未经任何校正) |
| `stm_magnitude` | Float32MultiArray | 完全校正后的磁场模长 |
| `stm_magnitude_raw` | Float32MultiArray | 原始磁场模长 |

### 订阅 (Subscribers)

| Topic | Type | Description |
|-------|------|-------------|
| `stm_downlink` | StmDownlink | 下行命令 (模式、位图、时序参数) |

## 数据校正流程

```
b_raw (原始读数)
    ↓
1. 椭球校正: b_corr = C_i × (b_raw - o_i)
    ↓
2. R_CORR 旋转: b_rot = R_CORR[sid] × b_corr
    ↓
3. 一致性校正: b_cons = D_i × b_rot + e_i
    ↓
4. 逆缩放 (amp_factor): b_final = b_cons / amp_factor
    ↓
stm_uplink 发布
```

## 参数文件

| 参数 | 来源 | 用途 |
|------|------|------|
| `o_i`, `C_i` | `intrinsic_params.json` | 椭球校正 |
| `R_CORR` | `sensor_array_params.json` | 传感器框架旋转 |
| `D_i`, `e_i` | `consistency_params.json` | 一致性校正 |
| `amp_factor` | `consistency_params.json` | 逆缩放因子 |

## amp_factor (逆缩放因子)

椭球校正 (C_i 矩阵) 会对测量结果产生各向异性缩放。一致性校准阶段计算的 `amp_factor` 用于恢复原始测量量级：

```
amp_factor = ||b_corr_mean|| / ||b_raw_mean|| (背景条件)
```

最终发布时：`b_final = b_cons / amp_factor`

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
