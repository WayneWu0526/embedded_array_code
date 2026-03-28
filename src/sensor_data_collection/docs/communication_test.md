# 传感器阵列通信测试文档

## 概述

本文档用于验证 PC → STM32 下行命令（12字节协议）和 STM32 → PC 上行数据传输是否正常工作。

## 前置条件

1. ROS Noetic 工作空间已构建
2. STM32 设备已连接（确认 `/dev/ttyACM0` 存在且权限正确）
3. 已配置 STM32 参数（settling_time, cycle_time 等）

## 测试步骤

---

### 步骤1: 启动 roscore

```bash
cd /home/zhang/embedded_array_ws
source devel/setup.bash
roscore
```

---

### 步骤2: 启动 serial_node_tdm.py

serial_node_tdm.py 负责：
- 监听 `/stm_downlink` 主题，接收下行命令并发送到 STM32
- 从 STM32 接收上行数据，发布到 `/stm_uplink` 主题

```bash
# 需要先 source zlab_robots (signal_generator dependency)
source ~/zlab_robots/devel/setup.bash
source /home/zhang/embedded_array_ws/devel/setup.bash

rosrun serial_processor serial_node_tdm.py \
    port:=/dev/ttyACM0 \
    baudrate:=921600 \
    _output:=screen
```

**验证点:** 应看到类似输出：
```
Connecting to /dev/ttyACM0 at 921600...
Serial connected successfully
Waiting for STM32 initialization...
```

---

### 步骤3: 手动发送下行命令

手动发布一个 StmDownlink 消息来测试下行通道：

```bash
source /home/zhang/embedded_array_ws/devel/setup.bash

rostopic pub /stm_downlink serial_processor/StmDownlink "{mode: 1, bitmap: 4095, settling_time: 6000, cycle_time: 28800}" -1
```

**参数说明:**
| 参数 | 值 | 含义 |
|------|-----|------|
| mode | 1 | CVT 模式 (恒压，4 slots) |
| bitmap | 4095 (0x0FFF) | 全部 12 个传感器 |
| settling_time | 6000 | 60.00 ms (0.01ms 单位) |
| cycle_time | 28800 | 288.00 ms (0.01ms 单位) |

**验证点 (步骤2终端):** 应看到类似输出：
```
Sent downlink: aa5501000fff17703070000000
```

**字节分解:**
```
aa55 01 00 0fff 1770 00007080
|----|--|--|----|----|-------|
Hdr  Ver Mode Bmp  Settling CycleTime(28800=0x7080)
```

#### STM32 初始化回复

发送下行命令后，STM32 会返回一个 3 字节的初始化回复：

```
| 字节偏移 | 大小 | 内容 |
|---------|------|------|
| 0 | 1 | 0xAA (Header 高字节) |
| 1 | 1 | 0x55 (Header 低字节) |
| 2 | 1 | Status 状态码 |

**Status 状态码含义:**

| Status | 含义 | serial_node_tdm.py 输出 |
|--------|------|------------------------|
| 0x00 | 成功 | `[INFO] STM32 initialized successfully` |
| 0x01 | 参数错误 | `[WARN] STM32: Parameter error in downlink command` |
| 0x02 | 传感器位图无效 | `[WARN] STM32: Invalid sensor bitmap` |
| 其他 | 未知错误 | `[WARN] STM32: Unknown error (status=0xXX)` |
| 无回复 | STM32 无响应 | `[WARN] STM32: No initialization reply received` |

**完整流程图:**

```
PC                              STM32
 |                                |
 |  发送 12 字节下行命令           |
 |------------------------------->|
 |                                |
 |  <--- 3 字节初始化回复 ---      |
 |       (Header + Status)       |
 |<-------------------------------|
 |                                |
 |  判断 Status 显示日志          |
 |  (成功/参数错误/无响应等)       |
```

**注意:** 后续的周期性 slot 数据（sensor_data）是在 `run()` 循环中持续接收并发布到 `/stm_uplink` topic 的，不是在下行命令回调中处理的。

---

### 步骤4: 监听上行数据

监听 `/stm_uplink` 主题，检查 STM32 是否正常响应：

```bash
source /home/zhang/embedded_array_ws/devel/setup.bash
rostopic echo /stm_uplink
```

**正常响应应包含:**
```yaml
cycle_id: 0        # 递增
slot: 0            # 0 → 1 → 2 → 3 (CVT模式)
bitmap: 4095
timestamp: 1234567890123  # 微秒
sensor_data:       # 12个传感器数据
  - {id: 1, x: 0.123, y: 0.456, z: 0.789}
  - ...
  - {id: 12, ...}
cycle_end: 0       # slot 3 时为 1
```

---

### 步骤5: 完整集成测试

使用 launch 文件启动完整系统：

```bash
source ~/zlab_robots/devel/setup.bash
source /home/zhang/embedded_array_ws/devel/setup.bash

roslaunch sensor_data_collection data_collection.launch
```

**预期日志输出:**

1. **serial_node_tdm.py 终端:**
   ```
   Sent downlink: aa5501000fff17703070000000
   ```

2. **data_collection_node.py 终端:**
   ```
   Received: cycle_id=0, slot=0, cycle_end=0, sensors=12
   Received: cycle_id=0, slot=1, cycle_end=0, sensors=12
   Received: cycle_id=0, slot=2, cycle_end=0, sensors=12
   Received: cycle_id=0, slot=3, cycle_end=1, sensors=12
   Saved cycle 0 to .../cycle_0000.json
   ```

---

### 步骤6: 检查输出文件

```bash
# 查看生成的文件
ls ~/sensor_data/cycle_*.json

# 查看 JSON 结构
cat ~/sensor_data/cycle_0000.json
```

**预期 JSON 结构:**
```json
{
  "header": {
    "cycle_id": 0,
    "mode": "CVT",
    "num_slots": 4
  },
  "stm_timestamp": 1234567890123,
  "pc_timestamp": 1234567895.123,
  "slot_data": [
    {
      "slot": 0,
      "sensor_data": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0}, ...],
      "pose": {"position": {"x": 0.1, "y": 0.2, "z": 0.3}, "rotation": {...}}
    },
    {"slot": 1, "sensor_data": [...], "pose": {...}},
    {"slot": 2, "sensor_data": [...], "pose": {...}},
    {"slot": 3, "sensor_data": [...]}
  ],
  "ground_truth_pose": {"position": {...}, "rotation": {...}}
}
```

---

## 验证清单

- [ ] 下行发送日志显示正确的 12 字节 hex
- [ ] 上行数据 topic 有数据输出
- [ ] cycle 完成时生成 JSON 文件
- [ ] JSON 包含所有 slot 的 sensor_data
- [ ] JSON 包含 pose 数据（TF lookup 成功）

---

## 常见问题

### 问题1: `port: /dev/ttyACM0` 不存在

```bash
# 检查设备
ls -l /dev/ttyACM* /dev/ttyUSB*

# 添加用户到 dialout 组
sudo usermod -a -G dialout $USER
# 然后重新登录
```

### 问题2: 上行数据为空

1. 检查 STM32 是否正常工作
2. 检查波特率是否匹配 (921600)
3. 检查 STM32 是否收到了下行命令

### 问题3: JSON 文件未生成

1. 检查 `is_complete()` 逻辑 - 所有 slots 必须都收到
2. 检查 `cycle_end` 标志是否正确设置
3. 查看 data_collection_node 日志是否有错误

---

## 相关文件

| 文件 | 路径 | 说明 |
|------|------|------|
| serial_node_tdm.py | `src/serial_processor/scripts/` | TDM 协议实现 |
| data_collection_node.py | `src/sensor_data_collection/scripts/` | 数据采集节点 |
| StmDownlink.msg | `src/serial_processor/msg/` | 下行消息定义 |
| StmUplink.msg | `src/serial_processor/msg/` | 上行消息定义 |
| data_collection.launch | `src/sensor_data_collection/launch/` | 启动文件 |

---

## 协议参考

### 下行命令包 (PC → STM32): 12 bytes

| 偏移 | 大小 | 字段 | 类型 | 说明 |
|-----|-----|------|------|------|
| 0 | 2 | Header | uint16 | 0xAA55 |
| 2 | 1 | Version | uint8 | 0x01 |
| 3 | 1 | Mode | uint8 | 0x01=CVT, 0x02=CCI |
| 4 | 2 | Bitmap | uint16 | bit0=sensor1, bit11=sensor12 |
| 6 | 2 | SettlingTime | uint16 | 0.01ms 单位 |
| 8 | 4 | CycleTime | uint32 | 0.01ms 单位, 最大 10000.00ms |

### 上行数据 (STM32 → PC): Variable

每 slot 发送一次，最后一个 slot 的 `cycle_end=1`
