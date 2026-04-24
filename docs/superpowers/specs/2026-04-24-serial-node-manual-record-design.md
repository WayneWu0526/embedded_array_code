# SerialNodeTDM Manual Record Feature Design

## Overview

Add a manual recording capability to `SerialNodeTDM` that saves raw Hall sensor data to CSV on demand, controlled via a ROS topic trigger.

## Functionality

### Trigger Control
- **Topic:** `~record_trigger` (`std_msgs/Bool`)
- **Behavior:**
  - `True` → Start/continue recording to CSV
  - `False` → Pause recording (file remains open, can resume)

### Data Source
- Subscribe to `stm_uplink_raw` (published by this node itself)
- Use raw data before any correction (ellipsoid, R_CORR, consistency)

### Averaging
- Buffer 10 consecutive `stm_uplink_raw` messages
- Compute per-sensor arithmetic mean of x, y, z
- Write one averaged row per buffer full

### Output Format

**CSV File:** `{output_dir}/manual_record_{timestamp}.csv` (default: `~/sensor_data_collection/data/`)

**Columns (wide format, 37 columns + header):**

| Column | Description |
|--------|-------------|
| `timestamp` | STM32 timestamp (uint64, microseconds or as-is from uplink) |
| `sensor1_x`, `sensor1_y`, `sensor1_z` | Sensor 1 average (Gs, 6 decimal places) |
| `sensor2_x`, `sensor2_y`, `sensor2_z` | Sensor 2 average |
| ... | ... |
| `sensor12_x`, `sensor12_y`, `sensor12_z` | Sensor 12 average |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `~output_dir` | string | `~/sensor_data_collection/data` | Directory for CSV output |
| `~frames_to_average` | int | 10 | Number of frames to average per output row |

### State Machine

```
IDLE (file closed)
  │ trigger(True)
  ▼
RECORDING (file open, appending)
  │ trigger(False)
  ▼
PAUSED (file open, not writing)
  │ trigger(True)
  ▼
RECORDING (continue appending)
```

- On `rospy.shutdown()`: close file safely
- File is created/opened on first `True`, kept open through `False`, closed on shutdown

### ROS API Summary

| Element | Type | Purpose |
|---------|------|---------|
| `~record_trigger` | Subscriber `std_msgs/Bool` | Start/stop/pause recording |
| `stm_uplink_raw` | Subscriber `StmUplink` | Source of raw sensor data |
| `stm_manual_record/status` | Publisher `std_msgs/String` | Current state: `idle`, `recording`, `paused` |
| `~output_dir` | Parameter | CSV output directory |
| `~frames_to_average` | Parameter | Frames to average per row (default 10) |

## Implementation Notes

- File write uses standard Python file I/O with buffering
- No external CSV library needed (manual string formatting with `f"{value:.6f}"`)
- Thread safety: buffer access protected by existing lock or simple deque operations (Python GIL)
- If `trigger(False)` received while buffering less than 10 frames, discard incomplete buffer
- Directory is created if it does not exist (`os.makedirs(output_dir, exist_ok=True)`)

## File Changes

- Modify: `src/serial_processor/scripts/serial_node_tdm.py`
