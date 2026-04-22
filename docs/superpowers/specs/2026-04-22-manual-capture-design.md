# Manual Capture Module Design

## Overview

Add a manual capture module to `serial_node_tdm` that listens on a trigger topic, collects 10 samples of corrected sensor data, averages them, and saves to CSV.

## New Components

### ROS Topic Subscription
- **Topic**: `/manual_capture/trigger`
- **Message type**: `std_msgs/Empty`
- **Behavior**: Any empty message triggers one capture cycle

### Launch Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `capture_output_filename` | string | `manual_capture.csv` | Output CSV filename |

### CSV Output Format

```
timestamp,sensor_1_x,sensor_1_y,sensor_1_z,...,sensor_12_z
2026-04-22-10-30-00,0.123,-0.456,0.789,...,0.111
```

### Output Path

`~/embedded_array_ws/src/serial_processor/data/<filename.csv>`

### Capture Flow

1. Receive trigger message → reset sample buffer
2. Collect next 10 `stm_uplink` messages (corrected data)
3. Average x/y/z for each of 12 sensors
4. Append timestamped row to CSV file

## Data Source

Use `stm_uplink` topic (fully corrected: ellipsoid + R_CORR + consistency correction).

## Changes

- `src/serial_processor/scripts/serial_node_tdm.py` — add `ManualCaptureModule` class
- `src/serial_processor/launch/stm32_manual.launch` — add `capture_output_filename` parameter
