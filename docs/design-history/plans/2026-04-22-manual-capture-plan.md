# Manual Capture Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add manual capture capability to `serial_node_tdm` — subscribe to trigger topic, collect 10 samples, average, save to CSV.

**Architecture:** Add a `ManualCaptureModule` class to `serial_node_tdm.py` that subscribes to both the trigger topic and `stm_uplink`. On trigger, buffer next 10 samples, compute per-sensor averages, append to CSV.

**Tech Stack:** ROS Python (rospy), `std_msgs`, CSV, threading.Lock for buffer safety.

---

## File Map

| File | Action |
|------|--------|
| `src/serial_processor/scripts/serial_node_tdm.py` | Modify — add `ManualCaptureModule` class and integrate |
| `src/serial_processor/launch/stm32_manual.launch` | Modify — add `capture_output_filename` param |
| `src/serial_processor/data/` | Create — output directory for CSV files |

---

## Task 1: Add ManualCaptureModule class to serial_node_tdm.py

**Files:**
- Modify: `src/serial_processor/scripts/serial_node_tdm.py`

- [ ] **Step 1: Add imports**

In `serial_node_tdm.py`, add after the existing imports:

```python
import csv
import os
import threading
from std_msgs.msg import Empty
```

- [ ] **Step 2: Add ManualCaptureModule class before SerialNodeTDM class**

```python
class ManualCaptureModule:
    """Subscribes to trigger topic, collects N samples of corrected stm_uplink, averages, saves to CSV."""

    SENSOR_COUNT = 12

    def __init__(self, output_filename, sample_count=10):
        """
        Args:
            output_filename: CSV filename (not full path). Saved to serial_processor/data/.
            sample_count: Number of samples to average (default 10).
        """
        self._sample_count = sample_count
        self._buffer = []  # list of stm_uplink messages
        self._lock = threading.Lock()
        self._capturing = False

        # Resolve output path: ~/embedded_array_ws/src/serial_processor/data/<filename>
        ws_root = os.path.expanduser('~/embedded_array_ws/src/serial_processor')
        self._data_dir = os.path.join(ws_root, 'data')
        os.makedirs(self._data_dir, exist_ok=True)
        self._output_path = os.path.join(self._data_dir, output_filename)

        # Ensure CSV header exists
        if not os.path.exists(self._output_path):
            header = ['timestamp'] + [
                f'sensor_{i}_x,sensor_{i}_y,sensor_{i}_z'
                for i in range(1, self.SENSOR_COUNT + 1)
            ]
            with open(self._output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(','.join(header).split(','))  # split to flatten

        # Subscribers
        self._trigger_sub = rospy.Subscriber(
            '/manual_capture/trigger', Empty, self._on_trigger
        )
        self._uplink_sub = rospy.Subscriber(
            'stm_uplink', StmUplink, self._on_uplink
        )

        rospy.loginfo(
            f"ManualCaptureModule: output={self._output_path}, samples={sample_count}"
        )

    def _on_trigger(self, _msg):
        """Reset buffer and start collecting."""
        with self._lock:
            self._buffer = []
            self._capturing = True
        rospy.loginfo("ManualCapture: trigger received, collecting samples...")

    def _on_uplink(self, msg):
        """Buffer messages while capturing."""
        with self._lock:
            if not self._capturing:
                return
            self._buffer.append(msg)
            if len(self._buffer) >= self._sample_count:
                self._write_average()
                self._capturing = False
                self._buffer = []

    def _write_average(self):
        """Compute per-sensor averages and append to CSV."""
        with self._lock:
            if not self._buffer:
                return
            # buffer: list of StmUplink, each has sensor_data[0..11]
            # Build per-sensor accumulators
            sums = [[0.0, 0.0, 0.0] for _ in range(self.SENSOR_COUNT)]
            for uplink_msg in self._buffer:
                for idx, sd in enumerate(uplink_msg.sensor_data):
                    sums[idx][0] += sd.x
                    sums[idx][1] += sd.y
                    sums[idx][2] += sd.z
            avg = [
                [s[0] / len(self._buffer), s[1] / len(self._buffer), s[2] / len(self._buffer)]
                for s in sums
            ]
            # Timestamp
            from datetime import datetime
            ts = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            row = [ts] + [coord for sensor_avg in avg for coord in sensor_avg]
            with open(self._output_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            rospy.loginfo(f"ManualCapture: wrote {len(self._buffer)} samples to {self._output_path}")
```

- [ ] **Step 3: Integrate ManualCaptureModule into SerialNodeTDM.__init__**

After the existing subscriber line (`self.sub = rospy.Subscriber(...)`), add:

```python
        # Manual capture module
        capture_filename = rospy.get_param('~capture_output_filename', 'manual_capture.csv')
        self._capture = ManualCaptureModule(capture_filename, sample_count=10)
```

- [ ] **Step 4: Run catkin build to verify no import/syntax errors**

Run: `cd /home/zhang/embedded_array_ws && catkin build serial_processor`
Expected: BUILD SUCCESSFUL

- [ ] **Step 5: Commit**

```bash
git add src/serial_processor/scripts/serial_node_tdm.py
git commit -m "feat(serial_node_tdm): add ManualCaptureModule for triggered CSV logging"
```

---

## Task 2: Add capture_output_filename parameter to stm32_manual.launch

**Files:**
- Modify: `src/serial_processor/launch/stm32_manual.launch`

- [ ] **Step 1: Add argument declaration**

In `stm32_manual.launch`, add after the existing `<arg>` block:

```xml
  <arg name="capture_output_filename" default="manual_capture.csv"/>
```

- [ ] **Step 2: Pass parameter to serial_node_tdm node**

In the `<node>` for `serial_node_tdm`, add inside the node tag (before `</node>`):

```xml
    <param name="capture_output_filename" value="$(arg capture_output_filename)"/>
```

- [ ] **Step 3: Commit**

```bash
git add src/serial_processor/launch/stm32_manual.launch
git commit -m "feat(stm32_manual): add capture_output_filename parameter"
```

---

## Verification

After both tasks complete, verify:

1. `roslaunch serial_processor stm32_manual.launch` starts without error
2. `rostopic pub /manual_capture/trigger std_msgs/Empty -1` triggers collection
3. CSV appears at `~/embedded_array_ws/src/serial_processor/data/manual_capture.csv`

---

## Spec Coverage Check

| Spec Requirement | Task |
|-----------------|------|
| Subscribe `/manual_capture/trigger` (Empty) | Task 1 Step 2 (_trigger_sub) |
| Collect 10 samples | Task 1 Step 2 (sample_count=10) |
| Use stm_uplink (corrected) | Task 1 Step 2 (_uplink_sub) |
| Average per-sensor x/y/z | Task 1 Step 2 (_write_average) |
| Append to CSV with timestamp | Task 1 Step 2 (_write_average) |
| Output filename as launch param | Task 2 |
| Output path: serial_processor/data/ | Task 1 Step 2 (_data_dir) |
