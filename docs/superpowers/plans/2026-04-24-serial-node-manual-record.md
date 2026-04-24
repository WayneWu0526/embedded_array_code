# SerialNodeTDM Manual Record Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a manual recording capability to `SerialNodeTDM` — triggered via ROS topic, saves averaged raw Hall sensor data to CSV.

**Architecture:** Extend `SerialNodeTDM` with a `ManualRecorder` inner class handling state machine, 10-frame averaging, and CSV output. Self-subscribe to `stm_uplink_raw` (internally published).

**Tech Stack:** Python 3, ROS Noetic, std_msgs, serial_processor.msg.StmUplink

---

## File Changes Summary

- **Modify:** `src/serial_processor/scripts/serial_node_tdm.py`

---

### Task 1: Add parameters and imports

**Files:**
- Modify: `src/serial_processor/scripts/serial_node_tdm.py`

- [ ] **Step 1: Add new imports**

After the existing imports (line 13-24), add:

```python
from std_msgs.msg import Bool, String
from datetime import datetime
```

- [ ] **Step 2: Add new ROS parameters in `__init__`**

After line 48 (after `rospy.loginfo` for sensor type), add:

```python
# Manual record parameters
self.output_dir = os.path.expanduser(rospy.get_param('~output_dir', '~/sensor_data'))
self.frames_to_average = int(rospy.get_param('~frames_to_average', 10))
os.makedirs(self.output_dir, exist_ok=True)
rospy.loginfo(f"Manual record output directory: {self.output_dir}")
```

- [ ] **Step 3: Commit**

```bash
git add src/serial_processor/scripts/serial_node_tdm.py
git commit -m "feat(serial_node_tdm): add manual record params and imports"
```

---

### Task 2: Add ManualRecorder class

**Files:**
- Modify: `src/serial_processor/scripts/serial_node_tdm.py`

- [ ] **Step 1: Add ManualRecorder class before SerialNodeTDM class (after imports, before line 27)**

```python
class ManualRecorder:
    """Records raw stm_uplink data to CSV on trigger."""

    STATE_IDLE = 'idle'
    STATE_RECORDING = 'recording'
    STATE_PAUSED = 'paused'

    def __init__(self, output_dir, frames_to_average, n_sensors=12):
        self.output_dir = output_dir
        self.frames_to_average = frames_to_average
        self.n_sensors = n_sensors
        self._state = self.STATE_IDLE
        self._file = None
        self._buffer = []  # list of StmUplink messages
        self._csv_path = None

    @property
    def state(self):
        return self._state

    def _write_header(self, f):
        header = ['timestamp']
        for i in range(1, self.n_sensors + 1):
            header.extend([f'sensor{i}_x', f'sensor{i}_y', f'sensor{i}_z'])
        f.write(','.join(header) + '\n')

    def _average_buffer(self):
        """Compute per-sensor average from buffered StmUplink messages."""
        if not self._buffer:
            return None
        # Use first message to determine sensor order
        first = self._buffer[0]
        n = len(self._buffer)
        # Collect per-sensor sums
        sum_x = [0.0] * self.n_sensors
        sum_y = [0.0] * self.n_sensors
        sum_z = [0.0] * self.n_sensors
        for msg in self._buffer:
            for idx, s in enumerate(msg.sensor_data):
                sum_x[idx] += s.x
                sum_y[idx] += s.y
                sum_z[idx] += s.z
        avg = [0.0] * self.n_sensors
        for i in range(self.n_sensors):
            avg[i] = (sum_x[i] / n, sum_y[i] / n, sum_z[i] / n)
        # Get timestamp from last message (most recent)
        timestamp = self._buffer[-1].timestamp
        self._buffer.clear()
        return timestamp, avg

    def trigger(self, enable):
        """Handle trigger message."""
        if enable:
            self._start_recording()
        else:
            self._pause_recording()

    def _start_recording(self):
        if self._state == self.STATE_IDLE:
            # Open new file
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._csv_path = os.path.join(self.output_dir, f'manual_record_{ts}.csv')
            self._file = open(self._csv_path, 'w')
            self._write_header(self._file)
            rospy.loginfo(f"[ManualRecorder] Recording started: {self._csv_path}")
        # Regardless of state, go to recording
        self._state = self.STATE_RECORDING

    def _pause_recording(self):
        if self._state == self.STATE_RECORDING:
            # Flush any remaining incomplete buffer
            self._buffer.clear()
            self._state = self.STATE_PAUSED
            rospy.loginfo("[ManualRecorder] Recording paused")

    def on_uplink_raw(self, msg):
        """Process incoming stm_uplink_raw message. Returns True if row was written."""
        if self._state != self.STATE_RECORDING:
            return False
        self._buffer.append(msg)
        if len(self._buffer) >= self.frames_to_average:
            result = self._average_buffer()
            if result is None:
                return False
            timestamp, avg = result
            # Write CSV row
            row = [str(timestamp)]
            for (x, y, z) in avg:
                row.extend([f'{x:.6f}', f'{y:.6f}', f'{z:.6f}'])
            self._file.write(','.join(row) + '\n')
            return True
        return False

    def flush_and_close(self):
        """Flush and close file. Called on shutdown."""
        if self._file:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._file = None
        self._buffer.clear()
        self._state = self.STATE_IDLE
        rospy.loginfo("[ManualRecorder] File closed")
```

- [ ] **Step 2: Commit**

```bash
git add src/serial_processor/scripts/serial_node_tdm.py
git commit -m "feat(serial_node_tdm): add ManualRecorder class"
```

---

### Task 3: Wire ManualRecorder into SerialNodeTDM

**Files:**
- Modify: `src/serial_processor/scripts/serial_node_tdm.py`

- [ ] **Step 1: Initialize ManualRecorder in SerialNodeTDM.__init__**

After line 70 (after `self._load_sensor_array_params()` and before subscriber setup):

```python
# Initialize manual recorder
self.recorder = ManualRecorder(self.output_dir, self.frames_to_average)
# Subscribe to own stm_uplink_raw for recording (self-subscribe)
self.sub_record = rospy.Subscriber('stm_uplink_raw', StmUplink, self._on_uplink_raw_record)
# Subscribe to trigger
self.sub_trigger = rospy.Subscriber('~record_trigger', Bool, self._on_record_trigger)
# Status publisher
self.pub_status = rospy.Publisher('stm_manual_record/status', String, queue_size=10)
```

- [ ] **Step 2: Add handler callbacks**

Add these two methods to SerialNodeTDM class (e.g., after `_load_sensor_array_params`):

```python
def _on_record_trigger(self, msg):
    """Handle record trigger (Bool)."""
    enable = msg.data
    self.recorder.trigger(enable)
    self.pub_status.publish(String(data=self.recorder.state))

def _on_uplink_raw_record(self, msg):
    """Process stm_uplink_raw for recording."""
    self.recorder.on_uplink_raw(msg)
```

- [ ] **Step 3: Register shutdown handler**

After line 75 (after rospy.loginfo in `__init__`), add:

```python
# Register shutdown handler
rospy.on_shutdown(self._on_shutdown_record)
```

Add the method to SerialNodeTDM (e.g., after `_load_sensor_array_params`):

```python
def _on_shutdown_record(self):
    self.recorder.flush_and_close()
```

- [ ] **Step 4: Run test manually — verify it compiles and basic wiring works**

Run: `python3 -c "from serial_processor.scripts.serial_node_tdm import SerialNodeTDM, ManualRecorder; print('OK')"`
Expected: No import errors

```bash
git add src/serial_processor/scripts/serial_node_tdm.py
git commit -m "feat(serial_node_tdm): wire ManualRecorder into SerialNodeTDM"
```

---

### Task 4: Verify full integration and spec compliance

**Files:**
- Review: `src/serial_processor/scripts/serial_node_tdm.py`

- [ ] **Step 1: Read full modified file, verify all spec requirements are met**

Checklist:
- [ ] `~output_dir` param — line should exist
- [ ] `~frames_to_average` param — line should exist
- [ ] `~record_trigger` subscriber — line should exist
- [ ] `stm_uplink_raw` subscriber (internal) — line should exist
- [ ] `stm_manual_record/status` publisher — line should exist
- [ ] State machine: idle → recording → paused → recording
- [ ] 10-frame averaging
- [ ] CSV wide format: timestamp + 12 sensors × (x,y,z) = 37 columns
- [ ] 6 decimal places for float values
- [ ] Gs unit (values from `stm_uplink_raw` are already in Gs via `_adu_to_gs`)
- [ ] File created on first True, kept open through False, closed on shutdown
- [ ] os.makedirs for output_dir

- [ ] **Step 2: Commit**

```bash
git add src/serial_processor/scripts/serial_node_tdm.py
git commit -m "feat(serial_node_tdm): complete manual record integration"
```

---

### Task 5: Update launch file (optional documentation)

**Files:**
- Modify: `src/sensor_data_collection/launch/stm32_manual.launch`

Add to the launch file after line 17 (after </node>):

```xml
  <!-- Manual record parameters -->
  <param name="output_dir" value="$(env HOME)/sensor_data"/>
  <param name="frames_to_average" value="10"/>
```

```bash
git add src/sensor_data_collection/launch/stm32_manual.launch
git commit -m "docs(stm32_manual): add manual record params documentation"
```
