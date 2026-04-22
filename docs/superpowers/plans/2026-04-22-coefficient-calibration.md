# Coefficient Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a coefficient calibration module that calibrates a single gain coefficient per sensor using positive/negative magnetic field excitations, based on the existing `consistency_calibration` framework.

**Architecture:** Reuses the `consistency_calibration` ROS node structure (FY8300 + STM32 hardware control) but with a simplified workflow: no background measurement, configurable channels, and a new differential-based gain fitting algorithm.

**Tech Stack:** ROS Noetic, Python 3, numpy, csv, rospy, serial_processor (STM32), signal_generator (FY8300)

---

## File Structure

```
src/calibration/
├── scripts/
│   └── coefficient_calibration.py      # Main node (data collection + post-processing)
├── launch/
│   └── coefficient_calibration.launch  # Launch file with all parameters
└── lib/
    └── coefficient_fit.py              # Gain coefficient algorithm (optional, kept minimal)
```

---

### Task 1: Create Launch File

**Files:**
- Create: `src/calibration/launch/coefficient_calibration.launch`

- [ ] **Step 1: Write the launch file**

```xml
<launch>
    <!--
        Coefficient Calibration Launch File

        Calibrates a single gain coefficient per sensor using positive/negative
        magnetic field excitations. Uses differential measurement to cancel background.

        Hardware required:
          - FY8300 signal generator (connected to EM coils)
          - STM32 sensor array

        Usage:
            roslaunch calibration coefficient_calibration.launch

        Example with single channel:
            roslaunch calibration coefficient_calibration.launch \
                ch1_enable:=true ch2_enable:=false ch3_enable:=false \
                ch1_magnitude:=0.010 ch2_magnitude:=0.0 ch3_magnitude:=0.0

        Example with multiple channels:
            roslaunch calibration coefficient_calibration.launch \
                ch1_enable:=true ch2_enable:=true ch3_enable:=true \
                ch1_magnitude:=0.010 ch2_magnitude:=0.010 ch3_magnitude:=0.010
    -->

    <!-- Arguments -->
    <arg name="sensor_type" default="QMC6309"/>

    <!-- Channel enable flags (true/false) -->
    <arg name="ch1_enable" default="false"/>
    <arg name="ch2_enable" default="false"/>
    <arg name="ch3_enable" default="false"/>

    <!-- Theoretical magnetic field magnitude (Tesla) for each channel -->
    <arg name="ch1_magnitude" default="0.0"/>
    <arg name="ch2_magnitude" default="0.0"/>
    <arg name="ch3_magnitude" default="0.0"/>

    <!-- Skip sampling mode: skip data collection and run post-processing only -->
    <arg name="skip_sampling" default="false"/>
    <arg name="skip_csv_dir" default=""/>

    <!-- ===================== -->
    <!-- 1. Signal Generator (FY8300) -->
    <!-- ===================== -->
    <node name="fy8300" pkg="signal_generator" type="fy8300_node" output="screen">
        <rosparam command="load"
                  file="$(find calibration)/config/signal_params_calib.yaml"/>
    </node>

    <!-- ===================== -->
    <!-- 2. Hall Sensor Serial Node (Manual Mode) -->
    <!-- ===================== -->
    <include file="$(find sensor_data_collection)/launch/stm32_manual.launch">
      <arg name="sensor_type" value="$(arg sensor_type)"/>
    </include>

    <!-- ===================== -->
    <!-- 3. Coefficient Calibration Node -->
    <!-- ===================== -->
    <node
        name="coefficient_calibration"
        pkg="calibration"
        type="coefficient_calibration.py"
        output="screen"
        required="False"
    >
        <!-- Sensor type -->
        <param name="sensor_type" value="$(arg sensor_type)"/>

        <!-- Channel enable flags -->
        <param name="ch1_enable" value="$(arg ch1_enable)"/>
        <param name="ch2_enable" value="$(arg ch2_enable)"/>
        <param name="ch3_enable" value="$(arg ch3_enable)"/>

        <!-- Theoretical magnetic field magnitudes (Tesla) -->
        <param name="ch1_magnitude" value="$(arg ch1_magnitude)"/>
        <param name="ch2_magnitude" value="$(arg ch2_magnitude)"/>
        <param name="ch3_magnitude" value="$(arg ch3_magnitude)"/>

        <!-- Timing parameters -->
        <param name="wait_init_time" value="10.0"/>
        <param name="wait_enable_time" value="3.0"/>

        <!-- Collection parameters -->
        <param name="num_groups" value="50"/>
        <param name="samples_per_group" value="10"/>

        <!-- Output directory -->
        <param name="output_dir" value="$(find calibration)/data/coefficient"/>

        <!-- Skip sampling mode -->
        <param name="skip_sampling" value="$(arg skip_sampling)"/>
        <param name="skip_csv_dir" value="$(arg skip_csv_dir)"/>
    </node>

</launch>
```

- [ ] **Step 2: Verify launch file syntax**

Run: `xmllint --noout src/calibration/launch/coefficient_calibration.launch`
Expected: No output (valid XML)

---

### Task 2: Create Coefficient Calibration Node

**Files:**
- Create: `src/calibration/scripts/coefficient_calibration.py`

- [ ] **Step 1: Write the coefficient_calibration.py node**

```python
#!/usr/bin/env python3
"""
Coefficient Calibration Node

Calibrates a single gain coefficient per sensor using differential measurement
with positive/negative magnetic field excitations.

Workflow:
  1. Load FY8300 config (offset=+5V for positive, offset=-5V for negative)
  2. Report parameter settings (which channels enabled, magnitudes)
  3. For each enabled channel:
     - Collect num_groups groups of averaged data with channel ON (positive polarity)
     - Collect num_groups groups of averaged data with channel ON (negative polarity)
  4. Run post-processing to compute gain coefficients
  5. Save results to CSV

Usage:
  roslaunch calibration coefficient_calibration.launch \
      ch1_enable:=true ch2_enable:=false ch3_enable:=false \
      ch1_magnitude:=0.010

Algorithm (per sensor):
  diff = mean_positive - mean_negative           # Cancels background
  diff_magnitude = sqrt(diff_x^2 + diff_y^2 + diff_z^2)
  gain = 2 * theoretical_magnitude / diff_magnitude

  For multiple enabled channels, gains are averaged per sensor.
"""

import os
import sys
import csv
import threading
from pathlib import Path
import numpy as np
import rospy
from datetime import datetime
from std_msgs.msg import Bool, Float32
from serial_processor.msg import StmUplink


class CoefficientCalibration:
    def __init__(self):
        # === Parameters ===
        self.skip_sampling = rospy.get_param('~skip_sampling', False)
        self.wait_init_time = rospy.get_param('~wait_init_time', 1.0)
        self.wait_enable_time = rospy.get_param('~wait_enable_time', 10.0)
        self.num_groups = rospy.get_param('~num_groups', 50)
        self.samples_per_group = rospy.get_param('~samples_per_group', 10)
        self.output_dir = rospy.get_param('~output_dir',
            os.path.expanduser('~/embedded_array_ws/src/calibration/data/coefficient'))
        self._sensor_type = rospy.get_param('~sensor_type', 'QMC6309')
        self.skip_csv_dir = rospy.get_param('~skip_csv_dir', None)

        # Channel enable flags
        self.ch_enable = {
            1: rospy.get_param('~ch1_enable', False),
            2: rospy.get_param('~ch2_enable', False),
            3: rospy.get_param('~ch3_enable', False),
        }

        # Theoretical magnetic field magnitudes (Tesla)
        self.ch_magnitude = {
            1: rospy.get_param('~ch1_magnitude', 0.0),
            2: rospy.get_param('~ch2_magnitude', 0.0),
            3: rospy.get_param('~ch3_magnitude', 0.0),
        }

        # === Report parameter settings ===
        rospy.loginfo("=" * 60)
        rospy.loginfo("COEFFICIENT CALIBRATION")
        rospy.loginfo("=" * 60)
        rospy.loginfo("Channel enable flags:")
        for ch in [1, 2, 3]:
            rospy.loginfo("  CH%d: %s (magnitude=%.6f T)",
                          ch, "ENABLED" if self.ch_enable[ch] else "DISABLED",
                          self.ch_magnitude[ch])
        rospy.loginfo("Collection: %d groups x %d samples each",
                      self.num_groups, self.samples_per_group)
        rospy.loginfo("Output dir: %s", self.output_dir)
        rospy.loginfo("=" * 60)

        # Check that at least one channel is enabled
        if not any(self.ch_enable.values()):
            rospy.logerr("No channels enabled! At least one channel must be enabled.")
            sys.exit(1)

        # Check that enabled channels have non-zero magnitude
        for ch in [1, 2, 3]:
            if self.ch_enable[ch] and self.ch_magnitude[ch] <= 0:
                rospy.logerr("Channel %d is enabled but magnitude is %.6f (must be > 0)", ch, self.ch_magnitude[ch])
                sys.exit(1)

        # State
        self.latest_sensor_data = None
        self.mutex = threading.Lock()
        self.recording = False
        self.sample_buffer = []

        # Skip sampling mode
        if self.skip_sampling:
            self.csv_dir = Path(self.skip_csv_dir) if self.skip_csv_dir else self._get_default_csv_dir()
            if not self.csv_dir.exists():
                rospy.logerr(f"CSV directory not found: {self.csv_dir}")
                sys.exit(1)
            rospy.loginfo(f"Skip sampling mode: processing {self.csv_dir}")
            self._run_postprocessing()
            return

        # Publishers for FY8300
        self.pub_fy8300_ch = [
            rospy.Publisher(f'/fy8300/ch{i}/output_en', Bool, queue_size=1)
            for i in range(1, 4)
        ]
        self.pub_fy8300_offset = [
            rospy.Publisher(f'/fy8300/ch{i}/offset', Float32, queue_size=1)
            for i in range(1, 4)
        ]

        # Subscriber for STM32
        self.sub_stm_uplink = rospy.Subscriber(
            'stm_uplink_raw', StmUplink, self._stm_uplink_callback, queue_size=100)

        # Register shutdown hook
        rospy.on_shutdown(self.cleanup)

        # Wait for connections
        rospy.loginfo("Waiting for connections...")
        rospy.sleep(1.0)
        rospy.loginfo("Ready.")

        # Setup CSV header
        os.makedirs(self.output_dir, exist_ok=True)
        # CSV format: timestamp, sensor_1_x, sensor_1_y, sensor_1_z, ..., sensor_12_x, sensor_12_y, sensor_12_z
        self.csv_header = ['timestamp']
        for i in range(1, 13):
            self.csv_header.extend([f'sensor_{i}_x', f'sensor_{i}_y', f'sensor_{i}_z'])

    def _stm_uplink_callback(self, msg: StmUplink):
        with self.mutex:
            self.latest_sensor_data = msg
            if self.recording:
                self._collect_sample(msg)

    def _collect_sample(self, msg: StmUplink):
        """Collect a single sample into buffer for averaging."""
        sensor_dict = {}
        for sensor in msg.sensor_data:
            if 1 <= sensor.id <= 12:
                sensor_dict[sensor.id] = (sensor.x, sensor.y, sensor.z)
        self.sample_buffer.append(sensor_dict)

    def _compute_averaged_row(self):
        """Compute averaged values from sample buffer."""
        if not self.sample_buffer:
            return None

        avg_data = {}
        for sensor_id in range(1, 13):
            xs = [s[sensor_id][0] for s in self.sample_buffer if sensor_id in s]
            ys = [s[sensor_id][1] for s in self.sample_buffer if sensor_id in s]
            zs = [s[sensor_id][2] for s in self.sample_buffer if sensor_id in s]
            if xs:
                avg_data[sensor_id] = (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
            else:
                avg_data[sensor_id] = (0.0, 0.0, 0.0)

        timestamp = datetime.now().isoformat()
        row = [timestamp]
        for i in range(1, 13):
            x, y, z = avg_data[i]
            row.extend([f"{x:.7g}", f"{y:.7g}", f"{z:.7g}"])
        return row

    def _set_polarity(self, polarity: str):
        """Set FY8300 polarity by changing offset on all channels."""
        offset = 5.0 if polarity == "positive" else -5.0
        rospy.loginfo(f"Setting polarity to {polarity} (offset={offset}V)")
        msg = Float32()
        msg.data = offset
        for pub in self.pub_fy8300_offset:
            pub.publish(msg)

    def _set_channel(self, channel: int, enabled: bool):
        """Enable or disable a FY8300 channel."""
        msg = Bool()
        msg.data = enabled
        self.pub_fy8300_ch[channel - 1].publish(msg)
        state = "ENABLED" if enabled else "DISABLED"
        rospy.loginfo(f"  FY8300 CH{channel} {state}")

    def _collect_data(self, channel: int, polarity: str):
        """Collect num_groups groups of averaged data for a channel/polarity."""
        rospy.loginfo("")
        rospy.loginfo("%s", "=" * 60)
        rospy.loginfo("=== Channel %d | Polarity: %s | Status: RUNNING ===", channel, polarity.upper())
        rospy.loginfo("%s", "=" * 60)

        # Enable channel and wait for hardware to activate
        self._set_channel(channel, True)
        rospy.loginfo("Waiting %.1fs for FY8300 to activate...", self.wait_enable_time)
        rospy.sleep(self.wait_enable_time)

        # Collect data
        rospy.loginfo("Collecting %d groups, %d samples each...", self.num_groups, self.samples_per_group)
        rospy.loginfo("PROGRESS: 0/%d groups (0%%)", self.num_groups)
        self.recording = True
        group_results = []
        total_samples = 0

        for group_idx in range(self.num_groups):
            self.sample_buffer = []
            sample_count = 0

            while len(self.sample_buffer) < self.samples_per_group and not rospy.is_shutdown():
                with self.mutex:
                    if self.latest_sensor_data is not None:
                        sensor_dict = {}
                        for sensor in self.latest_sensor_data.sensor_data:
                            if 1 <= sensor.id <= 12:
                                sensor_dict[sensor.id] = (sensor.x, sensor.y, sensor.z)
                        self.sample_buffer.append(sensor_dict)
                        sample_count += 1
                        self.latest_sensor_data = None
                if not rospy.is_shutdown():
                    rospy.sleep(0.001)

            if rospy.is_shutdown():
                break

            total_samples += sample_count
            row = self._compute_averaged_row()
            if row:
                group_results.append(row)

            if (group_idx + 1) % 5 == 0 or group_idx == self.num_groups - 1:
                pct = (group_idx + 1) * 100.0 / self.num_groups
                rospy.loginfo("PROGRESS: %d/%d groups (%.0f%%) | Samples: %d",
                              group_idx + 1, self.num_groups, pct, total_samples)

        self.recording = False

        # Disable channel and wait
        self._set_channel(channel, False)
        rospy.loginfo("Waiting %.1fs for FY8300 to deactivate...", self.wait_enable_time)
        rospy.sleep(self.wait_enable_time)

        # Save CSV
        if group_results:
            csv_path = os.path.join(self.output_dir, f"coefficient_calib_ch{channel}_{polarity}.csv")
            self._write_csv(csv_path, group_results)
            rospy.loginfo("Wrote %d groups to %s", len(group_results), csv_path)

        rospy.loginfo("=== Channel %d | Polarity: %s | Status: COMPLETE ===", channel, polarity.upper())
        return group_results

    def _write_csv(self, csv_path, rows):
        """Write rows to CSV file."""
        if not rows:
            return
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_header)
                writer.writerows(rows)
            rospy.loginfo(f"Wrote {len(rows)} row(s) to CSV")
        except Exception as e:
            rospy.logerr(f"Failed to write CSV: {e}")

    def _get_default_csv_dir(self) -> Path:
        """Get default CSV directory."""
        return Path(os.path.expanduser('~/embedded_array_ws/src/calibration/data/coefficient'))

    def _load_csv_data(self, csv_path: Path):
        """Load CSV and return (N, 12, 3) numpy array."""
        import pandas as pd
        df = pd.read_csv(csv_path)
        cols = [c for c in df.columns if c.startswith('sensor_')]
        data = df[cols].values
        n = data.shape[0]
        reshaped = np.zeros((n, 12, 3))
        for i in range(12):
            reshaped[:, i, :] = data[:, i*3:(i+1)*3]
        return reshaped

    def _compute_stable_mean(self, data: np.ndarray, skip: int = 5) -> np.ndarray:
        """Compute stable mean, skipping first/last skip samples."""
        n = data.shape[0]
        if n == 0:
            return np.zeros((12, 3))
        start = min(skip, max(0, n // 10))
        end = max(start + 1, min(n, n - skip, n * 9 // 10))
        segment = data[start:end]
        if segment.shape[0] == 0:
            return np.zeros((12, 3))
        return segment.mean(axis=0)

    def _run_postprocessing(self):
        """Run post-processing to compute gain coefficients."""
        rospy.loginfo("")
        rospy.loginfo("%s", "=" * 60)
        rospy.loginfo("=== POST-PROCESSING ===")
        rospy.loginfo("%s", "=" * 60)

        # Collect gain estimates per channel, then average
        sensor_gains = {sid: [] for sid in range(1, 13)}

        for ch in [1, 2, 3]:
            if not self.ch_enable[ch]:
                continue

            magnitude = self.ch_magnitude[ch]

            # Load positive and negative CSV data
            pos_path = self.csv_dir / f"coefficient_calib_ch{ch}_positive.csv"
            neg_path = self.csv_dir / f"coefficient_calib_ch{ch}_negative.csv"

            if not pos_path.exists():
                rospy.logerr(f"Positive CSV not found: {pos_path}")
                continue
            if not neg_path.exists():
                rospy.logerr(f"Negative CSV not found: {neg_path}")
                continue

            pos_data = self._load_csv_data(pos_path)
            neg_data = self._load_csv_data(neg_path)

            # Compute stable means
            mean_pos = self._compute_stable_mean(pos_data)  # (12, 3)
            mean_neg = self._compute_stable_mean(neg_data)  # (12, 3)

            # Differential measurement (cancels background)
            diff = mean_pos - mean_neg  # (12, 3)

            # Compute gain per sensor
            rospy.loginfo("")
            rospy.loginfo("Channel %d (magnitude=%.6f T):", ch, magnitude)
            rospy.loginfo(f"  {'Sensor':<8} {'diff_x':>12} {'diff_y':>12} {'diff_z':>12} {'|diff|':>12} {'gain':>12}")
            rospy.loginfo("  " + "-" * 70)

            for sid in range(1, 13):
                dx, dy, dz = diff[sid - 1]
                diff_mag = np.sqrt(dx*dx + dy*dy + dz*dz)

                if diff_mag < 1e-10:
                    rospy.logwarn(f"  Sensor {sid}: diff_magnitude too small ({diff_mag:.2e}), skipping")
                    continue

                gain = 2.0 * magnitude / diff_mag
                sensor_gains[sid].append(gain)

                rospy.loginfo(f"  {sid:<8} {dx:>12.6f} {dy:>12.6f} {dz:>12.6f} {diff_mag:>12.6f} {gain:>12.6f}")

        # Average gains across enabled channels
        rospy.loginfo("")
        rospy.loginfo("%s", "=" * 60)
        rospy.loginfo("FINAL GAIN COEFFICIENTS (averaged across channels)")
        rospy.loginfo("%s", "=" * 60)
        rospy.loginfo(f"  {'Sensor':<8} {'gain':>12} {'n_channels':>12}")
        rospy.loginfo("  " + "-" * 35)

        final_gains = {}
        for sid in range(1, 13):
            if not sensor_gains[sid]:
                rospy.logwarn(f"  Sensor {sid}: no valid gain measurements")
                final_gains[sid] = 1.0  # Default
            else:
                final_gains[sid] = np.mean(sensor_gains[sid])
                rospy.loginfo(f"  {sid:<8} {final_gains[sid]:>12.6f} {len(sensor_gains[sid]):>12}")

        # Save to CSV
        csv_path = os.path.join(self.output_dir, "coefficient_gains.csv")
        self._save_gains_csv(csv_path, final_gains)

        rospy.loginfo("")
        rospy.loginfo("Coefficient calibration complete.")
        rospy.loginfo("Gain coefficients saved to: %s", csv_path)

    def _save_gains_csv(self, csv_path, gains):
        """Save gain coefficients to CSV."""
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sensor_id', 'gain'])
                for sid in range(1, 13):
                    writer.writerow([sid, f"{gains[sid]:.6f}"])
            rospy.loginfo(f"Saved {len(gains)} sensor gains to {csv_path}")
        except Exception as e:
            rospy.logerr(f"Failed to save gains CSV: {e}")

    def cleanup(self):
        """Cleanup on shutdown - run post-processing."""
        rospy.loginfo("Running post-processing...")
        self._run_postprocessing()

    def run(self):
        """Main execution flow."""
        # Initialize polarity to positive
        rospy.loginfo("")
        rospy.loginfo(">>> INITIALIZATION: Setting initial polarity to positive (offset=+5V)...")
        self._set_polarity("positive")
        rospy.loginfo("Waiting %.1fs for FY8300 to initialize...", self.wait_init_time)
        rospy.sleep(self.wait_init_time)

        # Collect data for each enabled channel
        for ch in [1, 2, 3]:
            if not self.ch_enable[ch]:
                continue

            if rospy.is_shutdown():
                break

            # Phase 1: Positive polarity
            self._collect_data(ch, "positive")

            if rospy.is_shutdown():
                break

            # Switch to negative polarity
            rospy.loginfo("")
            rospy.loginfo(">>> Switching to negative polarity (offset=-5V)...")
            self._set_polarity("negative")
            rospy.sleep(self.wait_enable_time)

            # Phase 2: Negative polarity
            self._collect_data(ch, "negative")

            # Switch back to positive for next channel
            if ch < 3 and any(self.ch_enable[c] for c in range(ch+1, 4)):
                rospy.loginfo("")
                rospy.loginfo(">>> Resetting to positive polarity for next channel...")
                self._set_polarity("positive")
                rospy.sleep(self.wait_enable_time)

        # Shutdown all channels
        rospy.loginfo("")
        rospy.loginfo(">>> SHUTDOWN: Disabling all channels...")
        for ch in [1, 2, 3]:
            self._set_channel(ch, False)

        rospy.loginfo("=" * 60)
        rospy.loginfo("DATA COLLECTION COMPLETE")
        rospy.loginfo("=" * 60)
        rospy.loginfo("CSV files saved to: %s", self.output_dir)
        rospy.loginfo("Run post-processing to compute gain coefficients.")
        rospy.loginfo("=" * 60)

        # Run post-processing before shutdown
        rospy.sleep(1.0)  # Small delay to ensure cleanup is called properly
        self._run_postprocessing()

        rospy.signal_shutdown("Calibration complete")


if __name__ == '__main__':
    rospy.init_node('coefficient_calibration', anonymous=True)
    try:
        calib = CoefficientCalibration()
        if not calib.skip_sampling:
            calib.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted.")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        import traceback
        traceback.print_exc()
```

- [ ] **Step 2: Make the script executable**

Run: `chmod +x src/calibration/scripts/coefficient_calibration.py`
Expected: No output

- [ ] **Step 3: Verify Python syntax**

Run: `python3 -m py_compile src/calibration/scripts/coefficient_calibration.py`
Expected: No output (no syntax errors)

---

### Task 3: Verify Launch File and Node

**Files:**
- Modify: `src/calibration/scripts/coefficient_calibration.py` (add missing `from __future__ import annotations` if needed)

- [ ] **Step 1: Verify the launch file and node work together**

Run: `roscd calibration && roslaunch --dry-run coefficient_calibration.launch ch1_enable:=true ch2_enable:=false ch3_enable:=false ch1_magnitude:=0.010 2>&1 | head -50`
Expected: Dry-run output showing parameter values

---

## Self-Review Checklist

**1. Spec coverage:**
- [x] Data acquisition phase reuses consistency_calibration framework (FY8300 + STM32)
- [x] No background measurement
- [x] Channel enable/disable as launch parameters
- [x] Theoretical magnetic field magnitude as launch parameters
- [x] Parameter settings reported at startup
- [x] 50 groups × 10 samples per group
- [x] Two CSV files per enabled channel (positive + negative)
- [x] Post-processing algorithm: diff = mean_pos - mean_neg, gain = 2*magnitude/|diff|
- [x] Gains averaged across enabled channels
- [x] Results saved to CSV

**2. Placeholder scan:**
- No TODOs, no TBDs
- All file paths are absolute or properly constructed with `os.path.join`
- All method signatures have explicit parameter types where needed

**3. Type consistency:**
- `ch_enable` dict: `{int: bool}`
- `ch_magnitude` dict: `{int: float}`
- `sensor_gains` dict: `{int: list}`
- `final_gains` dict: `{int: float}`
- CSV columns: `sensor_id` (int), `gain` (float)

---

## Execution Options

**Plan complete and saved to `docs/superpowers/plans/2026-04-22-coefficient-calibration.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
