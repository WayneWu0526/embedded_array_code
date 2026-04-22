#!/usr/bin/env python3
"""
Consistency Calibration Node

Controls FY8300 channels and records STM32 sensor data for consistency calibration.
流程:
  1. Load initial config (signal_params_calib.yaml with offset=+5V)
  2. CH3 ON -> wait -> CH3 OFF -> wait
  3. CH2 ON -> wait -> CH2 OFF -> wait
  4. CH1 ON -> wait -> CH1 OFF -> wait
  5. Switch to negative polarity (offset=-5V via topic)
  6. Repeat steps 2-4
  7. Shutdown

Usage:
  roslaunch calibration consistency_calibration.launch
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
from calibration.consistency_fit.consistency_fit import batch_consistency_fit, validate_consistency


class ConsistencyCalibration:
    def __init__(self):
        # Parameters
        self.skip_sampling = rospy.get_param('~skip_sampling', False)
        self.wait_init_time = rospy.get_param('~wait_init_time', 1.0)
        self.wait_enable_time = rospy.get_param('~wait_enable_time', 10.0)
        self.num_groups = rospy.get_param('~num_groups', 100)
        self.samples_per_group = rospy.get_param('~samples_per_group', 10)
        self.output_dir = rospy.get_param('~output_dir',
            os.path.expanduser('~/embedded_array_ws/src/calibration/data/consistency'))
        self._sensor_type = rospy.get_param('~sensor_type', 'QMC6309')
        self.skip_csv_dir = rospy.get_param('~skip_csv_dir', None)  # For skip_sampling mode

        # Load ellipsoid correction parameters (intrinsic params) for post-processing
        from sensor_array_config.base import get_config
        self._sensor_config = get_config(self._sensor_type)
        self._intrinsic_params = self._sensor_config.intrinsic  # Contains o_i and C_i for each sensor
        # Load R_CORR rotation matrices from hardware params
        self._r_corr = self._build_r_corr_dict()

        # State
        self.latest_sensor_data = None
        self.mutex = threading.Lock()
        self.recording = False
        self.current_polarity = "positive"
        self.current_channel = None
        self.sample_buffer = []  # Buffer for averaging

        # Skip sampling mode: run consistency fit directly on existing CSV data
        if self.skip_sampling:
            self.csv_dir = Path(self.skip_csv_dir) if self.skip_csv_dir else self._get_default_csv_dir()
            if not self.csv_dir.exists():
                rospy.logerr(f"CSV directory not found: {self.csv_dir}")
                sys.exit(1)
            rospy.loginfo(f"Skip sampling mode: processing {self.csv_dir}")
            self._run_consistency_fit()
            # Do not return here; we need the object to stay alive so it can be accessed in main,
            # but we won't initialize hardware or subscribers.
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

        # Subscriber for STM32 (raw data - we apply ellipsoid correction in post-processing)
        self.sub_stm_uplink = rospy.Subscriber(
            'stm_uplink_raw', StmUplink, self._stm_uplink_callback, queue_size=100)

        # Register shutdown hook for cleanup
        rospy.on_shutdown(self.cleanup)

        # Wait for connections
        rospy.loginfo("Waiting for connections...")
        rospy.sleep(1.0)
        rospy.loginfo("Ready.")

        # Setup CSV header (filename set per-channel)
        os.makedirs(self.output_dir, exist_ok=True)
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
        # Build dict of sensor data: {sensor_id: (x, y, z)}
        sensor_dict = {}
        for sensor in msg.sensor_data:
            if 1 <= sensor.id <= 12:
                sensor_dict[sensor.id] = (sensor.x, sensor.y, sensor.z)

        # Store as list of dicts for easy averaging
        self.sample_buffer.append(sensor_dict)

    def _compute_averaged_row(self):
        """Compute averaged values from sample buffer."""
        if not self.sample_buffer:
            return None

        # Average each sensor
        avg_data = {}
        for sensor_id in range(1, 13):
            xs = [s[sensor_id][0] for s in self.sample_buffer if sensor_id in s]
            ys = [s[sensor_id][1] for s in self.sample_buffer if sensor_id in s]
            zs = [s[sensor_id][2] for s in self.sample_buffer if sensor_id in s]
            if xs:
                avg_data[sensor_id] = (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
            else:
                avg_data[sensor_id] = (0.0, 0.0, 0.0)

        # Build row
        timestamp = datetime.now().isoformat()
        row = [timestamp]
        for i in range(1, 13):
            x, y, z = avg_data[i]
            row.extend([f"{x:.7g}", f"{y:.7g}", f"{z:.7g}"])
        return row

    def _set_polarity(self, polarity: str):
        """Set FY8300 polarity by changing offset on all channels.

        Args:
            polarity: "positive" for +5V, "negative" for -5V
        """
        if polarity == "positive":
            offset = 5.0
        elif polarity == "negative":
            offset = -5.0
        else:
            rospy.logwarn(f"Unknown polarity {polarity}, using +5V")
            offset = 5.0

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

    def _collect_background(self, num_groups: int, samples_per_group: int):
        """Collect background data with all channels OFF (no magnetic field excitation)."""
        rospy.loginfo("")
        rospy.loginfo("%s", "=" * 60)
        rospy.loginfo("=== BACKGROUND MEASUREMENT | Status: RUNNING ===")
        rospy.loginfo("%s", "=" * 60)

        # Ensure all channels are OFF
        for ch in [1, 2, 3]:
            self._set_channel(ch, False)

        rospy.loginfo("Waiting %.1fs for all channels to settle...", self.wait_enable_time)
        rospy.sleep(self.wait_enable_time)

        rospy.loginfo("Collecting %d groups, %d samples each...", num_groups, samples_per_group)
        rospy.loginfo("PROGRESS: 0/%d groups (0%%)", num_groups)
        self.recording = True
        group_results = []
        total_samples_collected = 0

        for group_idx in range(num_groups):
            group_samples = []
            sample_count = 0
            while len(group_samples) < samples_per_group and not rospy.is_shutdown():
                with self.mutex:
                    if self.latest_sensor_data is not None:
                        sensor_dict = {}
                        for sensor in self.latest_sensor_data.sensor_data:
                            if 1 <= sensor.id <= 12:
                                sensor_dict[sensor.id] = (sensor.x, sensor.y, sensor.z)
                        group_samples.append(sensor_dict)
                        sample_count += 1
                        self.latest_sensor_data = None

                if not rospy.is_shutdown():
                    rospy.sleep(0.001)

            if rospy.is_shutdown():
                break

            total_samples_collected += sample_count

            if group_samples:
                avg_data = {}
                for sensor_id in range(1, 13):
                    xs = [s[sensor_id][0] for s in group_samples if sensor_id in s]
                    ys = [s[sensor_id][1] for s in group_samples if sensor_id in s]
                    zs = [s[sensor_id][2] for s in group_samples if sensor_id in s]
                    if xs:
                        avg_data[sensor_id] = (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
                    else:
                        avg_data[sensor_id] = (0.0, 0.0, 0.0)

                timestamp = datetime.now().isoformat()
                row = [timestamp]
                for i in range(1, 13):
                    x, y, z = avg_data[i]
                    row.extend([f"{x:.7g}", f"{y:.7g}", f"{z:.7g}"])
                group_results.append(row)

            if (group_idx + 1) % 5 == 0 or group_idx == num_groups - 1:
                pct = (group_idx + 1) * 100.0 / num_groups
                rospy.loginfo("PROGRESS: %d/%d groups (%.0f%%) | Samples: %d",
                              group_idx + 1, num_groups, pct, total_samples_collected)

        self.recording = False
        rospy.loginfo("Collection finished: %d/%d groups, %d samples",
                      len(group_results), num_groups, total_samples_collected)

        # Save to CSV
        if group_results:
            csv_path = os.path.join(self.output_dir, "consistency_calib_background.csv")
            self._write_csv(csv_path, group_results)
            rospy.loginfo("Wrote %d groups to %s", len(group_results), csv_path)
        else:
            rospy.logwarn("No background data to write")

        rospy.loginfo("=== BACKGROUND MEASUREMENT | Status: COMPLETE ===")
        rospy.loginfo("%s", "=" * 60)
        rospy.loginfo("Waiting %.1fs before next step...", self.wait_enable_time)
        rospy.sleep(self.wait_enable_time)


    def _run_channel_sequence(self, channel: int, polarity: str, num_groups: int, samples_per_group: int):
        """Run a single channel: ON -> wait -> collect 100 groups of averaged data -> OFF -> wait -> save.

        Three-phase approach to handle FY8300 command latency:
          Phase 1: Send ON command, wait for hardware to respond
          Phase 2: Collect num_groups groups, each group is avg of samples_per_group samples
          Phase 3: Send OFF command, wait for hardware to confirm
        """
        rospy.loginfo("")
        rospy.loginfo("%s", "=" * 60)
        rospy.loginfo("=== Channel %d | Polarity: %s | Status: RUNNING ===", channel, polarity.upper())
        rospy.loginfo("%s", "=" * 60)

        self.current_channel = channel
        self.current_polarity = polarity

        # Phase 1: Send ON command and wait for FY8300 to physically activate
        rospy.loginfo("[1/3] PHASE 1: Enabling CH%d...", channel)
        self._set_channel(channel, True)
        rospy.loginfo("[1/3] Waiting %.1fs for FY8300 to activate...", self.wait_enable_time)
        rospy.sleep(self.wait_enable_time)

        # Phase 2: Collect num_groups groups of averaged data
        rospy.loginfo("[2/3] PHASE 2: Collecting %d groups, %d samples each...", num_groups, samples_per_group)
        rospy.loginfo("[2/3] PROGRESS: 0/%d groups (0%%)", num_groups)
        self.recording = True
        group_results = []  # List of averaged rows (one per group)
        total_samples_collected = 0

        for group_idx in range(num_groups):
            # Accumulate samples_per_group samples for this group
            group_samples = []
            sample_count = 0
            while len(group_samples) < samples_per_group and not rospy.is_shutdown():
                with self.mutex:
                    if self.latest_sensor_data is not None:
                        # Store a copy of current sensor data
                        sensor_dict = {}
                        for sensor in self.latest_sensor_data.sensor_data:
                            if 1 <= sensor.id <= 12:
                                sensor_dict[sensor.id] = (sensor.x, sensor.y, sensor.z)
                        group_samples.append(sensor_dict)
                        sample_count += 1
                        self.latest_sensor_data = None  # Consume sample

                if not rospy.is_shutdown():
                    rospy.sleep(0.001)  # Small sleep to avoid busy loop

            if rospy.is_shutdown():
                break

            total_samples_collected += sample_count

            # Compute group average
            if group_samples:
                avg_data = {}
                for sensor_id in range(1, 13):
                    xs = [s[sensor_id][0] for s in group_samples if sensor_id in s]
                    ys = [s[sensor_id][1] for s in group_samples if sensor_id in s]
                    zs = [s[sensor_id][2] for s in group_samples if sensor_id in s]
                    if xs:
                        avg_data[sensor_id] = (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
                    else:
                        avg_data[sensor_id] = (0.0, 0.0, 0.0)

                timestamp = datetime.now().isoformat()
                row = [timestamp]
                for i in range(1, 13):
                    x, y, z = avg_data[i]
                    row.extend([f"{x:.7g}", f"{y:.7g}", f"{z:.7g}"])
                group_results.append(row)

            # Progress report every 5 groups or at specific intervals
            if (group_idx + 1) % 5 == 0 or group_idx == num_groups - 1:
                pct = (group_idx + 1) * 100.0 / num_groups
                rospy.loginfo("[2/3] PROGRESS: %d/%d groups (%.0f%%) | Samples: %d",
                              group_idx + 1, num_groups, pct, total_samples_collected)

        # Stop recording
        self.recording = False
        rospy.loginfo("[2/3] Collection finished: %d/%d groups, %d samples",
                      len(group_results), num_groups, total_samples_collected)

        # Phase 3: Send OFF command and wait for FY8300 to confirm
        rospy.loginfo("[3/3] PHASE 3: Disabling CH%d...", channel)
        self._set_channel(channel, False)
        rospy.loginfo("[3/3] Waiting %.1fs for FY8300 to deactivate...", self.wait_enable_time)
        rospy.sleep(self.wait_enable_time)

        # Write all group results to CSV (filename includes channel and polarity)
        if group_results:
            csv_path = os.path.join(self.output_dir, f"consistency_calib_ch{channel}_{polarity}.csv")
            self._write_csv(csv_path, group_results)
            rospy.loginfo("[3/3] Wrote %d groups for CH%d (%s) to %s", len(group_results), channel, polarity, csv_path)
        else:
            rospy.logwarn("[3/3] No data to write for CH%d (%s)", channel, polarity)

        # Wait before next channel
        rospy.loginfo("=== Channel %d | Polarity: %s | Status: COMPLETE ===", channel, polarity.upper())
        rospy.loginfo("%s", "=" * 60)
        rospy.loginfo("Waiting %.1fs before next channel...", self.wait_enable_time)
        rospy.sleep(self.wait_enable_time)

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

    def _build_r_corr_dict(self):
        """Build sensor_id -> R_CORR numpy array dictionary."""
        from sensor_array_config.base import SensorArrayHardwareParams
        r_corr = {}
        for entry in self._sensor_config.hardware.R_CORR:
            mat = np.array(entry.matrix).reshape(3, 3, order='F')
            for sid in entry.sensor_ids:
                r_corr[sid] = mat
        return r_corr

    def _get_default_csv_dir(self) -> Path:
        """Get default CSV directory."""
        return Path(os.path.expanduser('~/embedded_array_ws/src/calibration/data/consistency'))

    def _run_consistency_fit(self):
        """Run consistency fit calibration on existing CSV data."""
        rospy.loginfo("Running Phase 2 (consistency) calibration...")
        try:
            import numpy as np

            # Output to sensor_array_config/sensor_array_config/config/{sensor_type}/
            # Matches QMC6309Config._QMC6309_ROOT path structure
            sensor_type_dir = Path(__file__).parent.parent.parent / 'sensor_array_config' / 'sensor_array_config' / 'config' / self._sensor_type.lower()
            sensor_type_dir.mkdir(parents=True, exist_ok=True)

            # Run Phase 2 consistency calibration
            # Pass intrinsic_params and r_corr so the full correction chain is applied:
            # raw -> ellipsoid -> R_CORR rotation -> consistency fit
            results, amp_factor = batch_consistency_fit(
                csv_dir=self.csv_dir,
                output_path=sensor_type_dir / 'consistency_params.json',
                auto_detect=True,
                intrinsic_params=self._intrinsic_params,
                r_corr=self._r_corr,
                logger=rospy.loginfo
            )

            # Print amp factor info
            if amp_factor is not None:
                rospy.loginfo("  Amplification factor (background): %.4f", amp_factor)

            # Print per-sensor report
            rospy.loginfo("")
            rospy.loginfo("%s", "=" * 60)
            rospy.loginfo("Phase 2 Calibration Results (per sensor)")
            rospy.loginfo("%s", "=" * 60)
            rospy.loginfo(f"  {'Sensor':<8} {'D_ix':<10} {'D_iy':<10} {'D_iz':<10} {'e_ix':<9} {'e_iy':<9} {'e_iz':<9}")
            rospy.loginfo("%s", "-" * 70)
            for r in results:
                D = np.array(r.D_i)
                e = np.array(r.e_i)
                rospy.loginfo(f"  {r.sensor_id:<8} {D[0,0]:<10.4f} {D[1,1]:<10.4f} {D[2,2]:<10.4f} "
                              f"{e[0]:<+9.4f} {e[1]:<+9.4f} {e[2]:<+9.4f}")

            # Run validation (using same processing chain as consistency_fit)
            D_list = [np.array(r.D_i) for r in results]
            e_list = [np.array(r.e_i) for r in results]
            validation = validate_consistency(
                self.csv_dir, D_list, e_list,
                intrinsic_params=self._intrinsic_params,
                r_corr=self._r_corr
            )

            rospy.loginfo("")
            rospy.loginfo("%s", "=" * 60)
            rospy.loginfo("Validation Report")
            rospy.loginfo("%s", "=" * 60)
            rospy.loginfo(f"  {'Condition':<8} {'Axis':<6} {'校正前':<12} {'校正后':<12} {'改善'}")
            rospy.loginfo("%s", "-" * 50)

            improvements = []
            for i in range(len(validation['conditions'])):
                cond = validation['conditions'][i]
                axis = validation['axes'][i]
                std_b = validation['before'][i]
                std_a = validation['after'][i]
                imp = validation['improvement_pct'][i]
                improvements.append(imp)
                rospy.loginfo(f"  {cond:<8} {axis:<6} {std_b:<12.6f} {std_a:<12.6f} {imp:>+6.1f}%")

            improvements_filtered = [x for x in improvements if x > -50]
            rospy.loginfo("")
            rospy.loginfo("  Summary:")
            rospy.loginfo("  " + "-" * 40)
            rospy.loginfo(f"    Mean improvement:   {np.mean(improvements_filtered):>+.1f}%")
            rospy.loginfo(f"    Median improvement: {np.median(improvements_filtered):>+.1f}%")
            rospy.loginfo(f"    Max improvement:    {np.max(improvements_filtered):>+.1f}%")
            rospy.loginfo(f"    Min improvement:    {np.min(improvements_filtered):>+.1f}%")

            rospy.loginfo("%s", "=" * 60)
            rospy.loginfo("Phase 2 calibration complete. Consistency params saved.")
            rospy.loginfo("Output: %s", sensor_type_dir / 'consistency_params.json')

        except Exception as e:
            rospy.logerr(f"Phase 2 calibration failed: {e}")
            import traceback
            traceback.print_exc()
        rospy.loginfo("Consistency calibration complete.")

    def cleanup(self):
        """Cleanup on shutdown - run Phase 2 consistency post-processing."""
        rospy.loginfo("Running Phase 2 consistency post-processing...")
        try:
            from consistency_fit import batch_consistency_fit, validate_consistency
            import numpy as np

            # Output to sensor_array_config/sensor_array_config/config/{sensor_type}/
            # Matches QMC6309Config._QMC6309_ROOT path structure
            sensor_type_dir = Path(__file__).parent.parent.parent / 'sensor_array_config' / 'sensor_array_config' / 'config' / self._sensor_type.lower()
            sensor_type_dir.mkdir(parents=True, exist_ok=True)

            # Run Phase 2 consistency calibration
            # Pass intrinsic_params and r_corr so the full correction chain is applied:
            # raw -> ellipsoid -> R_CORR rotation -> consistency fit
            results, amp_factor = batch_consistency_fit(
                csv_dir=self.output_dir,
                output_path=sensor_type_dir / 'consistency_params.json',
                auto_detect=True,
                intrinsic_params=self._intrinsic_params,
                r_corr=self._r_corr
            )

            # Print per-sensor report
            rospy.loginfo("")
            rospy.loginfo("%s", "=" * 60)
            rospy.loginfo("Phase 2 Calibration Results (per sensor)")
            rospy.loginfo("%s", "=" * 60)
            rospy.loginfo(f"  {'Sensor':<8} {'D_ix':<10} {'D_iy':<10} {'D_iz':<10} {'e_ix':<9} {'e_iy':<9} {'e_iz':<9}")
            rospy.loginfo("%s", "-" * 70)
            for r in results:
                D = np.array(r.D_i)
                e = np.array(r.e_i)
                rospy.loginfo(f"  {r.sensor_id:<8} {D[0,0]:<10.4f} {D[1,1]:<10.4f} {D[2,2]:<10.4f} "
                              f"{e[0]:<+9.4f} {e[1]:<+9.4f} {e[2]:<+9.4f}")

            # Run validation (using same processing chain as consistency_fit)
            D_list = [np.array(r.D_i) for r in results]
            e_list = [np.array(r.e_i) for r in results]
            validation = validate_consistency(
                self.output_dir, D_list, e_list,
                intrinsic_params=self._intrinsic_params,
                r_corr=self._r_corr
            )

            rospy.loginfo("")
            rospy.loginfo("%s", "=" * 60)
            rospy.loginfo("Validation Report")
            rospy.loginfo("%s", "=" * 60)
            rospy.loginfo(f"  {'Condition':<8} {'Axis':<6} {'校正前':<12} {'校正后':<12} {'改善'}")
            rospy.loginfo("%s", "-" * 50)

            improvements = []
            for i in range(len(validation['conditions'])):
                cond = validation['conditions'][i]
                axis = validation['axes'][i]
                std_b = validation['before'][i]
                std_a = validation['after'][i]
                imp = validation['improvement_pct'][i]
                improvements.append(imp)
                rospy.loginfo(f"  {cond:<8} {axis:<6} {std_b:<12.6f} {std_a:<12.6f} {imp:>+6.1f}%")

            improvements_filtered = [x for x in improvements if x > -50]
            rospy.loginfo("")
            rospy.loginfo("  Summary:")
            rospy.loginfo("  " + "-" * 40)
            rospy.loginfo(f"    Mean improvement:   {np.mean(improvements_filtered):>+.1f}%")
            rospy.loginfo(f"    Median improvement: {np.median(improvements_filtered):>+.1f}%")
            rospy.loginfo(f"    Max improvement:    {np.max(improvements_filtered):>+.1f}%")
            rospy.loginfo(f"    Min improvement:    {np.min(improvements_filtered):>+.1f}%")

            rospy.loginfo("%s", "=" * 60)
            rospy.loginfo("Phase 2 calibration complete. Consistency params saved.")
            rospy.loginfo("Output: %s", sensor_type_dir / 'consistency_params.json')

        except Exception as e:
            rospy.logerr(f"Phase 2 calibration failed: {e}")
            import traceback
            traceback.print_exc()
        rospy.loginfo("Consistency calibration complete.")
        rospy.loginfo("Data saved to: %s", self.output_dir)

    def run(self):
        """Main execution flow."""
        total_steps = 8  # 1 background + 1 init + 3 channels positive + 3 channels negative
        current_step = 0

        rospy.loginfo("=" * 60)
        rospy.loginfo("CONSISTENCY CALIBRATION")
        rospy.loginfo("=" * 60)
        rospy.loginfo("Groups per channel: %d", self.num_groups)
        rospy.loginfo("Samples per group: %d", self.samples_per_group)
        rospy.loginfo("Wait init time: %.1fs", self.wait_init_time)
        rospy.loginfo("Wait enable time: %.1fs", self.wait_enable_time)
        rospy.loginfo("Output dir: %s", self.output_dir)
        rospy.loginfo("=" * 60)

        # ===== BACKGROUND MEASUREMENT: Collect data with all channels OFF =====
        current_step += 1
        rospy.loginfo("\n>>> [%d/%d] BACKGROUND MEASUREMENT", current_step, total_steps)
        rospy.loginfo("Overall progress: %d/%d steps (%.0f%%)", current_step, total_steps, current_step * 100.0 / total_steps)
        self._collect_background(self.num_groups, self.samples_per_group)

        # ===== INITIALIZATION: Set initial polarity and wait for FY8300 to initialize =====
        current_step += 1
        rospy.loginfo("\n>>> [%d/%d] INITIALIZATION", current_step, total_steps)
        rospy.loginfo("Setting initial polarity to positive (offset=+5V)...")
        self._set_polarity("positive")
        rospy.loginfo("Waiting %.1fs for FY8300 to initialize...", self.wait_init_time)
        rospy.sleep(self.wait_init_time)

        # Phase 1: Positive polarity (offset=+5V)
        rospy.loginfo("\n>>> [2/%d] POSITIVE CONFIG", total_steps)

        for channel in [1, 2, 3]:
            if rospy.is_shutdown():
                break
            current_step += 1
            rospy.loginfo("Overall progress: %d/%d steps (%.0f%%)",
                          current_step, total_steps, current_step * 100.0 / total_steps)
            self._run_channel_sequence(channel, "positive", self.num_groups, self.samples_per_group)

        # Phase 2: Negative polarity (offset=-5V, changed via topic)
        rospy.loginfo("\n>>> [5/%d] NEGATIVE CONFIG", total_steps)
        self._set_polarity("negative")
        rospy.sleep(self.wait_enable_time)  # Wait for FY8300 to settle after polarity change

        for channel in [1, 2, 3]:
            if rospy.is_shutdown():
                break
            current_step += 1
            rospy.loginfo("Overall progress: %d/%d steps (%.0f%%)",
                          current_step, total_steps, current_step * 100.0 / total_steps)
            self._run_channel_sequence(channel, "negative", self.num_groups, self.samples_per_group)

        # Shutdown all channels
        rospy.loginfo("\n>>> [%d/%d] SHUTDOWN", total_steps, total_steps)
        for ch in [1, 2, 3]:
            self._set_channel(ch, False)

        rospy.loginfo("=" * 60)
        rospy.loginfo("DONE")
        rospy.loginfo("Data saved to: %s/consistency_calib_ch#_positive.csv and negative.csv", self.output_dir)
        rospy.loginfo("=" * 60)

        # Gracefully shutdown ROS
        rospy.signal_shutdown("Calibration complete")


if __name__ == '__main__':
    rospy.init_node('consistency_calibration', anonymous=True)
    try:
        calib = ConsistencyCalibration()
        if not calib.skip_sampling:
            calib.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted.")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        import traceback
        traceback.print_exc()
