#!/usr/bin/env python3
"""
Calibration Sweep Ramp Trigger Node

Monitors Hall sensor readings and triggers recording when magnetic field
reaches near maximum (32Gs), capturing one complete cycle of the ramp signal.

Workflow:
  1. Enable FY8300 channel (CH3 -> CH2 -> CH1)
  2. Monitor sensor data, wait for jump to ~32Gs
  3. Record one complete cycle until back to 32Gs
  4. Save data to CSV
  5. Wait for user input (Enter) to proceed to next channel
  6. Repeat for all 3 channels

Channel order: CH3, CH2, CH1
"""

import rospy
import csv
import os
import sys
from datetime import datetime
from threading import Lock

from std_msgs.msg import Bool
from serial_processor.msg import StmUplink


class CalibrationSweepRampTrigger:
    # Magnetic field threshold for trigger detection (Gauss)
    TRIGGER_THRESHOLD_HIGH = 28.0   # Trigger when sensor reaches near 32Gs
    TRIGGER_THRESHOLD_LOW = -28.0    # End trigger when sensor returns to near -32Gs

    def __init__(self):
        # State
        self.latest_sensor_data = None
        self.mutex = Lock()
        self.recording = False
        self.recorded_data = []
        self.triggered = False
        self.cycle_complete = False

        # Output directory
        self.output_dir = rospy.get_param("~output_dir", "/home/zhang/embedded_array_ws/src/calibration/data")

        # FY8300 stabilization wait time
        self.fy8300_wait = rospy.get_param("~fy8300_wait", 10.0)

        # Subscribers
        rospy.loginfo("Subscribing to stm_uplink topic...")
        self.sub_stm_uplink = rospy.Subscriber(
            "stm_uplink",
            StmUplink,
            self._stm_uplink_callback,
            queue_size=100
        )

        # FY8300 channel publishers (CH1, CH2, CH3)
        self.pub_fy8300_ch = [
            rospy.Publisher(f"/fy8300/ch{i}/output_en", Bool, queue_size=1)
            for i in range(1, 4)
        ]

        # Wait for connections
        rospy.loginfo("Waiting for stm_uplink connection...")
        self._wait_for_topic("stm_uplink", StmUplink)

        rospy.loginfo("Waiting for FY8300 publishers to be ready...")
        rate = rospy.Rate(10)
        for i, pub in enumerate(self.pub_fy8300_ch, 1):
            for _ in range(30):
                if pub.get_num_connections() > 0:
                    break
                rate.sleep()
            rospy.loginfo(f"  /fy8300/ch{i}/output_en ready")

        rospy.loginfo("All connections established.")

    def _wait_for_topic(self, topic, msg_type, timeout=10.0):
        """Wait for a topic to be published."""
        try:
            msg = rospy.wait_for_message(topic, msg_type, timeout=timeout)
            rospy.loginfo(f"  {topic} connected")
            return msg
        except rospy.ROSException:
            rospy.logwarn(f"  Timeout waiting for {topic}, continuing anyway...")
            return None

    def _stm_uplink_callback(self, msg: StmUplink):
        """Store latest sensor data and handle trigger detection."""
        with self.mutex:
            self.latest_sensor_data = msg

            if self.recording:
                # Record data with timestamp
                timestamp = datetime.now()
                for sensor in msg.sensor_data:
                    if 1 <= sensor.id <= 12:
                        self.recorded_data.append({
                            'timestamp': timestamp.isoformat(),
                            'sensor_id': sensor.id,
                            'x': sensor.x,
                            'y': sensor.y,
                            'z': sensor.z
                        })

                # Check for cycle completion (field returned to high after going low)
                if self.triggered and self.cycle_complete:
                    # Check if any sensor has returned to high threshold
                    for sensor in msg.sensor_data:
                        if sensor.z > self.TRIGGER_THRESHOLD_HIGH:
                            rospy.loginfo(f"Cycle complete! Sensor {sensor.id} returned to {sensor.z:.2f}Gs")
                            self.recording = False
                            return

                # Check if field has gone below low threshold (indicating we passed the minimum)
                if self.triggered and not self.cycle_complete:
                    for sensor in msg.sensor_data:
                        if sensor.z < self.TRIGGER_THRESHOLD_LOW:
                            rospy.loginfo(f"Passed minimum. Sensor {sensor.id} at {sensor.z:.2f}Gs, waiting for return to high...")
                            self.cycle_complete = True
                            return

            # Check for initial trigger (field jumped to near 32Gs)
            elif not self.triggered:
                for sensor in msg.sensor_data:
                    if abs(sensor.z) > self.TRIGGER_THRESHOLD_HIGH:
                        rospy.loginfo(f"Trigger detected! Sensor {sensor.id} at {sensor.z:.2f}Gs - starting recording")
                        self.triggered = True
                        self.recording = True
                        self.recorded_data = []
                        self.cycle_complete = False
                        return

    def _set_fy8300_channel(self, channel, enabled):
        """Enable or disable a FY8300 channel."""
        if channel < 1 or channel > 3:
            rospy.logwarn(f"Invalid channel {channel}")
            return

        msg = Bool()
        msg.data = enabled
        self.pub_fy8300_ch[channel - 1].publish(msg)
        state = "ENABLED" if enabled else "DISABLED"
        rospy.loginfo(f"  FY8300 CH{channel} {state}")

    def _wait_for_fy8300_stabilize(self):
        """Wait for FY8300 output to stabilize."""
        rospy.loginfo(f"  Waiting {self.fy8300_wait}s for FY8300 to stabilize...")
        rospy.sleep(self.fy8300_wait)

    def _save_csv(self, channel):
        """Save recorded data to CSV file."""
        if not self.recorded_data:
            rospy.logwarn("No data recorded!")
            return None

        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%MS")
        csv_path = os.path.join(self.output_dir, f"ramp_calib_ch{channel}_{timestamp}.csv")

        # Group by timestamp to form complete readings
        readings = {}
        for entry in self.recorded_data:
            ts = entry['timestamp']
            if ts not in readings:
                readings[ts] = {}
            readings[ts][entry['sensor_id']] = (entry['x'], entry['y'], entry['z'])

        # Write CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestamp']
            for i in range(1, 13):
                header.extend([f'sensor_{i}_x', f'sensor_{i}_y', f'sensor_{i}_z'])
            writer.writerow(header)

            for ts in sorted(readings.keys()):
                row = [ts]
                for sensor_id in range(1, 13):
                    if sensor_id in readings[ts]:
                        x, y, z = readings[ts][sensor_id]
                        row.extend([f"{x:.2f}", f"{y:.2f}", f"{z:.2f}"])
                    else:
                        row.extend(["0.00", "0.00", "0.00"])
                writer.writerow(row)

        rospy.loginfo(f"Data saved to: {csv_path}")
        rospy.loginfo(f"Total readings: {len(readings)}")
        return csv_path

    def run(self):
        """Main procedure for 3-channel sweep."""
        rospy.loginfo("=" * 60)
        rospy.loginfo("CALIBRATION RAMP TRIGGER SWEEP")
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"Trigger threshold: +/- {self.TRIGGER_THRESHOLD_HIGH}Gs")
        rospy.loginfo(f"Output directory: {self.output_dir}")
        rospy.loginfo("=" * 60)

        csv_paths = []

        for channel in [3, 2, 1]:
            rospy.loginfo(f"\n{'='*60}")
            rospy.loginfo(f"CHANNEL {channel}")
            rospy.loginfo(f"{'='*60}")

            # Reset state
            with self.mutex:
                self.triggered = False
                self.recording = False
                self.recorded_data = []
                self.cycle_complete = False

            # Enable channel
            rospy.loginfo(f"Enabling FY8300 CH{channel}...")
            self._set_fy8300_channel(channel, True)

            # Wait for stabilization
            self._wait_for_fy8300_stabilize()

            # Wait for trigger and recording to complete
            rospy.loginfo("Monitoring sensors for trigger (~32Gs)... Press Enter to skip this channel")
            rospy.loginfo("(Recording will auto-stop after one complete cycle)")

            # Use a thread to check for user input while monitoring
            import select

            rate = rospy.Rate(10)
            timeout_count = 0
            max_timeout = 600  # 60 seconds max wait for trigger

            while not rospy.is_shutdown():
                with self.mutex:
                    if not self.triggered:
                        # Check if user pressed Enter (non-blocking)
                        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                            line = sys.stdin.readline()
                            rospy.loginfo(f"User skipped channel {channel}")
                            break

                        timeout_count += 1
                        if timeout_count > max_timeout:
                            rospy.logwarn("Timeout waiting for trigger, skipping channel")
                            break

                if not self.triggered:
                    rospy.loginfo_throttle(5, "Waiting for trigger...")

                with self.mutex:
                    if self.triggered and not self.recording:
                        # Recording just finished
                        break

                if self.triggered:
                    rospy.loginfo_throttle(1, "Recording in progress...")

                rate.sleep()

            # Save data
            with self.mutex:
                if self.recorded_data:
                    csv_path = self._save_csv(channel)
                    if csv_path:
                        csv_paths.append(csv_path)
                else:
                    rospy.logwarn(f"No data recorded for CH{channel}")

            # Disable channel
            rospy.loginfo(f"Disabling FY8300 CH{channel}...")
            self._set_fy8300_channel(channel, False)

            # Wait for stabilization before next channel
            self._wait_for_fy8300_stabilize()

            if channel > 1:
                # Wait for user input to proceed to next channel
                rospy.loginfo(f"\n=== CH{channel} complete! ===")
                rospy.loginfo("Press ENTER to continue to next channel (CH%d)..." % (channel - 1))
                raw_input()

        rospy.loginfo("\n" + "=" * 60)
        rospy.loginfo("ALL CHANNELS COMPLETE")
        rospy.loginfo("=" * 60)
        rospy.loginfo("Saved CSV files:")
        for path in csv_paths:
            rospy.loginfo(f"  - {path}")

        return csv_paths


if __name__ == "__main__":
    rospy.init_node("calibration_sweep_ramp_trigger", anonymous=True)

    try:
        sweeper = CalibrationSweepRampTrigger()
        output_paths = sweeper.run()
        if output_paths:
            rospy.loginfo(f"Done. Output: {output_paths}")
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted.")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        import traceback
        traceback.print_exc()
