#!/usr/bin/env python3
"""
Calibration Joint Sweep Node

Sweeps diana7 joint7 through its range of motion while recording Hall sensor data.
Only joint7 is rotated - all other joints remain stationary.

Direct joint control via FollowJointTrajectory topic.

FY8300 Signal Generator Control:
  - Enables one channel at a time (CH3 -> CH2 -> CH1)
  - Performs sweep, saves CSV, then disables channel
  - Waits 10s for stabilization between operations

Data collected per position:
  - 5 sensor samples averaged
  - 12 sensors x 3 axes (x, y, z) = 36 values
  - Joint angle in degrees

Output: CSV file with ~360 rows (one per degree)
"""

import rospy
import csv
import os
from datetime import datetime
from threading import Lock

from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from serial_processor.msg import StmUplink
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class CalibrationJointSweep:
    def __init__(self):
        # Joint7 name and index
        self.joint7_name = "diana7_joint_7"
        self.joint_names = [
            "diana7_joint_1", "diana7_joint_2", "diana7_joint_3",
            "diana7_joint_4", "diana7_joint_5", "diana7_joint_6",
            "diana7_joint_7"
        ]

        # Joint limits (radians) - approximately +/- 179 degrees
        self.joint7_min = rospy.get_param("~joint7_min", -3.12)
        self.joint7_max = rospy.get_param("~joint7_max", 3.12)

        # Step size (1 degree)
        self.step_deg = rospy.get_param("~step_deg", 1.0)

        # Number of samples to average at each position
        self.num_samples = rospy.get_param("~num_samples", 5)

        # Settling time after move (seconds)
        self.settling_time = rospy.get_param("~settling_time", 0.3)

        # Move duration (seconds per move)
        self.move_duration_initial = rospy.get_param("~move_duration_initial", 3.0)
        self.move_duration_step = rospy.get_param("~move_duration_step", 0.5)

        # FY8300 stabilization wait time (seconds)
        self.fy8300_wait = rospy.get_param("~fy8300_wait", 10.0)

        # Output directory
        self.output_dir = rospy.get_param("~output_dir", "/tmp/calibration_data")

        # State
        self.current_joints = None
        self.latest_sensor_data = None
        self.mutex = Lock()

        # Subscribers
        self.sub_joint_states = rospy.Subscriber(
            "/diana7/joint_states",
            JointState,
            self._joint_states_callback,
            queue_size=10
        )

        self.sub_stm_uplink = rospy.Subscriber(
            "stm_uplink",
            StmUplink,
            self._stm_uplink_callback,
            queue_size=100
        )

        # Publisher for trajectory command
        cmd_topic = "/diana7/position_trajectory_controller/command"
        rospy.loginfo(f"Publishing to command topic: {cmd_topic}")
        self.cmd_pub = rospy.Publisher(cmd_topic, JointTrajectory, queue_size=10)

        # FY8300 channel publishers (CH1, CH2, CH3)
        self.pub_fy8300_ch = [
            rospy.Publisher(f"/fy8300/ch{i}/output_en", Bool, queue_size=1)
            for i in range(1, 4)
        ]

        # Wait for connections
        rospy.loginfo("Waiting for /diana7/joint_states connection...")
        self._wait_for_topic("/diana7/joint_states", JointState)

        rospy.loginfo("Waiting for stm_uplink connection...")
        self._wait_for_topic("stm_uplink", StmUplink)

        # Wait for command publisher
        rospy.loginfo("Waiting for command publisher to be ready...")
        rate = rospy.Rate(10)
        for _ in range(50):
            if self.cmd_pub.get_num_connections() > 0:
                break
            rate.sleep()
        rospy.loginfo("  Command publisher ready")

        # Wait for FY8300 publishers
        rospy.loginfo("Waiting for FY8300 publishers to be ready...")
        for i, pub in enumerate(self.pub_fy8300_ch, 1):
            for _ in range(30):
                if pub.get_num_connections() > 0:
                    break
                rate.sleep()
            rospy.loginfo(f"  /fy8300/ch{i}/output_en ready")

        rospy.loginfo("All connections established.")

        # Get current position
        rospy.loginfo("Reading current joint positions...")
        self._wait_for_topic("/diana7/joint_states", JointState)
        with self.mutex:
            if self.current_joints:
                rospy.loginfo(f"Current joint7 angle: {self.current_joints[6]:.3f} rad ({self._rad_to_deg(self.current_joints[6]):.1f} deg)")

    def _wait_for_topic(self, topic, msg_type, timeout=10.0):
        """Wait for a topic to be published."""
        try:
            msg = rospy.wait_for_message(topic, msg_type, timeout=timeout)
            rospy.loginfo(f"  {topic} connected")
            return msg
        except rospy.ROSException:
            rospy.logwarn(f"  Timeout waiting for {topic}, continuing anyway...")
            return None

    def _joint_states_callback(self, msg: JointState):
        """Store current joint angles."""
        with self.mutex:
            self.current_joints = list(msg.position)

    def _stm_uplink_callback(self, msg: StmUplink):
        """Store latest sensor data."""
        with self.mutex:
            self.latest_sensor_data = msg

    def _set_fy8300_channel(self, channel, enabled):
        """Enable or disable a FY8300 channel.

        Args:
            channel: Channel number (1, 2, or 3)
            enabled: True to enable, False to disable
        """
        if channel < 1 or channel > 3:
            rospy.logwarn(f"Invalid channel {channel}, must be 1, 2, or 3")
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

    def _move_joint7_to(self, angle_rad, duration=None):
        """Move joint7 to target angle via trajectory topic."""
        if duration is None:
            duration = self.move_duration_step

        with self.mutex:
            if self.current_joints is None:
                rospy.logwarn("No joint states received yet!")
                return False
            target_positions = list(self.current_joints)

        target_positions[6] = angle_rad

        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        trajectory.header.stamp = rospy.Time.now()

        point = JointTrajectoryPoint()
        point.positions = target_positions
        point.velocities = [0.0] * 7
        point.time_from_start = rospy.Duration(duration)
        trajectory.points.append(point)

        rospy.loginfo(f"  Publishing: joint7 -> {angle_rad:.4f} rad ({self._rad_to_deg(angle_rad):.1f} deg)")

        self.cmd_pub.publish(trajectory)

        # Wait for move to complete and verify position
        timeout = duration + 2.0
        start_time = rospy.Time.now()
        tolerance = 0.02

        while (rospy.Time.now() - start_time).to_sec() < timeout:
            with self.mutex:
                if self.current_joints is not None:
                    actual = self.current_joints[6]
                    diff = abs(actual - angle_rad)
                    if diff < tolerance:
                        rospy.loginfo(f"  Reached target: {self._rad_to_deg(actual):.1f} deg (diff={diff:.4f} rad)")
                        return True
            rospy.sleep(0.05)

        rospy.logwarn(f"  Timeout waiting for target, continuing anyway")
        return True

    def _collect_samples_at_position(self):
        """Collect num_samples sensor readings and return averaged values."""
        samples = {i: [] for i in range(1, 13)}
        collected = 0
        rate = rospy.Rate(100)

        while collected < self.num_samples and not rospy.is_shutdown():
            with self.mutex:
                if self.latest_sensor_data is not None:
                    msg = self.latest_sensor_data
                    for sensor in msg.sensor_data:
                        if 1 <= sensor.id <= 12:
                            samples[sensor.id].append((sensor.x, sensor.y, sensor.z))
                    collected += 1
            rate.sleep()

        averaged = {}
        for sensor_id in range(1, 13):
            if samples[sensor_id]:
                xs = [s[0] for s in samples[sensor_id]]
                ys = [s[1] for s in samples[sensor_id]]
                zs = [s[2] for s in samples[sensor_id]]
                averaged[sensor_id] = (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
            else:
                averaged[sensor_id] = (0.0, 0.0, 0.0)
                rospy.logwarn(f"No samples collected for sensor {sensor_id}")

        return averaged

    def _rad_to_deg(self, rad):
        return rad * (180.0 / 3.14159265359)

    def _deg_to_rad(self, deg):
        return deg * (3.14159265359 / 180.0)

    def _perform_single_sweep(self, channel):
        """Perform a single sweep with specified channel enabled.

        Args:
            channel: FY8300 channel number (1, 2, or 3)

        Returns:
            Path to saved CSV file, or None if failed
        """
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.output_dir, f"calibration_ch{channel}_{timestamp}.csv")

        header = ["timestamp", "joint_angle_deg"]
        for i in range(1, 13):
            header.extend([f"sensor_{i}_x", f"sensor_{i}_y", f"sensor_{i}_z"])

        rospy.loginfo(f"=== Starting sweep for CH{channel} ===")

        current_angle_deg = self._rad_to_deg(self.joint7_min)
        end_angle_deg = self._rad_to_deg(self.joint7_max)

        # Move to starting position
        rospy.loginfo(f"Moving to starting position: {current_angle_deg:.1f} deg")
        success = self._move_joint7_to(self.joint7_min, duration=self.move_duration_initial)
        if not success:
            rospy.logerr("Failed to move to starting position, exiting")
            return None
        rospy.sleep(self.settling_time)

        row_count = 0

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            while current_angle_deg <= end_angle_deg + 0.01:
                if rospy.is_shutdown():
                    rospy.loginfo("Shutdown requested, stopping sweep.")
                    break

                with self.mutex:
                    if self.current_joints is None:
                        rospy.sleep(0.5)
                        continue
                    actual_angle_deg = self._rad_to_deg(self.current_joints[6])

                rospy.loginfo(f"Angle: {actual_angle_deg:.1f} deg - collecting {self.num_samples} samples...")
                sensor_data = self._collect_samples_at_position()

                row = [datetime.now().isoformat(), f"{actual_angle_deg:.3f}"]
                for i in range(1, 13):
                    x, y, z = sensor_data[i]
                    row.extend([f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"])

                writer.writerow(row)
                f.flush()
                row_count += 1

                current_angle_deg += self.step_deg
                next_angle_rad = self._deg_to_rad(current_angle_deg)

                if next_angle_rad > self.joint7_max:
                    next_angle_rad = self.joint7_max

                success = self._move_joint7_to(next_angle_rad, duration=self.move_duration_step)
                if success:
                    rospy.sleep(self.settling_time)
                else:
                    rospy.logwarn(f"Move to {current_angle_deg:.1f} deg failed, continuing...")

        rospy.loginfo(f"Sweep complete for CH{channel}! Data saved to: {csv_path}")
        rospy.loginfo(f"Total rows written: {row_count}")

        return csv_path

    def run(self):
        """Main 3-channel sweep procedure.

        For each channel (3, 2, 1):
          1. Enable channel
          2. Wait for stabilization
          3. Perform sweep and save CSV
          4. Disable channel
          5. Wait for stabilization
          6. Return robot to neutral position
        """
        rospy.loginfo("=" * 60)
        rospy.loginfo("CALIBRATION 3-CHANNEL SWEEP")
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"Joint7 range: {self._rad_to_deg(self.joint7_min):.1f} deg to {self._rad_to_deg(self.joint7_max):.1f} deg")
        rospy.loginfo(f"Step size: {self.step_deg} deg")
        rospy.loginfo(f"Samples per position: {self.num_samples}")
        rospy.loginfo(f"FY8300 wait time: {self.fy8300_wait} s")
        rospy.loginfo(f"Output directory: {self.output_dir}")
        rospy.loginfo("=" * 60)

        csv_paths = []

        # Initial move to neutral position before first channel
        rospy.loginfo("\nMoving robot to initial position (0 deg) before starting...")
        self._move_joint7_to(0.0, duration=self.move_duration_initial)
        rospy.sleep(self.settling_time)

        for channel in [3, 2, 1]:
            if rospy.is_shutdown():
                break

            rospy.loginfo(f"\n{'='*60}")
            rospy.loginfo(f"CHANNEL {channel}")
            rospy.loginfo(f"{'='*60}")

            # Enable channel
            rospy.loginfo(f"Enabling FY8300 CH{channel}...")
            self._set_fy8300_channel(channel, True)

            # Wait for stabilization
            self._wait_for_fy8300_stabilize()

            # Perform sweep
            csv_path = self._perform_single_sweep(channel)
            if csv_path:
                csv_paths.append(csv_path)

            # Disable channel
            rospy.loginfo(f"Disabling FY8300 CH{channel}...")
            self._set_fy8300_channel(channel, False)

            # Wait for stabilization
            self._wait_for_fy8300_stabilize()

            # Return robot to neutral
            rospy.loginfo("Returning robot to neutral position (0 deg)...")
            self._move_joint7_to(0.0, duration=self.move_duration_initial)
            rospy.sleep(self.settling_time)

        rospy.loginfo("\n" + "=" * 60)
        rospy.loginfo("ALL CHANNELS COMPLETE")
        rospy.loginfo("=" * 60)
        rospy.loginfo("Saved CSV files:")
        for path in csv_paths:
            rospy.loginfo(f"  - {path}")

        return csv_paths


if __name__ == "__main__":
    rospy.init_node("calibration_joint_sweep", anonymous=True)

    try:
        sweeper = CalibrationJointSweep()
        output_paths = sweeper.run()
        if output_paths:
            rospy.loginfo(f"Done. Output: {output_paths}")
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted.")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        import traceback
        traceback.print_exc()
