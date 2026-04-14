#!/usr/bin/env python3
"""
Ellipsoid calibration sampler for magnetometer array.

Collects magnetic field sensor data at different orientations while
keeping the end-effector position fixed. Uses Fibonacci hemisphere
distribution for orientation sampling.

Usage:
    roslaunch calibration ellipsoid_calibration.launch
"""

import sys
import rospy
import moveit_commander
import yaml
import os
import math
import csv
import threading
import numpy as np
from datetime import datetime
from geometry_msgs.msg import Pose, Quaternion
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from serial_processor.msg import StmUplink


class EllipsoidCalibrationSampler:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('ellipsoid_calibration_sampler', anonymous=True)

        self.robot = moveit_commander.RobotCommander()

        # Parameters
        self.connect_wait_time = rospy.get_param('~connect_wait_time', 5.0)
        self.max_retries = rospy.get_param('~max_retries', 3)
        self.speed_scaling = rospy.get_param('~speed_scaling', 0.08)

        rospy.loginfo(f"Waiting {self.connect_wait_time}s for move_group...")
        rospy.sleep(self.connect_wait_time)

        # Initialize MoveIt - use "diana7" like scan_controller.py
        retry_count = 0
        while not rospy.is_shutdown() and retry_count < self.max_retries:
            try:
                self.diana7_group = moveit_commander.MoveGroupCommander("diana7")
                rospy.loginfo("Connected to move_group (diana7)")
                break
            except RuntimeError as e:
                retry_count += 1
                rospy.logwarn(f"MoveGroupCommander attempt {retry_count}/{self.max_retries} failed: {e}")
                rospy.sleep(self.connect_wait_time)

        if retry_count == self.max_retries:
            rospy.logerr("Failed to connect to move_group. Exiting.")
            sys.exit(1)

        # Set speed scaling for safety
        self.diana7_group.set_max_velocity_scaling_factor(self.speed_scaling)
        self.diana7_group.set_max_acceleration_scaling_factor(self.speed_scaling)

        # Sensor collection parameters
        self.num_samples = rospy.get_param('~num_samples', 10)
        self.settling_time = rospy.get_param('~settling_time', 0.5)
        self.skip_poses = rospy.get_param('~skip_poses', 0)

        # Sensor data state
        self.latest_sensor_data = None
        self.mutex = threading.Lock()

        # Subscribe to stm_uplink for sensor data
        rospy.loginfo("Subscribing to stm_uplink topic...")
        self.sub_stm_uplink = rospy.Subscriber(
            "stm_uplink",
            StmUplink,
            self._stm_uplink_callback,
            queue_size=100
        )

        # Wait for stm_uplink to be ready
        rospy.sleep(1.0)
        rospy.loginfo("stm_uplink subscriber ready.")

        # Load diana7_home config
        self.home_config = self.load_home_config()
        if self.home_config is None:
            rospy.logerr("Failed to load home config. Exiting.")
            sys.exit(1)

        rospy.loginfo("diana7_home config loaded:")
        for name, data in self.home_config.items():
            rospy.loginfo(f"  {name}: {data['degrees']:.2f} deg ({data['radians']:.6f} rad)")

        # Convert home_config to joint list
        self.home_joints = self._config_to_joints()

        # Move to home position
        rospy.loginfo("Moving to home position...")
        self.diana7_group.set_joint_value_target(self.home_joints)
        success = self.diana7_group.go(wait=True)
        self.diana7_group.stop()

        if success:
            rospy.loginfo("Robot reached home position successfully.")
        else:
            rospy.logerr("Robot failed to reach home position.")
            # Continue anyway for testing orientation generation

        # Get reference pose at home position
        self.reference_pose = self.diana7_group.get_current_pose().pose
        rospy.loginfo(f"Reference pose: pos=({self.reference_pose.position.x:.4f}, "
                     f"{self.reference_pose.position.y:.4f}, {self.reference_pose.position.z:.4f})")

        # Generate orientations: Rx, Ry pairs using Fibonacci hemisphere
        # Each (Rx, Ry) will have joint7 sweep through 180 degrees
        self.test_angles = self._generate_rxry_hemisphere(num_poses=50)

        # Setup CSV output
        result = self._setup_csv()
        if result is None:
            rospy.logerr("Failed to setup CSV file. Exiting.")
            sys.exit(1)

        self.csv_file, self.csv_writer = result

        # Execute orientation test with data collection
        self.run_orientation_test()

        # Close CSV and cleanup
        self.csv_file.close()
        rospy.loginfo("Data collection complete.")

        rospy.signal_shutdown("done")

    def run_orientation_test(self):
        """Test orientation generation with joint7 180-degree sweep per Rx/Ry pose."""
        rospy.loginfo("Starting orientation test with joint7 sweep...")
        rospy.loginfo(f"Total Rx/Ry poses: {len(self.test_angles)}")

        # Skip poses if resuming
        if self.skip_poses > 0:
            rospy.loginfo(f"Skipping first {self.skip_poses} poses (resuming from pose {self.skip_poses+1})...")
            test_angles = self.test_angles[self.skip_poses:]
        else:
            test_angles = self.test_angles

        consecutive_failures = 0
        max_consecutive_failures = 3
        valid_pose_count = self.skip_poses  # Count already completed poses

        for i, (angle_x_deg, angle_y_deg) in enumerate(test_angles):
            global_i = i + self.skip_poses
            rospy.loginfo(f"\n--- Rx/Ry Pose {global_i+1}/{len(self.test_angles)}: "
                         f"Rx={angle_x_deg:.1f}°, Ry={angle_y_deg:.1f}° ---")

            # Generate target pose with rotation applied (Rz=0)
            target_pose = self._generate_rotated_pose(angle_x_deg, angle_y_deg, angle_z_deg=0)

            rospy.loginfo(f"Target orientation: qx={target_pose.orientation.x:.4f}, "
                         f"qy={target_pose.orientation.y:.4f}, "
                         f"qz={target_pose.orientation.z:.4f}, "
                         f"qw={target_pose.orientation.w:.4f}")

            # Move using Cartesian path
            success = self._move_cartesian(target_pose)
            if not success:
                rospy.logerr(f"Rx/Ry Pose {global_i+1} move failed")
                consecutive_failures += 1
                rospy.logwarn(f"Consecutive failures: {consecutive_failures}/{max_consecutive_failures}")

                if consecutive_failures >= max_consecutive_failures:
                    rospy.logwarn(f"Reached {max_consecutive_failures} consecutive failures. Going home...")
                    home_ok = self._move_to_home()

                    if home_ok:
                        rospy.loginfo("Home reached. Will continue with next orientation after settling...")
                        rospy.sleep(2.0)  # Wait for robot to fully settle
                        # Re-sync state before next orientation
                        current_state = self.robot.get_current_state()
                        self.diana7_group.set_start_state(current_state)
                    else:
                        rospy.logerr("Home move failed! Stopping for manual intervention.")
                        self.csv_file.close()
                        rospy.signal_shutdown("Home failed - manual intervention required")
                        return

                    consecutive_failures = 0
                    # Do NOT continue - let loop naturally proceed to next Rx/Ry after home completes

            rospy.loginfo(f"Rx/Ry Pose {global_i+1} move succeeded")
            consecutive_failures = 0
            valid_pose_count += 1

            # Wait for vibrations to settle
            rospy.sleep(self.settling_time)

            # Get current joint7 angle
            current_joints = self.diana7_group.get_current_joint_values()
            joint_names = self.diana7_group.get_active_joints()
            joint7_idx = None
            for idx, name in enumerate(joint_names):
                if 'joint_7' in name.lower():
                    joint7_idx = idx
                    break

            if joint7_idx is None:
                rospy.logerr("joint7 not found in active joints!")
                continue

            joint7_start = current_joints[joint7_idx]

            # Joint7 limits from URDF: ±3.12 rad (±178.8°)
            joint7_min = -3.0
            joint7_max = 3.0
            rospy.loginfo(f"Joint7 limits: [{math.degrees(joint7_min):.1f}°, {math.degrees(joint7_max):.1f}°]")

            # Ensure start is within bounds
            joint7_start = max(joint7_min, min(joint7_max, joint7_start))
            rospy.loginfo(f"Current joint7: {math.degrees(joint7_start):.1f}°")

            # Calculate joint7 sweep range (180 degrees in joint space)
            if joint7_start >= 0:
                joint7_end = joint7_start - math.radians(180)
            else:
                joint7_end = joint7_start + math.radians(180)

            # Clamp to joint limits
            joint7_end = max(joint7_min, min(joint7_max, joint7_end))

            # Ensure we have at least 90 degrees of sweep
            sweep_range = abs(joint7_end - joint7_start)
            if sweep_range < math.radians(90):
                rospy.logwarn(f"Sweep range {math.degrees(sweep_range):.1f}° < 90°, adjusting...")
                if joint7_start >= 0:
                    joint7_end = joint7_start - math.radians(90)
                else:
                    joint7_end = joint7_start + math.radians(90)
                joint7_end = max(joint7_min, min(joint7_max, joint7_end))

            rospy.loginfo(f"Joint7 sweep: {math.degrees(joint7_start):.1f}° -> {math.degrees(joint7_end):.1f}° (clamped)")

            # Sweep joint7 through 180 degrees with 10 sample points, plus 0 at end
            num_samples = 10
            joint7_positions = np.linspace(joint7_start, joint7_end, num_samples)
            joint7_positions = np.append(joint7_positions, 0.0)  # End with joint7=0

            for j, joint7_target in enumerate(joint7_positions):
                # Clamp to safe bounds
                joint7_target = max(joint7_min, min(joint7_max, joint7_target))
                rospy.loginfo(f"  Joint7 position {j+1}/{num_samples}: {math.degrees(joint7_target):.1f}°")

                # Move joint7 to target position (keep other joints fixed)
                move_ok = self._move_joint7_to(joint7_target)
                if not move_ok:
                    rospy.logwarn(f"  Joint7 position {j+1} move failed, skipping sample...")
                    continue

                # Wait for settling
                rospy.sleep(self.settling_time)

                # Collect sensor data: 10 samples averaged
                sensor_data = self._collect_samples_at_position()

                # Get current pose
                current_pose = self.diana7_group.get_current_pose().pose

                # Write to CSV
                self._write_csv_row(
                    timestamp=datetime.now().isoformat(),
                    pose=current_pose,
                    sensor_data=sensor_data,
                    joint7_deg=math.degrees(joint7_target)
                )

            rospy.loginfo(f"Rx/Ry Pose {global_i+1} completed: {len(joint7_positions)} samples saved.")

        rospy.loginfo(f"\n=== Orientation test complete. Valid Rx/Ry poses: {valid_pose_count}/{len(self.test_angles)} ===")

        # Fill remaining with random orientations if needed
        target_total = len(self.test_angles)
        if valid_pose_count < target_total:
            fill_count = target_total - valid_pose_count
            rospy.loginfo(f"Filling {fill_count} remaining poses with random orientations...")
            self._fill_random_orientations(fill_count)

        rospy.loginfo("All orientations processed.")

    def _move_to_home(self):
        """Move robot to home position and update reference pose."""
        rospy.loginfo("Moving to home position...")

        # Sync state before moving
        current_state = self.robot.get_current_state()
        self.diana7_group.set_start_state(current_state)

        self.diana7_group.set_joint_value_target(self.home_joints)
        success = self.diana7_group.go(wait=True)
        self.diana7_group.stop()

        if success:
            rospy.loginfo("Robot reached home position.")
            # Update reference pose to home pose
            self.reference_pose = self.diana7_group.get_current_pose().pose
        else:
            rospy.logerr("Failed to reach home position.")

    def _fill_random_orientations(self, fill_count):
        """
        Fill remaining poses with random (Rx, Ry) orientations and joint7 sweep.
        This is called when some Rx/Ry poses fail to execute.
        """
        rospy.loginfo(f"Generating {fill_count} random orientations...")

        for i in range(fill_count):
            # Generate random Rx, Ry in [-90, 90] degrees
            rx_deg = np.random.uniform(-90, 90)
            ry_deg = np.random.uniform(-90, 90)

            rospy.loginfo(f"\n--- Random Fill {i+1}/{fill_count}: Rx={rx_deg:.1f}°, Ry={ry_deg:.1f}° ---")

            # Move to random orientation
            target_pose = self._generate_rotated_pose(rx_deg, ry_deg, angle_z_deg=0)
            success = self._move_cartesian(target_pose)

            if not success:
                rospy.logwarn(f"Random Fill {i+1} move failed, skipping...")
                continue

            rospy.sleep(self.settling_time)

            # Get current joint7 angle
            current_joints = self.diana7_group.get_current_joint_values()
            joint_names = self.diana7_group.get_active_joints()
            joint7_idx = None
            for idx, name in enumerate(joint_names):
                if 'joint_7' in name.lower():
                    joint7_idx = idx
                    break

            if joint7_idx is None:
                continue

            joint7_start = current_joints[joint7_idx]
            joint7_min = -3.0
            joint7_max = 3.0

            # Random sweep direction
            if np.random.rand() > 0.5:
                joint7_end = joint7_start - math.radians(180)
            else:
                joint7_end = joint7_start + math.radians(180)

            joint7_end = max(joint7_min, min(joint7_max, joint7_end))

            # 10 sample points, plus 0 at end
            num_samples = 10
            joint7_positions = np.linspace(joint7_start, joint7_end, num_samples)
            joint7_positions = np.append(joint7_positions, 0.0)  # End with joint7=0

            for j, joint7_target in enumerate(joint7_positions):
                joint7_target = max(joint7_min, min(joint7_max, joint7_target))

                move_ok = self._move_joint7_to(joint7_target)
                if not move_ok:
                    continue

                rospy.sleep(self.settling_time)
                sensor_data = self._collect_samples_at_position()
                current_pose = self.diana7_group.get_current_pose().pose

                self._write_csv_row(
                    timestamp=datetime.now().isoformat(),
                    pose=current_pose,
                    sensor_data=sensor_data,
                    joint7_deg=math.degrees(joint7_target)
                )

            rospy.loginfo(f"Random Fill {i+1} completed.")

    def _move_joint7_to(self, target_angle):
        """Move joint7 to target angle while keeping other joints at current values."""
        # Sync state before moving
        current_state = self.robot.get_current_state()
        self.diana7_group.set_start_state(current_state)

        current_joints = self.diana7_group.get_current_joint_values()
        joint_names = self.diana7_group.get_active_joints()

        # Find joint7 index
        joint7_idx = None
        for idx, name in enumerate(joint_names):
            if 'joint_7' in name.lower():
                joint7_idx = idx
                break

        if joint7_idx is None:
            rospy.logerr("joint7 not found!")
            return False

        # Modify only joint7
        target_joints = list(current_joints)
        target_joints[joint7_idx] = target_angle

        self.diana7_group.set_joint_value_target(target_joints)
        success = self.diana7_group.go(wait=True)
        self.diana7_group.stop()

        return success

    def _generate_rxry_hemisphere(self, num_poses=50):
        """
        Generate (Rx, Ry) angles for hemisphere coverage using Fibonacci spiral.

        Args:
            num_poses: Number of (Rx, Ry) pairs to generate (default 50)

        Returns:
            List of (Rx_deg, Ry_deg) tuples covering the hemisphere
        """
        angles = []

        golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ~2.39996 radians

        for i in range(num_poses):
            # Fibonacci sphere distribution on hemisphere
            t = i / float(num_poses)  # [0, 1)
            phi = np.arccos(1 - t)  # [0, pi/2] for hemisphere (0 to 90° from north)
            theta = golden_angle * i

            # Map spherical coords to Rx, Ry (degrees)
            # phi=0 -> (0, 0), phi=90° -> max spread at equator
            rx_deg = 90 * np.sin(phi) * np.cos(theta)
            ry_deg = 90 * np.sin(phi) * np.sin(theta)

            angles.append((rx_deg, ry_deg))

        rospy.loginfo(f"Generated {len(angles)} (Rx, Ry) orientation pairs")
        return angles

    def _collect_samples_at_position(self):
        """
        Collect num_samples sensor readings and average them.

        Returns:
            dict: Averaged sensor data {sensor_id: (x, y, z), ...}
        """
        samples = {i: [] for i in range(1, 13)}  # sensors 1-12
        collected = 0

        rate = rospy.Rate(100)  # 100Hz polling
        timeout = rospy.Time.now() + rospy.Duration(5.0)  # 5 second timeout

        while collected < self.num_samples and rospy.Time.now() < timeout:
            with self.mutex:
                if self.latest_sensor_data is not None:
                    msg = self.latest_sensor_data
                    for sensor in msg.sensor_data:
                        if 1 <= sensor.id <= 12:
                            samples[sensor.id].append((sensor.x, sensor.y, sensor.z))
                    collected += 1
            rate.sleep()

        if collected < self.num_samples:
            rospy.logwarn(f"Only collected {collected}/{self.num_samples} samples")

        # Average the samples
        averaged = {}
        for sensor_id, data_list in samples.items():
            if len(data_list) > 0:
                avg_x = sum(d[0] for d in data_list) / len(data_list)
                avg_y = sum(d[1] for d in data_list) / len(data_list)
                avg_z = sum(d[2] for d in data_list) / len(data_list)
                averaged[sensor_id] = (avg_x, avg_y, avg_z)
            else:
                averaged[sensor_id] = (0.0, 0.0, 0.0)
                rospy.logwarn(f"No samples collected for sensor {sensor_id}")

        return averaged

    def _setup_csv(self):
        """Setup CSV file with headers."""
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data'
        )
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"ellipsoid_calib_{timestamp}.csv")

        try:
            csv_file = open(csv_path, 'w', newline='')
            writer = csv.writer(csv_file)

            # Header: timestamp, joint7_deg, pos_x, pos_y, pos_z, qx, qy, qz, qw, sensor_1_x, ...
            header = ['timestamp', 'joint7_deg',
                     'pos_x', 'pos_y', 'pos_z',
                     'qx', 'qy', 'qz', 'qw']
            for i in range(1, 13):
                header.extend([f'sensor_{i}_x', f'sensor_{i}_y', f'sensor_{i}_z'])
            writer.writerow(header)

            rospy.loginfo(f"CSV file created: {csv_path}")
            return csv_file, writer

        except Exception as e:
            rospy.logerr(f"Failed to create CSV file: {e}")
            return None

    def _write_csv_row(self, timestamp, pose, sensor_data, joint7_deg=None):
        """Write a data row to CSV."""
        row = [
            timestamp,
            f"{joint7_deg:.2f}" if joint7_deg is not None else "",
            f"{pose.position.x:.4f}", f"{pose.position.y:.4f}", f"{pose.position.z:.4f}",
            f"{pose.orientation.x:.4f}", f"{pose.orientation.y:.4f}",
            f"{pose.orientation.z:.4f}", f"{pose.orientation.w:.4f}"
        ]
        for i in range(1, 13):
            if i in sensor_data:
                x, y, z = sensor_data[i]
                row.extend([f"{x:.2f}", f"{y:.2f}", f"{z:.2f}"])
            else:
                row.extend(["0.00", "0.00", "0.00"])

        self.csv_writer.writerow(row)
        self.csv_file.flush()  # Ensure data is written immediately

    def _stm_uplink_callback(self, msg: StmUplink):
        """Store latest sensor data."""
        with self.mutex:
            self.latest_sensor_data = msg

    def _generate_rotated_pose(self, angle_x_deg, angle_y_deg, angle_z_deg=0):
        """
        Generate a pose with rotations around X, Y, Z axes applied to reference pose.

        Args:
            angle_x_deg: Rotation angle around X axis in degrees
            angle_y_deg: Rotation angle around Y axis in degrees
            angle_z_deg: Rotation angle around Z axis in degrees (default: 0, for future sweep 0->90)

        Returns:
            Pose: New pose with rotated orientation, same position
        """
        # Convert reference quaternion to Euler angles
        ref_quat = [
            self.reference_pose.orientation.x,
            self.reference_pose.orientation.y,
            self.reference_pose.orientation.z,
            self.reference_pose.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(ref_quat)

        # Add the test rotations
        new_roll = roll + math.radians(angle_x_deg)
        new_pitch = pitch + math.radians(angle_y_deg)
        new_yaw = yaw + math.radians(angle_z_deg)

        # Convert back to quaternion
        new_quat = quaternion_from_euler(new_roll, new_pitch, new_yaw)

        # Create new pose with rotated orientation
        pose = Pose()
        pose.position.x = self.reference_pose.position.x
        pose.position.y = self.reference_pose.position.y
        pose.position.z = self.reference_pose.position.z
        pose.orientation.x = new_quat[0]
        pose.orientation.y = new_quat[1]
        pose.orientation.z = new_quat[2]
        pose.orientation.w = new_quat[3]

        return pose

    def _move_cartesian(self, target_pose, max_attempts=3):
        """
        Move to target pose using Cartesian path (like scan_controller.py).

        Args:
            target_pose: Target Pose to move to
            max_attempts: Number of retry attempts

        Returns:
            bool: True if movement succeeded
        """
        for attempt in range(max_attempts):
            waypoints = [target_pose]

            # Sync start state with actual robot state before planning
            current_state = self.robot.get_current_state()
            self.diana7_group.set_start_state(current_state)

            # Compute Cartesian path
            (plan, fraction) = self.diana7_group.compute_cartesian_path(
                waypoints, 0.01, True  # eef_step=0.01m, avoid_collisions=True
            )

            if fraction < 0.9:
                rospy.logwarn(f"Cartesian path fraction {fraction:.2%} < 90%, aborting")
                return False

            rospy.loginfo(f"Cartesian path computed: {fraction:.2%}")

            # Retime trajectory with speed scaling (like scan_controller.py)
            plan = self.diana7_group.retime_trajectory(
                self.robot.get_current_state(),
                plan,
                velocity_scaling_factor=self.speed_scaling,
                acceleration_scaling_factor=self.speed_scaling
            )

            # Execute plan
            success = self.diana7_group.execute(plan, wait=True)
            self.diana7_group.stop()
            self.diana7_group.clear_pose_targets()

            if success:
                # Force state re-sync after successful execution
                rospy.sleep(0.1)  # Small delay for state to update
                return True

            # If failed, re-sync state and retry
            rospy.logwarn(f"Move attempt {attempt+1} failed, re-syncing state...")
            current_state = self.robot.get_current_state()
            self.diana7_group.set_start_state(current_state)

        rospy.logerr(f"Move failed after {max_attempts} attempts")
        return False

    def _config_to_joints(self):
        """Convert home config dict to ordered joint list."""
        joints = []
        for i in range(1, 8):
            joint_name = f'diana7_joint_{i}'
            if joint_name in self.home_config:
                joints.append(self.home_config[joint_name]['radians'])
            else:
                rospy.logwarn(f"Missing {joint_name} in home config, using 0.0")
                joints.append(0.0)
        rospy.loginfo(f"Home joints: {joints}")
        return joints

    def load_home_config(self):
        """Load diana7_home configuration from YAML file."""
        config_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'diana7_home.yaml'
        )

        if not os.path.exists(config_file):
            rospy.logerr(f"Home config not found: {config_file}")
            return None

        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            if 'diana7_home' not in config:
                rospy.logerr("No 'diana7_home' in config file.")
                return None

            return config['diana7_home']

        except Exception as e:
            rospy.logerr(f"Failed to load home config: {e}")
            return None


if __name__ == '__main__':
    sampler = EllipsoidCalibrationSampler()
