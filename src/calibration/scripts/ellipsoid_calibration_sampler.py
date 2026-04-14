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

        # Generate test orientations: ±10 degrees around X, Y, Z axes
        # Z rotation will be swept from 0 to 90 degrees in later experiments
        self.test_angles = [
            (0, 0, 0),           # home
            (10, 0, 0),          # +10 deg around X
            (-10, 0, 0),         # -10 deg around X
            (0, 10, 0),          # +10 deg around Y
            (0, -10, 0),         # -10 deg around Y
            (0, 0, 10),          # +10 deg around Z (for future sweep 0->90)
            (0, 0, -10),         # -10 deg around Z
        ]

        # Setup CSV output
        self.csv_file = self._setup_csv()
        if self.csv_file is None:
            rospy.logerr("Failed to setup CSV file. Exiting.")
            sys.exit(1)

        # Execute orientation test with data collection
        self.run_orientation_test()

        # Close CSV and cleanup
        self.csv_file.close()
        rospy.loginfo("Data collection complete.")

        rospy.signal_shutdown("done")

    def run_orientation_test(self):
        """Test orientation generation with ±10 degree rotations and sensor collection."""
        rospy.loginfo("Starting orientation test with ±10 degree rotations...")

        for i, (angle_x_deg, angle_y_deg, angle_z_deg) in enumerate(self.test_angles):
            rospy.loginfo(f"\n--- Test {i+1}/{len(self.test_angles)}: "
                         f"rot_x={angle_x_deg}°, rot_y={angle_y_deg}°, rot_z={angle_z_deg}° ---")

            # Generate target pose with rotation applied
            target_pose = self._generate_rotated_pose(angle_x_deg, angle_y_deg, angle_z_deg)

            rospy.loginfo(f"Target orientation: qx={target_pose.orientation.x:.4f}, "
                         f"qy={target_pose.orientation.y:.4f}, "
                         f"qz={target_pose.orientation.z:.4f}, "
                         f"qw={target_pose.orientation.w:.4f}")

            # Move using Cartesian path (like scan_controller.py)
            success = self._move_cartesian(target_pose)
            if success:
                rospy.loginfo(f"Test {i+1} move succeeded")
            else:
                rospy.logerr(f"Test {i+1} move failed")
                continue

            # Wait for vibrations to settle
            rospy.sleep(self.settling_time)

            # Collect sensor data: 10 samples averaged
            rospy.loginfo(f"Collecting {self.num_samples} sensor samples...")
            sensor_data = self._collect_samples_at_position()

            # Get current pose after movement
            current_pose = self.diana7_group.get_current_pose().pose

            # Write to CSV
            self._write_csv_row(
                timestamp=datetime.now().isoformat(),
                pose=current_pose,
                sensor_data=sensor_data,
                rot_x=angle_x_deg,
                rot_y=angle_y_deg,
                rot_z=angle_z_deg
            )

            rospy.loginfo(f"Test {i+1} completed and saved.")

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

            # Header: timestamp, rot_x, rot_y, rot_z, pos_x, pos_y, pos_z, qx, qy, qz, qw, sensor_1_x, sensor_1_y, sensor_1_z, ...
            header = ['timestamp', 'rot_x_deg', 'rot_y_deg', 'rot_z_deg',
                     'pos_x', 'pos_y', 'pos_z',
                     'qx', 'qy', 'qz', 'qw']
            for i in range(1, 13):
                header.extend([f'sensor_{i}_x', f'sensor_{i}_y', f'sensor_{i}_z'])
            writer.writerow(header)

            rospy.loginfo(f"CSV file created: {csv_path}")
            return csv_file

        except Exception as e:
            rospy.logerr(f"Failed to create CSV file: {e}")
            return None

    def _write_csv_row(self, timestamp, pose, sensor_data, rot_x, rot_y, rot_z):
        """Write a data row to CSV."""
        row = [
            timestamp,
            f"{rot_x:.3f}", f"{rot_y:.3f}", f"{rot_z:.3f}",
            f"{pose.position.x:.6f}", f"{pose.position.y:.6f}", f"{pose.position.z:.6f}",
            f"{pose.orientation.x:.6f}", f"{pose.orientation.y:.6f}",
            f"{pose.orientation.z:.6f}", f"{pose.orientation.w:.6f}"
        ]
        for i in range(1, 13):
            if i in sensor_data:
                x, y, z = sensor_data[i]
                row.extend([f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"])
            else:
                row.extend(["0.000", "0.000", "0.000"])

        self.csv_file.writerow(row)
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

    def _move_cartesian(self, target_pose):
        """
        Move to target pose using Cartesian path (like scan_controller.py).

        Args:
            target_pose: Target Pose to move to

        Returns:
            bool: True if movement succeeded
        """
        waypoints = [target_pose]

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

        return success

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
