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
import numpy as np
from geometry_msgs.msg import Pose, Quaternion
from tf.transformations import quaternion_from_euler, euler_from_quaternion


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

        # Generate test orientations: ±10 degrees around X and Y axes
        self.test_angles = [
            (0, 0),           # home
            (10, 0),          # +10 deg around X
            (-10, 0),         # -10 deg around X
            (0, 10),          # +10 deg around Y
            (0, -10),         # -10 deg around Y
        ]

        # Execute orientation test
        self.run_orientation_test()

        rospy.signal_shutdown("done")

    def run_orientation_test(self):
        """Test orientation generation with ±10 degree rotations."""
        rospy.loginfo("Starting orientation test with ±10 degree rotations...")

        for i, (angle_x_deg, angle_y_deg) in enumerate(self.test_angles):
            rospy.loginfo(f"\n--- Test {i+1}/{len(self.test_angles)}: "
                         f"rot_x={angle_x_deg}°, rot_y={angle_y_deg}° ---")

            # Generate target pose with rotation applied
            target_pose = self._generate_rotated_pose(angle_x_deg, angle_y_deg)

            rospy.loginfo(f"Target orientation: qx={target_pose.orientation.x:.4f}, "
                         f"qy={target_pose.orientation.y:.4f}, "
                         f"qz={target_pose.orientation.z:.4f}, "
                         f"qw={target_pose.orientation.w:.4f}")

            # Move using Cartesian path (like scan_controller.py)
            success = self._move_cartesian(target_pose)
            if success:
                rospy.loginfo(f"Test {i+1} succeeded")
            else:
                rospy.logerr(f"Test {i+1} failed")

            rospy.sleep(0.5)

    def _generate_rotated_pose(self, angle_x_deg, angle_y_deg):
        """
        Generate a pose with rotations around X and Y axes applied to reference pose.

        Args:
            angle_x_deg: Rotation angle around X axis in degrees
            angle_y_deg: Rotation angle around Y axis in degrees

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

        # Convert back to quaternion
        new_quat = quaternion_from_euler(new_roll, new_pitch, yaw)

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
