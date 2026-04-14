#!/usr/bin/env python3
"""
Move diana7 to home position.

Usage:
    roslaunch calibration ellipsoid_calibration.launch
"""

import sys
import rospy
import moveit_commander
import yaml
import os


class SimpleCalibrationSampler:
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

        rospy.signal_shutdown("done")

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
    sampler = SimpleCalibrationSampler()
