#!/usr/bin/env python3
"""
Save diana7 home position to config file.

Run this script after manually moving diana7 to a safe home position.
It will read the current joint states and save them to a YAML config file.

Usage:
    rosrun calibration save_diana7_home.py

Output:
    config/diana7_home.yaml - Contains joint positions for diana7
"""

import sys
import rospy
import yaml
import os
from sensor_msgs.msg import JointState


class SaveDiana7Home:
    def __init__(self):
        rospy.init_node('save_diana7_home', anonymous=True)

        # Output config file path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(os.path.dirname(script_dir), 'config')
        self.config_file = os.path.join(config_dir, 'diana7_home.yaml')

        # Ensure config directory exists
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        self.joint_names = [
            'diana7_joint_1', 'diana7_joint_2', 'diana7_joint_3',
            'diana7_joint_4', 'diana7_joint_5', 'diana7_joint_6',
            'diana7_joint_7'
        ]
        self.joint_positions = {}

        # Subscribe to joint states
        self.joint_sub = rospy.Subscriber(
            '/diana7/joint_states',
            JointState,
            self.joint_callback,
            queue_size=1
        )

        rospy.loginfo("Waiting for joint_states...")
        rospy.loginfo("Make sure diana7 is at the desired home position!")

        # Wait for joint data
        timeout = 30.0
        start_time = rospy.Time.now()
        while not rospy.is_shutdown() and len(self.joint_positions) < 7:
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                rospy.logerr("Timeout waiting for joint_states!")
                sys.exit(1)
            rospy.sleep(0.1)

        # Save to file
        self.save_config()

    def joint_callback(self, msg: JointState):
        """Store joint positions when received."""
        for name, pos in zip(msg.name, msg.position):
            if name in self.joint_names:
                self.joint_positions[name] = pos

    def save_config(self):
        """Save joint positions to YAML config file."""
        # Create config dict in rosparam style
        config_data = {
            'diana7_joint_1': self.joint_positions.get('diana7_joint_1', 0.0),
            'diana7_joint_2': self.joint_positions.get('diana7_joint_2', 0.0),
            'diana7_joint_3': self.joint_positions.get('diana7_joint_3', 0.0),
            'diana7_joint_4': self.joint_positions.get('diana7_joint_4', 0.0),
            'diana7_joint_5': self.joint_positions.get('diana7_joint_5', 0.0),
            'diana7_joint_6': self.joint_positions.get('diana7_joint_6', 0.0),
            'diana7_joint_7': self.joint_positions.get('diana7_joint_7', 0.0),
        }

        # Convert to degrees for readability
        config_degrees = {}
        for k, v in config_data.items():
            config_degrees[k] = {
                'radians': v,
                'degrees': round(v * 180.0 / 3.14159265359, 2)
            }

        output = {
            'diana7_home': config_degrees
        }

        with open(self.config_file, 'w') as f:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)

        rospy.loginfo(f"Saved diana7 home position to: {self.config_file}")
        for name, data in config_degrees.items():
            rospy.loginfo(f"  {name}: {data['degrees']} deg ({data['radians']} rad)")


if __name__ == '__main__':
    SaveDiana7Home()
