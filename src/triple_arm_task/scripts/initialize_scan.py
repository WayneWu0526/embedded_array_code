#!/usr/bin/env python3

import rospy
import moveit_commander
import sys
import yaml
import os
import rospkg
import roslaunch
import time
import geometry_msgs.msg

def initialize_scan_positions():
    """
    Initializes the scan by launching the robot bringup, getting current poses,
    and saving them as start positions for the scan controller.
    """
    # Initialize moveit_commander
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('initialize_scan_positions', anonymous=True)

    # 1. Launch triple_arm_bringup
    rospy.loginfo("Launching triple_arm_bringup...")
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    
    rospack = rospkg.RosPack()
    try:
        bringup_pkg_path = rospack.get_path('zlab_robots_bringup')
    except rospkg.ResourceNotFound:
        rospy.logerr("Package 'zlab_robots_bringup' not found.")
        return

    launch_file = os.path.join(bringup_pkg_path, 'launch', 'triple_arm', 'triple_arm_bringup.launch')
    
    if not os.path.exists(launch_file):
        rospy.logerr(f"Launch file not found: {launch_file}")
        return

    # Launch the bringup file
    launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_file])
    launch.start()
    
    rospy.loginfo("Waiting for MoveIt groups to be available...")
    
    # 2. Connect to MoveGroups
    groups = {
        'diana7': None,
        'arm1': None,
        'arm2': None
    }
    
    # Wait loop for MoveGroups
    timeout = 60 # seconds
    start_time = time.time()
    
    while not rospy.is_shutdown():
        all_connected = True
        for name in groups.keys():
            if groups[name] is None:
                try:
                    # Try to connect
                    groups[name] = moveit_commander.MoveGroupCommander(name)
                    rospy.loginfo(f"Connected to {name}")
                except RuntimeError:
                    all_connected = False
                except Exception as e:
                    # Sometimes other exceptions occur during startup
                    all_connected = False
        
        if all_connected:
            break
            
        if time.time() - start_time > timeout:
            rospy.logerr("Timeout waiting for MoveGroups")
            launch.shutdown()
            return
            
        rospy.sleep(2.0)

    # 3. Get Poses
    start_positions = {}
    
    try:
        # Give a little more time for everything to settle
        rospy.sleep(2.0)
        
        for name, group in groups.items():
            rospy.loginfo(f"Getting pose for {name}...")
            
            # Set end effector links to match scan_controller.py
            try:
                if name == 'diana7':
                    group.set_end_effector_link("diana7_magnetometer_array_tcp_link")
                elif name == 'arm1':
                    group.set_end_effector_link("arm1_electronic_magnet_tcp_link")
                elif name == 'arm2':
                    group.set_end_effector_link("arm2_electronic_magnet_tcp_link")
            except Exception as e:
                rospy.logwarn(f"Could not set end effector link for {name}: {e}. Using default.")

            # Get current pose
            pose_stamped = group.get_current_pose()
            pose = pose_stamped.pose
            
            start_positions[name] = {
                'position': [pose.position.x, pose.position.y, pose.position.z],
                'orientation': [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            }
            rospy.loginfo(f"{name} pose: {start_positions[name]}")

        # 4. Save to YAML
        try:
            task_pkg_path = rospack.get_path('triple_arm_task')
        except rospkg.ResourceNotFound:
            rospy.logerr("Package 'triple_arm_task' not found.")
            return

        config_path = os.path.join(task_pkg_path, 'config', 'start_positions.yaml')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        data_to_save = {'start_positions': start_positions}
        
        with open(config_path, 'w') as f:
            yaml.dump(data_to_save, f, default_flow_style=False)
            
        rospy.loginfo(f"Successfully saved start positions to {config_path}")
        
    except Exception as e:
        rospy.logerr(f"An error occurred during pose retrieval or saving: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        rospy.loginfo("Shutting down...")
        launch.shutdown()
        moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    initialize_scan_positions()
