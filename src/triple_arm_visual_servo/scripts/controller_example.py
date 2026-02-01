#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
import json
import csv
import os
import datetime
from serial_processor.srv import GetHallData
from triple_arm_task.msg import ScanData
from std_msgs.msg import Header

class TripleArmScanner:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('triple_arm_scanner', anonymous=True)

        self.robot = moveit_commander.RobotCommander()

        # Parameters
        self.path_file = rospy.get_param('~path_file', '')
        self.start_positions = rospy.get_param('~start_positions', {})
        self.loop_path = rospy.get_param('~loop_path', False)
        self.connect_wait_time = rospy.get_param('~connect_wait_time', 5.0)
        self.max_retries = rospy.get_param('~max_retries', 5)
        self.manual_confirm = rospy.get_param('~manual_confirm', False)
        self.speed_scaling = rospy.get_param('~speed_scaling', 0.01)
        self.num_samples = rospy.get_param('~num_samples', 10)
        
        # JSON Logging Setup
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_filename = os.path.join(self.data_dir, f"scan_results_{timestamp_str}.json")
        self.scan_results = []
        
        rospy.loginfo(f"Data logging initialized to {self.json_filename}")
        rospy.on_shutdown(self.cleanup)

        # Wait for move_group to be ready (especially when launched together)
        rospy.loginfo(f"Waiting {self.connect_wait_time}s for move_group action servers...")
        rospy.sleep(self.connect_wait_time) 
        
        # MoveIt Groups
        retry_count = 0
        while not rospy.is_shutdown() and retry_count < self.max_retries:
            try:
                self.diana7_group = moveit_commander.MoveGroupCommander("diana7")
                self.arm1_group = moveit_commander.MoveGroupCommander("arm1")
                self.arm2_group = moveit_commander.MoveGroupCommander("arm2")
                break
            except RuntimeError as e:
                retry_count += 1
                rospy.logwarn(f"MoveGroupCommander connection attempt {retry_count}/{self.max_retries} failed: {e}. Retrying in {self.connect_wait_time}s...")
                rospy.sleep(self.connect_wait_time)
        
        if retry_count == self.max_retries:
            rospy.logerr("Failed to connect to move_group after multiple attempts. Exiting.")
            sys.exit(1)

        # Set speed scaling for safety
        self.diana7_group.set_max_velocity_scaling_factor(self.speed_scaling)
        self.diana7_group.set_max_acceleration_scaling_factor(self.speed_scaling)
        self.arm1_group.set_max_velocity_scaling_factor(self.speed_scaling)
        self.arm2_group.set_max_velocity_scaling_factor(self.speed_scaling)

        # Set end effector links to ensure we control the TCP
        # These names should match the links in your URDF/SRDF
        try:
            self.diana7_group.set_end_effector_link("diana7_magnetometer_array_tcp_link")
            self.arm1_group.set_end_effector_link("arm1_electronic_magnet_tcp_link")
            self.arm2_group.set_end_effector_link("arm2_electronic_magnet_tcp_link")
            rospy.loginfo("End effector links set to TCP links.")
        except Exception as e:
            rospy.logwarn(f"Could not set end effector links: {e}. Using defaults.")
        
        # Service Client
        rospy.wait_for_service('get_hall_data')
        self.get_hall_data_srv = rospy.ServiceProxy('get_hall_data', GetHallData)

        # Publisher
        self.data_pub = rospy.Publisher('scan_data', ScanData, queue_size=10)

    def cleanup(self):
        self.save_json()
        rospy.loginfo("Cleanup complete.")

    def save_json(self):
        try:
            with open(self.json_filename, 'w') as f:
                json.dump(self.scan_results, f, indent=4)
        except Exception as e:
            rospy.logerr(f"Failed to save JSON: {e}")

    def go_to_start(self):
        rospy.loginfo("Moving to start positions...")
        
        def move_to_pose(group, pose_data):
            pose = geometry_msgs.msg.Pose()
            pose.position.x = pose_data['position'][0]
            pose.position.y = pose_data['position'][1]
            pose.position.z = pose_data['position'][2]
            pose.orientation.x = pose_data['orientation'][0]
            pose.orientation.y = pose_data['orientation'][1]
            pose.orientation.z = pose_data['orientation'][2]
            pose.orientation.w = pose_data['orientation'][3]
            
            # 方案 A: 强制使用笛卡尔路径到达起始点
            waypoints = [pose]
            # 参数说明: waypoints, eef_step, avoid_collisions
            (plan, fraction) = group.compute_cartesian_path(waypoints, 0.01, True)
            
            if fraction >= 0.9:
                rospy.loginfo(f"Executing Cartesian path to start (fraction: {fraction:.2%})...")
                # 显式重新计算轨迹时间，以应用速度和加速度缩放
                plan = group.retime_trajectory(self.robot.get_current_state(), plan, 
                                               velocity_scaling_factor=self.speed_scaling,
                                               acceleration_scaling_factor=self.speed_scaling)
                success = group.execute(plan, wait=True)
            else:
                rospy.logerr(f"Cartesian path to start failed (fraction: {fraction:.2%}). Aborting to prevent large joint reconfigurations.")
                success = False

            group.stop()
            group.clear_pose_targets()
            return success

        # Move arm1
        # if 'arm1' in self.start_positions:
        #     rospy.loginfo("Moving arm1 to start pose...")
        #     move_to_pose(self.arm1_group, self.start_positions['arm1'])
        
        # Move arm2
        # if 'arm2' in self.start_positions:
        #     rospy.loginfo("Moving arm2 to start pose...")
        #     move_to_pose(self.arm2_group, self.start_positions['arm2'])

        # Move diana7
        if 'diana7' in self.start_positions:
            rospy.loginfo("Moving diana7 to start pose...")
            move_to_pose(self.diana7_group, self.start_positions['diana7'])
            
        rospy.loginfo("Reached start positions.")

    def load_path(self):
        points = []
        if not os.path.exists(self.path_file):
            rospy.logerr(f"Path file not found: {self.path_file}")
            return points

        with open(self.path_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pose = geometry_msgs.msg.Pose()
                pose.position.x = float(row['x'])
                pose.position.y = float(row['y'])
                pose.position.z = float(row['z'])
                pose.orientation.x = float(row['qx'])
                pose.orientation.y = float(row['qy'])
                pose.orientation.z = float(row['qz'])
                pose.orientation.w = float(row['qw'])
                points.append(pose)
        return points

    def execute_scan(self):
        self.go_to_start()
        
        points = self.load_path()
        rospy.loginfo(f"Loaded {len(points)} points from path.")
        
        # 减小容差以保证精度
        self.diana7_group.set_goal_position_tolerance(0.005) # 5mm
        self.diana7_group.set_goal_orientation_tolerance(0.02) # ~1.1 degrees

        while not rospy.is_shutdown():
            total_points = len(points)
            for i, target_pose in enumerate(points):
                if rospy.is_shutdown():
                    break
                
                progress = (i + 1) / total_points * 100
                remaining = total_points - (i + 1)
                rospy.loginfo(f"--- Progress: {i+1}/{total_points} ({progress:.1f}%) | Remaining: {remaining} ---")
                rospy.loginfo(f"Moving to point {i+1}...")
                
                # ---------------------------------------------------------
                # 方案 A: 纯笛卡尔路径规划 (Pure Cartesian Path)
                # 强制末端走直线，并禁用 PTP 回退逻辑，确保不会出现构型跳变。
                # ---------------------------------------------------------
                waypoints = [target_pose]
                # eef_step=0.01: 1cm 步长插值
                # avoid_collisions=True: 开启碰撞检测
                (plan, fraction) = self.diana7_group.compute_cartesian_path(
                                    waypoints,   
                                    0.01,        
                                    True)        

                plan_success = False
                if fraction == 1.0:
                    rospy.loginfo("Cartesian path computed successfully (100%).")
                    # 显式重新计算轨迹时间，以应用速度和加速度缩放
                    plan = self.diana7_group.retime_trajectory(self.robot.get_current_state(), plan, 
                                                               velocity_scaling_factor=self.speed_scaling,
                                                               acceleration_scaling_factor=self.speed_scaling)
                    plan_success = True
                else:
                    rospy.logerr(f"Cartesian path incomplete (fraction={fraction:.2f}). Skipping this point to ensure safety.")
                    plan_success = False

                if self.manual_confirm:
                    if plan_success:
                        rospy.loginfo("Plan found! Visualizing in RViz...")
                        # Wait for user input
                        print(f"\n>>> Ready to move to point {i}. Check RViz for planned path.")
                        user_input = input(">>> Press ENTER to execute, 's' to skip, 'q' to quit: ")
                        
                        if user_input.lower() == 'q':
                            rospy.loginfo("User aborted task.")
                            rospy.signal_shutdown("User aborted")
                            return
                        elif user_input.lower() == 's':
                            rospy.loginfo("Skipping point...")
                            continue
                        
                        # Execute the plan
                        success = self.diana7_group.execute(plan, wait=True)
                    else:
                        rospy.logwarn("Planning failed! Cannot execute.")
                        success = False
                else:
                    # Automatic mode
                    if plan_success:
                        success = self.diana7_group.execute(plan, wait=True)
                    else:
                        success = False

                self.diana7_group.stop()
                self.diana7_group.clear_pose_targets()


                if success:
                    self.collect_and_publish_data(f"point_{i}")
                else:
                    rospy.logwarn(f"Failed to reach point {i}")
            
            # 路径执行结束后回到初始点
            rospy.loginfo("Path execution finished. Returning to initial start positions...")
            self.go_to_start()

            if not self.loop_path:
                rospy.loginfo("Task completed. Shutting down...")
                rospy.signal_shutdown("Scan task finished")
                break
            rospy.loginfo("Path completed. Looping...")

    def collect_and_publish_data(self, target_id):
        try:
            # Multi-sample averaging
            rospy.loginfo(f"Collecting {self.num_samples} samples for {target_id}...")
            
            avg_sensors = [geometry_msgs.msg.Vector3(0, 0, 0) for _ in range(12)]
            
            for s_idx in range(self.num_samples):
                if rospy.is_shutdown():
                    return
                response = self.get_hall_data_srv()
                for i in range(12):
                    avg_sensors[i].x += response.sensors[i].x
                    avg_sensors[i].y += response.sensors[i].y
                    avg_sensors[i].z += response.sensors[i].z
                # Optional: small sleep to ensure fresh data if the service is too fast
                # rospy.sleep(0.01) 
            
            # Calculate average
            for i in range(12):
                avg_sensors[i].x /= self.num_samples
                avg_sensors[i].y /= self.num_samples
                avg_sensors[i].z /= self.num_samples

            now = rospy.Time.now()
            
            # Get current poses
            diana7_pose = self.diana7_group.get_current_pose().pose
            arm1_pose = self.arm1_group.get_current_pose().pose
            arm2_pose = self.arm2_group.get_current_pose().pose
            
            # Create message
            msg = ScanData()
            msg.header = Header()
            msg.header.stamp = now
            msg.current_target_id = target_id
            msg.diana7_tool_pose = diana7_pose
            msg.arm1_tool_pose = arm1_pose
            msg.arm2_tool_pose = arm2_pose
            msg.hall_data = avg_sensors
            
            self.data_pub.publish(msg)
            
            # Prepare JSON entry
            entry = {
                "timestamp": now.to_sec(),
                "target_id": target_id,
                "num_samples": self.num_samples,
                "poses": {
                    "diana7": {
                        "position": [diana7_pose.position.x, diana7_pose.position.y, diana7_pose.position.z],
                        "orientation": [diana7_pose.orientation.x, diana7_pose.orientation.y, diana7_pose.orientation.z, diana7_pose.orientation.w]
                    },
                    "arm1": {
                        "position": [arm1_pose.position.x, arm1_pose.position.y, arm1_pose.position.z],
                        "orientation": [arm1_pose.orientation.x, arm1_pose.orientation.y, arm1_pose.orientation.z, arm1_pose.orientation.w]
                    },
                    "arm2": {
                        "position": [arm2_pose.position.x, arm2_pose.position.y, arm2_pose.position.z],
                        "orientation": [arm2_pose.orientation.x, arm2_pose.orientation.y, arm2_pose.orientation.z, arm2_pose.orientation.w]
                    }
                },
                "hall_data": [{"x": s.x, "y": s.y, "z": s.z} for s in avg_sensors]
            }
            
            self.scan_results.append(entry)
            self.save_json() # Save incrementally
            
            rospy.loginfo(f"Averaged data ({self.num_samples} samples) saved to JSON for {target_id}")
            
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

if __name__ == '__main__':
    try:
        scanner = TripleArmScanner()
        scanner.execute_scan()
    except rospy.ROSInterruptException:
        pass
