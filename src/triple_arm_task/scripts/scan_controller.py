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
from trajectory_loader import TrajectoryLoader
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_msgs.msg import PositionIKRequest, RobotState
from sensor_msgs.msg import JointState

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
        # self.relative_scale = rospy.get_param('~relative_scale', 1.0)
        self.num_samples = rospy.get_param('~num_samples', 10)
        self.optimize_frequency = rospy.get_param('~optimize_frequency', 10)
        
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

        rospy.loginfo("Waiting for /compute_ik service...")
        rospy.wait_for_service('/compute_ik')
        self.ik_srv = rospy.ServiceProxy('/compute_ik', GetPositionIK)

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

    def optimize_null_space(self):
        """
        尝试寻找一个保持当前末端位姿不变，但关节尽量靠近 0 位的构型（依赖 TRAC-IK Distance 模式）
        """
        rospy.loginfo("Performing null-space optimization for diana7...")
        
        # 获取当前位姿
        current_pose = self.diana7_group.get_current_pose()
        joint_names = self.diana7_group.get_active_joints()
        
        # 构造 IK 请求
        req = GetPositionIKRequest()
        req.ik_request.group_name = "diana7"
        req.ik_request.pose_stamped = current_pose
        req.ik_request.avoid_collisions = True
        
        # 设置 Seed State 为全 0，TRAC-IK 的 Distance 模式会寻找最靠近 Seed 的解
        seed_state = RobotState()
        seed_state.joint_state.name = joint_names
        seed_state.joint_state.position = [0.0] * len(joint_names)
        req.ik_request.robot_state = seed_state
        
        try:
            resp = self.ik_srv(req)
            if resp.error_code.val == resp.error_code.SUCCESS:
                # 提取 diana7 的关节值
                optimized_joints = []
                for name in joint_names:
                    if name in resp.solution.joint_state.name:
                        idx = resp.solution.joint_state.name.index(name)
                        optimized_joints.append(resp.solution.joint_state.position[idx])
                
                if len(optimized_joints) == len(joint_names):
                    rospy.loginfo("Optimized configuration found. Executing joint-space move...")
                    # 应用速度缩放
                    self.diana7_group.set_max_velocity_scaling_factor(self.speed_scaling)
                    self.diana7_group.set_joint_value_target(optimized_joints)
                    success = self.diana7_group.go(wait=True)
                    self.diana7_group.stop()
                    return success
                else:
                    rospy.logwarn("Could not find all joint values in IK solution.")
                    return False
            else:
                rospy.logwarn(f"Null-space optimization IK failed (Error code: {resp.error_code.val}). Keeping current config.")
                return False
        except Exception as e:
            rospy.logerr(f"Exception in optimize_null_space: {e}")
            return False

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
        if 'arm1' in self.start_positions:
            rospy.loginfo("Moving arm1 to start pose...")
            move_to_pose(self.arm1_group, self.start_positions['arm1'])
        
        # Move arm2
        if 'arm2' in self.start_positions:
            rospy.loginfo("Moving arm2 to start pose...")
            move_to_pose(self.arm2_group, self.start_positions['arm2'])

        # Move diana7
        if 'diana7' in self.start_positions:
            rospy.loginfo("Moving diana7 to start pose...")
            move_to_pose(self.diana7_group, self.start_positions['diana7'])
            # 这里的起始点虽然通常是手动设定的，但到达后执行一次零空间优化可以确保
            # TRAC-IK 找到此位姿下最接近 0 位的构型，为后续扫描腾出关节空间。
            self.optimize_null_space()
            
        rospy.loginfo("Reached start positions.")

    def load_path(self):
        points = []
        if not os.path.exists(self.path_file):
            rospy.logerr(f"Path file not found: {self.path_file}")
            return points

        ext = os.path.splitext(self.path_file)[1].lower()
        if ext == '.csv':
            rospy.loginfo(f"Loading CSV trajectory: {self.path_file}")
            points = TrajectoryLoader.load_csv(self.path_file)
        elif ext == '.json':
            # json轨迹的读取方法，其中每个json文件中都有设置的scale，可以进行缩放（默认对位置和姿态同时缩放）
            rospy.loginfo(f"Loading JSON relative trajectory: {self.path_file}")
            raw_points, scale = TrajectoryLoader.load_json(self.path_file)
            
            # For JSON relative trajectory, we need a base pose and scale
            # We use the current pose of diana7 as the base
            try:
                base_pose_stamped = self.diana7_group.get_current_pose()
                base_pose = base_pose_stamped.pose
                points = TrajectoryLoader.apply_relative_trajectory(raw_points, base_pose, scale)
                rospy.loginfo(f"Transformed {len(points)} relative points using base pose.")
            except Exception as e:
                rospy.logerr(f"Failed to get base pose for relative trajectory: {e}")
                return []
        else:
            rospy.logerr(f"Unsupported trajectory format: {ext}")
            
        return points

    def execute_scan(self):
        self.go_to_start()
        
        # return
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
                
                # 每 K 个点进行一次零空间优化，重置关节构型以避免关节限位
                if i > 0 and i % self.optimize_frequency == 0:
                    rospy.loginfo(f"Triggering periodic null-space optimization at point {i}...")
                    self.optimize_null_space()

                progress = (i + 1) / total_points * 100
                remaining = total_points - (i + 1)
                rospy.loginfo(f"--- Progress: {i+1}/{total_points} ({progress:.1f}%) | Remaining: {remaining} ---")
                rospy.loginfo(f"Moving to point {i+1}...")
                
                # ---------------------------------------------------------
                # 使用零空间偏置 IK (Null-space Biased IK)
                # ---------------------------------------------------------
                joint_names = self.diana7_group.get_active_joints()
                req = GetPositionIKRequest()
                req.ik_request.group_name = "diana7"
                req.ik_request.pose_stamped.header.frame_id = "world"
                req.ik_request.pose_stamped.pose = target_pose
                req.ik_request.avoid_collisions = True
                
                # 设置 Seed 为全 0，诱导 TRAC-IK 寻找最舒展的构型
                seed_state = RobotState()
                seed_state.joint_state.name = joint_names
                seed_state.joint_state.position = [0.0] * len(joint_names)
                req.ik_request.robot_state = seed_state
                
                plan_success = False
                try:
                    resp = self.ik_srv(req)
                    if resp.error_code.val == resp.error_code.SUCCESS:
                        # 提取关节解并过滤（应对 19 关节返回问题）
                        joint_dictionary = dict(zip(resp.solution.joint_state.name, resp.solution.joint_state.position))
                        joint_goal = [joint_dictionary[name] for name in joint_names]
                        
                        self.diana7_group.set_joint_value_target(joint_goal)
                        plan_success = True
                    else:
                        rospy.logerr(f"IK failed for point {i} (Error: {resp.error_code.val})")
                except Exception as e:
                    rospy.logerr(f"IK service error: {e}")

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
                        success = self.diana7_group.go(wait=True)
                    else:
                        rospy.logwarn("Planning failed! Cannot execute.")
                        success = False
                else:
                    # Automatic mode
                    if plan_success:
                        success = self.diana7_group.go(wait=True)
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
