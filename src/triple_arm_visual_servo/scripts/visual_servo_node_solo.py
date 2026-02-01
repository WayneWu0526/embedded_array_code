#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_geometry_msgs
import moveit_commander
import numpy as np
import sys
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped, Point, Quaternion
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R
import yaml
import os
import threading
from pose_filter import PoseFilter

def pose_to_matrix(pose):
    res = np.eye(4)
    res[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    res[:3, :3] = R.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]).as_matrix()
    return res

def matrix_to_pose(matrix):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = matrix[:3, 3]
    quat = R.from_matrix(matrix[:3, :3]).as_quat()
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
    return pose



class VisualServoNode:
    def __init__(self):
        rospy.init_node('visual_servo_node')
        self.lock = threading.Lock()
        
        # Load config
        config_path = rospy.get_param('~config_path', '')
        if not config_path:
            # Fallback to default path relative to this script
            config_path = os.path.join(os.path.dirname(__file__), '../config/move_group.yaml')
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # MoveIt setup
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        
        # Scaling parameters
        self.speed_scaling = rospy.get_param('~speed_scaling', 0.01)
        self.acc_scaling = rospy.get_param('~acc_scaling', 0.01)
        
        self.arms = {}
        for arm_cfg in self.config['arms']:
            name = arm_cfg['name']
            
            # Connection with retry
            group = None
            retry_count = 0
            max_retries = 5
            while not rospy.is_shutdown() and retry_count < max_retries:
                try:
                    rospy.loginfo(f"Connecting to MoveGroup '{arm_cfg['move_group']}' (attempt {retry_count+1})...")
                    group = moveit_commander.MoveGroupCommander(arm_cfg['move_group'])
                    break
                except RuntimeError as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        rospy.logwarn(f"[{name}] MoveGroup connection failed, retrying in 2s... ({e})")
                        rospy.sleep(2.0)
                    else:
                        rospy.logerr(f"[{name}] Failed to connect to MoveGroup after {max_retries} attempts.")
                        raise e

            # Ensure we control the intended TCP and plan in world
            group.set_end_effector_link(arm_cfg['ee_link'])
            # group.set_pose_reference_frame('world')

            self.arms[name] = {
                'group': group,
                'config': arm_cfg,
                'filter': PoseFilter(window_size=10),
                'vision_pose': None,
                'trajectory': []
            }

        # Apply initial scaling limits
        self.apply_scaling(self.speed_scaling, self.acc_scaling)

        # Target trajectory subscriber
        from geometry_msgs.msg import PoseArray
        # One topic per arm to support multi-arm control. PoseArray header.frame_id can be 'world' or vision frame.
        self.traj_subs = []
        for name in self.arms:
            topic = f'/servo/{name}/trajectory'
            self.traj_subs.append(rospy.Subscriber(topic, PoseArray, self.trajectory_cb, name))
        # # Backward compatibility: shared topic applies to diana7
        # self.legacy_traj_sub = rospy.Subscriber('/servo/trajectory', PoseArray, self.trajectory_cb, 'diana7')

        # Calibration data
        self.T_world_vision = None
        self.arm_bases_in_world = {} # {arm_name: T_world_base}
        
        rospy.loginfo("Visual Servo Node Initialized")

    def _get_nominal_base(self, arm_name):
        """Lookup the nominal world->base TF from the robot model/TF tree."""
        base_frame = self.arms[arm_name]['config']['base_link']
        try:
            trans = self.tf_buffer.lookup_transform('world', base_frame, rospy.Time(0), rospy.Duration(1.0))
            T = np.eye(4)
            T[:3, 3] = [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]
            T[:3, :3] = R.from_quat([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]).as_matrix()
            return T
        except Exception as e:
            rospy.logwarn(f"Failed to lookup nominal base for {arm_name}: {e}")
            return None

    def apply_scaling(self, vel, acc):
        """Set velocity/acceleration scaling for all arms."""
        self.speed_scaling = vel
        self.acc_scaling = acc
        for arm in self.arms.values():
            arm['group'].set_max_velocity_scaling_factor(vel)
            arm['group'].set_max_acceleration_scaling_factor(acc)
        rospy.loginfo(f"Scaling updated: vel={vel}, acc={acc}")

    def trajectory_cb(self, msg, arm_name):
        rospy.loginfo(f"[{arm_name}] Received trajectory with {len(msg.poses)} points (frame={msg.header.frame_id})")
        # Convert poses into world frame immediately using calibration if needed
        traj_world = []
        for p in msg.poses:
            world_pose = self._pose_to_world(p, msg.header.frame_id)
            if world_pose:
                compensated = self._compensate_target_for_base(world_pose, arm_name)
                traj_world.append(compensated)
        self.arms[arm_name]['trajectory'] = traj_world

    def _update_vision_poses(self, event=None):
        """Poll TF tree for all arms to update vision feedback."""
        # 修正访问路径：从 config['axes'][1]['name'] 获取
        cam_frame = self.config['axes'][1]['name']
        for name, arm in self.arms.items():
            flange_frame = arm['config']['flange_calib']
            try:
                # Lookup transform from camera frame (lab_table) to the flange
                trans = self.tf_buffer.lookup_transform(cam_frame, flange_frame, rospy.Time(0))
                
                pose = Pose()
                pose.position.x = trans.transform.translation.x
                pose.position.y = trans.transform.translation.y
                pose.position.z = trans.transform.translation.z
                
                # Use raw orientation from TF (rectification is now handled in launch/TF tree)
                pose.orientation.x = trans.transform.rotation.x
                pose.orientation.y = trans.transform.rotation.y
                pose.orientation.z = trans.transform.rotation.z
                pose.orientation.w = trans.transform.rotation.w
                
                filtered_pose = arm['filter'].update(pose)
                if filtered_pose:
                    with self.lock:
                        arm['vision_pose'] = filtered_pose
                    
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                # Only log warning occasionally to avoid spam
                pass

    def calibrate(self):
        """
        Step 1: Calibrate T_world_vision.
        diana7_base is the world origin.
        We lookup current TF for flange and compare with vision feedback.
        """
        rospy.loginfo("Starting Calibration (TF based)...")
        
        # We use diana7 as the reference for world-vision calibration
        ref_arm = 'diana7'
        flange_frame = self.arms[ref_arm]['config']['ee_link']
        
        # Wait for vision feedback
        rospy.loginfo(f"Waiting for vision feedback for reference arm '{ref_arm}' from TF...")
        
        # Start background vision update timer
        self.vision_timer = rospy.Timer(rospy.Duration(0.05), self._update_vision_poses)
        
        timeout = rospy.Time.now() + rospy.Duration(15.0)
        while not rospy.is_shutdown():
            with self.lock:
                if self.arms[ref_arm]['vision_pose'] is not None:
                    break
            if rospy.Time.now() > timeout:
                rospy.logerr(f"Vision feedback timeout: TF transform from {self.config['axes'][1]['name']} to {flange_frame} not found.")
                return
            rospy.sleep(0.5)
        rospy.loginfo(f"Received vision feedback for {ref_arm}, proceeding with calibration.")

        # 1. Get T_world_flange from TF
        try:
            trans = self.tf_buffer.lookup_transform('world', flange_frame, rospy.Time(0), rospy.Duration(2.0))
            T_world_flange = pose_to_matrix(matrix_to_pose(np.eye(4))) # init
            T_world_flange[:3, 3] = [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]
            T_world_flange[:3, :3] = R.from_quat([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]).as_matrix()
        except Exception as e:
            rospy.logerr(f"TF lookup failed for calibration: {e}")
            return

        # 2. Get T_vision_flange from vision feedback
        with self.lock:
            v_pose = self.arms[ref_arm]['vision_pose']
        T_vision_flange = pose_to_matrix(v_pose)
        rospy.loginfo(f"flange_pose from vision: \n{T_vision_flange}")

        # 3. Calculate T_world_vision = T_world_flange * inv(T_vision_flange)
        self.T_world_vision = T_world_flange @ np.linalg.inv(T_vision_flange)
        rospy.loginfo(f"Calibrated T_world_vision: \n{self.T_world_vision}")

        # 4. Find arm1 and arm2 base positions in world frame
        for arm_name in ['arm1', 'arm2']:
            rospy.loginfo(f"Calculating base for {arm_name}...")
            with self.lock:
                v_pose = self.arms[arm_name]['vision_pose']
            if v_pose is None:
                rospy.logwarn(f"No vision feedback for {arm_name}, skipping base calculation")
                continue
                
            T_vision_tcp = pose_to_matrix(v_pose)
            T_world_tcp = self.T_world_vision @ T_vision_tcp
            
            try:
                base_frame = self.arms[arm_name]['config']['base_link']
                flange_frame = self.arms[arm_name]['config']['ee_link']
                trans = self.tf_buffer.lookup_transform(base_frame, flange_frame, rospy.Time(0), rospy.Duration(1.0))
                T_base_tcp = np.eye(4)
                T_base_tcp[:3, 3] = [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]
                T_base_tcp[:3, :3] = R.from_quat([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]).as_matrix()
                
                # T_world_base = T_world_tcp * inv(T_base_tcp)
                self.arm_bases_in_world[arm_name] = T_world_tcp @ np.linalg.inv(T_base_tcp)
                rospy.loginfo(f"Base of {arm_name} in world: \n{self.arm_bases_in_world[arm_name]}")
            except Exception as e:
                rospy.logwarn(f"Failed to find base for {arm_name}: {e}")
        
        self._publish_calibration_tf()

    def _publish_calibration_tf(self):
        # Broadcast world -> lab_table (vision frame)
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "world"
        t.child_frame_id = self.config['axes'][1]['name']
        
        pose = matrix_to_pose(self.T_world_vision)
        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z
        t.transform.rotation = pose.orientation
        self.tf_broadcaster.sendTransform(t)
        
        # Broadcast world -> arm1_base, arm2_base
        for arm_name, T in self.arm_bases_in_world.items():
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "world"
            # 增加 _calib 后缀
            t.child_frame_id = f"{self.arms[arm_name]['config']['base_link']}"
            pose = matrix_to_pose(T)
            t.transform.translation.x = pose.position.x
            t.transform.translation.y = pose.position.y
            t.transform.translation.z = pose.position.z
            t.transform.rotation = pose.orientation
            self.tf_broadcaster.sendTransform(t)

    def _pose_to_world(self, pose, frame_id):
        """Convert a pose from given frame to world using calibration/TF."""
        if frame_id in ['', 'world', None]:
            return pose

        # If frame is the vision frame, use calibrated T_world_vision
        cam_frame = self.config['axes'][1]['name']
        if frame_id == cam_frame:
            if self.T_world_vision is None:
                rospy.logwarn("T_world_vision not calibrated yet; dropping pose")
                return None
            T = self.T_world_vision @ pose_to_matrix(pose)
            return matrix_to_pose(T)

        # Fallback: use TF transform
        try:
            ps = PoseStamped()
            ps.header.frame_id = frame_id
            ps.pose = pose
            trans = self.tf_buffer.transform(ps, 'world', rospy.Duration(1.0))
            return trans.pose
        except Exception as e:
            rospy.logwarn(f"Failed to transform pose from {frame_id} to world: {e}")
            return None

    def _compensate_target_for_base(self, target_world, arm_name):
        """Pre-distort target pose so that a calibrated base is compensated when MoveIt uses nominal base."""
        if arm_name not in self.arm_bases_in_world:
            return target_world

        T_world_base_calib = self.arm_bases_in_world[arm_name]
        T_world_base_nom = self._get_nominal_base(arm_name)
        if T_world_base_nom is None:
            return target_world

        # Error transform from nominal to calibrated: T_err = T_calib * inv(T_nom)
        T_err = T_world_base_calib @ np.linalg.inv(T_world_base_nom)

        # Command pose that compensates base error: target_cmd = inv(T_err) * target_desired
        T_target_desired = pose_to_matrix(target_world)
        T_target_cmd = np.linalg.inv(T_err) @ T_target_desired
        return matrix_to_pose(T_target_cmd)

    def iterative_move(self, arm_name, target_pose_world, tol=0.002, max_iter=20):
        """
        Receives target pose in world frame, uses vision feedback to adjust goal iteratively.
        """

        if self.T_world_vision is None:
            rospy.logerr("Calibration required first")
            return False

        group = self.arms[arm_name]['group']
        alpha = 0.5 # Gain for iteration
        
        current_goal_moveit = target_pose_world
        
        for i in range(max_iter):
            # Calculate plan using MoveIt
            waypoints = [current_goal_moveit]
            (plan, fraction) = group.compute_cartesian_path(waypoints, 0.01, True)
            
            if fraction >= 0.9:
                # Retime and execute according to scan_controller.py style
                plan = group.retime_trajectory(self.robot.get_current_state(), plan, 
                                               velocity_scaling_factor=self.speed_scaling,
                                               acceleration_scaling_factor=self.acc_scaling)
                group.execute(plan, wait=True)
            else:
                rospy.logwarn(f"Cartesian plan failed (fraction: {fraction}), fallback to set_pose_target")
                group.set_pose_target(current_goal_moveit)
                group.go(wait=True)

            rospy.sleep(0.5) 
            
            # Get vision feedback
            with self.lock:
                v_pose = self.arms[arm_name]['vision_pose']
            if not v_pose:
                rospy.logwarn("No vision feedback, stopping iteration")
                break
                
            # Convert vision pose to world frame
            T_v_tcp = pose_to_matrix(v_pose)
            T_world_tcp_feedback = self.T_world_vision @ T_v_tcp
            
            # Compute error
            target_mat = pose_to_matrix(target_pose_world)
            dist_err = np.linalg.norm(target_mat[:3, 3] - T_world_tcp_feedback[:3, 3])
            rospy.loginfo(f"Iteration {i}: dist error = {dist_err:.4f}")
            
            if dist_err < tol:
                rospy.loginfo("Converged!")
                group.stop()
                group.clear_pose_targets()
                return True
                
            # Adjust goal: P_new = P_old + alpha * (target - feedback)
            # Position adjustment
            delta_pos = (target_mat[:3, 3] - T_world_tcp_feedback[:3, 3])
            
            # Orientation adjustment using axis-angle representation of the rotation error
            R_feedback = T_world_tcp_feedback[:3, :3]
            R_target = target_mat[:3, :3]
            R_error = R_target @ R_feedback.T
            rot_vec = R.from_matrix(R_error).as_rotvec()
            
            current_goal_mat = pose_to_matrix(current_goal_moveit)
            current_goal_mat[:3, 3] += alpha * delta_pos
            
            # Apply fractional rotation error
            delta_R = R.from_rotvec(alpha * rot_vec).as_matrix()
            current_goal_mat[:3, :3] = delta_R @ current_goal_mat[:3, :3]
            
            current_goal_moveit = matrix_to_pose(current_goal_mat)

        group.stop()
        group.clear_pose_targets()
        return False

    def run_control_loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            for arm_name, data in self.arms.items():
                if data['trajectory']:
                    traj = data['trajectory']
                    data['trajectory'] = []  # consume
                    rospy.loginfo(f"[{arm_name}] Executing trajectory with {len(traj)} points")
                    for i, target_pose in enumerate(traj):
                        rospy.loginfo(f"[{arm_name}] Moving to point {i+1}/{len(traj)}")
                        self.iterative_move(arm_name, target_pose)
                        
            rate.sleep()

if __name__ == '__main__':
    try:
        node = VisualServoNode()
        node.calibrate()
        node.run_control_loop()
    except rospy.ROSInterruptException:
        pass
