#!/usr/bin/env python3
import os
import json
import math
import rospy
import tf2_ros
from geometry_msgs.msg import PoseArray, Pose


def load_trajectory(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Trajectory file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def to_pose_array(traj_data):
    frame_id = traj_data.get("frame_id", "world")
    poses_data = traj_data.get("poses", [])

    msg = PoseArray()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id

    for p in poses_data:
        pose = Pose()
        pos = p.get("position", {})
        ori = p.get("orientation", {})
        pose.position.x = float(pos.get("x", 0.0))
        pose.position.y = float(pos.get("y", 0.0))
        pose.position.z = float(pos.get("z", 0.0))
        pose.orientation.x = float(ori.get("x", 0.0))
        pose.orientation.y = float(ori.get("y", 0.0))
        pose.orientation.z = float(ori.get("z", 0.0))
        pose.orientation.w = float(ori.get("w", 1.0))
        msg.poses.append(pose)

    return msg


def _scale_relative_orientation(q_rel, scale):
    """Scale a relative quaternion by scale using axis-angle."""
    x, y, z, w = q_rel
    # Normalize to be safe
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return 0.0, 0.0, 0.0, 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    # Convert to axis-angle
    w = max(min(w, 1.0), -1.0)
    angle = 2.0 * math.acos(w)
    if abs(angle) < 1e-9:
        return 0.0, 0.0, 0.0, 1.0
    s = math.sqrt(1.0 - w * w)
    if s < 1e-9:
        axis = (1.0, 0.0, 0.0)
    else:
        axis = (x / s, y / s, z / s)

    # Scale the rotation angle
    angle_scaled = angle * scale
    half = 0.5 * angle_scaled
    sin_half = math.sin(half)
    return axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, math.cos(half)


def _quat_multiply(q1, q2):
    """Hamilton product q = q1 * q2."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _quat_conjugate(q):
    x, y, z, w = q
    return (-x, -y, -z, w)


def _rotate_vector_by_quat(v, q):
    """Rotate vector v by quaternion q (assumed to be unit)."""
    vx, vy, vz = v
    vq = (vx, vy, vz, 0.0)
    q_conj = _quat_conjugate(q)
    r = _quat_multiply(_quat_multiply(q, vq), q_conj)
    return r[0], r[1], r[2]


def apply_relative_trajectory(msg, base_pose, scale):
    """Apply pose = pose_0 + R(q0) * (k * pos_rel) and scaled relative rotation."""
    q0 = (
        base_pose.orientation.x,
        base_pose.orientation.y,
        base_pose.orientation.z,
        base_pose.orientation.w,
    )
    # Normalize q0 to be safe
    norm = math.sqrt(q0[0] * q0[0] + q0[1] * q0[1] + q0[2] * q0[2] + q0[3] * q0[3])
    if norm == 0.0:
        q0 = (0.0, 0.0, 0.0, 1.0)
    else:
        q0 = (q0[0] / norm, q0[1] / norm, q0[2] / norm, q0[3] / norm)
    for pose in msg.poses:
        # Position: p_world = p0 + R(q0) * (k * p_rel)
        rel = (scale * pose.position.x, scale * pose.position.y, scale * pose.position.z)
        dx, dy, dz = _rotate_vector_by_quat(rel, q0)
        pose.position.x = base_pose.position.x + dx
        pose.position.y = base_pose.position.y + dy
        pose.position.z = base_pose.position.z + dz

        # Orientation: q = q0 * (q_rel^k)
        q_rel = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        q_rel_scaled = _scale_relative_orientation(q_rel, scale)
        q = _quat_multiply(q0, q_rel_scaled)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q


def main():
    rospy.init_node("trajectory_executor")

    traj_pub = rospy.Publisher("/servo/diana7/trajectory", PoseArray, queue_size=1)

    # 启动 TF 监听并等待校准完成（lab_table 出现）
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    rospy.loginfo("Waiting for TF 'world' -> 'lab_table' to become available...")
    rate = rospy.Rate(1)
    calibration_done = False
    while not rospy.is_shutdown() and not calibration_done:
        try:
            tf_buffer.lookup_transform('world', 'lab_table', rospy.Time(0), rospy.Duration(0.5))
            rospy.loginfo("Calibration detected! T_world_vision (lab_table) is now available.")
            calibration_done = True
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.loginfo("Still waiting for calibration (lab_table frame)...")
            rate.sleep()
            continue

    if rospy.is_shutdown():
        return

    # 读取参数或使用默认轨迹文件路径
    default_path = \
        "/home/zhang/embedded_array_ws/src/triple_arm_visual_servo/config/trajectories/ring_12.json"
    traj_file = rospy.get_param("~trajectory_file", default_path)
    ee_link = rospy.get_param("~ee_link", "diana7_ee_link")
    scale = float(rospy.get_param("~relative_scale", 0.5))
    rospy.loginfo(f"Loading trajectory file: {traj_file}")

    try:
        traj_data = load_trajectory(traj_file)
        msg = to_pose_array(traj_data)
    except Exception as e:
        rospy.logerr(f"Failed to load trajectory: {e}")
        return

    # 获取当前末端位姿作为基准 pose_0
    try:
        trans = tf_buffer.lookup_transform('world', ee_link, rospy.Time(0), rospy.Duration(1.0))
        base_pose = Pose()
        base_pose.position.x = trans.transform.translation.x
        base_pose.position.y = trans.transform.translation.y
        base_pose.position.z = trans.transform.translation.z
        base_pose.orientation.x = trans.transform.rotation.x
        base_pose.orientation.y = trans.transform.rotation.y
        base_pose.orientation.z = trans.transform.rotation.z
        base_pose.orientation.w = trans.transform.rotation.w
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Failed to get base pose from TF (world -> {ee_link}): {e}")
        return

    # 应用相对轨迹：pose = pose_0 + k * pose_json
    apply_relative_trajectory(msg, base_pose, scale)

    # 等待订阅者连接
    rospy.sleep(1.0)

    rviz_only = True

    if rviz_only:
        # 发布器：与 example_move.py 不同话题
        traj_pub = rospy.Publisher("/servo/diana77/trajectory", PoseArray, queue_size=1)
        rospy.loginfo(f"Looping {len(msg.poses)} poses to /servo/diana77/trajectory at 1Hz...")
        loop_rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            msg.header.stamp = rospy.Time.now() # 更新时间戳
            traj_pub.publish(msg)
            loop_rate.sleep()
    else:
        # 发布一次完整轨迹
        # 发布器：与 example_move.py 相同话题
        traj_pub.publish(msg)
        rospy.loginfo(f"Published {len(msg.poses)} poses to /servo/diana7/trajectory (frame: {msg.header.frame_id}).")
        # 保持节点运行以便查看日志或后续扩展
        rospy.spin()
        

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
