#!/usr/bin/env python3
import json
import os
import math
import csv
from geometry_msgs.msg import Pose

class TrajectoryLoader:
    """加载轨迹点的工具类，能够支持两种轨迹的加载形式"""
    @staticmethod
    def load_csv(file_path):
        points = []
        if not os.path.exists(file_path):
            return points
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pose = Pose()
                pose.position.x = float(row['x'])
                pose.position.y = float(row['y'])
                pose.position.z = float(row['z'])
                pose.orientation.x = float(row['qx'])
                pose.orientation.y = float(row['qy'])
                pose.orientation.z = float(row['qz'])
                pose.orientation.w = float(row['qw'])
                points.append(pose)
        return points

    @staticmethod
    def load_json(file_path):
        if not os.path.exists(file_path):
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        poses = []
        poses_data = data.get("poses", [])
        scale = data.get("scale", 1.0)
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
            poses.append(pose)
        return poses, scale

    @staticmethod
    def _scale_relative_orientation(q_rel, scale):
        x, y, z, w = q_rel
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm == 0.0:
            return 0.0, 0.0, 0.0, 1.0
        x, y, z, w = x / norm, y / norm, z / norm, w / norm

        w = max(min(w, 1.0), -1.0)
        angle = 2.0 * math.acos(w)
        if abs(angle) < 1e-9:
            return 0.0, 0.0, 0.0, 1.0
        s = math.sqrt(1.0 - w * w)
        if s < 1e-9:
            axis = (1.0, 0.0, 0.0)
        else:
            axis = (x / s, y / s, z / s)

        angle_scaled = angle * scale
        half = 0.5 * angle_scaled
        sin_half = math.sin(half)
        return axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, math.cos(half)

    @staticmethod
    def _quat_multiply(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )

    @staticmethod
    def _quat_conjugate(q):
        x, y, z, w = q
        return (-x, -y, -z, w)

    @staticmethod
    def _rotate_vector_by_quat(v, q):
        vx, vy, vz = v
        vq = (vx, vy, vz, 0.0)
        q_conj = TrajectoryLoader._quat_conjugate(q)
        r = TrajectoryLoader._quat_multiply(TrajectoryLoader._quat_multiply(q, vq), q_conj)
        return r[0], r[1], r[2]

    @staticmethod
    def apply_relative_trajectory(poses, base_pose, scale):
        q0 = (
            base_pose.orientation.x,
            base_pose.orientation.y,
            base_pose.orientation.z,
            base_pose.orientation.w,
        )
        norm = math.sqrt(q0[0] * q0[0] + q0[1] * q0[1] + q0[2] * q0[2] + q0[3] * q0[3])
        if norm == 0.0:
            q0 = (0.0, 0.0, 0.0, 1.0)
        else:
            q0 = (q0[0] / norm, q0[1] / norm, q0[2] / norm, q0[3] / norm)
            
        transformed_poses = []
        for p in poses:
            new_pose = Pose()
            # Position: p_world = p0 + R(q0) * (k * p_rel)
            rel = (scale * p.position.x, scale * p.position.y, scale * p.position.z)
            dx, dy, dz = TrajectoryLoader._rotate_vector_by_quat(rel, q0)
            new_pose.position.x = base_pose.position.x + dx
            new_pose.position.y = base_pose.position.y + dy
            new_pose.position.z = base_pose.position.z + dz

            # Orientation: q = q0 * (q_rel^k)
            q_rel = (
                p.orientation.x,
                p.orientation.y,
                p.orientation.z,
                p.orientation.w,
            )
            # 如果不需要对角度进行相对缩放，将 scale 设置为 1.0 即可
            q_rel_scaled = TrajectoryLoader._scale_relative_orientation(q_rel, scale)
            q = TrajectoryLoader._quat_multiply(q0, q_rel_scaled)
            new_pose.orientation.x, new_pose.orientation.y, new_pose.orientation.z, new_pose.orientation.w = q
            transformed_poses.append(new_pose)
        
        return transformed_poses
