from collections import deque
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Pose

class PoseFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.pos_history = deque(maxlen=window_size)
        self.ori_history = deque(maxlen=window_size)

    def update(self, pose):
        # Position
        pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        # Orientation (as quaternion)
        ori = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        
        # Outlier rejection using distance from median
        if len(self.pos_history) >= 3:
            pts = np.array(self.pos_history)
            median_pos = np.median(pts, axis=0)
            dist = np.linalg.norm(pos - median_pos)
            if dist > 0.08: # 8cm threshold for singular points
                rospy.logdebug(f"Outlier rejected: dist={dist:.4f}")
                return self.get_filtered_pose()

        self.pos_history.append(pos)
        self.ori_history.append(ori)
        return self.get_filtered_pose()

    def get_filtered_pose(self):
        if not self.pos_history:
            return None
        
        # Sliding Window Average + Median combination
        pts = np.array(self.pos_history)
        if len(pts) < 3:
            filtered_pos = np.mean(pts, axis=0)
        else:
            # We can use median to be robust
            filtered_pos = np.median(pts, axis=0)
        
        # Orientation: normalize average of quaternions (approximate but OK for small windows)
        oris = np.array(self.ori_history)
        avg_ori = np.mean(oris, axis=0)
        avg_ori /= np.linalg.norm(avg_ori)
        
        res = Pose()
        res.position.x, res.position.y, res.position.z = filtered_pos
        res.orientation.x, res.orientation.y, res.orientation.z, res.orientation.w = avg_ori
        return res