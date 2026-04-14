#!/usr/bin/env python3
"""格式化打印 stm_magnitude 和 stm_magnitude_raw topic 对比"""
import rospy
from std_msgs.msg import Float32MultiArray

latest_magnitude = None
latest_magnitude_raw = None

def cb_magnitude(msg):
    global latest_magnitude
    latest_magnitude = msg.data

def cb_magnitude_raw(msg):
    global latest_magnitude_raw
    latest_magnitude_raw = msg.data
    print_comparison()

def print_comparison():
    global latest_magnitude, latest_magnitude_raw
    if latest_magnitude is None:
        return
    if latest_magnitude_raw is None:
        vals = ", ".join(f"{v:.2f}" for v in latest_magnitude)
        rospy.loginfo(f"[raw=N/A] [corr] [{vals}]")
        return

    raw_vals = ", ".join(f"{v:.2f}" for v in latest_magnitude_raw)
    corr_vals = ", ".join(f"{v:.2f}" for v in latest_magnitude)
    rospy.loginfo(f"[raw] [{raw_vals}]")
    rospy.loginfo(f"[corr] [{corr_vals}]")

if __name__ == '__main__':
    rospy.init_node('show_magnitude')
    rospy.Subscriber('stm_magnitude', Float32MultiArray, cb_magnitude)
    rospy.Subscriber('stm_magnitude_raw', Float32MultiArray, cb_magnitude_raw)
    rospy.loginfo("Subscribed to stm_magnitude and stm_magnitude_raw")
    rospy.spin()
