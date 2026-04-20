#!/usr/bin/env python3
"""格式化打印 stm_magnitude_raw 和 stm_magnitude 对比"""
import rospy
from std_msgs.msg import Float32MultiArray

latest_magnitude = None
latest_magnitude_raw = None

def cb_magnitude(msg):
    global latest_magnitude
    latest_magnitude = msg.data
    print_comparison()

def cb_magnitude_raw(msg):
    global latest_magnitude_raw
    latest_magnitude_raw = msg.data

def print_comparison():
    global latest_magnitude, latest_magnitude_raw

    # 构造输出行
    output = []

    if latest_magnitude_raw is not None:
        raw_vals = ", ".join(f"{v:.2f}" for v in latest_magnitude_raw)
        output.append(f"[raw] [{raw_vals}]")

    if latest_magnitude is not None:
        corr_vals = ", ".join(f"{v:.2f}" for v in latest_magnitude)
        output.append(f"[corr] [{corr_vals}]")

    # 打印非空行
    for line in output:
        rospy.loginfo(line)

if __name__ == '__main__':
    rospy.init_node('show_magnitude')
    rospy.Subscriber('stm_magnitude', Float32MultiArray, cb_magnitude)
    rospy.Subscriber('stm_magnitude_raw', Float32MultiArray, cb_magnitude_raw)

    rospy.loginfo("Subscribed to stm_magnitude and stm_magnitude_raw")
    rospy.spin()
