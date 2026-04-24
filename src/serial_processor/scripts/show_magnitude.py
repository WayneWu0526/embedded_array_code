#!/usr/bin/env python3
"""格式化打印 stm_magnitude_raw、stm_ellip_magnitude 和 stm_magnitude 对比"""
import rospy
from std_msgs.msg import Float32MultiArray

latest_magnitude = None
latest_magnitude_raw = None
latest_magnitude_ellip = None
last_print_time = None
print_interval = 0.2  # 最小打印间隔（秒）

def cb_magnitude(msg):
    global latest_magnitude
    latest_magnitude = msg.data
    print_comparison()

def cb_magnitude_raw(msg):
    global latest_magnitude_raw
    latest_magnitude_raw = msg.data

def cb_magnitude_ellip(msg):
    global latest_magnitude_ellip
    latest_magnitude_ellip = msg.data
    print_comparison()

def print_comparison():
    global latest_magnitude, latest_magnitude_raw, latest_magnitude_ellip, last_print_time

    # 延迟初始化 last_print_time
    if last_print_time is None:
        last_print_time = rospy.Time.now()
        return  # 第一次直接返回，不打印

    # 时间间隔限制
    now = rospy.Time.now()
    if (now - last_print_time).to_sec() < print_interval:
        return
    last_print_time = now

    # 构造输出行
    output = []

    if latest_magnitude_raw is not None:
        raw_vals = ", ".join(f"{v:.2f}" for v in latest_magnitude_raw)
        output.append(f"[raw] [{raw_vals}]")

    if latest_magnitude_ellip is not None:
        ellip_vals = ", ".join(f"{v:.2f}" for v in latest_magnitude_ellip)
        output.append(f"[ellip] [{ellip_vals}]")

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
    rospy.Subscriber('stm_ellip_magnitude', Float32MultiArray, cb_magnitude_ellip)

    rospy.loginfo("Subscribed to stm_magnitude, stm_magnitude_raw, and stm_ellip_magnitude")
    rospy.spin()
