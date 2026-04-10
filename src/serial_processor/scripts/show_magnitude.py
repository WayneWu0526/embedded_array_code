#!/usr/bin/env python3
"""格式化打印 stm_magnitude topic"""
import rospy
from std_msgs.msg import Float32MultiArray

def cb(msg):
    vals = ", ".join(f"{v:.2f}" for v in msg.data)
    rospy.loginfo(f"[stm_magnitude] [{vals}]")

if __name__ == '__main__':
    rospy.init_node('show_magnitude')
    rospy.Subscriber('stm_magnitude', Float32MultiArray, cb)
    rospy.spin()
