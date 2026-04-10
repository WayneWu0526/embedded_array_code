#!/usr/bin/env python3
"""发送 STM32 Manual Mode 初始化命令"""
import rospy
from serial_processor.msg import StmDownlink

if __name__ == '__main__':
    rospy.init_node('stm32_init')
    pub = rospy.Publisher('stm_downlink', StmDownlink, queue_size=1)

    # 等待节点就绪
    rospy.sleep(0.5)

    msg = StmDownlink()
    msg.mode = 0           # MANUAL mode
    msg.bitmap = 4095      # 全部 12 个传感器 (0x0FFF)
    msg.settling_time = 100   # 1ms
    msg.cycle_time = 10000   # 100ms
    msg.cycle_num = 1

    pub.publish(msg)
    rospy.loginfo("STM32 manual mode initialization sent")
