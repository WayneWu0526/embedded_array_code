#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
import json
import os
import datetime
import threading
from serial_processor.srv import GetHallData
from triple_arm_task.msg import ScanData
from std_msgs.msg import Header

class CalibratedRecorder:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('calibrated_recorder', anonymous=True)

        self.robot = moveit_commander.RobotCommander()

        # 参数设置
        self.sampling_rate = rospy.get_param('~sampling_rate', 50.0) # 采样频率 (Hz)
        self.max_samples = rospy.get_param('~max_samples', 200)      # 最大采样数
        self.save_interval = rospy.get_param('~save_interval', 100)  # 每 100 组数据保存一次
        self.connect_wait_time = rospy.get_param('~connect_wait_time', 5.0)

        # 异步位置更新
        self.current_poses = {
            "diana7": geometry_msgs.msg.Pose(),
            "arm1": geometry_msgs.msg.Pose(),
            "arm2": geometry_msgs.msg.Pose()
        }
        self.pose_lock = threading.Lock()
        
        # JSON 日志设置
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_filename = os.path.join(self.data_dir, f"calibrated_record_{timestamp_str}.json")
        self.scan_results = []
        self.count = 0
        
        rospy.loginfo(f"数据记录初始化至: {self.json_filename}")
        rospy.on_shutdown(self.cleanup)

        # MoveIt 规划组初始化
        rospy.loginfo(f"等待 {self.connect_wait_time}s 以连接 move_group...")
        rospy.sleep(self.connect_wait_time) 
        
        try:
            self.diana7_group = moveit_commander.MoveGroupCommander("diana7")
            self.arm1_group = moveit_commander.MoveGroupCommander("arm1")
            self.arm2_group = moveit_commander.MoveGroupCommander("arm2")
        except Exception as e:
            rospy.logerr(f"无法连接到 MoveGroup: {e}")
            sys.exit(1)

        # 设置端效器链接为 calibrated_em 组合工具中的电磁铁 TCP
        try:
            self.diana7_group.set_end_effector_link("diana7_electronic_magnet_tcp_link")
            self.arm1_group.set_end_effector_link("arm1_electronic_magnet_tcp_link")
            self.arm2_group.set_end_effector_link("arm2_electronic_magnet_tcp_link")
        except Exception as e:
            rospy.logwarn(f"无法设置端效器链接，请检查 URDF 命名: {e}")
        
        # 霍尔传感器服务客户端
        rospy.wait_for_service('get_hall_data')
        self.get_hall_data_srv = rospy.ServiceProxy('get_hall_data', GetHallData)

        # 发布实时数据
        self.data_pub = rospy.Publisher('scan_data', ScanData, queue_size=10)

        # 启动位置更新线程
        self.pose_thread = threading.Thread(target=self.update_poses_loop)
        self.pose_thread.daemon = True
        self.pose_thread.start()

    def update_poses_loop(self):
        """异步更新机械臂位姿的线程"""
        # 位置更新频率不需要太高，因为机械臂运动速度受限，10-20Hz 足够
        pose_rate = rospy.Rate(20) 
        while not rospy.is_shutdown():
            try:
                # 获取位姿是耗时操作
                d7 = self.diana7_group.get_current_pose().pose
                a1 = self.arm1_group.get_current_pose().pose
                a2 = self.arm2_group.get_current_pose().pose
                
                with self.pose_lock:
                    self.current_poses["diana7"] = d7
                    self.current_poses["arm1"] = a1
                    self.current_poses["arm2"] = a2
            except Exception as e:
                rospy.logwarn(f"位姿更新线程异常: {e}")
            pose_rate.sleep()

    def cleanup(self):
        self.save_json()
        rospy.loginfo(f"任务结束。总计录得 {self.count} 组数据。")

    def save_json(self):
        try:
            with open(self.json_filename, 'w') as f:
                json.dump(self.scan_results, f, indent=4)
        except Exception as e:
            rospy.logerr(f"JSON 保存失败: {e}")

    def run(self):
        rate = rospy.Rate(self.sampling_rate)
        rospy.loginfo(f"开始高频数据采集，频率: {self.sampling_rate}Hz，目标: {self.max_samples}组...")
        
        while not rospy.is_shutdown() and self.count < self.max_samples:
            try:
                # 1. 获取硬件数据 (耗时约 12.5ms)
                hall_resp = self.get_hall_data_srv()
                
                # 2. 从本地缓存获取当前世界位姿 (非阻塞)
                with self.pose_lock:
                    diana7_pose = self.current_poses["diana7"]
                    arm1_pose = self.current_poses["arm1"]
                    arm2_pose = self.current_poses["arm2"]
                
                now = rospy.Time.now()
                target_id = f"high_freq_{self.count}"
                
                # 3. 发布数据
                msg = ScanData()
                msg.header.stamp = now
                msg.current_target_id = target_id
                msg.diana7_tool_pose = diana7_pose
                msg.arm1_tool_pose = arm1_pose
                msg.arm2_tool_pose = arm2_pose
                msg.hall_data = hall_resp.sensors
                self.data_pub.publish(msg)
                
                # 4. 存入内存记录
                entry = {
                    "timestamp": now.to_sec(),
                    "sample_index": self.count,
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
                    "hall_data": [{"x": s.x, "y": s.y, "z": s.z} for s in hall_resp.sensors]
                }
                self.scan_results.append(entry)
                self.count += 1
                
                if self.count % self.save_interval == 0:
                    self.save_json()
                
                rate.sleep()
                
            except Exception as e:
                rospy.logerr(f"采样循环异常: {e}")
                break
                        },
                        "arm2": {
                            "position": [arm2_pose.position.x, arm2_pose.position.y, arm2_pose.position.z],
                            "orientation": [arm2_pose.orientation.x, arm2_pose.orientation.y, arm2_pose.orientation.z, arm2_pose.orientation.w]
                        }
                    },
                    "hall_data": [{"x": s.x, "y": s.y, "z": s.z} for s in hall_resp.sensors]
                }
                self.scan_results.append(entry)
                self.count += 1
                
                # 周期性增量保存，防止程序异常崩溃丢失大量数据
                if self.count % self.save_interval == 0:
                    self.save_json()
                    rospy.loginfo(f"已采集并保存 {self.count} 组数据...")
                
            except Exception as e:
                rospy.logwarn(f"在第 {self.count} 次采集时发生错误: {e}")
                
            rate.sleep()
            
        if self.count >= self.max_samples:
            rospy.loginfo("达到预设采样总数，正在自动关闭节点...")
            rospy.signal_shutdown("Max samples reached")

if __name__ == '__main__':
    try:
        recorder = CalibratedRecorder()
        recorder.run()
    except rospy.ROSInterruptException:
        pass
