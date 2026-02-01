#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
import geometry_msgs.msg

def check_setup():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('moveit_setup_checker', anonymous=True)

    robot = moveit_commander.RobotCommander()
    
    print("\n" + "="*50)
    print("1. 规划组信息 (Planning Groups)")
    print("="*50)
    group_names = robot.get_group_names()
    print(f"可用的规划组: {group_names}")

    for name in group_names:
        try:
            group = moveit_commander.MoveGroupCommander(name)
            print(f"\n--- 组: {name} ---")
            print(f"  规划坐标系 (Planning Frame): {group.get_planning_frame()}")
            print(f"  默认末端 Link (Default EE Link): {group.get_end_effector_link()}")
        except Exception as e:
            print(f"\n--- 组: {name} --- (无法加载: {e})")
    
    print("\n" + "="*50)
    print("2. 机器人模型中的所有 Link (All Links in Robot Model)")
    print("="*50)
    all_links = robot.get_link_names()
    print(f"总共发现 {len(all_links)} 个 Link。")
    
    interesting_keywords = ['tcp', 'magnet', 'tool', 'flange', 'array']
    print("\n可能感兴趣的 Link (包含关键字):")
    for link in all_links:
        if any(kw in link.lower() for kw in interesting_keywords):
            print(f"  - {link}")

    print("\n" + "="*50)
    print("3. 获取特定 Link 的实时位姿 (Get Link Pose)")
    print("="*50)
    
    # 尝试获取一些常见的 TCP Link 位姿
    test_links = [
        "diana7_magnetometer_array_tcp_link",
        "arm1_electronic_magnet_tcp_link",
        "arm2_electronic_magnet_tcp_link",
        "diana7_calibrated_magmeterarray_tcp_link",
        "arm1_calibrated_em_tcp_link",
        "arm2_calibrated_em_tcp_link"
    ]

    if group_names:
        any_group = moveit_commander.MoveGroupCommander(group_names[0])
        for link in test_links:
            if link in all_links:
                try:
                    pose_stamped = any_group.get_current_pose(link)
                    p = pose_stamped.pose
                    print(f"\nLink: {link}")
                    print(f"  Position: [{p.position.x:.4f}, {p.position.y:.4f}, {p.position.z:.4f}]")
                    print(f"  Orientation: [{p.orientation.x:.4f}, {p.orientation.y:.4f}, {p.orientation.z:.4f}, {p.orientation.w:.4f}]")
                except Exception as e:
                    print(f"\nLink: {link} -> 无法获取位姿 (TF 可能未就绪)")
            else:
                # print(f"\nLink: {link} -> 在模型中未找到")
                pass

    print("\n" + "="*50)
    print("诊断完成")
    print("="*50)

if __name__ == '__main__':
    try:
        check_setup()
    except rospy.ROSInterruptException:
        pass
