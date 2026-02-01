#!/usr/bin/env python3

import rospy
from moveit_msgs.srv import GetPlanningScene
from moveit_msgs.msg import PlanningSceneComponents

def check_collisions():
    rospy.init_node('collision_checker_tool')
    
    rospy.wait_for_service('/get_planning_scene')
    get_scene = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)
    
    # Request the planning scene with collision information
    request = PlanningSceneComponents()
    request.components = PlanningSceneComponents.ROBOT_STATE | \
                         PlanningSceneComponents.WORLD_OBJECT_GEOMETRY | \
                         PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
    
    try:
        # This is a bit complex to do purely via service without the C++ API or moveit_commander's internal wrappers
        # But we can check the move_group logs or use moveit_commander
        import moveit_commander
        import sys
        
        moveit_commander.roscpp_initialize(sys.argv)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        
        rospy.loginfo("Checking for collisions in the current planning scene...")
        
        # Note: moveit_commander doesn't have a direct "get_collisions" method that returns pairs.
        # The best way is to look at the move_group terminal or use the planning_scene_monitor in C++.
        
        print("\n" + "="*60)
        print("如何查看碰撞物体 (How to check collisions):")
        print("="*60)
        print("1. 查看终端输出 (Terminal):")
        print("   在运行 'roslaunch ...' 的终端中，寻找包含 'Contact' 或 'Collision' 的 INFO/WARN 日志。")
        print("   例如: 'Found a contact between 'arm1_pedestal_link' and 'floor''")
        print("\n2. 使用 RViz 可视化 (RViz Visualization):")
        print("   - 在 RViz 中添加 'Motion Planning' 插件。")
        print("   - 在 'Scene Robot' 标签页中勾选 'Show Robot Collision'。")
        print("   - 在 'Planning Scene' 标签页中勾选 'Show Scene Collision'。")
        print("   - 发生碰撞的物体会以 红色 (Red) 高亮显示。")
        print("\n3. 检查当前环境物体 (Current Scene Objects):")
        print(f"   已知物体: {scene.get_known_object_names()}")
        print("\n" + "="*60)

    except Exception as e:
        rospy.logerr(f"Error: {e}")

if __name__ == '__main__':
    check_collisions()
