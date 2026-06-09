cmake_minimum_required(VERSION 3.0.2)
project(triple_arm_visual_servo)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  control_msgs
  dynamic_reconfigure
  geometry_msgs
  message_filters
  moveit_commander
  moveit_msgs
  nav_msgs
  rospy
  std_msgs
  tf2_geometry_msgs
  tf2_ros
  trajectory_msgs
)

catkin_package(
  CATKIN_DEPENDS
    actionlib
    control_msgs
    dynamic_reconfigure
    geometry_msgs
    message_filters
    moveit_commander
    moveit_msgs
    nav_msgs
    rospy
    std_msgs
    tf2_geometry_msgs
    tf2_ros
    trajectory_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/example_move.py
  scripts/execute_trajectory.py
  scripts/generate_paths.py
  scripts/pose_filter.py
  scripts/visual_servo_node.py
  scripts/visual_servo_node_solo.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
