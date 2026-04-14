cmake_minimum_required(VERSION 3.0.2)
project(calibration)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Python scripts go in scripts/
# ROS nodes go in nodes/
# Launch files go in launch/
