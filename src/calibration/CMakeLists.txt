cmake_minimum_required(VERSION 3.0.2)
project(calibration)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/calibration_affine_model.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
