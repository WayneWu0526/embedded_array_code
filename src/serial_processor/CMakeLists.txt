cmake_minimum_required(VERSION 3.0.2)
project(serial_processor)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  roscpp
  rospy
  sensor_array_config
  sensor_msgs
  std_msgs
)

add_message_files(
  FILES
  SensorData.msg
  StmUplink.msg
  StmDownlink.msg
  MagGradImuRaw.msg
)

add_service_files(
  FILES
  GetHallData.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    message_runtime
    roscpp
    rospy
    sensor_array_config
    sensor_msgs
    std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/serial_node.py
  scripts/serial_node_tdm.py
  scripts/serial_node_maggrad.py
  scripts/tools/mock_stm_node.py
  scripts/tools/show_magnitude.py
  scripts/legacy_tdm/stm32_init.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
