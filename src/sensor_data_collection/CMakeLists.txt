cmake_minimum_required(VERSION 3.0.2)
project(sensor_data_collection)

find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs geometry_msgs sensor_msgs tf tf2_ros serial_processor message_generation)

add_message_files(
  FILES
  SlotData.msg
  SensorReading.msg
)

add_service_files(
  FILES
  LocalizeCycle.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime std_msgs geometry_msgs sensor_msgs tf2_ros)

catkin_install_python(PROGRAMS
  scripts/data_collection_node.py
  scripts/maggrad_continuous_collection_node.py
  scripts/maggrad_manual_record_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
