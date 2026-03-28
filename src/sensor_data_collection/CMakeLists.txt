cmake_minimum_required(VERSION 3.0.2)
project(sensor_data_collection)

find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs geometry_msgs tf serial_processor)

catkin_package()

catkin_install_python(PROGRAMS scripts/data_collection_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
