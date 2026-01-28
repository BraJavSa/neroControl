cmake_minimum_required(VERSION 3.5)
project(nero_drone)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)


include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)


add_executable(Joy2Cmd src/Joy2Cmd.cpp)
ament_target_dependencies(Joy2Cmd rclcpp sensor_msgs geometry_msgs std_msgs)
target_link_libraries(Joy2Cmd ${OpenCV_LIBS})

add_executable(apriltag_node src/apriltag_node.cpp)
ament_target_dependencies(apriltag_node rclcpp OpenCV)
target_link_libraries(apriltag_node apriltag ${OpenCV_LIBS})


add_executable(beboptag src/beboptag.cpp)
ament_target_dependencies(beboptag rclcpp sensor_msgs cv_bridge OpenCV)
target_link_libraries(beboptag apriltag ${OpenCV_LIBS})

add_executable(ibvs src/ibvs.cpp)
ament_target_dependencies(ibvs rclcpp sensor_msgs cv_bridge OpenCV)
target_link_libraries(ibvs apriltag ${OpenCV_LIBS})

add_executable(tf_cam src/tf_cam.cpp)
ament_target_dependencies(tf_cam rclcpp tf2 tf2_ros geometry_msgs nav_msgs std_msgs tf2_geometry_msgs)

add_executable(tf_tag_bebop src/tf_tag_bebop.cpp)
ament_target_dependencies(tf_tag_bebop rclcpp sensor_msgs cv_bridge OpenCV tf2_ros geometry_msgs visualization_msgs tf2 tf2_geometry_msgs)
target_link_libraries(tf_tag_bebop apriltag ${OpenCV_LIBS})


add_executable(RefPublisher src/RefPublisher.cpp)
ament_target_dependencies(RefPublisher rclcpp geometry_msgs)

add_executable(velocity_logger src/velocity_logger.cpp)
ament_target_dependencies(velocity_logger
  rclcpp
  nav_msgs
  geometry_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
)

add_executable(ref_vec_filter src/filtred.cpp)
ament_target_dependencies(ref_vec_filter
  rclcpp
  geometry_msgs
  std_msgs
  visualization_msgs
  tf2_geometry_msgs
)

add_executable(tagfollow src/tagfollow.cpp)
ament_target_dependencies(tagfollow
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  std_msgs
  tf2
  tf2_ros
  visualization_msgs
)
add_executable(bebop_odometry src/bebop_odometry.cpp)

ament_target_dependencies(bebop_odometry
  rclcpp
  geometry_msgs
  nav_msgs
  std_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)



add_executable(visual src/visual.cpp)
ament_target_dependencies(visual
  rclcpp
  geometry_msgs
  visualization_msgs
  tf2_ros
)
add_executable(land src/land.cpp)
ament_target_dependencies(land rclcpp std_msgs)


add_executable(safety_watchdog
  src/safe_watchdog.cpp
)
ament_target_dependencies(safety_watchdog
  rclcpp
  geometry_msgs
  std_msgs
)


add_executable(tag_pos src/tag_pos.cpp)

ament_target_dependencies(tag_pos
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  OpenCV
  tf2
  tf2_ros
  tf2_geometry_msgs
)

target_link_libraries(tag_pos
  apriltag
  ${OpenCV_LIBS}
)

add_executable(safe_bebop_republisher src/bridge_cmd.cpp)
ament_target_dependencies(safe_bebop_republisher rclcpp geometry_msgs)

add_executable(isfly src/isfly.cpp)
ament_target_dependencies(isfly rclcpp std_msgs)

add_executable(tf_odom_to_map src/tf_odom_map.cpp)

ament_target_dependencies(tf_odom_to_map
  rclcpp
  tf2_ros
  geometry_msgs
)



add_executable(bebop_odom_fusion src/bebop_odom_fusion.cpp)
ament_target_dependencies(bebop_odom_fusion rclcpp nav_msgs geometry_msgs sensor_msgs)


# ---------------- Install ----------------
install(TARGETS
  Joy2Cmd
  apriltag_node
  beboptag
  tf_cam
  tf_tag_bebop
  velocity_logger
  tagfollow
  visual
  safety_watchdog
  safe_bebop_republisher
  land
  ref_vec_filter
  isfly
  bebop_odometry
  ibvs
  tag_pos
  tf_odom_to_map
  bebop_odom_fusion
  DESTINATION lib/${PROJECT_NAME}
)

# ---------------- Python nodes ----------------
install(PROGRAMS
  nero_drone/graphical.py
  nero_drone/apriltag.py
  nero_drone/takeoff.py
  nero_drone/land.py
  nero_drone/calibration_publish.py
  nero_drone/bebop_control_gui.py
  nero_drone/ref_pos.py
  nero_drone/ref_trayectory.py
  nero_drone/controlador.py
  nero_drone/pre_controlador.py
  nero_drone/sim.py
  nero_drone/tf_to_gazebo.py
  nero_drone/altitude.py
  nero_drone/initial_frame.py
  DESTINATION lib/${PROJECT_NAME}
)

# ---------------- Launch files ----------------
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

# ---------------- Config files ----------------
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

# ---------------- URDF + meshes + others ----------------
install(DIRECTORY urdf meshes others DESTINATION share/${PROJECT_NAME})

ament_package()
