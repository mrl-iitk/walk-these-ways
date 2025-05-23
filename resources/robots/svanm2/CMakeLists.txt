set(CMAKE_CXX_STANDARD 14)
cmake_minimum_required(VERSION 3.8)
project(svanm2_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

# # Add executable for JointStateSensor plugin
# find_library(joint_state_sensor REQUIRED)
# find_library(base_state_sensor REQUIRED)
# find_library(motor_controller REQUIRED)
# find_library(contact_plugin_default REQUIRED)

# include_directories(
#   ${rclcpp_INCLUDE_DIRS}
#   ${rclpy_INCLUDE_DIRS}
#   ${std_msgs_INCLUDE_DIRS}
#   ${nav_msgs_INCLUDE_DIRS}
# )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY 
  launch meshes xacro urdf config rviz
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
