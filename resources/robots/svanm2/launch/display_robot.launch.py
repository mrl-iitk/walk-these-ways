from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # Declare the launch argument for the URDF file path
    pkg_share = FindPackageShare('svanm2_description').find('svanm2_description')
    urdf_file = os.path.join(pkg_share, 'urdf', 'm2.urdf')
    # Load the URDF file content
    with open(urdf_file, 'r') as file:
        robot_description_content = file.read()
        
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description_content}]
    )

    # Joint State Publisher node (without GUI)
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher'
    )

    # RViz2 node
    rviz_config_file = os.path.join(pkg_share, 'rviz', 'robot_urdf.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
    )

    # Return the launch description
    return LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_node,
        rviz_node
    ])