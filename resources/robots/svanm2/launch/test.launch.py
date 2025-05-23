import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.actions import ExecuteProcess

import launch_ros
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
pkg_share = launch_ros.substitutions.FindPackageShare(package="svanm2_description").find("svanm2_description")

bringup_dir = get_package_share_directory('svanm2_description')
launch_dir = os.path.join(pkg_share, "launch")
rviz_config_path = os.path.join(bringup_dir, 'rviz', 'navigation.rviz')

urdf_file = os.path.join(
            get_package_share_directory('svanm2_description'),
            'urdf',
            'm2.urdf'
    )
def generate_launch_description():
    use_sim_time = False

    bringup_dir = get_package_share_directory('svanm2_description')
    default_model_path = os.path.join(bringup_dir, 'urdf', 'm2.urdf')

    # Read URDF file
   


    gazebo_server = ExecuteProcess(
        cmd=[
            "gzserver",
            # "-u",
            "--verbose",
            "-s",
            "libgazebo_ros_init.so",
            "-s",
            "libgazebo_ros_factory.so",

        ],
        cwd=[launch_dir],
        output="screen",
    )
    gazebo_client = ExecuteProcess(
        cmd=["gzclient"],
        cwd=[launch_dir],
        output="screen",
    )

    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        output="screen",
        arguments=[
            "-entity",
            "svanM2",
            "-topic",
            "/robot_description",
           "-z", 
            "1"
        ],
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': open(urdf_file).read(),"use_tf_static":False, 'use_sim_time':use_sim_time}]
    )

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='screen',
        parameters=[{'use_sim_time':True}]
    
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"]
    )

    robot_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_group_effort_controller", "--controller-manager", "/controller_manager"]
    )
   
    load_joint_state_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_states_controller'],
        output='screen',
    )

    load_joint_trajectory_position_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_group_position_controller'],
        output='screen'
    )
    load_joint_trajectory_effort_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_group_effort_controller'],
        output='screen'
    )
    odom_tf_publisher_node = Node(
        package='odom_tf_publisher',
        executable='odom_to_tf',
        name='odom_to_tf',
        output='screen',
        parameters=[{'use_sim_time': True}]

    )

    return LaunchDescription([
        # RegisterEventHandler(
        #     event_handler=OnProcessExit(
        #         target_action=joint_state_broadcaster_spawner,
        #         on_exit=[robot_controller_spawner],
        #     )
        # ),
        DeclareLaunchArgument(name='model', default_value=default_model_path,
                                             description='Absolute path to robot urdf file'),

        joint_state_publisher,
        robot_state_publisher,

        gazebo_server,
        gazebo_client,
        spawn_entity,
        load_joint_state_controller,
        load_joint_trajectory_position_controller,
        load_joint_trajectory_effort_controller,
        odom_tf_publisher_node,
        # Start the robot state publisher and joint state publisher
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_path],
            parameters=[{'use_sim_time': True}]
        )
        # Spawn the robot in Gazebo after Gazebo has started

    ])


