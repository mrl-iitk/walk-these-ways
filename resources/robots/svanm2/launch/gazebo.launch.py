import launch
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler
from launch.substitutions import Command, LaunchConfiguration
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from ament_index_python.packages import get_package_share_directory

import launch_ros
import os
import launch_ros.descriptions

def generate_launch_description():
    use_sim_time = False
    pkg_share_gazebo = launch_ros.substitutions.FindPackageShare(package='gazebo_ros').find('gazebo_ros')
    default_rviz_config_path = os.path.join("", 'rviz/urdf.rviz')
    world_file = get_package_share_directory('svanm2_description') + '/worlds/room.world'
    bringup_dir = get_package_share_directory('svanm2_description')
    default_model_path = os.path.join(bringup_dir, 'urdf', 'm2.urdf')
    # rviz_config_path = PathJoinSubstitution([FindPackageShare('champ_navigation'), 'rviz', 'navigation.rviz'])
    rviz_config_path = os.path.join(bringup_dir, 'rviz', 'navigation.rviz')
    with open(default_model_path, 'r') as infp:
        urdf = infp.read() 
    params = {'robot_description':urdf, "use_tf_static":False, "use_sim_time": use_sim_time}

    robot_state_publisher_node = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    spawn_entity = launch_ros.actions.Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'testing', '-topic', 'robot_description', '-z', '1'],
        output='screen'
    )
    rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path]

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
    
    load_gazebo = ExecuteProcess(
        cmd=['gazebo', '-u', "-s", "libgazebo_ros_factory.so", "-s", "libgazebo_ros_init.so", world_file],
        output='screen'
    )

    odom_tf_publisher_node = Node(
        package='odom_tf_publisher',
        executable='odom_to_tf',
        name='odom_to_tf',
        output='screen',
        parameters=[{'use_sim_time': True}]

    )

    # slam_toolbox_node = Node(
    #         package='slam_toolbox',
    #         executable='sync_slam_toolbox_node',
    #         name='slam_toolbox',
    #         output='screen',
    #         parameters=[slam_toolbox_params, {'use_sim_time': True}],
    #         remappings=[
    #             ('/scan', '/svan/scan')  # Ensure this matches your scan topic
    #         ]
    #     )
    return launch.LaunchDescription([
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[robot_controller_spawner],
            )
        ),
        launch.actions.DeclareLaunchArgument(name='model', default_value=default_model_path,
                                             description='Absolute path to robot urdf file'),
        load_gazebo,
        robot_state_publisher_node,
        spawn_entity,
        # rviz_node,
        joint_state_broadcaster_spawner,
        odom_tf_publisher_node,
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_path],
            parameters=[{'use_sim_time': True}]
        )
    ])
