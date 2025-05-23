import launch
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler
from launch.substitutions import Command, LaunchConfiguration
from launch.event_handlers import OnProcessExit

import launch_ros
import os
import launch_ros.descriptions

def generate_launch_description():
    pkg_share_gazebo = launch_ros.substitutions.FindPackageShare(package='gazebo_ros').find('gazebo_ros')
    pkg_share = launch_ros.substitutions.FindPackageShare(package='svanm2_description').find('svanm2_description')
    # default_model_path = os.path.join("",'src/svan_sw2/svanm2_description/urdf/m2.urdf')
    default_model_path = os.path.join(pkg_share, 'urdf', 'm2.urdf')
    default_rviz_config_path = os.path.join("", 'rviz/urdf.rviz')
    with open(default_model_path, 'r') as infp:
        urdf = infp.read() 
    params = {'robot_description':urdf}

    robot_state_publisher_node = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output = 'screen',
        parameters=[params]
    )

    spawn_entity = launch_ros.actions.Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'testing', '-topic', 'robot_description'],
        output='screen'
    )
    rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
    )
    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_state_broadcaster'],
        output='screen'
    )

    load_joint_trajectory_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_trajectory_controller'],
        output='screen'
    )
    return launch.LaunchDescription([
        # RegisterEventHandler(
        #     event_handler=OnProcessExit(
        #         target_action=spawn_entity,
        #         on_exit=[load_joint_state_broadcaster],
        #     )
        # ),
        # RegisterEventHandler(
        #     event_handler=OnProcessExit(
        #         target_action=load_joint_state_broadcaster,
        #         on_exit=[load_joint_trajectory_controller],
        #     )
        # ),
        launch.actions.DeclareLaunchArgument(name='model', default_value=default_model_path,
                                            description='Absolute path to robot urdf file'),
        launch.actions.IncludeLaunchDescription(launch.launch_description_sources.PythonLaunchDescriptionSource(pkg_share_gazebo + '/launch/gazebo.launch.py')),                                            
        robot_state_publisher_node,
        spawn_entity,
        rviz_node
    ])