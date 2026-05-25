import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro
from os.path import join


def generate_launch_description():

    os.environ['RMW_IMPLEMENTATION'] = 'rmw_cyclonedds_cpp'

    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_cooka = get_package_share_directory('cooka_description')

    robot_description_file = os.path.join(pkg_cooka, 'urdf', 'cooka.xacro')
    ros_gz_bridge_config = os.path.join(pkg_cooka, 'config', 'ros_gz_bridge_gazebo.yaml')

    robot_description_config = xacro.process_file(robot_description_file)
    robot_description = {'robot_description': robot_description_config.toxml()}

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': True}],
        additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': '-r -v 4 empty.sdf'}.items()
    )

    spawn_robot = TimerAction(
        period=3.0,
        actions=[Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-topic', '/robot_description',
                '-name', 'cooka',
                '-allow_renaming', 'false',
                '-x', '0.0',
                '-y', '0.0',
                '-z', '0.0',
                '-Y', '0.0'
            ],
            output='screen',
            additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
        )]
    )

    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': ros_gz_bridge_config}],
        output='screen',
        additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
    )

    joint_state_broadcaster_spawner = TimerAction(
        period=6.0,
        actions=[Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
            output='screen',
            additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
        )]
    )

    arm_controller_spawner = TimerAction(
        period=8.0,
        actions=[Node(
            package='controller_manager',
            executable='spawner',
            arguments=['arm_controller', '--controller-manager', '/controller_manager'],
            output='screen',
            additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
        )]
    )

    gripper_controller_spawner = TimerAction(
        period=9.0,
        actions=[Node(
            package='controller_manager',
            executable='spawner',
            arguments=['gripper_controller', '--controller-manager', '/controller_manager'],
            output='screen',
            additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
        )]
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_robot,
        ros_gz_bridge,
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
        gripper_controller_spawner,
    ])
