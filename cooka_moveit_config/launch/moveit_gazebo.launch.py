"""
Launch Gazebo simulation, move_group, and RViz for the cooka robot.

Timeline:
  t= 3 s  robot spawned in Gazebo
  t= 6 s  joint_state_broadcaster active
  t= 8 s  arm_controller active
  t= 9 s  gripper_controller active
  t=12 s  move_group starts
  t=15 s  RViz starts (after move_group is ready)
"""

import os
import yaml
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def _load_yaml(package_name, rel_path):
    path = os.path.join(get_package_share_directory(package_name), rel_path)
    with open(path) as f:
        return yaml.safe_load(f)


def generate_launch_description():

    os.environ['RMW_IMPLEMENTATION'] = 'rmw_cyclonedds_cpp'

    pkg_moveit = get_package_share_directory('cooka_moveit_config')
    pkg_cooka = get_package_share_directory('cooka_description')

    # --- robot description (with Gazebo plugins active) ---
    robot_description_config = xacro.process_file(
        os.path.join(pkg_cooka, 'urdf', 'cooka.xacro'),
        mappings={'use_gazebo': 'true'},
    )
    robot_description = {'robot_description': robot_description_config.toxml()}

    with open(os.path.join(pkg_moveit, 'config', 'cooka.srdf')) as f:
        robot_description_semantic = {'robot_description_semantic': f.read()}

    robot_description_kinematics = {
        'robot_description_kinematics': _load_yaml('cooka_moveit_config', 'config/kinematics.yaml')
    }

    robot_description_planning = {
        'robot_description_planning': _load_yaml('cooka_moveit_config', 'config/joint_limits.yaml')
    }

    ompl_yaml = _load_yaml('cooka_moveit_config', 'config/ompl_planning.yaml')
    planning_pipelines = {
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'ompl': ompl_yaml,
    }

    moveit_controllers = _load_yaml('cooka_moveit_config', 'config/moveit_controllers.yaml')

    # --- Gazebo simulation (robot spawn + controllers at t=3/6/8/9 s) ---
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_cooka, 'launch', 'gazebo.launch.py')
        )
    )

    # --- move_group: wait for arm_controller to be active before starting ---
    move_group = TimerAction(
        period=12.0,
        actions=[Node(
            package='moveit_ros_move_group',
            executable='move_group',
            output='screen',
            parameters=[
                robot_description,
                robot_description_semantic,
                robot_description_kinematics,
                robot_description_planning,
                planning_pipelines,
                moveit_controllers,
                {
                    'use_sim_time': True,
                    'publish_robot_description_semantic': True,
                    'allow_trajectory_execution': True,
                    'capabilities': '',
                    'disable_capabilities': '',
                    'monitor_dynamics': False,
                },
            ],
            additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
        )]
    )

    # --- RViz: start after move_group is ready ---
    rviz = TimerAction(
        period=15.0,
        actions=[Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(pkg_moveit, 'config', 'moveit.rviz')],
            output='screen',
            parameters=[
                robot_description,
                robot_description_semantic,
                robot_description_kinematics,
                {'use_sim_time': True},
            ],
            additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
        )]
    )

    return LaunchDescription([gazebo, move_group, rviz])
