"""
Launch RViz2 with the MoveIt2 MotionPlanning panel.

Connects automatically to a running move_group (started via moveit_gazebo.launch.py).
The MotionPlanning panel shows "waiting for move_group" until it is available.
"""

import os
import yaml
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def _load_yaml(package_name, rel_path):
    path = os.path.join(get_package_share_directory(package_name), rel_path)
    with open(path) as f:
        return yaml.safe_load(f)


def generate_launch_description():

    os.environ['RMW_IMPLEMENTATION'] = 'rmw_cyclonedds_cpp'

    pkg_moveit = get_package_share_directory('cooka_moveit_config')
    pkg_cooka = get_package_share_directory('cooka_description')

    # use_gazebo=false: exclude Gazebo plugin elements from the RViz description
    robot_description_config = xacro.process_file(
        os.path.join(pkg_cooka, 'urdf', 'cooka.xacro'),
        mappings={'use_gazebo': 'false'},
    )
    robot_description = {'robot_description': robot_description_config.toxml()}

    with open(os.path.join(pkg_moveit, 'config', 'cooka.srdf')) as f:
        robot_description_semantic = {'robot_description_semantic': f.read()}

    robot_description_kinematics = {
        'robot_description_kinematics': _load_yaml('cooka_moveit_config', 'config/kinematics.yaml')
    }

    ompl_yaml = _load_yaml('cooka_moveit_config', 'config/ompl_planning.yaml')
    planning_pipelines = {
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'ompl': ompl_yaml,
    }

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg_moveit, 'config', 'moveit.rviz')],
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            planning_pipelines,
            {'use_sim_time': True},
        ],
        additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
    )

    return LaunchDescription([rviz])
