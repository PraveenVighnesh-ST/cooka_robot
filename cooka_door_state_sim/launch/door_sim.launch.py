"""
Launch file for door-state pick-and-place simulation.

Starts the full MoveIt2 + Gazebo stack, spawns the payload, then starts
door_manager and door_pnp_node so the door-aware sequence runs automatically.

Timeline
--------
  t= 0 s  Gazebo + robot spawned
  t= 4 s  joint_state_broadcaster active
  t= 5 s  arm_controller active
  t= 6 s  gripper_controller active
  t= 7 s  payload spawned at pick A position (gravity=false, floats in place)
  t=12 s  move_group starts
  t=15 s  RViz starts
  t=17 s  door_manager starts (manages planning scene collision zones)
  t=22 s  door_pnp_node starts (door-aware pick-and-place sequence)

What to watch in RViz
---------------------
  - "Planning Scene" display shows door zone boxes appearing/disappearing.
  - Both modules start CLOSED (entry-blocking zones visible).
  - Zones switch as the sequence progresses through OPENING → OPEN → CLOSING → CLOSED.
  - Robot arm stays retracted (HOME config) every time a zone is active.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    pkg_moveit = get_package_share_directory('cooka_moveit_config')
    pkg_door   = get_package_share_directory('cooka_door_state_sim')

    modules_config = os.path.join(pkg_door, 'config', 'modules.yaml')

    # ── Full MoveIt2 + Gazebo + RViz stack ────────────────────────────────────
    moveit_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_moveit, 'launch', 'moveit_gazebo.launch.py')
        )
    )

    # ── Spawn payload at pick A position (same coords as pnp_operation) ───────
    # gravity=false in the SDF keeps it floating until DetachableJoint attaches.
    # door_pnp_node sends detach at startup before any motion.
    spawn_payload = TimerAction(
        period=7.0,
        actions=[ExecuteProcess(
            cmd=[
                'ros2', 'run', 'pnp_operation', 'spawn_payload',
                '--x', '0.3',
                '--y', '0.5',
                '--z', '0.425',
                '--name', 'active_payload',
            ],
            additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
            output='screen',
        )]
    )

    # ── Door manager — publishes collision zones to MoveIt2 planning scene ─────
    door_manager = TimerAction(
        period=17.0,
        actions=[Node(
            package='cooka_door_state_sim',
            executable='door_manager',
            name='door_manager',
            output='screen',
            parameters=[{'modules_config': modules_config}],
            additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
        )]
    )

    # ── Door-aware pick-and-place node ────────────────────────────────────────
    door_pnp_node = TimerAction(
        period=22.0,
        actions=[Node(
            package='cooka_door_state_sim',
            executable='door_pnp_node',
            name='door_pnp_node',
            output='screen',
            additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
        )]
    )

    return LaunchDescription([
        moveit_gazebo,
        spawn_payload,
        door_manager,
        door_pnp_node,
    ])
