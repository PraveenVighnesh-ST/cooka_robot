import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    # ── Robot simulation (Gazebo + controllers) ────────────────────────────────
    cooka_gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('cooka_description'),
                'launch', 'gazebo.launch.py'
            )
        )
    )

    # ── Spawn payload at pick A position early so it is visible from the start ─
    # gravity=false (set in spawn_payload SDF) keeps it floating at this position
    # until the DetachableJoint receives an attach command from pnp_node.
    # pnp_node sends detach at startup (t≈15 s) before any motion, so the payload
    # stays free at (0.3, 0.319, 0.525) while the robot homes and aligns.
    spawn_payload = TimerAction(
        period=5.0,
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

    # ── Pick-and-place node ────────────────────────────────────────────────────
    # Starts at t=15 s. First action in _run() is to send detach × 3 so the
    # payload (auto-attached by DetachableJoint on spawn) is released back to
    # its spawn position before the robot moves.
    pnp_node = TimerAction(
        period=15.0,
        actions=[Node(
            package='pnp_operation',
            executable='pnp_node',
            name='pnp_node',
            output='screen',
            additional_env={'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'},
        )]
    )

    return LaunchDescription([
        cooka_gazebo_launch,
        spawn_payload,
        pnp_node,
    ])
