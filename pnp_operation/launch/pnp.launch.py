from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os


def generate_launch_description():

    cooka_gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('cooka_description'),
                'launch', 'gazebo.launch.py'
            )
        )
    )

    return LaunchDescription([
        cooka_gazebo_launch,
    ])
