"""Spawn the payload_pan mesh as a model in the running Gazebo simulation."""

import argparse
import subprocess
import sys

from ament_index_python.packages import get_package_share_directory

# STL is in mm (same convention as all cooka meshes) — scale 0.001 to metres.
_MESH_SCALE = '0.001 0.001 0.001'

_SDF_TEMPLATE = """\
<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <mesh>
            <uri>file://{mesh_path}</uri>
            <scale>{scale}</scale>
          </mesh>
        </geometry>
        <surface>
          <friction><ode><mu>0.8</mu><mu2>0.8</mu2></ode></friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>
          <mesh>
            <uri>file://{mesh_path}</uri>
            <scale>{scale}</scale>
          </mesh>
        </geometry>
        <material>
          <ambient>0.7 0.7 0.7 1</ambient>
          <diffuse>0.7 0.7 0.7 1</diffuse>
          <specular>0.3 0.3 0.3 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


def main(args=None):
    parser = argparse.ArgumentParser(description='Spawn payload_pan in Gazebo')
    parser.add_argument('--x', type=float, default=0.0, metavar='X')
    parser.add_argument('--y', type=float, default=0.0, metavar='Y')
    parser.add_argument('--z', type=float, default=0.0, metavar='Z')
    parser.add_argument('--yaw', type=float, default=0.0, metavar='YAW')
    parser.add_argument(
        '--name', default='active_payload',
        help='Model name (default: active_payload — matches DetachableJoint)')
    parsed = parser.parse_args(args)

    mesh_path = get_package_share_directory('pnp_operation') + '/meshes/payload_pan.stl'

    sdf = _SDF_TEMPLATE.format(
        name=parsed.name,
        mesh_path=mesh_path,
        scale=_MESH_SCALE,
    )

    cmd = [
        'ros2', 'run', 'ros_gz_sim', 'create',
        '-string', sdf,
        '-name', parsed.name,
        '-x', str(parsed.x),
        '-y', str(parsed.y),
        '-z', str(parsed.z),
        '-Y', str(parsed.yaw),
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        sys.exit(result.returncode)
