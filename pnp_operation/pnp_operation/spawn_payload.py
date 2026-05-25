"""Spawn a box payload in the running Gazebo simulation."""

import argparse
import subprocess
import sys

# Box: 80 mm × 80 mm × 50 mm, 0.5 kg.
# Link name must stay "link" — the DetachableJoint plugin in cooka.gazebo
# attaches to active_payload::link.
_PAYLOAD_SDF = """\
<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="active_payload">
    <static>false</static>
    <link name="link">
      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.000371</ixx>
          <iyy>0.000371</iyy>
          <izz>0.000533</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><box><size>0.08 0.08 0.05</size></box></geometry>
        <surface>
          <friction><ode><mu>0.8</mu><mu2>0.8</mu2></ode></friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry><box><size>0.08 0.08 0.05</size></box></geometry>
        <material>
          <ambient>1.0 0.5 0.0 1</ambient>
          <diffuse>1.0 0.5 0.0 1</diffuse>
          <specular>0.2 0.2 0.2 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


def main(args=None):
    parser = argparse.ArgumentParser(description='Spawn payload box in Gazebo')
    parser.add_argument('--x', type=float, default=0.0, metavar='X')
    parser.add_argument('--y', type=float, default=0.0, metavar='Y')
    parser.add_argument('--z', type=float, default=0.0, metavar='Z')
    parser.add_argument(
        '--name', default='active_payload',
        help='Model name (default: active_payload — matches DetachableJoint)')
    parsed = parser.parse_args(args)

    sdf = _PAYLOAD_SDF.replace('name="active_payload"', f'name="{parsed.name}"')

    cmd = [
        'ros2', 'run', 'ros_gz_sim', 'create',
        '-string', sdf,
        '-name', parsed.name,
        '-x', str(parsed.x),
        '-y', str(parsed.y),
        '-z', str(parsed.z),
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        sys.exit(result.returncode)
