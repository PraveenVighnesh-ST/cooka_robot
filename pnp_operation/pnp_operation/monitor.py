"""
Live terminal monitor for the cooka robot.

Shows a table of joint positions/velocities and the FK-computed EE position,
refreshing at ~10 Hz.  Run alongside the simulation:

    ros2 run pnp_operation monitor
"""

import sys
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

try:
    from cooka_description.cooka_kinematics import fk_position
except ImportError:
    fk_position = None

JOINT_ORDER = ['h_slider', 'v_slider', 'scara_j1', 'scara_j2', 'wrist_z', 'wrist_y']
GRIPPER_JOINT = 'finger_bottom'

# ANSI helpers
CLEAR_SCREEN = '\033[2J\033[H'
BOLD = '\033[1m'
RESET = '\033[0m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
RED = '\033[31m'
CYAN = '\033[36m'


def _fmt_pos(v, is_revolute):
    if is_revolute:
        return f'{np.degrees(v):+8.2f} deg'
    return f'{v:+8.4f} m  '


def _fmt_vel(v):
    if abs(v) > 0.5:
        color = RED
    elif abs(v) > 0.1:
        color = YELLOW
    else:
        color = GREEN
    return f'{color}{v:+8.4f}{RESET}'


class MonitorNode(Node):
    def __init__(self):
        super().__init__('cooka_monitor')
        self._lock = threading.Lock()
        self._js = {}
        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)
        self.create_timer(0.1, self._print)

    def _js_cb(self, msg):
        with self._lock:
            for name, pos, vel in zip(msg.name, msg.position, msg.velocity):
                self._js[name] = (pos, vel)

    def _print(self):
        with self._lock:
            js = dict(self._js)

        lines = [CLEAR_SCREEN]
        lines.append(f'{BOLD}{CYAN}=== cooka robot monitor ==={RESET}')
        lines.append(f'{"Joint":<16} {"Position":>18} {"Velocity (rad/s or m/s)":>26}')
        lines.append('-' * 62)

        revolute = {'scara_j1', 'scara_j2', 'wrist_z', 'wrist_y'}
        q = []
        for jname in JOINT_ORDER:
            pos, vel = js.get(jname, (0.0, 0.0))
            q.append(pos)
            is_rev = jname in revolute
            lines.append(f'{jname:<16} {_fmt_pos(pos, is_rev)} {_fmt_vel(vel)}')

        grip_pos, grip_vel = js.get(GRIPPER_JOINT, (0.0, 0.0))
        lines.append(f'{GRIPPER_JOINT:<16} {grip_pos:+8.4f} m   {_fmt_vel(grip_vel)}')

        lines.append('')
        if fk_position is not None and len(q) == 6:
            try:
                ee = fk_position(q)
                lines.append(f'{BOLD}EE position (FK):{RESET}  '
                              f'x={ee[0]:.4f}  y={ee[1]:.4f}  z={ee[2]:.4f}')
            except Exception as exc:
                lines.append(f'FK error: {exc}')
        else:
            lines.append('(cooka_kinematics not available — FK skipped)')

        lines.append('')
        lines.append("Press Ctrl-C to quit.")
        sys.stdout.write('\n'.join(lines))
        sys.stdout.flush()


def main(args=None):
    rclpy.init(args=args)
    node = MonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        sys.stdout.write('\n')
