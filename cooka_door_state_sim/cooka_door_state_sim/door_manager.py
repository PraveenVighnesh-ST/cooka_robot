"""
Door state manager for the cooka smart kitchen simulation.

Subscribes to /door_command  (std_msgs/String)
  Format:  "<module_name>:<STATE>"
  States:
    CLOSED  — box covers module opening; EE cannot enter module.
    OPENING — tall box covers door sweep zone (front + vertical travel area).
    OPEN    — box at top where door rests; front is clear so EE can enter.
    CLEAR   — remove all zones for this module (debug/reset use).

Each module has exactly ONE collision box active at a time.
The box is defined in modules.yaml as a corner (bottom-left-front) + dims (W×D×H).
Centre = corner + dims/2 per axis.  Frame: base_link.

Verification commands (run in a second terminal while door_sim.launch.py is active):
  ros2 topic pub --once /door_command std_msgs/msg/String "data: 'module1:CLOSED'"
  ros2 topic pub --once /door_command std_msgs/msg/String "data: 'module1:OPENING'"
  ros2 topic pub --once /door_command std_msgs/msg/String "data: 'module1:OPEN'"
  ros2 topic pub --once /door_command std_msgs/msg/String "data: 'module1:CLEAR'"
  ros2 topic pub --once /door_command std_msgs/msg/String "data: 'module2:CLOSED'"
  ros2 topic pub --once /door_command std_msgs/msg/String "data: 'module2:OPENING'"
  ros2 topic pub --once /door_command std_msgs/msg/String "data: 'module2:OPEN'"
  ros2 topic pub --once /door_command std_msgs/msg/String "data: 'module2:CLEAR'"
"""

import os
import yaml

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, Point, Quaternion


VALID_STATES = ('CLOSED', 'OPENING', 'OPEN', 'CLEAR')

# Accept common variants so the user isn't tripped up by capitalisation
_STATE_ALIASES = {
    'OPENED':  'OPEN',
    'CLOSE':   'CLOSED',
    'CLOSING': 'OPENING',   # door closing uses the same zone as door opening
    'RESET':   'CLEAR',
    'NONE':    'CLEAR',
}


class DoorManager(Node):

    def __init__(self):
        super().__init__('door_manager')

        self.declare_parameter('modules_config', '')
        config_path = self.get_parameter('modules_config').value

        if not config_path or not os.path.isfile(config_path):
            self.get_logger().fatal(f'modules_config not found: "{config_path}"')
            raise RuntimeError('Missing modules_config parameter')

        with open(config_path) as f:
            data = yaml.safe_load(f)
        self.modules: dict = data['modules']

        self._active_states: dict[str, str] = {}

        self._ps_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        self._cmd_sub = self.create_subscription(
            String, '/door_command', self._on_command, 10
        )

        names = ', '.join(self.modules.keys())
        self.get_logger().info(f'Door manager ready.  Modules: {names}')
        self.get_logger().info(
            'Commands → /door_command  as  "<module>:CLOSED|OPENING|OPEN|CLEAR"'
        )

    # ── public API (callable from other nodes) ────────────────────────────────

    def set_door_state(self, module_name: str, state: str) -> bool:
        state = _STATE_ALIASES.get(state.upper().strip(), state.upper().strip())
        if module_name not in self.modules:
            self.get_logger().error(f'Unknown module "{module_name}"')
            return False
        if state not in VALID_STATES:
            self.get_logger().error(
                f'Unknown state "{state}" — valid: {VALID_STATES}'
            )
            return False

        self._active_states[module_name] = state
        self._publish(module_name, state)
        return True

    # ── internals ─────────────────────────────────────────────────────────────

    def _on_command(self, msg: String):
        raw = msg.data.strip()
        if ':' not in raw:
            self.get_logger().error(
                f'Bad format "{raw}" — expected "<module>:<STATE>"'
            )
            return
        module_name, state = raw.split(':', 1)
        self.set_door_state(module_name.strip(), state.strip())

    def _publish(self, module_name: str, state: str):
        obj_id = f'{module_name}_door_zone'
        ps = PlanningScene()
        ps.is_diff = True

        if state == 'CLEAR':
            ps.world.collision_objects = [self._remove(obj_id)]
            self.get_logger().info(f'[{module_name}] CLEAR — zone removed')
        else:
            cfg = self.modules[module_name][state.lower()]
            co = self._make_box(obj_id, cfg['dims'], cfg['corner'])
            ps.world.collision_objects = [co]
            cx, cy, cz = self._centre(cfg['corner'], cfg['dims'])
            self.get_logger().info(
                f'[{module_name}] {state} — box '
                f'{[round(d*1000) for d in cfg["dims"]]} mm  '
                f'centre ({cx:.3f}, {cy:.3f}, {cz:.3f}) m'
            )

        self._ps_pub.publish(ps)

    # ── collision object helpers ───────────────────────────────────────────────

    @staticmethod
    def _centre(corner: list, dims: list) -> tuple:
        return (
            corner[0] + dims[0] / 2.0,
            corner[1] + dims[1] / 2.0,
            corner[2] + dims[2] / 2.0,
        )

    def _make_box(
        self, obj_id: str, dims: list, corner: list
    ) -> CollisionObject:
        co = CollisionObject()
        co.id = obj_id
        co.header.frame_id = 'base_link'
        co.operation = CollisionObject.ADD

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [float(d) for d in dims]

        cx, cy, cz = self._centre(corner, dims)
        pose = Pose(
            position=Point(x=cx, y=cy, z=cz),
            orientation=Quaternion(w=1.0),
        )

        co.primitives = [box]
        co.primitive_poses = [pose]
        return co

    def _remove(self, obj_id: str) -> CollisionObject:
        co = CollisionObject()
        co.id = obj_id
        co.operation = CollisionObject.REMOVE
        return co


def main(args=None):
    rclpy.init(args=args)
    node = DoorManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
