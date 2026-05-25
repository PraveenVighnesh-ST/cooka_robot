"""
Pick-and-place node for the cooka robot.

Workflow (to be implemented):
  1. Spawn payload at a named location (module A, B, etc.)
  2. Record / confirm EE pick position for that location
  3. Execute pick: move to pre-pick → pick → lift
  4. Execute place: move to pre-place → place → retreat
"""

import rclpy
from rclpy.node import Node


class PnPNode(Node):

    def __init__(self):
        super().__init__('pnp_node')
        self.get_logger().info('PnP node started — ready for pick-and-place operations.')

    # ------------------------------------------------------------------ #
    # Future methods:                                                       #
    #   spawn_payload(location_name)                                        #
    #   record_ee_position(label)                                           #
    #   execute_pick(location_name)                                         #
    #   execute_place(location_name)                                        #
    # ------------------------------------------------------------------ #


def main(args=None):
    rclpy.init(args=args)
    node = PnPNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
