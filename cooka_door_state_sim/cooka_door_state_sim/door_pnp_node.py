"""
Door-aware pick-and-place node for the cooka smart kitchen robot.

Picks payload from module1, transports it, drops at module2.
The robot always retracts its arm to the HOME (folded) configuration
before any door state changes — this is the "safe position" that
geometrically avoids all door collision zones.

Sequence
--------
INIT
  Both modules → CLOSED  (entry-blocking zones active in planning scene)
  Robot → HOME (arm fully retracted)

PICK  (module1)
  1. Align gantry to module1, arm retracted  ← safe position
  2. module1 → OPENING   (front zone activates — door travelling up)
  3. Wait for door to open
  4. module1 → OPEN      (front zone removed, top zone activates)
  5. Extend arm → pick payload at PICK_A
  6. Grip + attach payload (DetachableJoint)
  7. Retract arm           ← back to safe position

CLOSE module1
  8. module1 → CLOSING   (= OPENING zone — same zone, door travelling down)
  9. Wait for door to close
 10. module1 → CLOSED

TRANSPORT
 11. Align gantry to module2, arm retracted  ← safe position

DROP  (module2)
 12. module2 → OPENING
 13. Wait for door to open
 14. module2 → OPEN
 15. Extend arm → drop payload at DROP_B
 16. Detach + release payload (DetachableJoint + gripper)
 17. Retract arm           ← safe position

CLOSE module2
 18. module2 → CLOSING
 19. Wait for door to close
 20. module2 → CLOSED

COMPLETE
 21. Return to absolute HOME
"""

import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from cooka_description.cooka_kinematics import ik, JOINT_LIMITS

# ── Robot constants (identical to pnp_node) ───────────────────────────────────

ARM_JOINTS = ['h_slider', 'v_slider', 'scara_j1', 'scara_j2', 'wrist_z', 'wrist_y']

PNP_HOME = [
    0.0,
    0.0,
    np.radians(120),
    np.radians(-240),
    -2.094,
    0.0,
]

PICK_A = (0.3,  0.319, 0.525)
DROP_B = (1.6,  0.319, 1.1)

Y_SPEED      = 0.200
SAMPLE_DT    = 0.020
SAFE_Y       = 0.165
GANTRY_SPEED = 0.20
HOME_SPEED   = 1.0
HOME_SETTLE  = 2.0

# ── Door simulation timing ────────────────────────────────────────────────────

DOOR_TRAVEL_TIME = 3.5   # seconds simulated door travel (opening or closing)
INIT_DISPLAY_WAIT = 2.0  # seconds to show initial CLOSED zones before starting


class DoorPnPNode(Node):
    """Door-aware pick-and-place: module1 → module2 with door state management."""

    def __init__(self):
        super().__init__('door_pnp_node')

        # ── Arm trajectory action client ──────────────────────────────────────
        self._arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')

        # ── Gripper (ForwardCommandController) ────────────────────────────────
        self._gripper_pub = self.create_publisher(
            Float64MultiArray, '/gripper_controller/commands', 10)

        # ── DetachableJoint (Gazebo plugin via ros_gz_bridge) ─────────────────
        self._attach_pub = self.create_publisher(
            Empty, '/cooka/detachable_joint/attach', 10)
        self._detach_pub = self.create_publisher(
            Empty, '/cooka/detachable_joint/detach', 10)

        # ── Door state publisher → door_manager node ──────────────────────────
        self._door_pub = self.create_publisher(String, '/door_command', 10)

        # ── Joint state subscriber ────────────────────────────────────────────
        self._joint_pos = {j: 0.0 for j in ARM_JOINTS}
        self._joint_pos['finger_bottom'] = 0.0
        self._js_received = False
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)

        self.get_logger().info(
            'DoorPnP: waiting for arm_controller action server...')
        self._arm_client.wait_for_server()
        self.get_logger().info(
            'DoorPnP: arm_controller ready — waiting for joint states...')
        self._wait_js()

        self._run()

    # ── Joint state helpers ───────────────────────────────────────────────────

    def _on_js(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in self._joint_pos:
                self._joint_pos[name] = pos
        self._js_received = True

    def _wait_js(self):
        while not self._js_received:
            rclpy.spin_once(self, timeout_sec=0.5)

    def _q_now(self) -> list:
        return [self._joint_pos[j] for j in ARM_JOINTS]

    # ── Door helpers ──────────────────────────────────────────────────────────

    def _set_door(self, module: str, state: str):
        """Publish door state command and log."""
        msg = String()
        msg.data = f'{module}:{state}'
        self._door_pub.publish(msg)
        self.get_logger().info(f'  DOOR [{module}] → {state}')

    def _wait_door(self, seconds: float, label: str):
        """Spin the node while simulating door travel time."""
        self.get_logger().info(
            f'  Waiting {seconds:.1f}s for door to {label}...')
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    # ── Gripper / DetachableJoint ─────────────────────────────────────────────

    def _grip(self, depth_m: float = 0.015):
        cmd = Float64MultiArray()
        cmd.data = [-min(abs(depth_m), 0.015)]
        self._gripper_pub.publish(cmd)
        self.get_logger().info(f'  Gripper close {depth_m * 1000:.0f} mm')
        time.sleep(0.8)

    def _release(self):
        cmd = Float64MultiArray()
        cmd.data = [0.0]
        self._gripper_pub.publish(cmd)
        self.get_logger().info('  Gripper open')
        time.sleep(0.3)

    def _attach(self):
        self._attach_pub.publish(Empty())
        self.get_logger().info('  DetachableJoint: attach')
        time.sleep(0.2)

    def _detach(self, count: int = 3):
        for _ in range(count):
            self._detach_pub.publish(Empty())
            time.sleep(0.15)
        self.get_logger().info('  DetachableJoint: detach')

    # ── Trajectory helpers (identical to pnp_node) ────────────────────────────

    @staticmethod
    def _pt(q: list, t_s: float) -> JointTrajectoryPoint:
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        pt.time_from_start.sec = int(t_s)
        pt.time_from_start.nanosec = int(round((t_s % 1) * 1e9))
        return pt

    def _send_traj(self, waypoints: list, timestamps: list) -> bool:
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        traj.points = [self._pt(q, t) for q, t in zip(waypoints, timestamps)]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        send_fut = self._arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut)
        gh = send_fut.result()
        if not gh.accepted:
            self.get_logger().error('Trajectory rejected by controller')
            return False

        result_fut = gh.get_result_async()
        timeout = timestamps[-1] * 4.0 + 60.0
        rclpy.spin_until_future_complete(self, result_fut, timeout_sec=timeout)
        wrapped = result_fut.result()
        if wrapped is None:
            self.get_logger().error('Trajectory timed out')
            return False
        return wrapped.result.error_code == 0

    # ── Motion primitives (identical to pnp_node) ─────────────────────────────

    def _go_home(self) -> bool:
        q_start = self._q_now()
        deltas = [abs(PNP_HOME[i] - q_start[i]) for i in range(6)]
        t_move = max(max(deltas) / HOME_SPEED, 2.0)
        t_settle = t_move + HOME_SETTLE
        self.get_logger().info(f'  Homing ({t_move:.1f}s move + {HOME_SETTLE}s settle)')
        return self._send_traj(
            [q_start, PNP_HOME, PNP_HOME],
            [0.0,     t_move,   t_settle])

    def _align_gantry(self, q0: float, q1: float) -> bool:
        q_start = self._q_now()
        q_target = list(q_start)
        q_target[0] = q0
        q_target[1] = q1
        q_target[2:] = PNP_HOME[2:]
        dist = max(abs(q0 - q_start[0]), abs(q1 - q_start[1]), 1e-4)
        t_move = max(dist / GANTRY_SPEED, 1.0)
        self.get_logger().info(
            f'  Align gantry h={q0:.3f}m v={q1:.3f}m ({t_move:.1f}s)')
        return self._send_traj(
            [q_start, q_target, q_target],
            [0.0,     t_move,   t_move + 1.5])

    def _fold_scara(self, q0: float, q1: float) -> bool:
        q_start = self._q_now()
        q_home = list(q_start)
        q_home[0], q_home[1] = q0, q1
        q_home[2:] = PNP_HOME[2:]
        max_delta = max(abs(q_home[i] - q_start[i]) for i in [2, 3, 4])
        dur = max(max_delta / 1.0, 0.5)
        self.get_logger().info(f'  Fold SCARA → home ({dur:.1f}s)')
        return self._send_traj([q_start, q_home], [0.0, dur])

    def _unfold_to_safe_y(
            self, q0: float, q1: float, px: float, pz: float) -> bool:
        q_safe = ik(px, SAFE_Y, pz, yaw_ee=0.0, elbow='down')
        if q_safe is None:
            self.get_logger().error(f'IK failed at SAFE_Y px={px} pz={pz}')
            return False
        q_safe[0], q_safe[1] = q0, q1
        q_start = self._q_now()
        max_delta = max(abs(q_safe[i] - q_start[i]) for i in [2, 3, 4])
        dur = max(max_delta / 1.0, 0.5)
        self.get_logger().info(f'  Unfold HOME → SAFE_Y={SAFE_Y}m ({dur:.1f}s)')
        return self._send_traj([q_start, q_safe], [0.0, dur])

    def _y_cartesian(
            self, q0: float, q1: float, px: float, pz: float,
            y_start: float, y_end: float) -> bool:
        n = max(2, int(abs(y_end - y_start) / (Y_SPEED * SAMPLE_DT)) + 1)
        y_arr = np.linspace(y_start, y_end, n)
        waypoints, timestamps, t = [], [], 0.0
        for i, py in enumerate(y_arr):
            q = ik(px, float(py), pz, yaw_ee=0.0, elbow='down')
            if q is None:
                self.get_logger().error(f'IK failed at y={py:.4f}m')
                return False
            q[0], q[1] = q0, q1
            waypoints.append(q)
            timestamps.append(t)
            if i < n - 1:
                t += abs(float(y_arr[i + 1]) - float(py)) / Y_SPEED
        direction = 'extend' if y_end > y_start else 'retract'
        self.get_logger().info(
            f'  Cartesian Y {direction} {y_start:.3f}→{y_end:.3f}m '
            f'({n} pts, {t:.1f}s)')
        return self._send_traj(waypoints, timestamps)

    def _extend_arm(
            self, q0: float, q1: float, px: float, pz: float,
            y_target: float) -> bool:
        self.get_logger().info(f'  Extend arm → y={y_target:.3f}m')
        if not self._unfold_to_safe_y(q0, q1, px, pz):
            return False
        return self._y_cartesian(q0, q1, px, pz, SAFE_Y, y_target)

    def _retract_arm(
            self, q0: float, q1: float, px: float, pz: float,
            y_current: float) -> bool:
        self.get_logger().info(f'  Retract arm y={y_current:.3f}m → HOME')
        if y_current > SAFE_Y + 0.005:
            if not self._y_cartesian(q0, q1, px, pz, y_current, SAFE_Y):
                return False
        return self._fold_scara(q0, q1)

    @staticmethod
    def _gantry_for(px: float, pz: float) -> tuple:
        q0 = float(np.clip(px - 0.275, JOINT_LIMITS[0, 0], JOINT_LIMITS[0, 1]))
        q1 = float(np.clip(0.125 - pz, JOINT_LIMITS[1, 0], JOINT_LIMITS[1, 1]))
        return q0, q1

    # ── Logging banner ────────────────────────────────────────────────────────

    def _banner(self, step: int, total: int, label: str):
        self.get_logger().info(
            f'\n{"="*60}\n  Step {step}/{total}: {label}\n{"="*60}')

    # ── Main sequence ─────────────────────────────────────────────────────────

    def _run(self):
        TOTAL = 21
        px_a, py_a, pz_a = PICK_A
        px_b, py_b, pz_b = DROP_B
        q0_a, q1_a = self._gantry_for(px_a, pz_a)
        q0_b, q1_b = self._gantry_for(px_b, pz_b)

        self.get_logger().info(
            f'Pick A : EE=({px_a},{py_a},{pz_a})  '
            f'gantry h={q0_a:.3f}m v={q1_a:.3f}m')
        self.get_logger().info(
            f'Drop B : EE=({px_b},{py_b},{pz_b})  '
            f'gantry h={q0_b:.3f}m v={q1_b:.3f}m')

        # ── INIT ─────────────────────────────────────────────────────────────
        self._banner(0, TOTAL, 'INIT — set initial door states + release payload')
        self._set_door('module1', 'CLOSED')
        self._set_door('module2', 'CLOSED')
        self.get_logger().info(
            f'  Both modules CLOSED. '
            f'Waiting {INIT_DISPLAY_WAIT}s for RViz to show zones...')
        self._wait_door(INIT_DISPLAY_WAIT, 'display')

        # Release payload that was auto-attached on spawn
        self._release()
        self._detach()

        # ── Step 1: Home ──────────────────────────────────────────────────────
        self._banner(1, TOTAL, 'Home robot')
        if not self._go_home():
            self.get_logger().error('Homing failed — aborting')
            return

        # ── Step 2: Safe position near module1 ───────────────────────────────
        self._banner(2, TOTAL, 'Align gantry → module1 (arm retracted = safe position)')
        if not self._align_gantry(q0_a, q1_a):
            self.get_logger().error('Gantry align (module1) failed — aborting')
            return

        # ── Step 3: Open module1 door ─────────────────────────────────────────
        self._banner(3, TOTAL, 'module1 door → OPENING')
        self.get_logger().info(
            '  Arm is retracted — geometrically avoids OPENING zone.')
        self._set_door('module1', 'OPENING')
        self._wait_door(DOOR_TRAVEL_TIME, 'finish opening')

        # ── Step 4: Door fully open ───────────────────────────────────────────
        self._banner(4, TOTAL, 'module1 door → OPEN (front clear, top blocked)')
        self._set_door('module1', 'OPEN')

        # ── Step 5: Pick payload ──────────────────────────────────────────────
        self._banner(5, TOTAL, 'Extend arm → Pick payload at module1')
        if not self._extend_arm(q0_a, q1_a, px_a, pz_a, py_a):
            self.get_logger().error('Extend (pick) failed — aborting')
            return

        # ── Step 6: Grip ──────────────────────────────────────────────────────
        self._banner(6, TOTAL, 'Grip + attach payload')
        self._grip()
        self._attach()

        # ── Step 7: Retract arm → safe position ──────────────────────────────
        self._banner(7, TOTAL, 'Retract arm → safe position (module1)')
        if not self._retract_arm(q0_a, q1_a, px_a, pz_a, py_a):
            self.get_logger().error('Retract (pick) failed — aborting')
            return

        # ── Step 8: Close module1 door ────────────────────────────────────────
        self._banner(8, TOTAL, 'module1 door → CLOSING (arm retracted = safe)')
        self._set_door('module1', 'CLOSING')
        self._wait_door(DOOR_TRAVEL_TIME, 'finish closing')

        # ── Step 9: Door closed ───────────────────────────────────────────────
        self._banner(9, TOTAL, 'module1 door → CLOSED')
        self._set_door('module1', 'CLOSED')

        # ── Step 10: Transport to module2 ─────────────────────────────────────
        self._banner(10, TOTAL, 'Transport — align gantry → module2 (arm retracted)')
        if not self._align_gantry(q0_b, q1_b):
            self.get_logger().error('Gantry align (module2) failed — aborting')
            return

        # ── Step 11: Open module2 door ────────────────────────────────────────
        self._banner(11, TOTAL, 'module2 door → OPENING')
        self.get_logger().info(
            '  Arm is retracted — geometrically avoids OPENING zone.')
        self._set_door('module2', 'OPENING')
        self._wait_door(DOOR_TRAVEL_TIME, 'finish opening')

        # ── Step 12: Door fully open ──────────────────────────────────────────
        self._banner(12, TOTAL, 'module2 door → OPEN (front clear, top blocked)')
        self._set_door('module2', 'OPEN')

        # ── Step 13: Drop payload ─────────────────────────────────────────────
        self._banner(13, TOTAL, 'Extend arm → Drop payload at module2')
        if not self._extend_arm(q0_b, q1_b, px_b, pz_b, py_b):
            self.get_logger().error('Extend (drop) failed — aborting')
            return

        # ── Step 14: Release ──────────────────────────────────────────────────
        self._banner(14, TOTAL, 'Detach + release payload')
        self._detach()
        self._release()

        # ── Step 15: Retract arm → safe position ─────────────────────────────
        self._banner(15, TOTAL, 'Retract arm → safe position (module2)')
        if not self._retract_arm(q0_b, q1_b, px_b, pz_b, py_b):
            self.get_logger().error('Retract (drop) failed — aborting')
            return

        # ── Step 16: Close module2 door ───────────────────────────────────────
        self._banner(16, TOTAL, 'module2 door → CLOSING (arm retracted = safe)')
        self._set_door('module2', 'CLOSING')
        self._wait_door(DOOR_TRAVEL_TIME, 'finish closing')

        # ── Step 17: Door closed ──────────────────────────────────────────────
        self._banner(17, TOTAL, 'module2 door → CLOSED')
        self._set_door('module2', 'CLOSED')

        # ── Step 18: Return to absolute HOME ──────────────────────────────────
        self._banner(18, TOTAL, 'Return to absolute HOME')
        if not self._go_home():
            self.get_logger().error('Final homing failed')
            return

        self.get_logger().info(
            '\n' + '='*60 +
            '\n  DOOR-AWARE PICK-AND-PLACE COMPLETE' +
            '\n' + '='*60)


def main(args=None):
    rclpy.init(args=args)
    node = DoorPnPNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
