"""
Door-aware pick-and-place node for the cooka smart kitchen robot.

Picks payload from module1, transports it, drops at module2.

Key concept — AWAY position
---------------------------
When a door is OPENING or CLOSING, the gantry carriage must not be
within the door's sweep envelope.  Problem with the naive approach:
aligning gantry to pick A (h=0.025, v=-0.400) puts the carriage at
Z≈0.475 m, which is inside the OPENING zone (Z=0.425–1.825 m).
The door would physically hit the carriage as it slides upward.

Fix: before commanding OPENING, move the gantry to a safe AWAY
position that clears the door in both X and Z:
  AWAY_H = 0.700 m  →  carriage X = 0.700 m
                         outside module1 X range (0.050–0.600 m) ✓
                         outside module2 X range (1.300–1.850 m) ✓
  AWAY_V = 0.000 m  →  carriage Z ≈ 0.075 m
                         below all OPENING zones (Z starts at 0.425 m) ✓

Door timing (total 5 s per open or close)
-----------------------------------------
  t=0   Command OPENING / CLOSING  → zone activates in planning scene
  t=0→2 Robot moves gantry to AWAY position (~2 s movement)
  t=2→5 Wait remaining ~3 s for door to finish travelling
  t=5   Command OPEN / CLOSED
  t=5+  Align gantry to pick/drop position, then extend arm

Full sequence
-------------
INIT
  Both modules → CLOSED
  Robot → absolute HOME

PICK (module1)
  1. Move gantry to AWAY          ← clears module1 door envelope
  2. module1 → OPENING            ← zone activates
  3. Wait remaining door-open time (5 s total − movement time)
  4. module1 → OPEN
  5. Align gantry to pick A (h=0.025, v=-0.400)
  6. Extend arm → pick payload
  7. Grip + attach (DetachableJoint)
  8. Retract arm

CLOSE module1
  9. module1 → CLOSING (= OPENING zone)
 10. Move gantry to AWAY          ← clears door envelope for closing
 11. Wait remaining door-close time
 12. module1 → CLOSED

TRANSPORT
 13. Gantry already at AWAY; move to module2 AWAY alignment

DROP (module2)
 14. module2 → OPENING
 15. Wait remaining door-open time
 16. module2 → OPEN
 17. Align gantry to drop B (h=1.250, v=-0.975)
 18. Extend arm → drop payload
 19. Detach + release
 20. Retract arm

CLOSE module2
 21. module2 → CLOSING
 22. Move gantry to AWAY
 23. Wait remaining door-close time
 24. module2 → CLOSED

COMPLETE
 25. Return to absolute HOME
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

# ── Robot motion constants (identical to pnp_node) ────────────────────────────

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

# ── Door / safety constants ───────────────────────────────────────────────────

# Gantry position that is clear of ALL module door envelopes.
# h=0.700 → carriage X outside module1 (0.05–0.60 m) and module2 (1.30–1.85 m).
# v=0.000 → carriage Z ≈ 0.075 m, below every OPENING zone (start Z=0.425 m).
AWAY_H = 0.700   # m
AWAY_V = 0.000   # m

# Total simulated door travel time (open or close).
# Robot moves to AWAY in the first ~2 s; waits the remainder.
DOOR_TRAVEL_TIME = 5.0   # s

INIT_DISPLAY_WAIT = 2.0  # s — pause so RViz shows initial CLOSED zones


class DoorPnPNode(Node):
    """Door-aware pick-and-place: module1 → module2 with gantry safe-retract."""

    def __init__(self):
        super().__init__('door_pnp_node')

        self._arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')

        self._gripper_pub = self.create_publisher(
            Float64MultiArray, '/gripper_controller/commands', 10)

        self._attach_pub = self.create_publisher(
            Empty, '/cooka/detachable_joint/attach', 10)
        self._detach_pub = self.create_publisher(
            Empty, '/cooka/detachable_joint/detach', 10)

        self._door_pub = self.create_publisher(String, '/door_command', 10)

        self._joint_pos = {j: 0.0 for j in ARM_JOINTS}
        self._joint_pos['finger_bottom'] = 0.0
        self._js_received = False
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)

        self.get_logger().info(
            'DoorPnP: waiting for arm_controller...')
        self._arm_client.wait_for_server()
        self.get_logger().info(
            'DoorPnP: arm_controller ready — waiting for joint states...')
        self._wait_js()

        self._run()

    # ── Joint state ───────────────────────────────────────────────────────────

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
        msg = String()
        msg.data = f'{module}:{state}'
        self._door_pub.publish(msg)
        self.get_logger().info(f'  DOOR [{module}] → {state}')

    def _wait_spin(self, seconds: float, label: str):
        """Spin the node for `seconds` seconds (callbacks stay alive)."""
        if seconds <= 0.0:
            return
        self.get_logger().info(f'  Waiting {seconds:.1f}s {label}')
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    # ── Door + safe-move sequences ────────────────────────────────────────────

    def _open_door(self, module: str, q0_target: float, q1_target: float):
        """
        Safe door-open sequence:
          1. Command OPENING
          2. Move gantry to AWAY position  (~2 s — clears door envelope)
          3. Wait remaining door-travel time
          4. Command OPEN
          5. Align gantry to pick/drop position  (ready to extend arm)
        """
        self._banner(f'{module} door OPENING — moving gantry to AWAY first')
        t_start = time.time()

        self._set_door(module, 'OPENING')

        # Move gantry clear of the door envelope while door begins to travel
        self.get_logger().info(
            f'  Moving gantry to AWAY (h={AWAY_H}, v={AWAY_V}) '
            f'— clears door X and Z ranges')
        if not self._align_gantry(AWAY_H, AWAY_V):
            self.get_logger().error('AWAY move failed — aborting')
            return False

        # Wait out any remaining door travel time
        elapsed = time.time() - t_start
        self._wait_spin(
            max(0.0, DOOR_TRAVEL_TIME - elapsed),
            'for door to finish opening')

        self._set_door(module, 'OPEN')

        # Now safe to align to pick/drop position
        self.get_logger().info(
            f'  Door open — aligning gantry to target '
            f'(h={q0_target:.3f}, v={q1_target:.3f})')
        if not self._align_gantry(q0_target, q1_target):
            self.get_logger().error('Pick/drop gantry align failed — aborting')
            return False

        return True

    def _close_door(self, module: str):
        """
        Safe door-close sequence:
          1. Command CLOSING (= OPENING zone)
          2. Move gantry to AWAY position  (~2 s — clears door envelope)
          3. Wait remaining door-travel time
          4. Command CLOSED
        """
        self._banner(f'{module} door CLOSING — moving gantry to AWAY first')
        t_start = time.time()

        self._set_door(module, 'CLOSING')

        self.get_logger().info(
            f'  Moving gantry to AWAY (h={AWAY_H}, v={AWAY_V}) '
            f'— clears door X and Z ranges for closing')
        if not self._align_gantry(AWAY_H, AWAY_V):
            self.get_logger().error('AWAY move failed during close — aborting')
            return False

        elapsed = time.time() - t_start
        self._wait_spin(
            max(0.0, DOOR_TRAVEL_TIME - elapsed),
            'for door to finish closing')

        self._set_door(module, 'CLOSED')
        return True

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

    # ── Trajectory helpers ────────────────────────────────────────────────────

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
            self.get_logger().error('Trajectory rejected')
            return False

        result_fut = gh.get_result_async()
        timeout = timestamps[-1] * 4.0 + 60.0
        rclpy.spin_until_future_complete(self, result_fut, timeout_sec=timeout)
        wrapped = result_fut.result()
        if wrapped is None:
            self.get_logger().error('Trajectory timed out')
            return False
        return wrapped.result.error_code == 0

    # ── Motion primitives ─────────────────────────────────────────────────────

    def _go_home(self) -> bool:
        q_start = self._q_now()
        deltas = [abs(PNP_HOME[i] - q_start[i]) for i in range(6)]
        t_move = max(max(deltas) / HOME_SPEED, 2.0)
        t_settle = t_move + HOME_SETTLE
        self.get_logger().info(
            f'  Homing ({t_move:.1f}s move + {HOME_SETTLE}s settle)')
        return self._send_traj(
            [q_start, PNP_HOME, PNP_HOME],
            [0.0,     t_move,   t_settle])

    def _align_gantry(self, q0: float, q1: float) -> bool:
        """Move h_slider → q0, v_slider → q1 with SCARA locked in HOME angles."""
        q_start = self._q_now()
        q_target = list(q_start)
        q_target[0] = q0
        q_target[1] = q1
        q_target[2:] = PNP_HOME[2:]
        dist = max(abs(q0 - q_start[0]), abs(q1 - q_start[1]), 1e-4)
        t_move = max(dist / GANTRY_SPEED, 1.0)
        self.get_logger().info(
            f'  Gantry → h={q0:.3f}m  v={q1:.3f}m  ({t_move:.1f}s)')
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
        self.get_logger().info(
            f'  Unfold HOME → SAFE_Y={SAFE_Y}m ({dur:.1f}s)')
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

    # ── Logging ───────────────────────────────────────────────────────────────

    def _banner(self, label: str):
        self.get_logger().info(f'\n{"─"*60}\n  {label}\n{"─"*60}')

    # ── Main sequence ─────────────────────────────────────────────────────────

    def _run(self):
        px_a, py_a, pz_a = PICK_A
        px_b, py_b, pz_b = DROP_B
        q0_a, q1_a = self._gantry_for(px_a, pz_a)
        q0_b, q1_b = self._gantry_for(px_b, pz_b)

        self.get_logger().info(
            f'Pick A : EE=({px_a},{py_a},{pz_a})  '
            f'gantry h={q0_a:.3f}m  v={q1_a:.3f}m')
        self.get_logger().info(
            f'Drop B : EE=({px_b},{py_b},{pz_b})  '
            f'gantry h={q0_b:.3f}m  v={q1_b:.3f}m')
        self.get_logger().info(
            f'AWAY   : h={AWAY_H}m  v={AWAY_V}m  '
            f'(carriage clears all door envelopes)')

        # ── INIT ─────────────────────────────────────────────────────────────
        self._banner('INIT — set initial door states')
        self._set_door('module1', 'CLOSED')
        self._set_door('module2', 'CLOSED')
        self._wait_spin(INIT_DISPLAY_WAIT, '(RViz shows initial CLOSED zones)')

        # Release payload auto-attached on spawn
        self._release()
        self._detach()

        # ── Home ─────────────────────────────────────────────────────────────
        self._banner('Home robot (arm retracted)')
        if not self._go_home():
            self.get_logger().error('Homing failed — aborting')
            return

        # ═══════════════════════════════════════════════════════════════════
        #  PICK  (module1)
        # ═══════════════════════════════════════════════════════════════════

        # Open module1 door with safe gantry retract
        # → moves to AWAY, waits, then aligns to pick A
        if not self._open_door('module1', q0_a, q1_a):
            self.get_logger().error('Door open (module1) failed — aborting')
            return

        # Extend arm into module1 and pick
        self._banner('Extend arm → pick payload at module1')
        if not self._extend_arm(q0_a, q1_a, px_a, pz_a, py_a):
            self.get_logger().error('Extend (pick) failed — aborting')
            return

        self._banner('Grip + attach payload')
        self._grip()
        self._attach()

        self._banner('Retract arm from module1')
        if not self._retract_arm(q0_a, q1_a, px_a, pz_a, py_a):
            self.get_logger().error('Retract (pick) failed — aborting')
            return

        # Close module1 door with safe gantry retract
        # → moves to AWAY while door closes
        if not self._close_door('module1'):
            self.get_logger().error('Door close (module1) failed — aborting')
            return

        # ═══════════════════════════════════════════════════════════════════
        #  DROP  (module2)
        # ═══════════════════════════════════════════════════════════════════
        # Gantry is already at AWAY after closing module1.
        # _open_door will move from AWAY → AWAY (no-op gantry move, fast),
        # wait for door, then align to drop B.

        if not self._open_door('module2', q0_b, q1_b):
            self.get_logger().error('Door open (module2) failed — aborting')
            return

        self._banner('Extend arm → drop payload at module2')
        if not self._extend_arm(q0_b, q1_b, px_b, pz_b, py_b):
            self.get_logger().error('Extend (drop) failed — aborting')
            return

        self._banner('Detach + release payload')
        self._detach()
        self._release()

        self._banner('Retract arm from module2')
        if not self._retract_arm(q0_b, q1_b, px_b, pz_b, py_b):
            self.get_logger().error('Retract (drop) failed — aborting')
            return

        if not self._close_door('module2'):
            self.get_logger().error('Door close (module2) failed — aborting')
            return

        # ═══════════════════════════════════════════════════════════════════
        #  COMPLETE
        # ═══════════════════════════════════════════════════════════════════
        self._banner('Return to absolute HOME')
        self._go_home()

        self.get_logger().info(
            '\n' + '═' * 60 +
            '\n  DOOR-AWARE PICK-AND-PLACE COMPLETE' +
            '\n' + '═' * 60)


def main(args=None):
    rclpy.init(args=args)
    node = DoorPnPNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
