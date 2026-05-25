"""
Pick-and-place node for the cooka gantry-SCARA robot.

Full sequence executed once on startup:
  1. Read current joint positions
  2. Home robot to PNP_HOME (h=0 m, v=0 m, j1=120°, j2=-210°, wz=-90°, wy=0°)
  3. Align gantry: h_slider + v_slider to closest point for pick A
  4. Extend SCARA to pick A  (EE → 0.3, 0.319, 0.525 m)
  5. Grip payload (close finger to 15 mm)
  6. Retract SCARA to home angles (payload clears modules)
  7. Align gantry: h_slider + v_slider to closest point for drop B
  8. Extend SCARA to drop B  (EE → 1.6, 0.319, 1.1 m)
  9. Release payload (open finger)
 10. Retract SCARA to home angles
"""

import os
import subprocess
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from cooka_description.cooka_kinematics import ik, JOINT_LIMITS

# ── Constants ──────────────────────────────────────────────────────────────────

ARM_JOINTS = ['h_slider', 'v_slider', 'scara_j1', 'scara_j2', 'wrist_z', 'wrist_y']

# Home joint config for PnP operations (angles in radians for revolutes)
PNP_HOME = [
    0.0,               # h_slider  [m]
    0.0,               # v_slider  [m]
    np.radians(120),   # scara_j1  [rad]
    np.radians(-240),  # scara_j2  [rad]
    -2.094,            # wrist_z   [rad]  ≈ -119.97° (limit is -2.094395)
    0.0,               # wrist_y   [rad]
]

# EE target positions in base_link frame [x, y, z] metres
PICK_A = (0.3,  0.319, 0.525)
DROP_B  = (1.6, 0.319, 1.1)

# Motion parameters
Y_SPEED      = 0.200   # m/s  — straight-line EE speed along Y during extension
SAMPLE_DT    = 0.020   # s    — trajectory sample period (50 Hz)
SAFE_Y       = 0.165   # m    — unfold/fold handoff (avoids singularity at y≈0.16 m)
GANTRY_SPEED = 0.20    # m/s  — h/v slider speed for alignment moves
HOME_SPEED   = 1.0     # rad/s (and m/s) — joint speed during homing
HOME_SETTLE  = 2.0     # s    — dwell at home for v_slider to settle against gravity


class PnPNode(Node):
    """One-shot pick-and-place node: executes full sequence then exits."""

    def __init__(self):
        super().__init__('pnp_node')

        self._arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')

        self._gripper_pub = self.create_publisher(
            Float64MultiArray, '/gripper_controller/commands', 10)

        # DetachableJoint control — bridged via ros_gz_bridge (ROS_TO_GZ, gz.msgs.Empty)
        self._attach_pub = self.create_publisher(
            Empty, '/cooka/detachable_joint/attach', 10)
        self._detach_pub = self.create_publisher(
            Empty, '/cooka/detachable_joint/detach', 10)

        self._joint_pos = {j: 0.0 for j in ARM_JOINTS}
        self._joint_pos['finger_bottom'] = 0.0
        self._joint_states_received = False
        self.create_subscription(
            JointState, '/joint_states', self._on_joint_states, 10)

        self.get_logger().info('PnP node ready — waiting for arm_controller action server...')
        self._arm_client.wait_for_server()
        self.get_logger().info('arm_controller ready — waiting for joint states...')
        self._wait_for_joint_states()
        self.get_logger().info(f'Current joints: {[round(v, 4) for v in self._q_now()]}')

        self._run()

    # ── Joint state tracking ───────────────────────────────────────────────────

    def _on_joint_states(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in self._joint_pos:
                self._joint_pos[name] = pos
        self._joint_states_received = True

    def _wait_for_joint_states(self):
        while not self._joint_states_received:
            rclpy.spin_once(self, timeout_sec=0.5)

    def _q_now(self) -> list:
        """Current joint vector in ARM_JOINTS order."""
        return [self._joint_pos[j] for j in ARM_JOINTS]

    # ── Gripper ────────────────────────────────────────────────────────────────

    def _grip(self, depth_m: float = 0.015):
        """Close gripper to depth_m (clamped to 15 mm max)."""
        cmd = Float64MultiArray()
        cmd.data = [-min(abs(depth_m), 0.015)]
        self._gripper_pub.publish(cmd)
        self.get_logger().info(f'Grip  target={depth_m * 1000:.1f} mm — waiting for engagement...')
        time.sleep(0.8)

    def _release(self):
        """Fully open gripper."""
        cmd = Float64MultiArray()
        cmd.data = [0.0]
        self._gripper_pub.publish(cmd)
        self.get_logger().info('Gripper open')
        time.sleep(0.3)

    def _attach_payload(self):
        """Rigidly join active_payload to y_rotation_1 via DetachableJoint plugin."""
        self._attach_pub.publish(Empty())
        self.get_logger().info('DetachableJoint: attach')
        time.sleep(0.2)

    def _detach_payload(self):
        """Release active_payload from y_rotation_1 via DetachableJoint plugin."""
        for _ in range(3):
            self._detach_pub.publish(Empty())
            time.sleep(0.15)
        self.get_logger().info('DetachableJoint: detach')

    def _spawn_payload_model(self, x: float, y: float, z: float) -> bool:
        """
        Spawn the payload at (x, y, z) in the running Gazebo simulation.

        Called from within pnp_node after the EE has reached SAFE_Y so the
        payload appears close to the gripper.  Immediately after this returns,
        _detach_payload() must be called before Gazebo's physics step processes
        the new entity — that window prevents the DetachableJoint plugin from
        auto-attaching.
        """
        self.get_logger().info(
            f'Spawning payload at ({x:.3f}, {y:.3f}, {z:.3f})...')
        env = {**os.environ, 'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'}
        cmd = [
            'ros2', 'run', 'pnp_operation', 'spawn_payload',
            '--x', str(x), '--y', str(y), '--z', str(z),
            '--name', 'active_payload',
        ]
        result = subprocess.run(cmd, env=env, timeout=15)
        if result.returncode == 0:
            self.get_logger().info('Payload spawned successfully')
            return True
        self.get_logger().error('Payload spawn failed')
        return False

    # ── Trajectory helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _pt(q: list, t_s: float) -> JointTrajectoryPoint:
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        pt.time_from_start.sec = int(t_s)
        pt.time_from_start.nanosec = int(round((t_s % 1) * 1e9))
        return pt

    def _send_traj(self, waypoints: list, timestamps: list) -> bool:
        """Send JointTrajectory via FollowJointTrajectory action; blocks until done."""
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        traj.points = [self._pt(q, t) for q, t in zip(waypoints, timestamps)]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        send_fut = self._arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut)
        gh = send_fut.result()
        if not gh.accepted:
            self.get_logger().error('Trajectory goal rejected by controller')
            return False

        result_fut = gh.get_result_async()
        timeout = timestamps[-1] * 4.0 + 60.0
        rclpy.spin_until_future_complete(self, result_fut, timeout_sec=timeout)
        wrapped = result_fut.result()
        if wrapped is None:
            self.get_logger().error(f'Trajectory timed out after {timeout:.0f} s')
            return False
        ec = wrapped.result.error_code
        if ec != 0:
            self.get_logger().error(f'Trajectory failed  error_code={ec}')
        return ec == 0

    # ── Motion primitives ──────────────────────────────────────────────────────

    def _go_home(self) -> bool:
        """Joint-space motion from current config to PNP_HOME, then settle."""
        q_start = self._q_now()
        deltas = [abs(PNP_HOME[i] - q_start[i]) for i in range(6)]
        t_move = max(max(deltas) / HOME_SPEED, 2.0)
        t_settle = t_move + HOME_SETTLE
        self.get_logger().info(
            f'Homing → {[round(np.degrees(v) if i > 1 else v, 2) for i, v in enumerate(PNP_HOME)]}'
            f'  ({t_move:.1f} s move + {HOME_SETTLE:.1f} s settle)')
        return self._send_traj(
            [q_start, PNP_HOME,  PNP_HOME],
            [0.0,     t_move,    t_settle])

    def _align_gantry(self, q0: float, q1: float) -> bool:
        """
        Move h_slider to q0 and v_slider to q1 while SCARA stays at home angles.
        Phase-1 of every pick/place: position the gantry column and row first.
        """
        q_start = self._q_now()
        q_target = list(q_start)
        q_target[0] = q0
        q_target[1] = q1
        q_target[2] = PNP_HOME[2]
        q_target[3] = PNP_HOME[3]
        q_target[4] = PNP_HOME[4]
        q_target[5] = PNP_HOME[5]

        dist = max(abs(q0 - q_start[0]), abs(q1 - q_start[1]), 1e-4)
        t_move = max(dist / GANTRY_SPEED, 1.0)
        t_settle = t_move + 1.5
        self.get_logger().info(
            f'Align gantry  h={q0:.3f} m  v={q1:.3f} m  ({t_move:.1f} s)')
        return self._send_traj(
            [q_start,  q_target,  q_target],
            [0.0,      t_move,    t_settle])

    def _fold_scara(self, q0: float, q1: float) -> bool:
        """Joint-space fold: SCARA joints → PNP_HOME angles (gantry locked at q0, q1)."""
        q_start = self._q_now()
        q_home = list(q_start)
        q_home[0], q_home[1] = q0, q1
        q_home[2] = PNP_HOME[2]
        q_home[3] = PNP_HOME[3]
        q_home[4] = PNP_HOME[4]
        q_home[5] = PNP_HOME[5]
        max_delta = max(abs(q_home[i] - q_start[i]) for i in [2, 3, 4])
        duration = max(max_delta / 1.0, 0.5)
        self.get_logger().info(f'Fold SCARA → home angles  ({duration:.1f} s)')
        return self._send_traj([q_start, q_home], [0.0, duration])

    def _unfold_to_safe_y(self, q0: float, q1: float,
                           px: float, pz: float, yaw_ee: float = 0.0) -> bool:
        """Joint-space unfold: PNP_HOME angles → IK config at SAFE_Y (handoff point)."""
        q_safe = ik(px, SAFE_Y, pz, yaw_ee=yaw_ee, elbow='down')
        if q_safe is None:
            self.get_logger().error(
                f'IK failed at SAFE_Y  px={px:.3f}  pz={pz:.3f}')
            return False
        q_safe[0], q_safe[1] = q0, q1  # lock gantry at aligned position

        q_start = self._q_now()
        max_delta = max(abs(q_safe[i] - q_start[i]) for i in [2, 3, 4])
        duration = max(max_delta / 1.0, 0.5)
        self.get_logger().info(f'Unfold HOME → SAFE_Y={SAFE_Y:.3f} m  ({duration:.1f} s)')
        return self._send_traj([q_start, q_safe], [0.0, duration])

    def _y_cartesian(self, q0: float, q1: float, px: float, pz: float,
                     y_start: float, y_end: float, yaw_ee: float = 0.0) -> bool:
        """IK-sampled straight-line trajectory along Y at Y_SPEED m/s."""
        n = max(2, int(abs(y_end - y_start) / (Y_SPEED * SAMPLE_DT)) + 1)
        y_arr = np.linspace(y_start, y_end, n)

        waypoints, timestamps = [], []
        t = 0.0
        for i, py in enumerate(y_arr):
            q = ik(px, float(py), pz, yaw_ee=yaw_ee, elbow='down')
            if q is None:
                self.get_logger().error(f'IK failed at y={py:.4f} m (step {i + 1}/{n})')
                return False
            q[0], q[1] = q0, q1
            waypoints.append(q)
            timestamps.append(t)
            if i < n - 1:
                t += abs(float(y_arr[i + 1]) - float(py)) / Y_SPEED

        dir_str = 'extend' if y_end > y_start else 'retract'
        self.get_logger().info(
            f'Cartesian Y {dir_str}  {y_start:.3f} → {y_end:.3f} m  '
            f'({n} pts, {t:.1f} s)')
        return self._send_traj(waypoints, timestamps)

    def _extend_arm(self, q0: float, q1: float, px: float, pz: float,
                    y_target: float, yaw_ee: float = 0.0) -> bool:
        """
        Two-phase extend: HOME angles → SAFE_Y (joint-space), then SAFE_Y → y_target (Cartesian).
        Avoids the IK singularity at y ≈ 0.160 m.
        """
        self.get_logger().info(f'Extend arm  y: HOME → {y_target:.3f} m')
        if not self._unfold_to_safe_y(q0, q1, px, pz, yaw_ee):
            return False
        return self._y_cartesian(q0, q1, px, pz, SAFE_Y, y_target, yaw_ee)

    def _retract_arm(self, q0: float, q1: float, px: float, pz: float,
                     y_current: float, yaw_ee: float = 0.0) -> bool:
        """
        Two-phase retract: y_current → SAFE_Y (Cartesian), then SAFE_Y → HOME (joint-space).
        """
        self.get_logger().info(f'Retract arm  y: {y_current:.3f} m → HOME')
        if y_current > SAFE_Y + 0.005:
            if not self._y_cartesian(q0, q1, px, pz, y_current, SAFE_Y, yaw_ee):
                return False
        return self._fold_scara(q0, q1)

    # ── Gantry coordinate helpers ──────────────────────────────────────────────

    @staticmethod
    def _gantry_for(px: float, pz: float) -> tuple:
        """
        Compute h_slider (q0) and v_slider (q1) that position the SCARA base
        as close as possible to the EE target (px, pz).

        h_slider places base at X = 0.275 + q0 → q0 = px - 0.275, clamped.
        v_slider places base at Z = 0.125 - q1 → q1 = 0.125 - pz, clamped.
        """
        q0 = float(np.clip(px - 0.275, JOINT_LIMITS[0, 0], JOINT_LIMITS[0, 1]))
        q1 = float(np.clip(0.125 - pz, JOINT_LIMITS[1, 0], JOINT_LIMITS[1, 1]))
        return q0, q1

    # ── Main pick-and-place sequence ───────────────────────────────────────────

    def _run(self):
        px_a, py_a, pz_a = PICK_A
        px_b, py_b, pz_b = DROP_B

        q0_a, q1_a = self._gantry_for(px_a, pz_a)
        q0_b, q1_b = self._gantry_for(px_b, pz_b)

        self.get_logger().info(
            f'Pick A  EE=({px_a}, {py_a}, {pz_a})  '
            f'gantry h={q0_a:.3f} m  v={q1_a:.3f} m')
        self.get_logger().info(
            f'Drop B  EE=({px_b}, {py_b}, {pz_b})  '
            f'gantry h={q0_b:.3f} m  v={q1_b:.3f} m')

        # Detach payload (spawned by launch file at pick A, auto-attached by
        # DetachableJoint plugin) so it stays free at its spawn position while
        # the robot homes and aligns.  Gravity=false in the payload SDF keeps
        # it floating at (px_a, py_a, pz_a) until we explicitly attach.
        self._release()
        self._detach_payload()

        # ── 1. Home ────────────────────────────────────────────────────────────
        self.get_logger().info('=== Step 1/10: Home ===')
        if not self._go_home():
            self.get_logger().error('Homing failed — aborting')
            return

        # ── 2. Align gantry to pick column / row ───────────────────────────────
        self.get_logger().info('=== Step 2/10: Align gantry → pick A ===')
        if not self._align_gantry(q0_a, q1_a):
            self.get_logger().error('Gantry align (pick) failed — aborting')
            return

        # ── 3. Extend SCARA to pick A ──────────────────────────────────────────
        self.get_logger().info('=== Step 3/10: Extend arm → pick A ===')
        if not self._extend_arm(q0_a, q1_a, px_a, pz_a, py_a):
            self.get_logger().error('Arm extend (pick) failed — aborting')
            return

        # ── 4. Grip and attach ─────────────────────────────────────────────────
        self.get_logger().info('=== Step 4/10: Grip payload ===')
        self._grip()
        self._attach_payload()   # rigidly connect payload to EE via DetachableJoint

        # ── 5. Retract SCARA to home angles (payload clears module face) ────────
        self.get_logger().info('=== Step 5/10: Retract arm after pick ===')
        if not self._retract_arm(q0_a, q1_a, px_a, pz_a, py_a):
            self.get_logger().error('Arm retract (pick) failed — aborting')
            return

        # ── 6. Align gantry to drop column / row ───────────────────────────────
        self.get_logger().info('=== Step 6/10: Align gantry → drop B ===')
        if not self._align_gantry(q0_b, q1_b):
            self.get_logger().error('Gantry align (drop) failed — aborting')
            return

        # ── 7. Extend SCARA to drop B ──────────────────────────────────────────
        self.get_logger().info('=== Step 7/10: Extend arm → drop B ===')
        if not self._extend_arm(q0_b, q1_b, px_b, pz_b, py_b):
            self.get_logger().error('Arm extend (drop) failed — aborting')
            return

        # ── 8. Release payload ─────────────────────────────────────────────────
        self.get_logger().info('=== Step 8/10: Release payload ===')
        self._detach_payload()   # release rigid joint first so payload can fall free
        self._release()          # then open gripper fingers

        # ── 9. Retract SCARA to home angles ────────────────────────────────────
        self.get_logger().info('=== Step 9/10: Retract arm after drop ===')
        if not self._retract_arm(q0_b, q1_b, px_b, pz_b, py_b):
            self.get_logger().error('Arm retract (drop) failed')
            return

        self.get_logger().info('=== Step 10/10: Pick-and-place COMPLETE ===')


def main(args=None):
    rclpy.init(args=args)
    node = PnPNode()
    node.destroy_node()
    rclpy.shutdown()
