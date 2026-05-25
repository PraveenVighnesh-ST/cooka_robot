#!/usr/bin/env python3
"""
Arm trajectory controller for the Cooka gantry-SCARA robot.

Two-phase Y motion (avoids IK singularity at y=0.160 m):
  SAFE_Y    = 0.165 m — Cartesian/joint-space handoff point
  HOME_EE_Y = 0.060 m — EE y at HOME joint config (payload clears 0.275 m frame)
  MAX_Y     = 0.560 m — fully extended (both SCARA links straight)
  Speed     = 200 mm/s along the EE Y axis

HOME joint configuration (scara_j1=145°, scara_j2=-260°, wrist_z=-115°):
  Cumulative yaw = 0° (EE points +Y).  EE_y=-0.088 m.
  250 mm-diameter payload front at y=0.037 m, safely inside the 0.275 m frame.

Motion sequence for retract_arm (collision-safe):
  Phase 1 — Cartesian -Y from y_current → SAFE_Y  (IK-safe region)
  Phase 2 — joint-space fold SAFE_Y config → HOME joint angles

Motion sequence for extend_arm:
  Phase 1 — joint-space unfold HOME → SAFE_Y IK config
  Phase 2 — Cartesian +Y from SAFE_Y → y_target

Public methods
--------------
  align_gantry(q0, q1)            move gantry to column/row, SCARA folds to HOME
  extend_arm(q0, q1, y_target)    HOME → y_target (two-phase)
  retract_arm(q0, q1, y_current)  y_current → HOME (two-phase)
  grip(depth_m)                   close gripper
  wait_for_grip(depth_m)          block until finger reaches commanded position
  release()                       open gripper
  pick(x_col, z_row, y_depth)     full ALIGN → EXTEND → GRIP → RETRACT
"""
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from cooka_description.cooka_kinematics import fk_position, ik

# ── Constants ─────────────────────────────────────────────────────────────────

ARM_JOINTS = ['h_slider', 'v_slider', 'scara_j1', 'scara_j2', 'wrist_z', 'wrist_y']

Y_SPEED   = 0.200  # m/s — EE straight-line speed along Y
SAMPLE_DT = 0.020  # s   — trajectory sample period (50 Hz)

# Cartesian/joint-space handoff: deepest y reachable by straight-line IK without
# crossing the singularity at y=0.160 m (where SCARA pivot dy→0).
SAFE_Y = 0.165  # m

# HOME joint configuration: scara_j1=145°, scara_j2=-260°, wrist_z=-115°.
# Cumulative yaw = -145+260-115 = 0° (EE points +Y).
# FK gives EE_y=-0.088 m; 250 mm payload front at 0.037 m clears 0.275 m frame.
HOME_EE_Y     = -0.088              # m — EE y at HOME config (informational)
HOME_SCARA_J1 =  np.radians( 145)  # 145°  = +2.5307 rad
HOME_SCARA_J2 =  np.radians(-260)  # -260° = -4.5379 rad
HOME_WRIST_Z  =  np.radians(-115)  # -115° = -2.0071 rad

GRIP_TOLERANCE = 0.0015  # m — finger_bottom within this of target = grip confirmed
GRIP_TIMEOUT   = 0.5     # s — wait for gripper to mechanically engage

# EE position in base_link frame is determined by:
#   px = 0.275 + q0        (gantry X offset + h_slider position)
#   pz = 0.125 - q1        (fixed Z offset  - v_slider position)
#   py = varies            (SCARA arm controls this)


class CookaArmNode(Node):

    def __init__(self):
        super().__init__('cooka_arm_node')

        self._arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')

        self._gripper_pub = self.create_publisher(
            Float64MultiArray, '/gripper_controller/commands', 10)

        # Track arm joints + gripper finger for grip confirmation
        self._joint_pos = {j: 0.0 for j in ARM_JOINTS}
        self._joint_pos['finger_bottom'] = 0.0
        self.create_subscription(
            JointState, '/joint_states', self._on_joint_states, 10)

        self.get_logger().info('Waiting for arm_controller action server...')
        self._arm_client.wait_for_server()
        self.get_logger().info('CookaArmNode ready.')

    # ── Joint state tracking ──────────────────────────────────────────────────

    def _on_joint_states(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in self._joint_pos:
                self._joint_pos[name] = pos

    def _q_now(self) -> list:
        """Return current joint vector in ARM_JOINTS order."""
        return [self._joint_pos[j] for j in ARM_JOINTS]

    # ── Gripper ───────────────────────────────────────────────────────────────

    def grip(self, depth_m: float = 0.015):
        """Close gripper. depth_m clamped to [0, 0.015] m."""
        msg = Float64MultiArray()
        msg.data = [-min(abs(depth_m), 0.015)]
        self._gripper_pub.publish(msg)
        self.get_logger().info(f'Grip  {depth_m * 1000:.1f} mm')

    def wait_for_grip(self, depth_m: float = 0.015,
                      timeout_sec: float = GRIP_TIMEOUT) -> bool:
        """
        Wait for the gripper to engage before retracting.

        Uses time.sleep (not spin_once) to avoid interfering with the
        action client's callback queue.  Returns True always — the caller
        should retract regardless of whether grip was confirmed.
        """
        time.sleep(timeout_sec)
        pos = self._joint_pos.get('finger_bottom', 0.0)
        target = -min(abs(depth_m), 0.015)
        if abs(pos - target) <= GRIP_TOLERANCE:
            self.get_logger().info(
                f'Grip confirmed  finger={pos * 1000:.1f} mm')
        else:
            self.get_logger().warning(
                f'Grip position not confirmed  '
                f'finger={pos * 1000:.1f} mm  target={target * 1000:.1f} mm')
        return True

    def release(self):
        """Fully open gripper."""
        msg = Float64MultiArray()
        msg.data = [0.0]
        self._gripper_pub.publish(msg)
        self.get_logger().info('Gripper open')

    # ── Trajectory helpers ────────────────────────────────────────────────────

    @staticmethod
    def _make_point(q: list, t_s: float) -> JointTrajectoryPoint:
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        pt.time_from_start.sec = int(t_s)
        pt.time_from_start.nanosec = int(round((t_s % 1) * 1e9))
        return pt

    def _send_traj(self, waypoints: list, timestamps: list) -> bool:
        """Send a JointTrajectory via FollowJointTrajectory action; blocks until done."""
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        traj.points = [
            self._make_point(q, t) for q, t in zip(waypoints, timestamps)
        ]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        send_fut = self._arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut)
        gh = send_fut.result()
        if not gh.accepted:
            self.get_logger().error('Trajectory goal rejected by controller')
            return False

        result_fut = gh.get_result_async()
        # 4× trajectory duration + 60 s headroom covers slow Gazebo (≤0.25× real-time)
        timeout = timestamps[-1] * 4.0 + 60.0
        rclpy.spin_until_future_complete(self, result_fut, timeout_sec=timeout)
        wrapped = result_fut.result()
        if wrapped is None:
            self.get_logger().error(
                f'Trajectory result timed out after {timeout:.1f} s '
                '(controller may still be executing)')
            return False
        ec = wrapped.result.error_code
        if ec != 0:
            self.get_logger().error(f'Trajectory failed  error_code={ec}')
        return ec == 0

    # ── Gantry alignment ─────────────────────────────────────────────────────

    def align_gantry(self, q0: float, q1: float, speed: float = 0.2,
                     px: float = None, settle_sec: float = 2.0) -> bool:
        """
        Move gantry to h_slider=q0 [m], v_slider=q1 [m] with arm at HOME.

        Three-phase trajectory:
          Phase A — gantry moves in X+Z while SCARA joints are locked.
          Phase B — SCARA folds to HOME joint angles (1.5 s).
          Dwell   — hold target for settle_sec sim-seconds so v_slider
                    fully settles against gravity before the next move.

        Parameters
        ----------
        q0         : h_slider target position [0.0, 1.25] m
        q1         : v_slider target position [-1.475, 0.0] m
        speed      : gantry joint speed in m/s
        px         : unused (kept for API compatibility)
        settle_sec : extra sim-time dwell at target (s). Default 2 s.
        """
        q_start = self._q_now()

        q_gantry_moved = list(q_start)
        q_gantry_moved[0] = q0
        q_gantry_moved[1] = q1

        q_home = list(q_gantry_moved)
        q_home[2] = HOME_SCARA_J1
        q_home[3] = HOME_SCARA_J2
        q_home[4] = HOME_WRIST_Z

        dist     = max(abs(q0 - q_start[0]), abs(q1 - q_start[1]), 1e-4)
        t_travel = dist / speed
        t_done   = t_travel + 1.5        # Phase B: SCARA fold to HOME
        t_settle = t_done + settle_sec   # Dwell: let v_slider settle

        self.get_logger().info(
            f'Align gantry  h={q0:.3f} m  v={q1:.3f} m  '
            f'(travel={t_travel:.1f}s  dwell={settle_sec:.1f}s sim)')
        return self._send_traj(
            [q_start,  q_gantry_moved, q_home,  q_home  ],
            [0.0,      t_travel,       t_done,  t_settle ])

    # ── Joint-space HOME fold / unfold ────────────────────────────────────────

    def _fold_to_home(self) -> bool:
        """Joint-space motion: current config → HOME angles (1 rad/s, min 0.5 s)."""
        q_start = self._q_now()
        q_home = list(q_start)
        q_home[2] = HOME_SCARA_J1
        q_home[3] = HOME_SCARA_J2
        q_home[4] = HOME_WRIST_Z
        max_delta = max(abs(q_home[i] - q_start[i]) for i in [2, 3, 4])
        duration = max(max_delta / 1.0, 0.5)
        self.get_logger().info(f'Fold → HOME  ({duration:.1f} s)')
        return self._send_traj([q_start, q_home], [0.0, duration])

    def _unfold_from_home(self, q0: float, q1: float,
                          px: float = None,
                          yaw_ee: float = 0.0,
                          wrist_z_bias: float = 0.0) -> bool:
        """Joint-space motion: HOME angles → IK config at SAFE_Y (1 rad/s, min 0.5 s)."""
        if px is None:
            px = 0.275 + q0
        pz = 0.125 - q1
        q_safe = ik(px, SAFE_Y, pz, yaw_ee=yaw_ee, elbow='down')
        if q_safe is None:
            self.get_logger().error(
                f'_unfold_from_home: IK failed  px={px:.3f}  SAFE_Y={SAFE_Y}  pz={pz:.3f}')
            return False
        q_safe[0], q_safe[1] = q0, q1
        q_safe[4] += wrist_z_bias
        q_start = self._q_now()
        max_delta = max(abs(q_safe[i] - q_start[i]) for i in [2, 3, 4])
        duration = max(max_delta / 1.0, 0.5)
        self.get_logger().info(f'Unfold HOME → SAFE_Y  ({duration:.1f} s)')
        return self._send_traj([q_start, q_safe], [0.0, duration])

    # ── Y-axis straight-line trajectory ──────────────────────────────────────

    def _y_waypoints(self,
                     q0: float, q1: float,
                     y_start: float, y_end: float,
                     yaw_ee: float = 0.0,
                     px: float = None,
                     wrist_z_bias: float = 0.0):
        """
        Compute IK waypoints for EE straight-line motion y_start → y_end.

        Gantry stays fixed at (q0, q1). px is the actual target EE x in
        base_link — defaults to 0.275+q0 but should be set explicitly when
        the h_slider is at its limit and SCARA contributes extra X reach.
        wrist_z_bias is added to the IK-computed wrist_z angle at every step
        (use to correct residual EE rotation when SCARA arm angles in X).
        Returns (waypoints, timestamps) or (None, None) on IK failure.
        """
        n = max(2, int(abs(y_end - y_start) / (Y_SPEED * SAMPLE_DT)) + 1)
        y_arr = np.linspace(y_start, y_end, n)
        if px is None:
            px = 0.275 + q0
        pz = 0.125 - q1

        waypoints, timestamps = [], []
        t = 0.0
        for i, py in enumerate(y_arr):
            q = ik(px, float(py), pz, yaw_ee=yaw_ee, elbow='down')
            if q is None:
                self.get_logger().error(
                    f'IK failed at y={py:.4f} m (step {i + 1}/{n})')
                return None, None
            q[0], q[1] = q0, q1   # lock gantry at commanded position
            q[4] += wrist_z_bias  # direct wrist_z correction after IK
            waypoints.append(q)
            timestamps.append(t)
            if i < n - 1:
                t += abs(float(y_arr[i + 1]) - float(py)) / Y_SPEED

        return waypoints, timestamps

    def extend_arm(self,
                   q0: float, q1: float,
                   y_target: float,
                   yaw_ee: float = 0.0,
                   px: float = None,
                   wrist_z_bias: float = 0.0) -> bool:
        """
        Extend arm from HOME to y_target (two-phase, avoids IK singularity).

        Phase 1 — joint-space: HOME angles → IK config at SAFE_Y
        Phase 2 — Cartesian:   SAFE_Y → y_target at Y_SPEED

        Parameters
        ----------
        q0, q1        : gantry joint positions
        y_target      : target EE y in base_link frame [SAFE_Y, 0.56] m
        yaw_ee        : desired EE yaw about Z (rad), default 0 = arm points +Y
        px            : actual target EE x [m]. Defaults to 0.275+q0.
        wrist_z_bias  : angular offset added to IK wrist_z result [rad]
        """
        self.get_logger().info(
            f'Extend  y: HOME({HOME_EE_Y:.3f}) → {y_target:.3f} m  '
            f'({Y_SPEED * 1000:.0f} mm/s)')
        if not self._unfold_from_home(q0, q1, px=px, yaw_ee=yaw_ee,
                                      wrist_z_bias=wrist_z_bias):
            return False
        wps, ts = self._y_waypoints(
            q0, q1, SAFE_Y, y_target, yaw_ee, px=px, wrist_z_bias=wrist_z_bias)
        if wps is None:
            return False
        return self._send_traj(wps, ts)

    def retract_arm(self,
                    q0: float, q1: float,
                    y_current: float,
                    yaw_ee: float = 0.0,
                    px: float = None,
                    wrist_z_bias: float = 0.0) -> bool:
        """
        Retract arm from y_current to HOME (two-phase, avoids IK singularity).

        Phase 1 — Cartesian:   y_current → SAFE_Y at Y_SPEED  (skipped if already ≤ SAFE_Y)
        Phase 2 — joint-space: IK config at SAFE_Y → HOME angles

        Parameters
        ----------
        q0, q1        : gantry joint positions
        y_current     : current EE y in base_link frame
        yaw_ee        : must match the yaw used during extend_arm
        px            : actual target EE x [m]. Defaults to 0.275+q0.
        wrist_z_bias  : must match the bias used during extend_arm
        """
        self.get_logger().info(
            f'Retract  y: {y_current:.3f} → HOME({HOME_EE_Y:.3f}) m  '
            f'({Y_SPEED * 1000:.0f} mm/s)')
        if y_current > SAFE_Y + 0.005:
            wps, ts = self._y_waypoints(
                q0, q1, y_current, SAFE_Y, yaw_ee, px=px, wrist_z_bias=wrist_z_bias)
            if wps is None or not self._send_traj(wps, ts):
                return False
        return self._fold_to_home()

    # ── Full pick sequence ────────────────────────────────────────────────────

    def pick(self,
             x_col: float,
             z_row: float,
             y_depth: float,
             yaw_ee: float = 0.0) -> bool:
        """
        Full pick sequence: ALIGN → EXTEND → GRIP → WAIT → RETRACT.

        Parameters
        ----------
        x_col  : h_slider position [m] for the target module column
        z_row  : v_slider position [m] for the target module row (negative)
        y_depth: SCARA reach [m] to the module face, e.g. 0.45 m
        yaw_ee : EE yaw (rad), default 0 = gripper faces +Y

        Returns True on success, False if any step fails.
        """
        self.get_logger().info(
            f'=== PICK  col={x_col:.3f} m  row={z_row:.3f} m  '
            f'depth={y_depth:.3f} m (HOME_EE_Y={HOME_EE_Y:.3f} m) ===')

        self.release()

        if not self.align_gantry(x_col, z_row):
            self.get_logger().error('PICK aborted: gantry align failed')
            return False

        if not self.extend_arm(x_col, z_row, y_depth, yaw_ee):
            self.get_logger().error('PICK aborted: arm extend failed')
            return False

        self.grip()
        # Block until finger reaches commanded position before retracting
        self.wait_for_grip()

        if not self.retract_arm(x_col, z_row, y_depth, yaw_ee):
            self.get_logger().error('PICK aborted: arm retract failed')
            return False

        self.get_logger().info('=== PICK complete ===')
        return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = CookaArmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
