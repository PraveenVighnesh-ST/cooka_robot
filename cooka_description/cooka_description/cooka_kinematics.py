#!/usr/bin/env python3
"""
Kinematics for the Cooka gantry-SCARA robot.

Joint ordering (q index):
  0  h_slider  prismatic  +X    [0,      1.25 ] m
  1  v_slider  prismatic  -Z    [-1.475, 0    ] m
  2  scara_j1  revolute   -Z    [-1.047, 4.014] rad
  3  scara_j2  revolute   -Z    [-4.712, 1.571] rad
  4  wrist_z   revolute   +Z    [-2.094, 2.094] rad
  5  wrist_y   revolute   +Y    [-pi,    pi   ] rad

End-effector frame: y_rotation_1 (gripper mounting face).

DH Parameter Table (standard DH: T = Rz(θ)·Tz(d)·Tx(a)·Rx(α)):
  Joint  θ_i            d_i              a_i    α_i
  1      0              q0+0.025         0      0        prismatic +X
  2      0              -q1+0.075        0.0375 π/2      prismatic -Z
  3      -q2            0.2375           0.200  0        revolute -Z (SCARA 1)
  4      -q3            -0.025           0.200  0        revolute -Z (SCARA 2)
  5       q4            -0.140           0      0        revolute +Z (wrist yaw)
  6      -π/2→q5 split  0.040           0      -π/2     revolute +Y (wrist pitch)

SCARA link lengths:  L1 = L2 = 0.200 m
Wrist Y-offset:      W  = 0.060 m  (origin of wrist_y in z_rotation_1 frame, Y comp)
"""

import numpy as np

# ── Joint limits ──────────────────────────────────────────────────────────────
JOINT_NAMES = ['h_slider', 'v_slider', 'scara_j1', 'scara_j2', 'wrist_z', 'wrist_y']
JOINT_LIMITS = np.array([
    [0.0,       1.25    ],   # h_slider  (m)
    [-1.475,    0.0     ],   # v_slider  (m)
    [-1.047198, 4.014257],   # scara_j1  (rad)
    [-4.712389, 1.570796],   # scara_j2  (rad)
    [-2.094395, 2.094395],   # wrist_z   (rad)
    [-np.pi,    np.pi   ],   # wrist_y   (rad)
])

# ── Kinematic constants from URDF joint origins ───────────────────────────────
L1 = 0.200   # scara_link1_1 reach (scara_j1 → scara_j2 Y distance)
L2 = 0.200   # scara_link2_1 reach (scara_j2 → wrist_z  Y distance)
W  = 0.060   # wrist_y origin Y offset inside z_rotation_1 frame


# ── Primitive transforms ──────────────────────────────────────────────────────

def _t(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]], dtype=float)


def _rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[ c, -s, 0, 0],
                     [ s,  c, 0, 0],
                     [ 0,  0, 1, 0],
                     [ 0,  0, 0, 1]], dtype=float)


def _ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]], dtype=float)


# ── Forward Kinematics ────────────────────────────────────────────────────────

def fk(q):
    """
    Full FK: base_link → y_rotation_1 (EE frame).

    Parameters
    ----------
    q : array-like, length 6
        [h_slider, v_slider, scara_j1, scara_j2, wrist_z, wrist_y]

    Returns
    -------
    T : np.ndarray (4, 4)
        Homogeneous transform.
    """
    q0, q1, q2, q3, q4, q5 = q

    # h_slider: slide +X by q0 from joint origin (0.025, 0, 0.025)
    T01 = _t(0.025, 0.0, 0.025) @ _t(q0, 0, 0)

    # v_slider: slide -Z by |q1| from joint origin (0.05, 0.0375, 0.05)
    #   axis = (0,0,-1)  →  displacement = q1*(0,0,-1) = (0,0,-q1)
    T12 = _t(0.05, 0.0375, 0.05) @ _t(0, 0, -q1)

    # scara_j1: revolute about -Z  →  Rz(-q2)
    T23 = _t(0.2, 0.0625, 0.175) @ _rz(-q2)

    # scara_j2: revolute about -Z  →  Rz(-q3)
    T34 = _t(0.0, 0.2, -0.025) @ _rz(-q3)

    # wrist_z: revolute about +Z  →  Rz(+q4)
    T45 = _t(0.0, 0.2, -0.14) @ _rz(q4)

    # wrist_y: revolute about +Y  →  Ry(+q5)
    T56 = _t(0.0, W, 0.04) @ _ry(q5)

    return T01 @ T12 @ T23 @ T34 @ T45 @ T56


def fk_position(q):
    """Return (x, y, z) of the EE origin in base_link frame."""
    return fk(q)[:3, 3]


def fk_full(q):
    """
    Return EE position and ZYX Euler angles (roll, pitch, yaw).

    Returns
    -------
    pos : np.ndarray (3,)
    rpy : np.ndarray (3,)  [roll, pitch, yaw] in radians
    """
    T = fk(q)
    pos = T[:3, 3]
    R = T[:3, :3]
    sy = np.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        roll  = np.arctan2( R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = np.arctan2( R[1, 0], R[0, 0])
    else:
        roll  = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = 0.0
    return pos, np.array([roll, pitch, yaw])


# ── Inverse Kinematics ────────────────────────────────────────────────────────

def ik(px, py, pz, yaw_ee=0.0, q5=0.0, elbow='up'):
    """
    Analytical IK for cooka robot.

    The robot has an X-redundancy (h_slider + SCARA both affect X).
    Resolution: h_slider places the SCARA base at px; the SCARA arm then
    handles Y reach (and small X deviations) using the 2R formula.

    Parameters
    ----------
    px, py, pz : float
        Desired EE position in base_link frame (metres).
    yaw_ee : float
        Desired EE yaw about Z in base frame (rad). Default 0 = arm pointing +Y.
    q5 : float
        wrist_y angle (rad). 0 = gripper vertical. Default 0.
    elbow : 'up' or 'down'
        SCARA elbow configuration. Default 'up'.

    Returns
    -------
    q : list [q0..q5] or None if target is out of reach / violates limits.
    """
    # ── Step 1: v_slider from Z ──────────────────────────────────────────────
    # Z_EE = 0.125 - q1   (with q5=0; wrist_y adds a small Z when q5 != 0)
    # Full: Z_EE = 0.125 - q1 + 0.04*cos(q5) - 0.04   ... but wrist_y origin
    # already has z=0.04 baked in, so Z_EE = 0.125 - q1 at q5=0.
    # For non-zero q5 the EE Z shifts by 0.04*(cos(q5)-1) ≈ small; correct here:
    z_offset = 0.04 * (np.cos(q5) - 1.0)   # wrist_y Z contribution when q5 != 0
    q1 = 0.125 - pz - z_offset

    # ── Step 2: back-propagate wrist_y offset to get wrist_z target ─────────
    # EE = wrist_z_origin + Rz(yaw_ee) * [0, W, 0]
    #    → x_EE = x_wz - W*sin(yaw_ee),  y_EE = y_wz + W*cos(yaw_ee)
    # Invert:
    x_wz = px + W * np.sin(yaw_ee) * np.cos(q5)
    y_wz = py - W * np.cos(yaw_ee) * np.cos(q5)

    # ── Step 3: h_slider places SCARA base as close as possible to x_wz ────────
    # scara_j1 is at X = 0.275 + q0,  Y = 0.100  in base frame.
    # Clamp q0 to its joint limits — the SCARA arm handles any remaining X offset
    # (e.g. when a module is beyond the h_slider stroke).
    q0 = float(np.clip(x_wz - 0.275, JOINT_LIMITS[0, 0], JOINT_LIMITS[0, 1]))
    x_j1 = 0.275 + q0
    y_j1 = 0.1

    # ── Step 4: 2R SCARA IK ───────────────────────────────────────────────────
    # FK geometry: x_reach = L1*sin(q[2]) + L2*sin(q[2]+q[3])
    #              y_reach = L1*cos(q[2]) + L2*cos(q[2]+q[3])
    # where q[2]=scara_j1 (shoulder) and q[3]=scara_j2 (elbow).
    # dx is non-zero when h_slider is clamped; the 2R formula handles it.
    dx = x_wz - x_j1
    dy = y_wz - y_j1

    r2 = dx**2 + dy**2
    c_elbow = (r2 - L1**2 - L2**2) / (2.0 * L1 * L2)

    if abs(c_elbow) > 1.0:
        return None   # target out of SCARA reach

    s_elbow = np.sqrt(1.0 - c_elbow**2) * (1.0 if elbow == 'up' else -1.0)
    q_scara_j2 = np.arctan2(s_elbow, c_elbow)
    q_scara_j1 = np.arctan2(dx, dy) - np.arctan2(L2 * s_elbow, L1 + L2 * c_elbow)

    # ── Step 5: wrist_z absorbs remaining yaw ────────────────────────────────
    # FK cumulative yaw = -q_scara_j1 - q_scara_j2 + q_wrist_z
    q_wrist_z = yaw_ee + q_scara_j1 + q_scara_j2

    result = [q0, q1, q_scara_j1, q_scara_j2, q_wrist_z, q5]

    if not check_limits(result):
        return None

    return result


# ── Utilities ─────────────────────────────────────────────────────────────────

def check_limits(q):
    """Return True if all joints are within their limits."""
    q = np.asarray(q)
    return bool(np.all(q >= JOINT_LIMITS[:, 0]) and np.all(q <= JOINT_LIMITS[:, 1]))


def print_dh_table():
    print("\nDH Parameter Table (standard DH convention)")
    print("T_i = Rz(θ_i) · Tz(d_i) · Tx(a_i) · Rx(α_i)")
    print(f"{'Joint':<12} {'θ_i':>18} {'d_i':>18} {'a_i':>8} {'α_i':>8}  Notes")
    print("-" * 80)
    rows = [
        ("h_slider",  "0 (fixed)",   "q0 + 0.025",      "0",      "0",       "prismatic +X"),
        ("v_slider",  "0 (fixed)",   "-q1 + 0.075",     "0.0375", "π/2",     "prismatic -Z"),
        ("scara_j1",  "-q2 (var)",   "0.2375",           "0.200",  "0",       "revolute  -Z"),
        ("scara_j2",  "-q3 (var)",   "-0.025",           "0.200",  "0",       "revolute  -Z"),
        ("wrist_z",   "+q4 (var)",   "-0.140",           "0",      "0",       "revolute  +Z"),
        ("wrist_y",   "+q5 (var)",   "0.040",            "0",      "-π/2",    "revolute  +Y"),
    ]
    for r in rows:
        print(f"  {r[0]:<10} {r[1]:>18} {r[2]:>18} {r[3]:>8} {r[4]:>8}  {r[5]}")
    print()


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print_dh_table()

    q_zero = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    p_zero = fk_position(q_zero)
    print(f"FK at zero config:  p = {p_zero.round(4)}")
    print(f"  Expected:             p = [0.275  0.56   0.125]")

    # IK round-trip: pick a target position, solve IK, verify FK matches
    ik_tests = [
        (0.50, 0.50, 0.40,  0.0,  0.0, 'up'),
        (0.40, 0.45, 0.20,  0.3,  0.0, 'up'),
        (0.70, 0.35, 0.60, -0.4,  0.0, 'down'),
    ]
    print()
    for px, py, pz, yaw, q5_t, elbow in ik_tests:
        q_ik = ik(px, py, pz, yaw_ee=yaw, q5=q5_t, elbow=elbow)
        if q_ik is None:
            print(f"IK ({px},{py},{pz}) yaw={yaw} {elbow}: None")
            continue
        p_check = fk_position(q_ik)
        err = np.linalg.norm(np.array([px, py, pz]) - p_check)
        print(f"IK ({px},{py},{pz}) yaw={yaw:+.1f} {elbow}: "
              f"q={[round(v,3) for v in q_ik]}")
        print(f"  FK={p_check.round(4)}  err={err:.2e} m")
