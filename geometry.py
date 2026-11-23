"""Geometry of the planar 2-link arm: forward kinematics and clearance to disc obstacles.

The arm lives in the vertical x-z plane (both joints turn about the y axis, see arm2.urdf).
Angles theta = (theta1, theta2) are the URDF joint angles: theta2 is relative to link 1.
At theta = (0, 0) the arm hangs straight down from the base at (x, z) = (0, 2.5).

Run:  python geometry.py     checks the kinematics against Pinocchio's frame placements.
"""
import numpy as np

BASE = np.array([0.0, 2.5])   # (x, z) of joint 1
L1, L2 = 1.0, 1.0

# Obstacles: one row per disc, (x, z, radius). Change them freely.
OBSTACLES = np.array([
    [1.2, 1.4, 0.30],
    [-1.1, 3.5, 0.35],
    [0.0, 0.7, 0.25],
])


def forward_kinematics(theta):
    """Base, elbow and tip positions in the x-z plane.

    theta has shape (..., 2); each output has shape (..., 2).
    A rotation by angle a about the y axis sends the downward direction (0, -1) to (-sin a, -cos a).
    """
    theta = np.asarray(theta, dtype=float)
    t1, t12 = theta[..., 0], theta[..., 0] + theta[..., 1]
    base = np.broadcast_to(BASE, theta.shape).copy()
    elbow = base + L1 * np.stack([-np.sin(t1), -np.cos(t1)], axis=-1)
    tip = elbow + L2 * np.stack([-np.sin(t12), -np.cos(t12)], axis=-1)
    return base, elbow, tip


def point_segment_distance(p, a, b):
    """Distance from points p to segments [a, b]; all arrays broadcast, last axis is (x, z)."""
    ab = b - a
    t = np.clip(np.sum((p - a) * ab, axis=-1) / np.sum(ab * ab, axis=-1), 0.0, 1.0)
    closest = a + t[..., None] * ab
    return np.linalg.norm(p - closest, axis=-1)


def pair_clearances(theta, obstacles=OBSTACLES):
    """Signed clearance of every (disc, link) pair: distance from the disc centre to the link
    segment, minus the radius. Shape (..., n_discs, 2). Negative means collision.
    Kept separate from clearance() because iLQR will want a smooth penalty per pair."""
    base, elbow, tip = forward_kinematics(theta)
    c = obstacles[:, :2]                      # (n, 2)
    r = obstacles[:, 2]                       # (n,)
    # insert a disc axis: (..., 1, 2) against (n, 2)
    a1, b1 = base[..., None, :], elbow[..., None, :]
    a2, b2 = elbow[..., None, :], tip[..., None, :]
    d1 = point_segment_distance(c, a1, b1) - r
    d2 = point_segment_distance(c, a2, b2) - r
    return np.stack([d1, d2], axis=-1)


def clearance(theta, obstacles=OBSTACLES):
    """d(theta): smallest signed distance from any obstacle disc to either link. Shape (...)."""
    return pair_clearances(theta, obstacles).min(axis=(-2, -1))


def wrap(theta):
    """Wrap angles to [-pi, pi): the configuration space is the torus T^2 = S^1 x S^1."""
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi


def check_against_pinocchio(n=200, seed=0):
    """Compare forward_kinematics with Pinocchio's placements of the two child frames of the joints."""
    import os
    import pinocchio as pin
    urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm2.urdf")
    model = pin.buildModelFromUrdf(urdf)
    data = model.createData()
    # Pinocchio's joint 1 sits at the base, joint 2 at the elbow. The tip is 1 m below joint 2,
    # so add a frame there to read it off.
    j2 = model.getJointId("joint2")
    tip_id = model.addFrame(pin.Frame("tip", j2, 0, pin.SE3(np.eye(3), np.array([0, 0, -L2])),
                                      pin.FrameType.OP_FRAME))
    data = model.createData()
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        q = rng.uniform(-np.pi, np.pi, 2)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        p_base = data.oMi[model.getJointId("joint1")].translation[[0, 2]]
        p_elbow = data.oMi[j2].translation[[0, 2]]
        p_tip = data.oMf[tip_id].translation[[0, 2]]
        base, elbow, tip = forward_kinematics(q)
        worst = max(worst, np.abs(base - p_base).max(), np.abs(elbow - p_elbow).max(),
                    np.abs(tip - p_tip).max())
    return worst


if __name__ == "__main__":
    err = check_against_pinocchio()
    print(f"Forward kinematics vs Pinocchio, 200 random angles: max position error {err:.1e} m")
    assert err < 1e-9, "kinematics disagree with Pinocchio: check the angle convention"
    # a few sanity values
    print("hanging down, theta = (0, 0):    tip at", forward_kinematics([0.0, 0.0])[2])
    print("link 1 horizontal, theta = (-pi/2, 0): tip at", forward_kinematics([-np.pi / 2, 0.0])[2])
    print("clearance at (0, 0):", round(float(clearance([0.0, 0.0])), 3), "m  (negative = collision)")
