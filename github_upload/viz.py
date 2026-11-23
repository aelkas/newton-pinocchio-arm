"""Meshcat viewer for the 2-link arm (the same browser viewer Pinocchio's viz.viewer uses).

Run:  python viz.py          free swing: Newton (orange) on top of the Pinocchio RK4 reference (blue ghost)
      python viz.py --once   play it once and exit instead of looping

A browser tab opens at http://127.0.0.1:7000/static/ (if not, open the printed URL yourself).
Works on the CPU: Meshcat only draws, it never touches Warp or CUDA.

The URDF has no <visual> tags, so the arm is drawn from geometry.forward_kinematics: one cylinder per
link, one flat cylinder per obstacle disc. Reuse it in later steps:

    from viz import ArmViewer
    v = ArmViewer()                 # opens the browser
    v.add_arm("plan", color=0x3366ff, opacity=0.4)
    v.play({"newton": Q_newton, "plan": Q_plan}, dt=1e-3)   # Q arrays of shape (n_steps, 2)
"""
import sys
import time
import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

from geometry import forward_kinematics, OBSTACLES, L1, L2

ROD_RADIUS = 0.03   # same as the capsule radius of the links in step 4


def _link_pose(start, end):
    """Transform placing a y-axis Meshcat cylinder between two (x, z) points in the y = 0 plane."""
    mid = 0.5 * (start + end)
    d = end - start
    angle = np.arctan2(-d[0], -d[1])          # rotation about y that sends (0, 0, -1) to (dx, 0, dz)
    return (tf.translation_matrix([mid[0], 0.0, mid[1]])
            @ tf.rotation_matrix(angle, [0, 1, 0])
            @ tf.rotation_matrix(np.pi / 2, [1, 0, 0]))   # cylinder axis y -> z


class ArmViewer:
    def __init__(self, obstacles=OBSTACLES, open_browser=True):
        self.vis = meshcat.Visualizer()
        print("Meshcat viewer at", self.vis.url())
        if open_browser:
            self.vis.open()
        self.vis.delete()
        self.arms = []
        for i, (x, z, r) in enumerate(obstacles):
            self.vis[f"obstacles/{i}"].set_object(
                g.Cylinder(0.1, r), g.MeshLambertMaterial(color=0xcc3333, opacity=0.8))
            self.vis[f"obstacles/{i}"].set_transform(tf.translation_matrix([x, 0.0, z]))
        # Camera looking along y at the x-z plane, roughly centred on the workspace.
        self.vis["/Cameras/default"].set_transform(tf.translation_matrix([0.0, 0.0, 2.0]))
        self.vis["/Cameras/default/rotated/<object>"].set_property("position", [0.0, 0.0, 6.0])

    def add_arm(self, name, color=0xff8800, opacity=1.0):
        mat = g.MeshLambertMaterial(color=color, opacity=opacity, transparent=opacity < 1)
        self.vis[f"{name}/link1"].set_object(g.Cylinder(L1, ROD_RADIUS), mat)
        self.vis[f"{name}/link2"].set_object(g.Cylinder(L2, ROD_RADIUS), mat)
        self.vis[f"{name}/tip"].set_object(g.Sphere(1.5 * ROD_RADIUS), mat)
        self.arms.append(name)

    def display(self, name, theta):
        base, elbow, tip = forward_kinematics(theta)
        self.vis[f"{name}/link1"].set_transform(_link_pose(base, elbow))
        self.vis[f"{name}/link2"].set_transform(_link_pose(elbow, tip))
        self.vis[f"{name}/tip"].set_transform(tf.translation_matrix([tip[0], 0.0, tip[1]]))

    def play(self, trajectories, dt, fps=50, speed=1.0):
        """Replay {arm name: (n, 2) joint angles} in real time (times `speed`)."""
        for name in trajectories:
            if name not in self.arms:
                self.add_arm(name)
        n = min(len(Q) for Q in trajectories.values())
        stride = max(1, int(round(1.0 / (fps * dt))))
        frame_time = stride * dt / speed
        for k in range(0, n, stride):
            t0 = time.perf_counter()
            for name, Q in trajectories.items():
                self.display(name, Q[k])
            time.sleep(max(0.0, frame_time - (time.perf_counter() - t0)))


def _demo(once):
    import os
    import pinocchio as pin
    import warp as wp
    import newton

    wp.config.quiet = True
    urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm2.urdf")
    q0, T, dt = np.array([1.0, 0.5]), 5.0, 1e-3
    n = int(round(T / dt))

    # Pinocchio reference, RK4 at the same dt (accurate enough to see by eye).
    pm = pin.buildModelFromUrdf(urdf)
    pd = pm.createData()
    tau = np.zeros(2)
    acc = lambda q, v: pin.aba(pm, pd, q, v, tau)
    q, v = q0.copy(), np.zeros(2)
    Q_ref = [q.copy()]
    for _ in range(n):
        k1q, k1v = v, acc(q, v)
        k2q, k2v = v + 0.5 * dt * k1v, acc(q + 0.5 * dt * k1q, v + 0.5 * dt * k1v)
        k3q, k3v = v + 0.5 * dt * k2v, acc(q + 0.5 * dt * k2q, v + 0.5 * dt * k2v)
        k4q, k4v = v + dt * k3v, acc(q + dt * k3q, v + dt * k3v)
        q = q + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
        v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        Q_ref.append(q.copy())

    # Newton, semi-implicit Euler (Featherstone) on the CPU.
    b = newton.ModelBuilder()
    b.add_urdf(urdf, floating=False, enable_self_collisions=False)
    model = b.finalize()
    solver = newton.solvers.SolverFeatherstone(model)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    s0.joint_q.assign(q0.astype(np.float32))
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    Q_new = [q0.copy()]
    for _ in range(n):
        s0.clear_forces()
        solver.step(s0, s1, ctrl, None, dt)
        s0, s1 = s1, s0
        Q_new.append(s0.joint_q.numpy().astype(float))
    print(f"Simulated {T} s on {model.device}; final angle gap Newton - RK4: "
          f"{np.abs(Q_new[-1] - Q_ref[-1]).max():.2e} rad")

    viewer = ArmViewer()
    viewer.add_arm("pinocchio_rk4", color=0x3366ff, opacity=0.35)
    viewer.add_arm("newton", color=0xff8800)
    time.sleep(1.0)   # give the browser a moment to connect
    print("Playing. Ctrl+C to stop.")
    try:
        while True:
            viewer.play({"pinocchio_rk4": Q_ref, "newton": Q_new}, dt)
            if once:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    _demo(once="--once" in sys.argv)
