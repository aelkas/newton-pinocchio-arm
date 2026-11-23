"""Step 4: execute the iLQR plan in Newton, with the obstacles as real collision bodies.

Run:  python step4_execute.py          (needs ilqr_plan.npz; saves figures/step4_execution.png)
      python step4_execute.py --viz    then replays the nominal runs in the Meshcat viewer

What Newton gets that the planner did not have
----------------------------------------------
- Physical shapes: each link is a capsule of radius ARM_RADIUS (density 0, so the URDF masses are
  unchanged), each disc a static cylinder across the arm's plane. Touching one gives a penalty
  contact force (stiffness CONTACT_KE): a bad trajectory bounces off instead of passing through.
- A finer time step: Newton integrates at 1 ms, the plan was optimised at 10 ms. Each plan torque is
  held for 10 Newton steps (zero-order hold, like a 100 Hz motor command).
- Wrong masses, in the batch: every world scales link 1 and link 2 masses (and inertias) by factors
  drawn in [0.7, 1.3]. The plan still assumes 1 kg each.

Two controllers, both updated at 100 Hz
---------------------------------------
open loop:    tau_k = TAU_MAX tanh(w_k)                              (replay the planned torques)
closed loop:  tau_k = TAU_MAX tanh(w_k + K_k (x_measured - x_planned_k))
K_k is the feedback gain iLQR computed in its last backward pass, so the controller is the
local optimal policy around the plan, not a hand-tuned PD.

Batched worlds: Newton puts all arms in one model (one world each) and steps them together.
World 0 has the nominal masses; the other worlds have perturbed ones.
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import warp as wp
import newton

from geometry import OBSTACLES, clearance, forward_kinematics, wrap

wp.config.quiet = True
HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "arm2.urdf")
PLAN_FILE = os.path.join(HERE, "ilqr_plan.npz")
FIG = os.path.join(HERE, "figures", "step4_execution.png")

ARM_RADIUS = 0.03        # m, capsule radius of each link
CONTACT_KE, CONTACT_KD = 1e5, 200.0
DT_SIM = 1e-3
N_PERTURBED = 64
MASS_RANGE = (0.7, 1.3)
GOAL_TOL = 0.05          # rad, final angle error counted as "reached"

plan = np.load(PLAN_FILE)
X_PLAN, W_PLAN, K_PLAN = plan["X"], plan["W"], plan["K"]
DT_PLAN, TAU_MAX, GOAL = float(plan["dt"]), plan["tau_max"], plan["goal"]
N = len(W_PLAN)
SUB = int(round(DT_PLAN / DT_SIM))


def arm_builder(scale1=1.0, scale2=1.0):
    """One world: the URDF arm with capsule links, plus the three static discs."""
    b = newton.ModelBuilder()
    b.add_urdf(URDF, floating=False, enable_self_collisions=False)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=CONTACT_KE, kd=CONTACT_KD, mu=0.3)
    for label, s in [("link1", scale1), ("link2", scale2)]:
        i = next(k for k, name in enumerate(b.body_label) if name.endswith(label))
        b.body_mass[i] *= s
        b.body_inv_mass[i] = 1.0 / b.body_mass[i]
        inertia = np.array(b.body_inertia[i]).reshape(3, 3) * s
        b.body_inertia[i] = wp.mat33(*inertia.ravel())
        b.body_inv_inertia[i] = wp.mat33(*np.linalg.inv(inertia).ravel())
        # body frame sits at the joint, the link runs 1 m along -z: capsule axis is z by default
        b.add_shape_capsule(i, xform=wp.transform((0.0, 0.0, -0.5), wp.quat_identity()),
                            radius=ARM_RADIUS, half_height=0.5, cfg=cfg)
    across_plane = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2)   # cylinder axis z -> y
    for x, z, r in OBSTACLES:
        b.add_shape_cylinder(-1, xform=wp.transform((x, 0.0, z), across_plane), radius=r,
                             half_height=0.3, cfg=cfg)
    return b


def build(scales):
    main = newton.ModelBuilder()
    for s1, s2 in scales:
        main.add_world(arm_builder(s1, s2))
    return main.finalize()


def simulate(model, closed_loop):
    """Run the plan in every world. Returns q (N*SUB+1, worlds, 2), v likewise, tau (N, worlds, 2), contacts."""
    nw = model.world_count
    solver = newton.solvers.SolverFeatherstone(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    s0, s1, ctrl = model.state(), model.state(), model.control()
    s0.joint_q.assign(np.tile(X_PLAN[0, :2], nw).astype(np.float32))
    s0.joint_qd.zero_()
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)

    # .numpy() on the CPU is a view of the simulation buffer, which the s0/s1 swap reuses: copy it
    Q, V = [s0.joint_q.numpy().reshape(nw, 2).copy()], [s0.joint_qd.numpy().reshape(nw, 2).copy()]
    taus, n_contacts = [], []
    for k in range(N):
        w = np.tile(W_PLAN[k], (nw, 1))
        if closed_loop:
            dx = np.concatenate([wrap(Q[-1] - X_PLAN[k, :2]), V[-1] - X_PLAN[k, 2:]], axis=1)
            w = w + dx @ K_PLAN[k].T
        tau = TAU_MAX * np.tanh(w)
        taus.append(tau)
        ctrl.joint_f.assign(tau.ravel().astype(np.float32))
        for _ in range(SUB):
            pipeline.collide(s0, contacts)
            n_contacts.append(int(contacts.rigid_contact_count.numpy()[0]))
            s0.clear_forces()
            solver.step(s0, s1, ctrl, contacts, DT_SIM)
            s0, s1 = s1, s0
            Q.append(s0.joint_q.numpy().reshape(nw, 2).astype(float))
            V.append(s0.joint_qd.numpy().reshape(nw, 2).astype(float))
    return np.array(Q), np.array(V), np.array(taus), np.array(n_contacts)


def metrics(Q):
    """Per world: tracking error at the plan knots, physical gap to the discs, final goal error."""
    knots = Q[::SUB]                                             # (N+1, worlds, 2)
    track = np.abs(wrap(knots - X_PLAN[:, None, :2])).max(axis=2)   # (N+1, worlds)
    gap = clearance(Q) - ARM_RADIUS                              # (steps, worlds); < 0 means touching
    final = np.abs(wrap(Q[-1] - GOAL)).max(axis=1)
    return track, gap, final


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    scales = np.vstack([[1.0, 1.0], rng.uniform(*MASS_RANGE, (N_PERTURBED, 2))])
    t0 = time.perf_counter()
    model = build(scales)
    print(f"Newton: {model.world_count} worlds on {model.device}, {model.shape_count} shapes, "
          f"plan from the '{plan['start_name']}' start, {N} commands x {SUB} steps of {DT_SIM} s")

    runs = {}
    for name, closed in [("open loop", False), ("closed loop", True)]:
        t1 = time.perf_counter()
        Q, V, tau, nc = simulate(model, closed)
        track, gap, final = metrics(Q)
        touched = (gap < 0).any(axis=0)
        reached = final < GOAL_TOL
        runs[name] = dict(Q=Q, V=V, tau=tau, nc=nc, track=track, gap=gap, final=final,
                          touched=touched, reached=reached)
        print(f"\n{name} ({time.perf_counter() - t1:.1f} s for all worlds)")
        print(f"   nominal masses: max tracking error {track[:, 0].max():.3f} rad, final error {final[0]:.3f} rad, "
              f"min gap {gap[:, 0].min() * 100:+.1f} cm, touched an obstacle: {'yes' if touched[0] else 'no'}")
        pert = slice(1, None)
        ok = reached[pert] & ~touched[pert]
        print(f"   {N_PERTURBED} perturbed worlds: success {ok.sum()}/{N_PERTURBED} "
              f"(reached within {GOAL_TOL} rad and never touched), touched {touched[pert].sum()}, "
              f"median final error {np.median(final[pert]):.3f} rad, worst {final[pert].max():.3f} rad")
        print(f"   Newton contact candidates (pairs within detection range, all worlds): up to {nc.max()} "
              f"at once, in {100 * (nc > 0).mean():.0f} % of the steps; forces act only when the gap is < 0")

    # ---------- figure ----------
    colors = {"open loop": "tab:red", "closed loop": "tab:blue"}
    t_sim = np.arange(N * SUB + 1) * DT_SIM
    t_knots = np.arange(N + 1) * DT_PLAN
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    w = ax[0, 0]
    for x, z, r in OBSTACLES:
        w.add_patch(plt.Circle((x, z), r, color="0.45"))
    tip_plan = forward_kinematics(X_PLAN[:, :2])[2]
    w.plot(tip_plan[:, 0], tip_plan[:, 1], "k--", lw=1.5, label="plan (Pinocchio)")
    for name, r in runs.items():
        tip = forward_kinematics(r["Q"][:, 0])[2]
        w.plot(tip[:, 0], tip[:, 1], color=colors[name], lw=1.3, label=f"Newton, {name}")
    for idx, col in [(0, "tab:green"), (-1, "k")]:
        pts = np.array(forward_kinematics(X_PLAN[idx, :2]))
        w.plot(pts[:, 0], pts[:, 1], "o-", color=col, lw=3, alpha=0.5)
    w.set_aspect("equal")
    w.set_xlim(-2.4, 2.4)
    w.set_ylim(0.2, 4.8)
    w.set_xlabel("x (m)")
    w.set_ylabel("z (m)")
    w.set_title("Tip path, nominal masses")
    w.legend(loc="lower left", fontsize=8)

    for name, r in runs.items():
        ax[0, 1].semilogy(t_knots, np.maximum(r["track"][:, 0], 1e-6), color=colors[name], label=name)
        ax[0, 2].plot(t_sim, 100 * r["gap"][:, 0], color=colors[name], label=name)
    ax[0, 1].set_xlabel("time (s)")
    ax[0, 1].set_ylabel("max |q_Newton - q_plan| (rad)")
    ax[0, 1].set_title("Tracking error, nominal masses")
    ax[0, 1].legend()
    ax[0, 2].plot(t_knots, 100 * (clearance(X_PLAN[:, :2]) - ARM_RADIUS), "k--", label="plan")
    ax[0, 2].axhline(0, color="k", lw=0.8)
    ax[0, 2].set_xlabel("time (s)")
    ax[0, 2].set_ylabel("gap between link surface and disc (cm)")
    ax[0, 2].set_title("Physical gap, nominal masses (< 0: contact)")
    ax[0, 2].legend()

    mass_ratio = scales[1:].sum(axis=1) / 2
    all_gaps = np.concatenate([100 * r["gap"][:, 1:].min(axis=0) for r in runs.values()])
    gap_bins = np.linspace(all_gaps.min() - 0.5, all_gaps.max() + 0.5, 40)
    for name, r in runs.items():
        f = r["final"][1:]
        ax[1, 0].scatter(scales[1:, 0], scales[1:, 1], s=20 + 400 * np.minimum(f, 1.0), facecolors="none",
                         edgecolors=colors[name], label=f"{name} (marker size ~ final error)")
        ax[1, 1].scatter(mass_ratio, np.maximum(f, 1e-5), color=colors[name], s=14,
                         label=f"{name}: {(r['reached'][1:] & ~r['touched'][1:]).sum()}/{N_PERTURBED} succeed")
        ax[1, 2].hist(100 * r["gap"][:, 1:].min(axis=0), bins=gap_bins, color=colors[name], alpha=0.6, label=name)
    ax[1, 0].plot(1, 1, "k+", ms=12, label="nominal")
    ax[1, 0].set_xlabel("mass scale of link 1")
    ax[1, 0].set_ylabel("mass scale of link 2")
    ax[1, 0].set_title("Perturbed worlds")
    ax[1, 0].legend(fontsize=8, loc="upper left")
    ax[1, 1].axhline(GOAL_TOL, ls="--", color="k", lw=0.8, label=f"tolerance {GOAL_TOL} rad")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel("mean mass scale of the two links")
    ax[1, 1].set_ylabel("final angle error (rad)")
    ax[1, 1].set_title("Final error vs mass error")
    ax[1, 1].legend(fontsize=8)
    ax[1, 2].axvline(0, color="k", lw=0.8)
    ax[1, 2].set_xlabel("smallest gap during the run (cm), < 0: contact")
    ax[1, 2].set_ylabel("worlds")
    ax[1, 2].set_title("Closest approach to the discs, perturbed worlds")
    ax[1, 2].legend()
    fig.tight_layout()
    fig.savefig(FIG, dpi=110)
    print(f"\nSaved {FIG}  (total {time.perf_counter() - t0:.0f} s)")

    if "--viz" in sys.argv:
        from viz import ArmViewer
        v = ArmViewer()
        v.add_arm("plan", color=0x222222, opacity=0.25)
        v.add_arm("open_loop", color=0xdd3333)
        v.add_arm("closed_loop", color=0x3366ff)
        plan_fine = np.repeat(X_PLAN[:, :2], SUB, axis=0)[: N * SUB + 1]
        time.sleep(1.0)
        print("Playing the nominal runs: grey = plan, red = open loop, blue = closed loop. Ctrl+C to stop.")
        try:
            while True:
                v.play({"plan": plan_fine, "open_loop": runs["open loop"]["Q"][:, 0],
                        "closed_loop": runs["closed loop"]["Q"][:, 0]}, dt=DT_SIM)
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
