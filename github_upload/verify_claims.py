"""Independent checks of the README's explanations (not just its numbers).

Run:  python verify_claims.py      (needs ilqr_plan.npz; about 1 minute on CPU)

A. "Open loop fails because of the time step (1 ms vs the plan's 10 ms) and float32, not Newton."
   Replay the planned torques with no obstacles, four ways:
     1. Pinocchio, float64, 10 ms steps     -> must reproduce the plan exactly (sanity check)
     2. Newton,   float32, 10 ms steps      -> if Newton matches, float32 alone is harmless here
     3. Pinocchio, float64, 1 ms steps      -> if this diverges, the time step alone causes it
     4. Newton,   float32, 1 ms steps       -> should look like 3 (Newton faithful to Pinocchio)
B. Which disc does the nominal open-loop run touch first?
C. Are the contacts real forces?  Re-run the nominal open loop with the contact forces off:
   the trajectories must coincide until the first touch and differ after it.
D. Are the perturbed masses really in the Newton model?  Read them back from the finalized model.
E. Closed loop: when is the tracking error largest, is the plan saturated then, and is the one
   failing perturbed world the heaviest?
"""
import numpy as np
import pinocchio as pin
import warp as wp
import newton

import step4_execute as s4
from geometry import OBSTACLES, pair_clearances, wrap

wp.config.quiet = True
X, W, TAU_MAX = s4.X_PLAN, s4.W_PLAN, s4.TAU_MAX
TAU = TAU_MAX * np.tanh(W)                    # planned torques, (300, 2)
N, DT_PLAN = len(W), s4.DT_PLAN

pmodel = pin.buildModelFromUrdf(s4.URDF)
pdata = pmodel.createData()


def pinocchio_replay(dt):
    """Semi-implicit Euler in float64, torques held for DT_PLAN (zero-order hold). Returns q at the knots."""
    sub = int(round(DT_PLAN / dt))
    q, v = X[0, :2].copy(), X[0, 2:].copy()
    out = [q.copy()]
    for k in range(N):
        for _ in range(sub):
            v = v + dt * pin.aba(pmodel, pdata, q, v, TAU[k])
            q = q + dt * v
        out.append(q.copy())
    return np.array(out)


def newton_replay(dt, with_obstacles=False, contacts_on=True):
    """Newton open-loop replay; returns q at every sim step."""
    if with_obstacles:
        model = s4.build([(1.0, 1.0)])
    else:
        b = newton.ModelBuilder()
        b.add_urdf(s4.URDF, floating=False, enable_self_collisions=False)
        model = b.finalize()
    sub = int(round(DT_PLAN / dt))
    solver = newton.solvers.SolverFeatherstone(model)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    s0.joint_q.assign(X[0, :2].astype(np.float32))
    s0.joint_qd.zero_()
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    pipeline = contacts = None
    if with_obstacles and contacts_on:
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
    Q = [s0.joint_q.numpy().astype(float)]
    for k in range(N):
        ctrl.joint_f.assign(TAU[k].astype(np.float32))
        for _ in range(sub):
            if pipeline is not None:
                pipeline.collide(s0, contacts)
            s0.clear_forces()
            solver.step(s0, s1, ctrl, contacts, dt)
            s0, s1 = s1, s0
            Q.append(s0.joint_q.numpy().astype(float))
    return np.array(Q), sub


def track(Qknots):
    return np.abs(wrap(Qknots - X[:, :2])).max(axis=1)


def first_time_above(err, thr):
    i = np.argmax(err > thr)
    return i * DT_PLAN if err[i] > thr else None


if __name__ == "__main__":
    print("A. Open-loop replay without obstacles: max |q - q_plan| (rad) at t = 0.5, 1, 1.5, 2 s")
    rows = []
    p10 = pinocchio_replay(1e-2)
    rows.append(("Pinocchio float64, 10 ms", p10))
    n10, _ = newton_replay(1e-2)
    rows.append(("Newton    float32, 10 ms", n10))
    p1 = pinocchio_replay(1e-3)
    rows.append(("Pinocchio float64,  1 ms", p1))
    n1, sub = newton_replay(1e-3)
    rows.append(("Newton    float32,  1 ms", n1[::sub]))
    for name, Qk in rows:
        e = track(Qk)
        vals = "  ".join(f"{e[int(t / DT_PLAN)]:.1e}" for t in [0.5, 1.0, 1.5, 2.0])
        t_off = first_time_above(e, 0.05)
        print(f"   {name}: {vals}   first > 0.05 rad at t = {t_off if t_off is None else round(t_off, 2)} s")
    gap = np.abs(wrap(n1[::sub] - p1)).max(axis=1)
    i = int(1.0 / DT_PLAN)
    print(f"   Newton 1 ms vs Pinocchio 1 ms (same discretisation): {gap[i]:.1e} rad at 1 s, "
          f"{gap[int(1.5 / DT_PLAN)]:.1e} at 1.5 s")

    print("\nB. Nominal open loop with obstacles: which disc is touched first?")
    Qo, sub = newton_replay(1e-3, with_obstacles=True)
    gaps = pair_clearances(Qo) - s4.ARM_RADIUS            # (steps, discs, links)
    k = np.argmax((gaps < 0).any(axis=(1, 2)))
    disc, link = np.argwhere(gaps[k] < 0)[0]
    print(f"   first contact at t = {k * 1e-3:.3f} s: disc {disc} at (x, z) = {OBSTACLES[disc, :2]}, "
          f"link {link + 1}; deepest penetration {-gaps.min() * 100:.2f} cm")

    print("\nC. Same run with contact forces switched off:")
    Qn, _ = newton_replay(1e-3, with_obstacles=True, contacts_on=False)
    diff = np.abs(wrap(Qo - Qn)).max(axis=1)
    print(f"   before first contact: max difference {diff[:k].max():.1e} rad; "
          f"0.2 s after it: {diff[min(k + 200, len(diff) - 1)]:.2f} rad; "
          f"lowest gap without contacts: {(pair_clearances(Qn) - s4.ARM_RADIUS).min() * 100:.1f} cm")

    print("\nD. Masses read back from the Newton model (first 4 worlds):")
    rng = np.random.default_rng(0)
    scales = np.vstack([[1.0, 1.0], rng.uniform(*s4.MASS_RANGE, (s4.N_PERTURBED, 2))])[:4]
    model = s4.build(scales)
    m = model.body_mass.numpy()
    per = len(m) // len(scales)
    for w_i, (s1, s2) in enumerate(scales):
        mw = m[w_i * per:(w_i + 1) * per]
        print(f"   world {w_i}: asked ({s1:.3f}, {s2:.3f}) kg, model has {np.round(mw[mw > 0], 3)} kg")

    print("\nE. Closed loop: tracking error vs torque saturation, and the failing world")
    rng = np.random.default_rng(0)
    scales = np.vstack([[1.0, 1.0], rng.uniform(*s4.MASS_RANGE, (s4.N_PERTURBED, 2))])
    Qc, _, tau_c, _ = s4.simulate(s4.build(scales), closed_loop=True)
    trk, gap_c, final_c = s4.metrics(Qc)
    t_cmd = np.arange(N) * DT_PLAN                            # times of the N torque commands
    t_knot = np.arange(N + 1) * DT_PLAN                       # times of the N + 1 plan states
    sat = np.abs(tau_c[:, 0]) / TAU_MAX                       # applied torque, nominal world
    for lo, hi in [(0.0, 0.3), (0.3, 2.9), (2.9, 3.0)]:
        mc = (t_cmd >= lo - 1e-9) & (t_cmd < hi - 1e-9)
        mk = (t_knot > lo + 1e-9) & (t_knot <= hi + 1e-9)    # states reached by those commands
        i = np.argmax(trk[mk, 0])
        print(f"   commands in [{lo}, {hi}) s: max tracking error {trk[mk, 0][i]:.4f} rad "
              f"(at t = {t_knot[mk][i]:.2f} s), max |tau|/tau_max (joint 1, joint 2) {sat[mc].max(0).round(3)}")
    ok = (final_c < s4.GOAL_TOL) & ~(gap_c < 0).any(0)
    fail = np.where(~ok[1:])[0] + 1
    mean_scale = scales.mean(1)
    print(f"   failing perturbed worlds: {fail.tolist()}, mean mass scale {mean_scale[fail].round(3).tolist()}; "
          f"heaviest perturbed world: {1 + int(np.argmax(mean_scale[1:]))} "
          f"(mean {mean_scale[1:].max():.3f})")
