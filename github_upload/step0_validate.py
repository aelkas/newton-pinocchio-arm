"""Step 0: check that Newton and Pinocchio simulate the same 2-link arm.

Run:  python step0_validate.py          (uses the GPU if Warp finds one, else the CPU)

Three checks, all without contacts:
  1. same instant: Newton's one-step acceleration (v1 - v)/dt against Pinocchio's aba(q, v, tau)
  2. free swing: Newton against a Runge-Kutta 4 reference from Pinocchio, for step sizes down to
     5e-5 s. The error is still first order there: float32 rounding (~1e-7 per step, so ~1e-5 rad
     after 2e4 steps) stays below the integration error at every step size tested.
  3. energy: drift of the total energy during a 5 s free swing, at three step sizes. Newton's
     semi-implicit Euler is first order and, because the mass matrix depends on q, it does not
     keep the energy bounded: the drift should roughly halve each time dt halves.
Prints the numbers and saves figures/step0_validation.png.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
import warp as wp
import newton

wp.config.quiet = True
HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "arm2.urdf")
FIG = os.path.join(HERE, "figures", "step0_validation.png")
os.makedirs(os.path.dirname(FIG), exist_ok=True)

# ---------- Pinocchio: the reference ----------
pmodel = pin.buildModelFromUrdf(URDF)
pdata = pmodel.createData()
assert pmodel.nq == pmodel.nv == 2


def accel(q, v, tau):
    """Exact joint acceleration from the equation of motion M(q) dv/dt + b(q, v) = tau."""
    return pin.aba(pmodel, pdata, q, v, tau)


def rk4_step(q, v, tau, dt):
    """One Runge-Kutta 4 step of dq/dt = v, dv/dt = aba(q, v, tau)."""
    k1q, k1v = v, accel(q, v, tau)
    k2q, k2v = v + 0.5 * dt * k1v, accel(q + 0.5 * dt * k1q, v + 0.5 * dt * k1v, tau)
    k3q, k3v = v + 0.5 * dt * k2v, accel(q + 0.5 * dt * k2q, v + 0.5 * dt * k2v, tau)
    k4q, k4v = v + dt * k3v, accel(q + dt * k3q, v + dt * k3v, tau)
    return (q + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q),
            v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v))


def energy(q, v):
    return pin.computeKineticEnergy(pmodel, pdata, q, v) + pin.computePotentialEnergy(pmodel, pdata, q)


# ---------- Newton: the simulator under test ----------
builder = newton.ModelBuilder()
builder.add_urdf(URDF, floating=False, enable_self_collisions=False)
nmodel = builder.finalize()
solver = newton.solvers.SolverFeatherstone(nmodel)
s0, s1 = nmodel.state(), nmodel.state()
control = nmodel.control()
print("Newton device:", nmodel.device, "| armature", nmodel.joint_armature.numpy(),
      "| gravity", nmodel.gravity.numpy()[0])


def newton_set(q, v, tau):
    s0.joint_q.assign(q.astype(np.float32))
    s0.joint_qd.assign(v.astype(np.float32))
    control.joint_f.assign(tau.astype(np.float32))
    newton.eval_fk(nmodel, s0.joint_q, s0.joint_qd, s0)


def newton_rollout(q, v, tau, dt, n):
    """Simulate n steps of size dt in Newton, return the joint angles and velocities at every step."""
    global s0, s1
    newton_set(q, v, tau)
    Q, V = [q.copy()], [v.copy()]
    for _ in range(n):
        s0.clear_forces()
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
        Q.append(s0.joint_q.numpy().astype(float))
        V.append(s0.joint_qd.numpy().astype(float))
    return np.array(Q), np.array(V)


rng = np.random.default_rng(0)
t_start = time.time()

# ---------- 1. same instant ----------
dt1 = 1e-3
rel_err, abs_err, a_norm = [], [], []
for _ in range(200):
    q = rng.uniform(-np.pi, np.pi, 2)
    v = rng.uniform(-3, 3, 2)
    tau = rng.uniform(-10, 10, 2)
    _, V = newton_rollout(q, v, tau, dt1, 1)
    a_newton = (V[1] - v) / dt1
    a_pin = accel(q, v, tau)
    abs_err.append(np.linalg.norm(a_newton - a_pin))
    a_norm.append(np.linalg.norm(a_pin))
rel_err = np.array(abs_err) / np.array(a_norm)
print(f"\n1. Same instant, 200 random states: relative acceleration error "
      f"median {np.median(rel_err):.1e}, max {rel_err.max():.1e}; "
      f"absolute error median {np.median(abs_err):.1e} rad/s^2")
print(f"   (float32 rounding of v ~ 3 rad/s, divided by dt = 1e-3, gives ~ {3 * 1.2e-7 / dt1:.0e} rad/s^2)")

# ---------- 2. free swing against a Runge-Kutta 4 reference ----------
q0, v0, tau0 = np.array([1.0, 0.5]), np.zeros(2), np.zeros(2)
T = 1.0
dt_ref = 1e-5
print("\n2. Computing the Runge-Kutta 4 reference at dt = 1e-5 (a few seconds)...")
ref_q = [q0.copy()]
q, v = q0.copy(), v0.copy()
for _ in range(int(round(T / dt_ref))):
    q, v = rk4_step(q, v, tau0, dt_ref)
    ref_q.append(q.copy())
ref_q = np.array(ref_q)

dts = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5]
errs = []
for dt in dts:
    n = int(round(T / dt))
    Q, _ = newton_rollout(q0, v0, tau0, dt, n)
    stride = int(round(dt / dt_ref))
    errs.append(np.abs(Q - ref_q[::stride][: n + 1]).max())
slope = np.polyfit(np.log(dts[:4]), np.log(errs[:4]), 1)[0]
print("   Free swing over 1 s, max angle error against Runge-Kutta 4:")
for dt, e in zip(dts, errs):
    print(f"   dt = {dt:.0e} s   error = {e:.2e} rad")
print(f"   fitted slope on the four largest steps: {slope:.2f} (1 means first-order integrator)")

# ---------- 3. energy during a 5 s free swing, three step sizes ----------
T3 = 5.0
E0 = energy(q0, v0)
drift = {}
for dt3 in [1e-3, 5e-4, 2.5e-4]:
    Q, V = newton_rollout(q0, v0, tau0, dt3, int(round(T3 / dt3)))
    E = np.array([energy(a, b) for a, b in zip(Q, V)])
    drift[dt3] = E - E0
mgl = 2 * 1.0 * 9.81 * 1.0   # energy scale: total mass x g x link length
print(f"\n3. Energy over 5 s (energy scale m g l ~ {mgl:.1f} J):")
for dt3, dE in drift.items():
    print(f"   dt = {dt3:.1e} s   max |E - E0| = {np.abs(dE).max():.2e} J "
          f"({100 * np.abs(dE).max() / mgl:.2f} % of m g l),  final E - E0 = {dE[-1]:.2e} J")
ratios = [np.abs(drift[a]).max() / np.abs(drift[b]).max() for a, b in [(1e-3, 5e-4), (5e-4, 2.5e-4)]]
print(f"   ratio of max drifts when dt halves: {ratios[0]:.2f}, {ratios[1]:.2f} "
      f"(about 2 means first-order integration error, not a model error)")

# ---------- figure ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].scatter(a_norm, rel_err, s=8)
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel("|a| from Pinocchio (rad/s²)")
ax[0].set_ylabel("relative acceleration error")
ax[0].set_title("1. Same instant: Newton vs Pinocchio aba")
ax[1].loglog(dts, errs, "o-", label="Newton (float32)")
ax[1].loglog(dts, errs[0] * np.array(dts) / dts[0], "--", label="slope 1")
ax[1].set_xlabel("time step (s)")
ax[1].set_ylabel("max angle error over 1 s (rad)")
ax[1].set_title("2. Free swing vs Runge-Kutta 4")
ax[1].legend()
for dt3, dE in drift.items():
    ax[2].plot(np.arange(len(dE)) * dt3, dE, label=f"dt = {dt3:.1e} s")
ax[2].set_xlabel("time (s)")
ax[2].set_ylabel("E(t) - E(0)  (J)")
ax[2].set_title("3. Energy drift during a free swing")
ax[2].legend()
fig.tight_layout()
fig.savefig(FIG, dpi=130)
print(f"\nSaved {FIG}  (total {time.time() - t_start:.0f} s)")
