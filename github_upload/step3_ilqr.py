"""Step 3: iLQR turns a path into torques the motors can produce, using Pinocchio's dynamics.

Run:  python step3_ilqr.py     (needs rrt_path.npz from rrt.py; saves ilqr_plan.npz, figures/step3_ilqr.png)

Problem, over T = 3 s in N = 300 steps of dt = 0.01 s
------------------------------------------------------
state x = (q, v), control w, torque tau = TAU_MAX * tanh(w)     (squashing keeps |tau_i| < TAU_MAX_i)
dynamics: the same semi-implicit Euler as Newton,  v+ = v + dt * aba(q, v, tau),  q+ = q + dt * v+
cost:     sum_k dt * [ W_TAU |tau|^2 + W_W |w|^2 + W_V |v|^2 + W_OBS sum_pairs max(0, D_SAFE - d_pair(q))^2 ]
          + W_GOAL sum_i (1 - cos(q_i - goal_i)) + W_VEL |v_N|^2
The goal term uses cosines so any representative of the goal angles mod 2 pi counts.
d_pair is the distance from one disc to one link segment (geometry.pair_clearances); D_SAFE = 0.12 m
leaves room for the 0.03 m link radius used in Newton and for tracking errors.

iLQR needs the linearised dynamics at every step: Pinocchio's computeABADerivatives gives
d(aba)/dq, d(aba)/dv and d(aba)/dtau = M(q)^-1 analytically.

Two starts: the RRT path from step 2, and the straight segment from start to goal in [-pi, pi)^2,
which passes through the obstacles. Each start is turned into initial controls by tracking a
minimum-jerk timing of the path with inverse dynamics (rnea) plus a PD term.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin

from geometry import OBSTACLES, clearance, forward_kinematics, pair_clearances, wrap
from cspace import START, GOAL

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "arm2.urdf")
RRT_FILE = os.path.join(HERE, "rrt_path.npz")
PLAN_FILE = os.path.join(HERE, "ilqr_plan.npz")
FIG = os.path.join(HERE, "figures", "step3_ilqr.png")

DT, T = 0.01, 3.0
N = int(round(T / DT))
TAU_MAX = np.array([25.0, 10.0])   # N m; holding the straight arm horizontal needs 19.6 and 4.9
D_SAFE = 0.12
W_TAU, W_W, W_OBS, W_GOAL, W_VEL = 1e-2, 1e-4, 1e5, 1e3, 1e2
W_V = 0.2   # running speed penalty: without it the optimum whips the elbow at up to 25 rad/s

model = pin.buildModelFromUrdf(URDF)
data = model.createData()


def squash(w):
    return TAU_MAX * np.tanh(w)


def dsquash(w):
    return TAU_MAX * (1.0 - np.tanh(w) ** 2)


# ---------- dynamics ----------
def step(x, w):
    q, v = x[:2], x[2:]
    v_next = v + DT * pin.aba(model, data, q, v, squash(w))
    return np.concatenate([q + DT * v_next, v_next])


def step_derivatives(x, w):
    q, v = x[:2], x[2:]
    a_q, a_v, a_tau = pin.computeABADerivatives(model, data, q, v, squash(w))
    eye = np.eye(2)
    fx = np.block([[eye + DT ** 2 * a_q, DT * eye + DT ** 2 * a_v],
                   [DT * a_q, eye + DT * a_v]])
    a_w = a_tau * dsquash(w)                 # chain rule through tanh (scales the columns)
    fu = np.vstack([DT ** 2 * a_w, DT * a_w])
    return fx, fu


def rollout(x0, W):
    X = [x0]
    for w in W:
        X.append(step(X[-1], w))
    return np.array(X)


# ---------- cost ----------
def obstacle_terms(Q):
    """Penalty, gradient and Gauss-Newton Hessian of W_OBS sum max(0, D_SAFE - d)^2 for each row of Q."""
    n = len(Q)
    d = pair_clearances(Q).reshape(n, -1)
    h = np.maximum(0.0, D_SAFE - d)
    G = np.empty(d.shape + (2,))
    eps = 1e-6
    for i in range(2):                       # d is C^1 away from the segment: central differences are fine
        e = np.zeros(2)
        e[i] = eps
        G[..., i] = (pair_clearances(Q + e).reshape(n, -1) - pair_clearances(Q - e).reshape(n, -1)) / (2 * eps)
    cost = W_OBS * (h ** 2).sum(1)
    grad = -2 * W_OBS * np.einsum("np,npi->ni", h, G)
    hess = 2 * W_OBS * np.einsum("np,npi,npj->nij", (h > 0).astype(float), G, G)
    return cost, grad, hess


def terminal_terms(x):
    e, v = x[:2] - GOAL, x[2:]
    oc, og, oh = obstacle_terms(x[None, :2])
    cost = W_GOAL * (1 - np.cos(e)).sum() + W_VEL * v @ v + DT * oc[0]
    grad = np.concatenate([W_GOAL * np.sin(e) + DT * og[0], 2 * W_VEL * v])
    hess = np.zeros((4, 4))
    hess[:2, :2] = np.diag(W_GOAL * np.maximum(np.cos(e), 1e-3)) + DT * oh[0]
    hess[2:, 2:] = 2 * W_VEL * np.eye(2)
    return cost, grad, hess


def cost_parts(X, W):
    tau = squash(W)
    effort = DT * (W_TAU * (tau ** 2).sum() + W_W * (W ** 2).sum() + W_V * (X[:-1, 2:] ** 2).sum())
    obstacle = DT * obstacle_terms(X[:-1, :2])[0].sum()
    terminal = terminal_terms(X[-1])[0]
    return effort, obstacle, terminal


def total_cost(X, W):
    return sum(cost_parts(X, W))


# ---------- iLQR ----------
def backward_pass(X, W, mu):
    fxs, fus = zip(*(step_derivatives(x, w) for x, w in zip(X[:-1], W)))
    _, og, oh = obstacle_terms(X[:-1, :2])
    tau, s = squash(W), dsquash(W)
    lu = DT * (2 * W_TAU * tau * s + 2 * W_W * W)
    luu_diag = DT * (2 * W_TAU * s ** 2 + 2 * W_W)
    _, Vx, Vxx = terminal_terms(X[-1])
    k_ff, K_fb = np.zeros_like(W), np.zeros((N, 2, 4))
    dv1 = dv2 = 0.0
    for i in reversed(range(N)):
        fx, fu = fxs[i], fus[i]
        lx = np.concatenate([DT * og[i], 2 * DT * W_V * X[i, 2:]])
        lxx = np.zeros((4, 4))
        lxx[:2, :2] = DT * oh[i]
        lxx[2:, 2:] = 2 * DT * W_V * np.eye(2)
        Qx = lx + fx.T @ Vx
        Qu = lu[i] + fu.T @ Vx
        Qxx = lxx + fx.T @ Vxx @ fx
        Quu = np.diag(luu_diag[i]) + fu.T @ Vxx @ fu
        Qux = fu.T @ Vxx @ fx
        Quu_reg = Quu + mu * np.eye(2)
        try:
            L = np.linalg.cholesky(Quu_reg)
        except np.linalg.LinAlgError:
            return None
        solve = lambda b: np.linalg.solve(L.T, np.linalg.solve(L, b))
        k, K = -solve(Qu), -solve(Qux)
        Vx = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
        Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
        Vxx = 0.5 * (Vxx + Vxx.T)
        k_ff[i], K_fb[i] = k, K
        dv1 += k @ Qu
        dv2 += 0.5 * k @ Quu @ k
    return k_ff, K_fb, dv1, dv2


def forward_pass(X, W, k_ff, K_fb, alpha):
    Xn, Wn = np.empty_like(X), np.empty_like(W)
    Xn[0] = X[0]
    for i in range(N):
        Wn[i] = W[i] + alpha * k_ff[i] + K_fb[i] @ (Xn[i] - X[i])
        Xn[i + 1] = step(Xn[i], Wn[i])
    return Xn, Wn


def ilqr(X, W, max_iter=200, tol=1e-6, label=""):
    J, mu = total_cost(X, W), 1e-3
    history = [J]
    K_fb = np.zeros((N, 2, 4))
    for it in range(max_iter):
        bp = backward_pass(X, W, mu)
        if bp is None:
            mu *= 10
            continue
        k_ff, K_fb, dv1, dv2 = bp
        accepted = False
        for alpha in 0.5 ** np.arange(8):
            Xn, Wn = forward_pass(X, W, k_ff, K_fb, alpha)
            if not np.all(np.isfinite(Xn)):
                continue
            Jn = total_cost(Xn, Wn)
            expected = -(alpha * dv1 + alpha ** 2 * dv2)
            if J - Jn > 1e-4 * max(expected, 0.0):
                accepted = True
                break
        if not accepted:
            mu *= 10
            if mu > 1e8:
                break
            continue
        rel = (J - Jn) / max(J, 1e-12)
        X, W, J = Xn, Wn, Jn
        history.append(J)
        mu = max(mu / 3, 1e-6)
        if rel < tol:
            break
    # gains of the final trajectory, for closed-loop execution in step 4
    bp = backward_pass(X, W, 1e-6)
    if bp is not None:
        K_fb = bp[1]
    print(f"   {label}: {len(history) - 1} accepted iterations, cost {history[0]:.1f} -> {J:.2f}")
    return X, W, K_fb, np.array(history)


# ---------- warm start: track a timed path with inverse dynamics + PD ----------
def timed_reference(P):
    """Minimum-jerk timing of the polyline P (m, 2) over [0, T]: q, v, a at the N + 1 knots."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    u = np.arange(N + 1) * DT / T
    s_t = s[-1] * (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)
    Q = np.stack([np.interp(s_t, s, P[:, i]) for i in range(2)], axis=1)
    V = np.gradient(Q, DT, axis=0)
    A = np.gradient(V, DT, axis=0)
    return Q, V, A


def warm_start(P, kp=np.array([150.0, 60.0]), kd=np.array([25.0, 10.0])):
    Qr, Vr, Ar = timed_reference(P)
    x = np.concatenate([Qr[0], np.zeros(2)])
    W = np.zeros((N, 2))
    for k in range(N):
        tau = pin.rnea(model, data, Qr[k], Vr[k], Ar[k]) + kp * (Qr[k] - x[:2]) + kd * (Vr[k] - x[2:])
        W[k] = np.arctanh(np.clip(tau / TAU_MAX, -0.995, 0.995))
        x = step(x, W[k])
    return rollout(np.concatenate([Qr[0], np.zeros(2)]), W), W, Qr


def summary(X, W):
    d = clearance(X[:, :2])
    goal_err = np.abs(wrap(X[-1, :2] - GOAL)).max()
    return {"min_clearance": d.min(), "goal_error": goal_err, "final_speed": np.abs(X[-1, 2:]).max(),
            "max_tau_ratio": (np.abs(squash(W)) / TAU_MAX).max()}


if __name__ == "__main__":
    rrt = np.load(RRT_FILE)
    starts = {"rrt": rrt["dense"], "straight": np.array([START, GOAL])}
    results = {}
    t_all = time.perf_counter()
    for name, P in starts.items():
        X0, W0, Qr = warm_start(P)
        print(f"\n{name} start: warm-start rollout min clearance {clearance(X0[:, :2]).min():.3f} m, "
              f"cost {total_cost(X0, W0):.1f}")
        t0 = time.perf_counter()
        X, W, K, hist = ilqr(X0, W0, label=name)
        s = summary(X, W)
        e, o, g = cost_parts(X, W)
        print(f"   {time.perf_counter() - t0:.1f} s; cost = effort + speed {e:.2f} + obstacle {o:.3f} + goal {g:.3f}")
        print(f"   min clearance {s['min_clearance']:.3f} m (target {D_SAFE}), goal error {s['goal_error']:.1e} rad, "
              f"final speed {s['final_speed']:.1e} rad/s, peak |tau|/tau_max {s['max_tau_ratio']:.2f}")
        results[name] = dict(X=X, W=W, K=K, hist=hist, X0=X0, Qr=Qr, cost=hist[-1], **s)

    feasible = {k: r for k, r in results.items() if r["min_clearance"] > D_SAFE - 0.01 and r["goal_error"] < 0.02}
    best = min(feasible or results, key=lambda k: results[k]["cost"])
    print(f"\nBest plan: from the {best} start (cost {results[best]['cost']:.2f}); total {time.perf_counter() - t_all:.0f} s")
    r = results[best]
    np.savez(PLAN_FILE, X=r["X"], W=r["W"], K=r["K"], tau=squash(r["W"]), dt=DT, tau_max=TAU_MAX,
             d_safe=D_SAFE, goal=GOAL, start_name=best,
             **{f"{k}_X": v["X"] for k, v in results.items()})
    print(f"Saved {PLAN_FILE}")

    # ---------- figure ----------
    from rrt import torus_segments
    from matplotlib.collections import LineCollection
    try:
        from cspace_warp import clearance_grid_warp
        D = clearance_grid_warp(512)
    except Exception:
        from cspace import clearance_grid
        D = clearance_grid(512)[1]
    colors = {"rrt": "magenta", "straight": "tab:orange"}
    t = np.arange(N + 1) * DT
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    c = ax[0, 0]
    ext = [-np.pi, np.pi, -np.pi, np.pi]
    c.imshow(np.where(D >= D_SAFE, D, np.nan).T, origin="lower", extent=ext, cmap="viridis", alpha=0.5)
    c.imshow(np.where(D < D_SAFE, np.where(D < 0, 1.0, 0.55), np.nan).T, origin="lower", extent=ext,
             cmap="Greys", vmin=0, vmax=1.6)
    for name, res in results.items():
        for Q, ls, lw, lab in [(res["Qr"], ":", 1.2, f"{name}: start path"),
                               (res["X"][:, :2], "-", 2.2, f"{name}: iLQR")]:
            Qw = wrap(Q)
            lc = LineCollection(torus_segments(list(zip(Qw[:-1], Qw[1:]))), colors=colors[name],
                                linestyles=ls, lw=lw, label=lab)
            c.add_collection(lc)
    c.plot(*START, "o", color="tab:green", ms=9, mec="k")
    c.plot(*GOAL, "o", color="tab:red", ms=9, mec="k")
    c.set_xlim(-np.pi, np.pi)
    c.set_ylim(-np.pi, np.pi)
    c.set_aspect("equal")
    c.set_xlabel(r"$\theta_1$ (rad)")
    c.set_ylabel(r"$\theta_2$ (rad)")
    c.set_title(f"Torus (grey: closer than D_SAFE = {D_SAFE} m)")
    c.legend(fontsize=7, loc="upper right")

    w = ax[0, 1]
    for x, z, rad in OBSTACLES:
        w.add_patch(plt.Circle((x, z), rad, color="0.45"))
        w.add_patch(plt.Circle((x, z), rad + D_SAFE, fill=False, ls="--", color="0.45"))
    Xb = results[best]["X"]
    for k, idx in enumerate(np.linspace(0, N, 13).astype(int)):
        pts = np.array(forward_kinematics(Xb[idx, :2]))
        w.plot(pts[:, 0], pts[:, 1], "o-", color=plt.cm.plasma(k / 12), lw=2, ms=3)
    w.set_aspect("equal")
    w.set_xlim(-2.4, 2.4)
    w.set_ylim(0.2, 4.8)
    w.set_title(f"Workspace: best plan ({best} start), every 0.25 s")
    w.set_xlabel("x (m)")
    w.set_ylabel("z (m)")

    for name, res in results.items():
        ax[0, 2].semilogy(res["hist"], color=colors[name], label=name)
        ax[1, 0].plot(t, clearance(res["X"][:, :2]), color=colors[name], label=name)
        tau = squash(res["W"])
        for j, ls in [(0, "-"), (1, "--")]:
            ax[1, 1].plot(t[:-1], tau[:, j], ls, color=colors[name], label=f"{name}, joint {j + 1}")
        ax[1, 2].plot(t, res["X"][:, 2], "-", color=colors[name], label=f"{name}, joint 1")
        ax[1, 2].plot(t, res["X"][:, 3], "--", color=colors[name], label=f"{name}, joint 2")
    ax[0, 2].set_xlabel("accepted iteration")
    ax[0, 2].set_ylabel("cost")
    ax[0, 2].set_title("iLQR convergence")
    ax[0, 2].legend()
    ax[1, 0].axhline(D_SAFE, ls="--", color="tab:red", label="D_SAFE")
    ax[1, 0].axhline(0, color="k", lw=0.5)
    ax[1, 0].set_xlabel("time (s)")
    ax[1, 0].set_ylabel("clearance (m)")
    ax[1, 0].set_title("Clearance of the optimised trajectories")
    ax[1, 0].legend()
    for j, ls in [(0, "-"), (1, "--")]:
        ax[1, 1].axhline(TAU_MAX[j], ls=ls, color="0.5", lw=0.8)
        ax[1, 1].axhline(-TAU_MAX[j], ls=ls, color="0.5", lw=0.8)
    ax[1, 1].set_xlabel("time (s)")
    ax[1, 1].set_ylabel("torque (N m)")
    ax[1, 1].set_title("Torques (grey: limits)")
    ax[1, 1].legend(fontsize=7)
    ax[1, 2].set_xlabel("time (s)")
    ax[1, 2].set_ylabel("joint velocity (rad/s)")
    ax[1, 2].set_title("Velocities")
    ax[1, 2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG, dpi=110)
    print(f"Saved {FIG}")
