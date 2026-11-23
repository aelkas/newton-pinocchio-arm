"""Step 2: RRT-Connect on the torus T^2 with certified edges, then shortcut smoothing.

Run:  python rrt.py          plans, checks, saves figures/rrt.png and rrt_path.npz
      python rrt.py --viz    same, then plays the smoothed path in the Meshcat viewer

Edge certificate (no sampling along edges, so no edge can be missed between two samples)
------------------------------------------------------------------------------------------
Move the angles along a straight segment theta(t) = a + t * delta. A point of link 1 moves at most
L1 |delta1|, a point of link 2 at most L1 |delta1| + L2 |delta1 + delta2|. Call that rho(delta).
The clearance d is the smallest distance from a point of the arm to a disc, and every such distance
changes by at most how far the point moves, so

        d(theta') >= d(theta) - rho(theta' - theta).

With c = d - MARGIN, the ball {theta' : rho(theta' - a) < c(a)} is free. A segment [a, b] is covered
by the two balls around its ends when rho(b - a) < c(a) + c(b). Otherwise split it at the midpoint
and try again on both halves; a midpoint with c <= 0 means the edge is rejected.
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from geometry import L1, L2, OBSTACLES, clearance, forward_kinematics, wrap
from cspace import START, GOAL

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures", "rrt.png")
PATH_FILE = os.path.join(HERE, "rrt_path.npz")

MARGIN = 0.05     # m: the planned arm keeps at least this far from every disc
STEP = 0.3        # rad: longest edge RRT-Connect adds in one extension
MAX_DEPTH = 30    # bisection depth after which an edge is rejected (conservative, never reached here)


def rho(delta):
    """Bound on how far any point of the arm moves when the angles move by delta (metres)."""
    delta = np.asarray(delta)
    return L1 * np.abs(delta[..., 0]) + L2 * np.abs(delta[..., 0] + delta[..., 1])


class Checker:
    """Clearance with the margin, plus the edge certificate. Counts clearance evaluations."""

    def __init__(self, margin=MARGIN):
        self.margin = margin
        self.evals = 0

    def c(self, theta):
        self.evals += 1
        return float(clearance(theta)) - self.margin

    def edge_free(self, a, b):
        """True if the shortest torus segment from a to b keeps clearance > margin everywhere."""
        ca, cb = self.c(a), self.c(b)
        if ca <= 0 or cb <= 0:
            return False
        delta = wrap(b - a)
        stack = [(0.0, 1.0, ca, cb, 0)]           # sub-segment [s, t] of the edge, c at its ends
        while stack:
            s, t, cs, ct, depth = stack.pop()
            if (t - s) * rho(delta) < cs + ct:
                continue                          # covered by the two end balls
            if depth >= MAX_DEPTH:
                return False
            m = 0.5 * (s + t)
            cm = self.c(a + m * delta)
            if cm <= 0:
                return False
            stack += [(s, m, cs, cm, depth + 1), (m, t, cm, ct, depth + 1)]
        return True


class Tree:
    def __init__(self, root):
        self.nodes = [np.array(root, dtype=float)]
        self.parent = [-1]

    def nearest(self, x):
        d = wrap(np.array(self.nodes) - x)
        return int(np.argmin(np.einsum("ij,ij->i", d, d)))

    def add(self, x, parent):
        self.nodes.append(np.array(x, dtype=float))
        self.parent.append(parent)
        return len(self.nodes) - 1

    def branch(self, i):
        """Nodes from the root to node i."""
        out = []
        while i >= 0:
            out.append(self.nodes[i])
            i = self.parent[i]
        return out[::-1]


def extend(tree, target, checker):
    """Grow the tree one step towards target. Returns ('reached' | 'advanced' | 'trapped', index)."""
    i = tree.nearest(target)
    near = tree.nodes[i]
    delta = wrap(target - near)
    dist = np.linalg.norm(delta)
    new = wrap(near + delta * min(1.0, STEP / max(dist, 1e-12)))
    if not checker.edge_free(near, new):
        return "trapped", None
    j = tree.add(new, i)
    return ("reached" if dist <= STEP else "advanced"), j


def connect(tree, target, checker):
    while True:
        status, j = extend(tree, target, checker)
        if status != "advanced":
            return status, j


def rrt_connect(start, goal, checker, rng, max_iter=5000):
    ta, tb = Tree(start), Tree(goal)
    for it in range(max_iter):
        x = rng.uniform(-np.pi, np.pi, 2)
        status, ja = extend(ta, x, checker)
        if status != "trapped":
            status_b, jb = connect(tb, ta.nodes[ja], checker)
            if status_b == "reached":
                pa, pb = ta.branch(ja), tb.branch(jb)
                path = pa + pb[::-1][1:]          # ta.nodes[ja] and tb.nodes[jb] coincide
                if np.allclose(ta.nodes[0], goal):
                    path, ta, tb = path[::-1], tb, ta
                return np.array(path), (ta, tb), it + 1
        ta, tb = tb, ta
    raise RuntimeError("RRT-Connect found no path")


def shortcut(path, checker, rng, iters=300):
    """Random vertex shortcuts: replace path[i..j] by a straight edge when that edge is certified."""
    path = list(path)
    for _ in range(iters):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i >= 2 and checker.edge_free(path[i], path[j]):
            path = path[: i + 1] + path[j:]
    return np.array(path)


def unwrap_path(path):
    """Continuous angles (no 2 pi jumps): each edge follows the shortest torus segment."""
    out = [path[0]]
    for p in path[1:]:
        out.append(out[-1] + wrap(p - out[-1]))
    return np.array(out)


def densify(path_unwrapped, spacing=0.01):
    """Resample a polyline at roughly constant joint-space spacing (rad)."""
    pts = [path_unwrapped[0]]
    for a, b in zip(path_unwrapped[:-1], path_unwrapped[1:]):
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        pts += [a + (b - a) * s for s in np.arange(1, k + 1) / k]
    return np.array(pts)


def joint_length(path_unwrapped):
    return np.linalg.norm(np.diff(path_unwrapped, axis=0), axis=1).sum()


def test_certificate(checker, rng, n=1000):
    """Certified edges must be free at every one of 2000 dense samples; count rejected free edges too."""
    false_free = rejected_free = n_free = 0
    s = np.linspace(0, 1, 2000)[:, None]
    for _ in range(n):
        a = rng.uniform(-np.pi, np.pi, 2)
        b = wrap(a + rng.uniform(-1.0, 1.0, 2))
        dense_ok = clearance(a + s * wrap(b - a)).min() > checker.margin
        cert_ok = checker.edge_free(a, b)
        n_free += dense_ok
        false_free += cert_ok and not dense_ok
        rejected_free += dense_ok and not cert_ok
    return false_free, rejected_free, n_free


def torus_segments(path):
    """Line segments for plotting an edge list on [-pi, pi)^2, drawing wrapping edges twice."""
    segs = []
    for a, b in path:
        e = a + wrap(b - a)
        segs.append([a, e])
        if np.any(np.abs(e) > np.pi):
            segs.append([wrap(e) - (e - a), wrap(e)])
    return segs


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    checker = Checker()

    false_free, rejected_free, n_free = test_certificate(checker, rng)
    print(f"Certificate test, 1000 random edges up to 1 rad per joint ({n_free} free by dense sampling):")
    print(f"   certified but not free: {false_free} (must be 0); free but rejected: {rejected_free}")
    assert false_free == 0

    checker.evals = 0
    t0 = time.perf_counter()
    raw, (tree_s, tree_g), iters = rrt_connect(START, GOAL, checker, rng)
    t_plan, ev_plan = time.perf_counter() - t0, checker.evals
    smooth = shortcut(raw, checker, rng)
    t_all = time.perf_counter() - t0
    raw_u, smooth_u = unwrap_path(raw), unwrap_path(smooth)
    dense = densify(smooth_u)
    d_dense = clearance(dense)

    print(f"\nRRT-Connect: {iters} iterations, {len(tree_s.nodes) + len(tree_g.nodes)} nodes, "
          f"{ev_plan} clearance evaluations, {1e3 * t_plan:.0f} ms")
    print(f"   raw path:      {len(raw)} waypoints, joint-space length {joint_length(raw_u):.2f} rad")
    print(f"   after shortcut: {len(smooth)} waypoints, length {joint_length(smooth_u):.2f} rad "
          f"(total {1e3 * t_all:.0f} ms)")
    print(f"   smoothed path sampled every 0.01 rad: min clearance {d_dense.min():.3f} m "
          f"(margin {MARGIN} m)")
    print(f"   goal reached as {smooth_u[-1].round(3)} (= goal {GOAL} up to 2 pi turns)")
    assert d_dense.min() > MARGIN - 1e-9

    np.savez(PATH_FILE, raw=raw_u, waypoints=smooth_u, dense=dense, start=START, goal=GOAL, margin=MARGIN)
    print(f"Saved {PATH_FILE}")

    # ---------- figure ----------
    try:
        from cspace_warp import clearance_grid_warp
        D = clearance_grid_warp(512)
    except Exception:
        from cspace import clearance_grid
        D = clearance_grid(512)[1]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={"width_ratios": [1.15, 1, 1]})

    c = ax[0]
    ext = [-np.pi, np.pi, -np.pi, np.pi]
    c.imshow(np.where(D >= MARGIN, D, np.nan).T, origin="lower", extent=ext, cmap="viridis", alpha=0.55)
    c.imshow(np.where(D < MARGIN, np.where(D < 0, 1.0, 0.55), np.nan).T, origin="lower", extent=ext,
             cmap="Greys", vmin=0, vmax=1.6)
    for tree, col in [(tree_s, "tab:green"), (tree_g, "tab:red")]:
        edges = [(tree.nodes[p], n) for n, p in zip(tree.nodes, tree.parent) if p >= 0]
        c.add_collection(LineCollection(torus_segments(edges), colors=col, lw=0.6, alpha=0.6))
    for p, col, lw, lab in [(raw, "k", 1.2, "raw RRT path"), (smooth, "magenta", 2.5, "after shortcuts")]:
        lc = LineCollection(torus_segments(list(zip(p[:-1], p[1:]))), colors=col, lw=lw, label=lab)
        c.add_collection(lc)
    c.plot(*START, "o", color="tab:green", ms=9, mec="k", label="start")
    c.plot(*GOAL, "o", color="tab:red", ms=9, mec="k", label="goal")
    c.set_xlim(-np.pi, np.pi)
    c.set_ylim(-np.pi, np.pi)
    c.set_aspect("equal")
    c.set_xlabel(r"$\theta_1$ (rad)")
    c.set_ylabel(r"$\theta_2$ (rad)")
    c.set_title("Torus: trees, raw and smoothed path\n(dark grey: collision, light grey: within margin)")
    c.legend(loc="upper right", fontsize=8)

    w = ax[1]
    for x, z, r in OBSTACLES:
        w.add_patch(plt.Circle((x, z), r, color="0.45"))
        w.add_patch(plt.Circle((x, z), r + MARGIN, fill=False, ls="--", color="0.45"))
    snaps = dense[np.linspace(0, len(dense) - 1, 12).astype(int)]
    for k, th in enumerate(snaps):
        pts = np.array(forward_kinematics(th))
        w.plot(pts[:, 0], pts[:, 1], "o-", color=plt.cm.plasma(k / (len(snaps) - 1)), lw=2, ms=3)
    tip = forward_kinematics(dense)[2]
    w.plot(tip[:, 0], tip[:, 1], ":", color="0.3", lw=1, label="tip trace")
    w.set_aspect("equal")
    w.set_xlim(-2.4, 2.4)
    w.set_ylim(0.2, 4.8)
    w.set_xlabel("x (m)")
    w.set_ylabel("z (m)")
    w.set_title("Workspace: smoothed path (dark = start, yellow = goal)")
    w.legend(loc="lower left")

    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    ax[2].plot(s, d_dense, color="magenta", label="smoothed path")
    d_raw = clearance(densify(raw_u))
    s_raw = np.linspace(0, joint_length(raw_u), len(d_raw))
    ax[2].plot(s_raw, d_raw, color="k", lw=0.8, alpha=0.6, label="raw path")
    ax[2].axhline(MARGIN, ls="--", color="tab:red", label=f"margin {MARGIN} m")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_xlabel("joint-space arc length (rad)")
    ax[2].set_ylabel(r"clearance $d(\theta)$ (m)")
    ax[2].set_title("Clearance along the path")
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    print(f"Saved {FIG}")

    if "--viz" in sys.argv:
        from viz import ArmViewer
        v = ArmViewer()
        v.add_arm("rrt", color=0xff00ff)
        time.sleep(1.0)
        print("Playing the smoothed path at 1 rad/s in joint space. Ctrl+C to stop.")
        try:
            while True:
                v.play({"rrt": dense}, dt=0.01)
                time.sleep(0.7)
        except KeyboardInterrupt:
            pass
