"""Obstacle region of the 2-link arm, drawn on the configuration torus T^2.

Run:  python cspace.py          (saves figures/cspace.png)

Left: the workspace (x-z plane) with the discs and the arm at the start and goal configurations.
Right: the square [-pi, pi)^2 with opposite edges glued, i.e. the torus. Grey is the obstacle
region {theta : d(theta) < 0}; colour is the clearance d(theta) on the free part.

This is the CPU (NumPy) version. The Warp GPU version comes next and must produce the same grid.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from geometry import OBSTACLES, forward_kinematics, clearance

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures", "cspace.png")
os.makedirs(os.path.dirname(FIG), exist_ok=True)

START = np.array([-2.2, 0.4])   # arm up and to the right
GOAL = np.array([2.0, -0.9])    # arm up and to the left
N = 256                         # grid resolution per angle


def clearance_grid(n=N):
    """d(theta) on an n x n grid of [-pi, pi)^2, computed with vectorised NumPy."""
    t = np.linspace(-np.pi, np.pi, n, endpoint=False)
    T1, T2 = np.meshgrid(t, t, indexing="ij")
    D = clearance(np.stack([T1, T2], axis=-1))
    return t, D


if __name__ == "__main__":
    for name, th in [("start", START), ("goal", GOAL)]:
        d = float(clearance(th))
        print(f"{name} {th}: clearance {d:.3f} m")
        assert d > 0, f"{name} configuration is in collision: move it"

    t0 = time.perf_counter()
    t, D = clearance_grid()
    print(f"CPU grid {N}x{N}: {1e3 * (time.perf_counter() - t0):.1f} ms, "
          f"{100 * (D < 0).mean():.1f} % of the torus is in collision")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))

    # workspace
    w = ax[0]
    for x, z, r in OBSTACLES:
        w.add_patch(plt.Circle((x, z), r, color="0.45"))
    for th, col, lab in [(START, "tab:green", "start"), (GOAL, "tab:red", "goal")]:
        pts = np.array(forward_kinematics(th))
        w.plot(pts[:, 0], pts[:, 1], "o-", color=col, lw=3, label=lab)
    w.add_patch(plt.Circle((0, 2.5), 2.0, fill=False, ls=":", color="0.6"))
    w.set_aspect("equal")
    w.set_xlim(-2.4, 2.4)
    w.set_ylim(0.2, 4.8)
    w.set_xlabel("x (m)")
    w.set_ylabel("z (m)")
    w.set_title("Workspace (dotted: reach of the arm)")
    w.legend(loc="lower left")

    # configuration torus
    c = ax[1]
    ext = [-np.pi, np.pi, -np.pi, np.pi]
    im = c.imshow(np.where(D >= 0, D, np.nan).T, origin="lower", extent=ext, cmap="viridis")
    c.imshow(np.where(D < 0, 1.0, np.nan).T, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.6)
    c.plot(*START, "o", color="tab:green", ms=9, mec="k", label="start")
    c.plot(*GOAL, "o", color="tab:red", ms=9, mec="k", label="goal")
    c.set_xlabel(r"$\theta_1$ (rad)")
    c.set_ylabel(r"$\theta_2$ (rad)")
    c.set_title(r"Configuration space $\mathbb{T}^2$ (opposite edges glued)")
    c.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    c.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    c.legend(loc="upper right")
    fig.colorbar(im, ax=c, label=r"clearance $d(\theta)$ (m)")

    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    print(f"Saved {FIG}")
