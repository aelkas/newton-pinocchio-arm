"""Step 1: the clearance grid d(theta) on the torus, computed by a Warp kernel.

Run:  python cspace_warp.py

One kernel thread per grid cell: forward kinematics, then the distance from every disc to both links.
Checks the result against the NumPy version in cspace.py and times both at several resolutions.

Warp runs the same kernel on a CUDA GPU if it finds one, otherwise it compiles it to native CPU code.
Without a GPU the comparison is "NumPy (vectorised, float64)" against "compiled loop (float32)":
the kernel avoids NumPy's large temporary arrays, which is where most of its gain comes from.
"""
import time
import numpy as np
import warp as wp

from geometry import BASE, L1, L2, OBSTACLES
from cspace import clearance_grid as clearance_grid_numpy

wp.config.quiet = True
wp.init()
DEVICE = wp.get_preferred_device()


@wp.func
def segment_distance(p: wp.vec2, a: wp.vec2, b: wp.vec2):
    ab = b - a
    t = wp.clamp(wp.dot(p - a, ab) / wp.dot(ab, ab), 0.0, 1.0)
    return wp.length(p - (a + t * ab))


@wp.kernel
def clearance_kernel(n: int, base: wp.vec2, l1: float, l2: float,
                     obstacles: wp.array(dtype=wp.vec3), out: wp.array2d(dtype=float)):
    i, j = wp.tid()
    step = 2.0 * wp.pi / float(n)
    t1 = -wp.pi + float(i) * step
    t12 = t1 + (-wp.pi + float(j) * step)
    elbow = base + l1 * wp.vec2(-wp.sin(t1), -wp.cos(t1))
    tip = elbow + l2 * wp.vec2(-wp.sin(t12), -wp.cos(t12))
    d = float(1.0e9)
    for k in range(obstacles.shape[0]):
        o = obstacles[k]
        c = wp.vec2(o[0], o[1])
        d = wp.min(d, segment_distance(c, base, elbow) - o[2])
        d = wp.min(d, segment_distance(c, elbow, tip) - o[2])
    out[i, j] = d


def clearance_grid_warp(n, obstacles=OBSTACLES, device=DEVICE):
    """d(theta) on the same n x n grid as cspace.clearance_grid (row index = theta1)."""
    obs = wp.array(obstacles.astype(np.float32), dtype=wp.vec3, device=device)
    out = wp.empty((n, n), dtype=float, device=device)
    wp.launch(clearance_kernel, dim=(n, n),
              inputs=[n, wp.vec2(*BASE), float(L1), float(L2), obs, out], device=device)
    return out.numpy()


def best_time(f, repeats):
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        f()
        wp.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


if __name__ == "__main__":
    print("Warp device:", DEVICE)
    t0 = time.perf_counter()
    clearance_grid_warp(8)   # first launch compiles the kernel (cached on disk afterwards)
    print(f"kernel build/load: {time.perf_counter() - t0:.2f} s")

    _, D_np = clearance_grid_numpy(256)
    D_wp = clearance_grid_warp(256)
    err = np.abs(D_np - D_wp).max()
    flips = np.sum((D_np < 0) != (D_wp < 0))
    print(f"256x256: max |NumPy - Warp| = {err:.1e} m, cells with a different free/collision label: {flips}")
    assert err < 1e-5, "Warp and NumPy disagree"

    print(f"\n{'n':>6} {'cells':>10} {'NumPy (ms)':>11} {'Warp (ms)':>10} {'speed-up':>9}")
    for n in [128, 256, 512, 1024, 2048]:
        t_wp = best_time(lambda: clearance_grid_warp(n), 5)
        if n <= 1024:   # NumPy needs ~ n^2 x 6 x 8 bytes per temporary: skip the largest grid
            t_np = best_time(lambda: clearance_grid_numpy(n), 3)
            print(f"{n:>6} {n * n:>10} {1e3 * t_np:>11.1f} {1e3 * t_wp:>10.1f} {t_np / t_wp:>8.1f}x")
        else:
            print(f"{n:>6} {n * n:>10} {'skipped':>11} {1e3 * t_wp:>10.1f}")
