"""Profile the 3D reciprocal-space mapping pipeline on a synthetic rocking stack.

Nothing here is measured data. The volume is a Gaussian Bragg peak plus one streak, Poisson
sampled with a fixed seed. Each scale is run in a fresh timing loop (best of REPEAT) and a
separate tracemalloc run for peak memory.

Usage::

    uv run python scripts/bench_rsm.py [--repeat 3] [--scales small medium large]
"""

import argparse
import time
import tracemalloc
from collections.abc import Callable

import numpy as np
from qorageom import wavelength
from scipy import ndimage

from cxd.core.transform import correct_rocking
from cxd.mapping.qspace import correct_to_qspace, get_full_spherical_map

SCALES = {"small": (32, 128, 128), "medium": (64, 256, 256), "large": (96, 384, 384)}
SDD = 500.0  # mm
PIXEL = 0.055  # mm
ENERGY = 9.0  # keV
THETA_STEP = 0.01  # deg


def make_volume(shape: tuple[int, int, int], seed: int = 0) -> np.ndarray:
    """(frame, y, x) counts, float64 like the arrays the pipeline receives."""
    c = (np.array(shape) - 1) / 2
    sigma = np.array(shape) / 32
    z, y, x = (np.arange(n) for n in shape)
    gz = np.exp(-((z - c[0]) ** 2) / (2 * sigma[0] ** 2))
    gy = np.exp(-((y - c[1]) ** 2) / (2 * sigma[1] ** 2))
    gx = np.exp(-((x - c[2]) ** 2) / (2 * sigma[2] ** 2))
    vol = 5000.0 * gz[:, None, None] * gy[None, :, None] * gx[None, None, :]
    return np.random.default_rng(seed).poisson(vol + 0.05).astype(np.float64)


def staged_correct_rocking(frames, beam_center, thetas):
    """The same steps as cxd.core.transform.correct_rocking, timed one by one."""
    t = {}
    clock = time.perf_counter
    num_omega, ny, nx = frames.shape
    k0 = 2 * np.pi / wavelength(ENERGY)

    t0 = clock()
    center_theta = np.interp(beam_center[0], np.arange(len(thetas)), thetas)
    y_idx, x_idx = np.indices((ny, nx))
    dy = (y_idx - beam_center[1]) * PIXEL
    dx = (x_idx - beam_center[2]) * PIXEL
    dist = np.sqrt(dx**2 + dy**2 + SDD**2)
    kf_lab = k0 * np.stack((dx / dist, dy / dist, SDD / dist), axis=0)
    q_lab_flat = (kf_lab - np.array([0, 0, k0]).reshape(3, 1, 1)).reshape(3, -1)
    t["detector q_lab"] = clock() - t0

    t0 = clock()
    omegas = np.deg2rad(thetas - center_theta)
    cos_w, sin_w = np.cos(-omegas), np.sin(-omegas)
    r_inv = np.zeros((num_omega, 3, 3))
    r_inv[:, 0, 0] = cos_w
    r_inv[:, 0, 2] = sin_w
    r_inv[:, 1, 1] = 1.0
    r_inv[:, 2, 0] = -sin_w
    r_inv[:, 2, 2] = cos_w
    q_crystal = np.einsum("oij,jk->oik", r_inv, q_lab_flat)
    t["rotation einsum"] = clock() - t0

    t0 = clock()
    points = np.transpose(q_crystal, (0, 2, 1)).reshape(-1, 3)
    t["transpose copy"] = clock() - t0

    t0 = clock()
    out, _ = np.histogramdd(points, bins=[num_omega, ny, nx], weights=frames.ravel())
    t["histogramdd"] = clock() - t0
    return out, t


def bincount_regrid(frames, beam_center, thetas):
    """Candidate CPU route: compute each voxel's bin index and accumulate with np.bincount.

    Same q mapping as correct_rocking; bins span the data range like np.histogramdd does.
    """
    n, ny, nx = frames.shape
    k0 = 2 * np.pi / wavelength(ENERGY)
    y_idx, x_idx = np.indices((ny, nx))
    dy = (y_idx - beam_center[1]) * PIXEL
    dx = (x_idx - beam_center[2]) * PIXEL
    dist = np.sqrt(dx**2 + dy**2 + SDD**2)
    kf = k0 * np.stack((dx / dist, dy / dist, SDD / dist))
    q = (kf - np.array([0, 0, k0]).reshape(3, 1, 1)).reshape(3, -1)
    w = np.deg2rad(thetas - np.interp(beam_center[0], np.arange(n), thetas))
    c, s = np.cos(-w), np.sin(-w)
    comps = (
        (c[:, None] * q[0] + s[:, None] * q[2], n),
        (np.broadcast_to(q[1], (n, ny * nx)), ny),
        (-s[:, None] * q[0] + c[:, None] * q[2], nx),
    )
    flat = 0
    for a, nb in comps:
        lo, hi = a.min(), a.max()
        i = np.minimum(((a - lo) / (hi - lo) * nb).astype(np.intp), nb - 1)
        flat = flat * nb + i
    return np.bincount(flat.ravel(), weights=frames.ravel(), minlength=frames.size).reshape(
        frames.shape
    )


def best_of(fn: Callable[[], object], repeat: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def peak_mib(fn: Callable[[], object]) -> float:
    tracemalloc.start()
    fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak / 2**20


def run(name: str, shape: tuple[int, int, int], repeat: int) -> None:
    frames = make_volume(shape)
    n = frames.size
    thetas = 30.0 + THETA_STEP * np.arange(shape[0])
    center = np.array([shape[0] / 2, shape[1] / 2, shape[2] / 2])

    def full():
        return correct_rocking(frames, center, thetas, SDD, ENERGY, PIXEL)

    ref = full()
    staged, _ = staged_correct_rocking(frames, center, thetas)
    assert np.array_equal(ref, staged), "staged copy differs from correct_rocking"
    assert np.isclose(ref.sum(), frames.sum(), rtol=1e-12), "total counts not conserved"

    print(
        f"\n## {name}: shape {shape}, {n / 1e6:.2f} Mvoxel, frames {frames.nbytes / 2**20:.0f} MiB"
    )
    total = best_of(full, repeat)
    print(f"correct_rocking total      {total:8.3f} s   {n / total / 1e6:6.2f} Mvoxel/s")
    stages = [staged_correct_rocking(frames, center, thetas)[1] for _ in range(repeat)]
    for key in stages[0]:
        s = min(st[key] for st in stages)
        print(f"  {key:<22} {s:8.3f} s   {100 * s / total:5.1f} %")
    print(f"correct_rocking peak RAM   {peak_mib(full):8.0f} MiB (tracemalloc)")

    def alt():
        return bincount_regrid(frames, center, thetas)

    equal = np.array_equal(alt(), ref)
    t_alt = best_of(alt, repeat)
    print(f"bincount_regrid            {t_alt:8.3f} s   {total / t_alt:5.2f}x   identical: {equal}")
    print(f"bincount_regrid peak RAM   {peak_mib(alt):8.0f} MiB (tracemalloc)")

    def zoom():
        return correct_to_qspace(ref, 1.5, 1.0, 1.0)

    print(f"correct_to_qspace (zoom)   {best_of(zoom, repeat):8.3f} s")
    com = np.array(ndimage.center_of_mass(ref))
    max_r = min(shape) / 2

    def spherical():
        return get_full_spherical_map(ref, com, max_r)

    print(f"get_full_spherical_map     {best_of(spherical, repeat):8.3f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--scales", nargs="+", choices=SCALES, default=list(SCALES))
    args = ap.parse_args()
    for scale in args.scales:
        run(scale, SCALES[scale], args.repeat)
