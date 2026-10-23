# Profile of the 3D reciprocal-space mapping

Reproduce with `uv run python scripts/bench_rsm.py --repeat 5`. The volume is synthetic (a
Gaussian Bragg peak, Poisson sampled, seed 0, float64), so nothing measured is needed. Geometry:
sdd 500 mm, pixel 0.055 mm, 9 keV, 0.01 deg per frame. Machine: AMD64 Family 25 Model 97 (12 logical cores),
RTX 5070, Windows 11, Python 3.13, NumPy/SciPy from `uv.lock`. Times are the best of 5 runs;
peak memory is from `tracemalloc` (it sees NumPy allocations, not the interpreter).

## What was measured

`correct_rocking` (detector pixels and rocking angles to a q-space histogram) is the regridding
step. `correct_to_qspace` (`ndimage.zoom`) and `get_full_spherical_map` (`map_coordinates` on a
spherical grid) are the two later steps that run on the regridded volume. SPEC and TIFF reading,
and visualization, are not in the benchmark: the volume is built in memory, so disk I/O is not
accounted for here.

| scale | shape (frame, y, x) | voxels | `correct_rocking` | throughput | peak RAM | `correct_to_qspace` | `get_full_spherical_map` |
|---|---|---|---|---|---|---|---|
| small | 32 x 128 x 128 | 0.52 M | 0.036 s | 14 Mvoxel/s | 51 MiB | 0.020 s | 0.073 s |
| medium | 64 x 256 x 256 | 4.19 M | 0.252 s | 17 Mvoxel/s | 397 MiB | 0.161 s | 0.156 s |
| large | 96 x 384 x 384 | 14.16 M | 0.925 s | 15 Mvoxel/s | 1329 MiB | 0.525 s | 0.263 s |

Inside `correct_rocking` (same steps timed one by one; the script checks that this copy returns
exactly the same array):

| step | small | medium | large |
|---|---|---|---|
| detector q_lab | 1.0 % | 0.7 % | 0.5 % |
| rotation (`einsum`) | 6.6 % | 7.3 % | 7.2 % |
| transpose copy to (N, 3) points | 7.2 % | 8.7 % | 9.2 % |
| `np.histogramdd` | 64 % | 81 % | 81 % |

Correctness checks in the script: the regridded total equals the input total (relative 1e-12),
and the staged copy equals `correct_rocking` bit for bit.

## Findings

- `np.histogramdd` is about 80 % of the regridding at medium and large scale. It is single
  threaded and works on an (N, 3) float64 point array: 340 MiB of points for 14 M voxels, plus the
  `einsum` output and the transposed copy. Peak memory is about 98 bytes per voxel (12 float64 values), i.e.
  1.3 GiB at 14 M voxels, and grows linearly with voxel count.
- The detector geometry (`q_lab`) is a fixed 2D computation and is negligible; the rotation is a
  few percent.
- `get_full_spherical_map` is nearly independent of volume size: its cost is set by the
  `num_theta x num_phi x max_r` sampling grid (180 x 360 x `max_r`), not by the voxel count.
- `correct_to_qspace` (linear `zoom`) scales with the number of output voxels.

## CPU candidate, measured

`bincount_regrid` in `scripts/bench_rsm.py` computes the bin index of each voxel directly
(rotate, scale to the data range, flatten) and accumulates with `np.bincount`:

| scale | `correct_rocking` | `bincount_regrid` | speed-up | peak RAM |
|---|---|---|---|---|
| small | 0.036 s | 0.018 s | 2.0x | 25 MiB (was 51) |
| medium | 0.252 s | 0.162 s | 1.6x | 198 MiB (was 397) |
| large | 0.925 s | 0.539 s | 1.7x | 660 MiB (was 1329) |

On this fixture the output equals `correct_rocking` exactly. Only these three volumes were
compared; a voxel that falls on a bin edge can round differently from `histogramdd`, so a
production replacement needs a test on edge cases (voxels on the last edge, constant axes)
before it replaces the current code. This is not applied to `src/` in this change.

## GPU: not justified yet

A GPU was not prototyped, so there is no GPU measurement here. Reasoning from the numbers:

- The regridding is a scatter-add (`bincount`), where many voxels can land in one bin. On a GPU
  that needs atomic adds or a sort, and the result then depends on summation order, so float
  tolerances must be tested instead of exact equality.
- The whole 14 M voxel regridding takes 0.5 to 0.9 s on the CPU. The input is 108 MiB (float64),
  so the host-to-device copy is a small part of it, but the step is already shorter than reading
  that many frames from disk is likely to be; that was not measured here.
- A GPU would pay off only if volumes reach the 10^9 voxel range, or if the regridding is repeated
  many times (parameter scans). Neither is the case in the current workflow.

Next steps, in order: (1) measure the SPEC/TIFF read stage on real-sized files, because whole-
process time decides whether the regridding matters at all; (2) harden and adopt the `bincount`
route with edge-case tests; (3) only then consider a JAX or CuPy scatter-add, comparing
whole-process time including transfers and reporting the cases where the GPU is slower.
