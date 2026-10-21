"""Rocking scan to reciprocal space with the detector angles, through QoraGeom's diffractometers.

Lab frame and circles follow ``qorageom.Diffractometer`` (beam along +z; for "34idc" the
detector arm turns delta about +y after gamma about −x, and the sample circles are th, chi, phi
with th outermost). For a detector pixel at (y, x) and a sample rocked to angle ω_n:

    P     = R_det(δ, γ) · (L ẑ + p (x − x0) î + p (y − y0) ĵ)        pixel position
    q_lab = k P / |P| − k ẑ,   k = 2π / λ                          (qorageom.pixel_to_q_lab)
    q     = R_s(ω_c) R_s(ω_n)ᵀ q_lab                               crystal frame at the centre frame

î, ĵ are the lab directions the detector columns and rows advance along at zero detector angles
(``orientation``). The last line is the change of the q a fixed pixel samples as the sample
turns, the convention of ``Diffractometer.rocking_basis``; at the centre frame q = q_lab. q is in
1/Å with the 2π included, like the rest of CXD. Closed-form checks: the centre pixel has
|q| = 2k sin θ with cos 2θ = cos δ cos γ, and one rocking step of Δω moves a q by
2 |q⊥| sin(Δω / 2), q⊥ its component perpendicular to the rocking axis (|q⊥| = |q| when gamma
is 0 and the axis is th).

The points are then binned without interpolation: each bin holds the mean intensity of the
(frame, pixel) samples that fall into it, and ``hits`` says how many did, so empty bins (0) are
told apart from measured zeros.
"""

from dataclasses import dataclass

import numpy as np
from qorageom import get_diffractometer, pixel_to_q_lab
from qorageom.diffractometer import axis_vector


@dataclass(frozen=True)
class QSpaceMap:
    """Binned rocking scan. Axes are (q_x, q_y, q_z) of the crystal frame, in 1/Å."""

    intensity: np.ndarray  # mean intensity per bin, 0 where nothing was measured
    hits: np.ndarray  # (frame, pixel) samples per bin
    edges: tuple[np.ndarray, np.ndarray, np.ndarray]  # bin edges along q_x, q_y, q_z

    @property
    def centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(0.5 * (e[1:] + e[:-1]) for e in self.edges)


def rocking_q(
    shape: tuple[int, int, int],
    thetas: np.ndarray,
    delta: float,
    gamma: float,
    center: tuple[float, float, float],
    sdd: float,
    pixel_size: float,
    energy_kev: float,
    *,
    diffractometer: str = "34idc",
    rock: str = "th",
    sample_motors_deg: tuple[float, ...] | None = None,
    orientation: tuple[str, str] = ("x+", "y+"),
) -> np.ndarray:
    """q [1/Å] of every (frame, y, x) sample of a rocking scan, shape ``shape + (3,)``.

    shape: (frames, ny, nx). thetas: reading of the rocked motor per frame [deg].
    delta, gamma: detector motor readings [deg]. center: (frame, y, x) of the reference, the
    pixel the detector arm points at and the frame whose sample orientation defines the crystal
    frame; the frame index may be fractional (the angle is interpolated). sdd and pixel_size
    share one length unit. sample_motors_deg: readings of all sample motors at the centre frame,
    in the order of the diffractometer's motors; defaults to readings that put every circle
    at 0 except the rocked one.
    """
    n_frames, ny, nx = shape
    thetas = np.asarray(thetas, dtype=float)
    if thetas.shape != (n_frames,):
        raise ValueError(f"{n_frames} frames but {thetas.size} rocking angles")
    geometry = get_diffractometer(diffractometer)
    if rock not in geometry.sample_motors:
        raise ValueError(f"cannot rock {rock!r}; sample motors: {geometry.sample_motors}")
    offsets = np.array(geometry.sample_offsets_deg or np.zeros(len(geometry.sample_axes)))
    motors = offsets.copy() if sample_motors_deg is None else np.array(sample_motors_deg, float)
    r = geometry.sample_motors.index(rock)

    def sample_rotation(angle):
        circles = motors - offsets
        circles[r] = angle - offsets[r]
        return geometry.sample_rotation(circles)

    beam = np.asarray(geometry.incident_axis, dtype=float)
    R_det = geometry.detector_rotation([delta, gamma])
    columns, rows = (R_det @ axis_vector(o) for o in orientation)
    yy, xx = np.meshgrid(np.arange(ny) - center[1], np.arange(nx) - center[2], indexing="ij")
    q_lab = pixel_to_q_lab(
        energy_kev, beam, R_det @ (sdd * beam), columns, rows, pixel_size, xx, yy
    )

    reference = sample_rotation(np.interp(center[0], np.arange(n_frames), thetas))
    turns = np.stack([reference @ sample_rotation(t).T for t in thetas])  # (frames, 3, 3)
    return np.einsum("fab,yxb->fyxa", turns, q_lab)


def map_rocking_scan(
    frames: np.ndarray,
    thetas: np.ndarray,
    delta: float,
    gamma: float,
    center: tuple[float, float, float],
    sdd: float,
    pixel_size: float,
    energy_kev: float,
    *,
    bins: int | tuple[int, int, int] | None = None,
    **geometry,
) -> QSpaceMap:
    """Bin a (frame, y, x) rocking stack onto a q_x, q_y, q_z grid; see ``rocking_q``.

    bins: count per axis, default the stack's shape; the grid spans the measured q.
    Further keyword arguments (diffractometer, rock, sample_motors_deg, orientation) go to
    ``rocking_q``.
    """
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 3:
        raise ValueError(f"frames must be (frame, y, x); got shape {frames.shape}")
    q = rocking_q(
        frames.shape, thetas, delta, gamma, center, sdd, pixel_size, energy_kev, **geometry
    )
    points = q.reshape(-1, 3)
    counts, edges = np.histogramdd(points, bins=bins or frames.shape)
    total, _ = np.histogramdd(points, bins=edges, weights=frames.ravel())
    mean = np.divide(total, counts, out=np.zeros_like(total), where=counts > 0)
    return QSpaceMap(mean, counts.astype(np.int64), tuple(edges))
