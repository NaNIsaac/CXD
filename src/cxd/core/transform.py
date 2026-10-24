import warnings

import numpy as np
from qorageom import wavelength


def correct_rocking(
    frames: np.ndarray,
    beam_center: np.ndarray,  # [omega_idx, y_idx, x_idx]
    thetas: np.ndarray,
    sdd: float,
    beam_energy: float,
    pixel_size: float,
) -> np.ndarray:
    """
    frames: 3D array of shape (num_omega, ny, nx)
    beam_center: 3-element array indicating the center position in (omega_idx, y_idx, x_idx)
    thetas: 1D array of rocking angles corresponding to each omega frame
    sdd: sample-to-detector distance [mm]
    beam_energy: X-ray beam energy [keV]
    pixel_size: size of each pixel in [mm]
    """
    warnings.warn(
        "correct_rocking ignores the detector angles (CXD#10) and is kept to reproduce old "
        "analyses; use cxd.core.rocking.map_rocking_scan",
        DeprecationWarning,
        stacklevel=2,
    )
    if frames.ndim != 3:
        raise ValueError(f"Frames must be a 3D array. Got shape {frames.shape}.")

    num_omega, ny, nx = frames.shape
    wavelen: float = wavelength(beam_energy)
    k0: float = 2 * np.pi / wavelen

    if num_omega != len(thetas):
        raise ValueError(f"Depth of frames {num_omega} mismatch with theta length {len(thetas)}.")

    cen_omega_idx, cen_y, cen_x = beam_center[0], beam_center[1], beam_center[2]

    # the beam center index is fractional, so interpolate theta linearly
    # (1.5 -> mean of thetas[1] and thetas[2])
    center_theta = np.interp(cen_omega_idx, np.arange(len(thetas)), thetas)

    y_idx, x_idx = np.indices((ny, nx))
    dy = (y_idx - cen_y) * pixel_size
    dx = (x_idx - cen_x) * pixel_size

    dist = np.sqrt(dx**2 + dy**2 + sdd**2)

    # kf_lab: (3, ny, nx)
    kf_lab = k0 * np.stack((dx / dist, dy / dist, sdd / dist), axis=0)
    ki_lab = np.array([0, 0, k0]).reshape(3, 1, 1)

    q_lab = kf_lab - ki_lab
    q_lab_flat = q_lab.reshape(3, -1)  # (3, ny*nx)

    # rotate q_lab into the crystal frame by the angle relative to center_theta
    omegas = np.deg2rad(thetas - center_theta)

    cos_w = np.cos(-omegas)
    sin_w = np.sin(-omegas)

    # Rotation Matrix (y-axis rotation assumed)
    # R_inv: (num_omega, 3, 3)
    R_inv = np.zeros((num_omega, 3, 3))
    R_inv[:, 0, 0] = cos_w
    R_inv[:, 0, 2] = sin_w
    R_inv[:, 1, 1] = 1.0
    R_inv[:, 2, 0] = -sin_w
    R_inv[:, 2, 2] = cos_w

    # q_crystal: (num_omega, 3, ny*nx)
    q_crystal = np.einsum("oij,jk->oik", R_inv, q_lab_flat)

    # q_crystal_points: (num_omega * ny * nx, 3)
    q_crystal_points = np.transpose(q_crystal, (0, 2, 1)).reshape(-1, 3)
    intensities = frames.ravel()

    # bins should come from the data range or be a parameter
    bins = [num_omega, ny, nx]

    regridded_q_space, edges = np.histogramdd(q_crystal_points, bins=bins, weights=intensities)

    return regridded_q_space
