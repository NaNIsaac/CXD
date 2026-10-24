import numpy as np
from qorageom import tth_to_q, wavelength
from scipy import ndimage


def calculate_qspace_params(energy_kev, tth_deg, sdd_mm, pixel_mm, omega_step_deg):
    lam = wavelength(energy_kev)  # A
    k_in = 2 * np.pi / lam  # A^-1
    q_cen = tth_to_q(tth_deg, lam)  # Bragg q

    # detector dq (Y, X)
    dq_pixel = k_in * (pixel_mm / sdd_mm)  # A^-1 / pixel

    # rocking dq (Z)
    dq_rock = q_cen * np.radians(omega_step_deg)

    return dq_rock, dq_pixel, dq_pixel


def correct_to_qspace(raw_imgs, dq_z, dq_y, dq_x):
    dq_min = min(dq_z, dq_y, dq_x)
    zoom_factors = (dq_z / dq_min, dq_y / dq_min, dq_x / dq_min)

    print(f"Resampling q-space (scale: {zoom_factors})")
    corrected_imgs = ndimage.zoom(raw_imgs, zoom_factors, order=1, mode="nearest")
    return corrected_imgs, dq_min


def get_full_spherical_map(imgs, com, max_r, num_theta=180, num_phi=360):
    theta_axis = np.linspace(0, np.pi, num_theta)
    phi_axis = np.linspace(0, 2 * np.pi, num_phi)
    r_axis = np.linspace(0, max_r, int(max_r))

    T, P, R = np.meshgrid(theta_axis, phi_axis, r_axis, indexing="ij")

    # spherical to Cartesian around the COM, ordered (Z, Y, X) as map_coordinates expects
    z_coords = com[0] + R * np.cos(T)
    y_coords = com[1] + R * np.sin(T) * np.sin(P)
    x_coords = com[2] + R * np.sin(T) * np.cos(P)

    coords = np.vstack((z_coords.ravel(), y_coords.ravel(), x_coords.ravel()))

    # max projection along r
    sampled_values = ndimage.map_coordinates(imgs, coords, order=1, mode="nearest")
    sampled_values = sampled_values.reshape(T.shape)
    spherical_map = np.max(sampled_values, axis=2)

    return theta_axis, phi_axis, spherical_map


def find_robust_peaks(spherical_map, theta_ax, phi_ax, threshold=0.6, min_dist=15):

    local_max = spherical_map == ndimage.maximum_filter(spherical_map, size=min_dist)
    high_intensity = spherical_map > np.max(spherical_map) * threshold
    peaks = np.argwhere(local_max & high_intensity)

    results = []
    for r, c in peaks:
        results.append((theta_ax[r], phi_ax[c], spherical_map[r, c]))

    results.sort(key=lambda x: x[2], reverse=True)
    return results


def get_radial_mask_radius(q_imgs, com, sigma_multiplier=3.0):
    """Mask radius of the central blob from the mean intensity per shell, times sigma_multiplier."""
    z, y, x = np.ogrid[: q_imgs.shape[0], : q_imgs.shape[1], : q_imgs.shape[2]]
    r_dist = np.sqrt((z - com[0]) ** 2 + (y - com[1]) ** 2 + (x - com[2]) ** 2)

    # mean intensity per spherical shell
    r_int = r_dist.astype(np.int32)
    radial_sum = ndimage.sum(q_imgs, labels=r_int, index=np.arange(r_int.max() + 1))
    radial_count = ndimage.sum(np.ones_like(q_imgs), labels=r_int, index=np.arange(r_int.max() + 1))
    radial_mean = radial_sum / (radial_count + 1e-9)

    # outer edge of the central blob: the last shell above 10% of the peak shell mean
    threshold = np.max(radial_mean) * 0.1
    mask_indices = np.where(radial_mean > threshold)[0]

    r_mask = mask_indices[-1] if len(mask_indices) > 0 else 10
    return r_mask * sigma_multiplier


def extract_orientation_map(q_imgs, com, r_mask, num_theta=180, num_phi=360):
    z, y, x = np.ogrid[: q_imgs.shape[0], : q_imgs.shape[1], : q_imgs.shape[2]]
    r_dist = np.sqrt((z - com[0]) ** 2 + (y - com[1]) ** 2 + (x - com[2]) ** 2)

    mask = r_dist > r_mask
    valid_z, valid_y, valid_x = np.where(mask & (q_imgs > 0))
    intensities = q_imgs[valid_z, valid_y, valid_x]

    dz, dy, dx = valid_z - com[0], valid_y - com[1], valid_x - com[2]
    r_valid = np.sqrt(dz**2 + dy**2 + dx**2)

    theta = np.arccos(dz / r_valid)
    phi = np.arctan2(dy, dx)

    # weight by r^2 to strengthen streaks far from the center, then log10 to compress the range
    compensated_intensities = intensities * (r_valid**2)
    log_intensities = np.log10(compensated_intensities + 1)

    s_map, t_edges, p_edges = np.histogram2d(
        theta,
        phi,
        bins=[num_theta, num_phi],
        range=[[0, np.pi], [-np.pi, np.pi]],
        weights=log_intensities,
    )

    t_ax = (t_edges[:-1] + t_edges[1:]) / 2
    p_ax = (p_edges[:-1] + p_edges[1:]) / 2

    return t_ax, p_ax, s_map
