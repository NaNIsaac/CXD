from pathlib import Path
import numpy as np
from scipy import ndimage

from io_util import get_matarr
from processing import calculate_qspace_params, correct_to_qspace, get_radial_mask_radius, find_robust_peaks, extract_orientation_map
from visualizer import plot_full_lambert_projection, plot_3d_pattern_with_orientations

def main() -> None:
    root = Path(r"D:\05_Resources\Data\9C_260324\onsite\ROI")
    mat_file = root / "scan_0255.mat"

    raw_data = get_matarr(mat_file)
    print(f"Loaded Raw Shape: {raw_data.shape}")

    dq_params = calculate_qspace_params(
        energy_kev=5.55,
        tth_deg=59.0,
        sdd_mm=5110,
        pixel_mm=0.055,
        omega_step_deg=0.005
    )
    
    q_imgs, dq_res = correct_to_qspace(raw_data, *dq_params)
    print(f"Corrected Shape: {q_imgs.shape} (dq={dq_res:.6f} A^-1)")

    com = ndimage.center_of_mass(q_imgs)
    
    z, y, x = np.ogrid[:q_imgs.shape[0], :q_imgs.shape[1], :q_imgs.shape[2]]
    r_dist = np.sqrt((z - com[0])**2 + (y - com[1])**2 + (x - com[2])**2)
    q_imgs_masked = q_imgs.copy()
    q_imgs_masked[r_dist < 10] = 0 
    r_mask = get_radial_mask_radius(q_imgs, com, sigma_multiplier=1.5)
    print(f"Calculated Mask Radius: {r_mask:.2f}")

    theta_ax, phi_ax, s_map = extract_orientation_map(q_imgs, com, r_mask)

    top_dirs = find_robust_peaks(s_map, theta_ax, phi_ax, threshold=0.5, min_dist=15)

    plot_full_lambert_projection(theta_ax, phi_ax, s_map, peaks=top_dirs[:6])
    plot_3d_pattern_with_orientations(q_imgs, com, top_dirs[:6], scale=1.5)

if __name__ == "__main__":
    main()