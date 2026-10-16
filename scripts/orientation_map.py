import argparse
from pathlib import Path

from qorageom.io import load_mat
from scipy import ndimage

from cxd.mapping.orientation_view import plot_3d_pattern_with_orientations, plot_full_lambert_projection
from cxd.mapping.qspace import (
    calculate_qspace_params,
    correct_to_qspace,
    extract_orientation_map,
    find_robust_peaks,
    get_radial_mask_radius,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resample a BCDI rocking scan to isotropic q, then map facet/streak directions (Lambert plot + 3D view)."
    )
    parser.add_argument("mat_file", type=Path, help=".mat file with 'data' (y, x, frame)")
    parser.add_argument("--energy", type=float, default=5.55, help="keV")
    parser.add_argument("--tth", type=float, default=59.0, help="deg")
    parser.add_argument("--sdd", type=float, default=5110, help="mm")
    parser.add_argument("--pixel", type=float, default=0.055, help="mm")
    parser.add_argument("--omega-step", type=float, default=0.005, help="deg")
    parser.add_argument("--n-peaks", type=int, default=6)
    args = parser.parse_args()

    raw_data = load_mat(args.mat_file)
    print(f"Loaded Raw Shape: {raw_data.shape}")

    dq_params = calculate_qspace_params(
        energy_kev=args.energy,
        tth_deg=args.tth,
        sdd_mm=args.sdd,
        pixel_mm=args.pixel,
        omega_step_deg=args.omega_step,
    )
    q_imgs, dq_res = correct_to_qspace(raw_data, *dq_params)
    print(f"Corrected Shape: {q_imgs.shape} (dq={dq_res:.6f} A^-1)")

    com = ndimage.center_of_mass(q_imgs)
    r_mask = get_radial_mask_radius(q_imgs, com, sigma_multiplier=1.5)
    print(f"Calculated Mask Radius: {r_mask:.2f}")

    theta_ax, phi_ax, s_map = extract_orientation_map(q_imgs, com, r_mask)
    top_dirs = find_robust_peaks(s_map, theta_ax, phi_ax, threshold=0.5, min_dist=15)

    plot_full_lambert_projection(theta_ax, phi_ax, s_map, peaks=top_dirs[: args.n_peaks])
    plot_3d_pattern_with_orientations(q_imgs, com, top_dirs[: args.n_peaks], scale=1.5)


if __name__ == "__main__":
    main()
