import argparse
from pathlib import Path

import numpy as np
from qorageom import shear_correction, wavelength
from qorageom.io import load_mat
from scipy.ndimage import center_of_mass, gaussian_filter

from cxd.mapping.profiles import get_symmetric_profiles, plot_origin_style
from cxd.mapping.spherical import extract_peak_from_2d_map, map2spherical
from cxd.mapping.vtk_view import vtk_visualize_cxd


def main():
    parser = argparse.ArgumentParser(
        description="Streak directions of a BCDI ROI stack: line profiles (plot_<stem>.png) and a 3D view (vtk_<stem>.png)."
    )
    parser.add_argument(
        "mat_files", type=Path, nargs="+", help="ROI .mat files with 'data' (y, x, frame)"
    )
    parser.add_argument("--save-dir", type=Path, default=Path("."))
    parser.add_argument("--energy", type=float, default=5.55, help="keV")
    parser.add_argument("--rocking-step", type=float, default=0.8 / 120, help="deg per frame")
    parser.add_argument(
        "--distance", type=float, default=0.511, help="sample-detector distance [m]"
    )
    parser.add_argument("--gamma", type=float, default=0.0, help="deg")
    parser.add_argument("--delta", type=float, default=69.0, help="deg")
    parser.add_argument("--pixel", type=float, default=5.5e-5, help="pixel size [m]")
    parser.add_argument("--img-thres", type=float, default=2.5)
    parser.add_argument("--hist-thres", type=float, default=1.5)
    args = parser.parse_args()

    B_recip = shear_correction(
        wavelength=wavelength(args.energy) * 1e-10,
        delta_theta=np.radians(args.rocking_step),
        D=args.distance,
        gamma=np.radians(args.gamma),
        delta=np.radians(args.delta),
        pixel_size=args.pixel,
    )

    for data_file in args.mat_files:
        raw_imgs: np.ndarray = load_mat(data_file)
        print(f"Images shape: {raw_imgs.shape}")

        imgs = raw_imgs > args.img_thres
        com = np.array(center_of_mass(imgs))
        print(f"Center of Mass: {com}")
        u_vecs, hist, u_edges, phi_edges = map2spherical(imgs, com, bins=150)
        hist = np.where(hist > args.hist_thres, hist, 0)
        hist = gaussian_filter(hist, sigma=1)
        peaks = extract_peak_from_2d_map(hist, u_edges, phi_edges, eps=0.2, threshold_rel=0.3)
        print(f"검출된 피크 개수: {len(peaks)}")

        results = get_symmetric_profiles(imgs, com, peaks, length=50, thickness=3)
        fig = plot_origin_style(results, show=False)
        fig.savefig(args.save_dir / f"plot_{data_file.stem}.png", dpi=300, bbox_inches="tight")
        vtk_visualize_cxd(
            imgs,
            com,
            results,
            peaks,
            B_recip,
            save_path=args.save_dir / f"vtk_{data_file.stem}.png",
            show=False,
        )


if __name__ == "__main__":
    main()
