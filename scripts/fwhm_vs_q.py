import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qorageom import tth_to_q, wavelength
from scipy.optimize import curve_fit

from cxd.core.spec import Spec


def calculate_q_geometry(delta, gamma, energy_kev, g_off=0.0, d_off=0.0):
    """|Q| [1/Å] from detector gamma/delta [deg] minus their offsets; cos 2θ = cos γ · cos δ."""
    gam_corr = np.radians(gamma - g_off)
    del_corr = np.radians(delta - d_off)
    two_theta = np.degrees(np.arccos(np.clip(np.cos(gam_corr) * np.cos(del_corr), -1.0, 1.0)))
    return tth_to_q(two_theta, wavelength(energy_kev))


def gaussian(x, a, xc, sigma, y0):
    return y0 + a * np.exp(-(x - xc)**2 / (2 * sigma**2))


def fit_gaussian_fwhm(x, y):
    """Fits data to Gaussian and returns FWHM."""
    if len(y) < 4:
        return np.nan

    p0 = [np.max(y)-np.min(y), x[np.argmax(y)], (np.max(x)-np.min(x))/10.0, np.min(y)]

    popt, _ = curve_fit(gaussian, x, y, p0=p0, maxfev=2000)
    return 2.3548 * abs(popt[2])


def main():
    parser = argparse.ArgumentParser(description="Gaussian FWHM of rocking scans vs Q, one series per scan-list CSV.")
    parser.add_argument("spec_file", type=Path)
    parser.add_argument("csv_files", type=Path, nargs="+", help="headerless CSV: scan, gamma, delta, theta")
    parser.add_argument("--labels", nargs="+", help="one legend label per CSV (default: file stem)")
    parser.add_argument("--energy", type=float, default=5.55, help="keV")
    parser.add_argument("--gamma-offset", type=float, default=1.4367, help="deg")
    parser.add_argument("--delta-offset", type=float, default=0.0112, help="deg")
    parser.add_argument("--counter", default="imroi1")
    parser.add_argument("--save", type=Path, help="write the figure here instead of showing it")
    args = parser.parse_args()
    labels = args.labels or [f.stem for f in args.csv_files]

    plt.figure(figsize=(10, 7))

    with Spec(args.spec_file) as spec:
        for csv_path, label in zip(args.csv_files, labels, strict=True):
            csv_df = pd.read_csv(csv_path, header=None, names=["scan", "gamma", "delta", "theta"])
            group_results = []

            for _, row in csv_df.iterrows():
                scan = spec.get_scan(int(row['scan']))
                if not scan.data:
                    continue

                q_val = calculate_q_geometry(row['delta'], row['gamma'], args.energy,
                                             g_off=args.gamma_offset, d_off=args.delta_offset)

                df_spec = scan.to_dataframe()
                x_data = df_spec.iloc[:, 0].values.astype(float)
                y_data = df_spec[args.counter].values.astype(float)

                fwhm = fit_gaussian_fwhm(x_data, y_data)

                if not np.isnan(fwhm):
                    group_results.append({'q': q_val, 'fwhm': fwhm, "intensity": sum(y_data)})

            if group_results:
                res_df = pd.DataFrame(group_results).sort_values('q')
                plt.scatter(res_df['q'], res_df['fwhm'], label=label, s=50, edgecolors='k')

    plt.xlabel(r"$Q$ ($\AA^{-1}$)", fontsize=13)
    plt.ylabel(r"fwhm ($^\circ$)", fontsize=13)
    plt.title(fr"Broadening (Offsets: $\gamma={args.gamma_offset}^\circ, \delta={args.delta_offset}^\circ$)", fontsize=13)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches="tight")
    else:
        plt.show()


if __name__ == '__main__':
    main()
