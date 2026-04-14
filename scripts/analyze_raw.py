from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from roi_rectangle import RoiRectangle

from cxd.core.constants import TIMEPIX_PIXEL_SIZE
from cxd.core.spec import Spec
from cxd.core.statistics import get_com
from scripts.prep_frames import merge_raw_tif_to_npy


def get_positions(run_ns: list[int], spec_file: Path) -> pd.DataFrame:
    data = []
    with Spec(spec_file) as sp:
        for run_n in run_ns:
            scan = sp.get_scan(run_n)
            row = {"run_n": run_n, **scan.motors}
            row['sdd'] = scan.sdd
            row['bc'] = scan.beam_center
            data.append(row)
    df: pd.DataFrame = pd.DataFrame(data).set_index('run_n')
    return df

def get_tth_of_peak(frame, rect: RoiRectangle, delta, gamma, sdd, bc_y, bc_x) -> float:
    roi_frame = rect.slice(frame)
    roi_com = get_com(roi_frame)

    cy = rect.y1 + roi_com[0]
    cx = rect.x1 + roi_com[1]

    dy = (cy - bc_y) * TIMEPIX_PIXEL_SIZE
    dx = (cx - bc_x) * TIMEPIX_PIXEL_SIZE
    dtth_v = np.rad2deg(np.arctan2(dy, sdd))
    dtth_h = np.rad2deg(np.arctan2(dx, sdd))
    true_gamma = gamma + dtth_v
    true_delta = delta + dtth_h
    tth = np.rad2deg(np.arccos(
        np.cos(np.deg2rad(true_delta)) * np.cos(np.deg2rad(true_gamma))
    ))
    return tth

def plot_curve(names, xs, ys):
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for name, x, y in zip(names, xs, ys):
        ax.plot(x, y, label=name)
    
    border_width = 2.0
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(border_width)

    ax.set_xlabel("$Angle [\\circ]$", fontsize=14, fontweight='bold')
    ax.set_ylabel("Intensity [A.U]", fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.tick_params(
        direction='in',
        length=6,
        width=border_width,
        labelsize=12,
        top=True,
        right=True
    )
    ax.legend(frameon=True, fontsize=12)

def main():
    spec_file: Path = Path(r"D:\05_Resources\Data\9C_260324\260324")
    npy_folder: Path = Path(r"D:\05_Resources\Data\9C_260324\analysis\merged")
    data_dir: Path = Path(r"D:\05_Resources\Data\9C_260324\data")
    run_ns: list[int] = [3105, 3089]

    df = get_positions(run_ns, spec_file)

    run_tth_map: dict[int, dict[str, np.ndarray]] = {}
    sp: Spec = Spec(spec_file)
    for run_n in run_ns:
        npy_file: Path = npy_folder / f"scan_{run_n:04d}.npy"
        tif_dir: Path = data_dir / f"S{run_n:04d}"
        merge_raw_tif_to_npy(tif_dir, npy_file)
        
        theta = df.loc[run_n, 'Theta']
        delta = df.loc[run_n, 'Delta']
        # chi = df.loc[run_n, 'Chi']
        gamma = df.loc[run_n, 'Gamma']
        sdd = df.loc[run_n, 'sdd']
        bc = df.loc[run_n, 'bc']

        frames = np.load(npy_file)
        mid_frame = frames[len(frames) // 2]
        rect: RoiRectangle = RoiRectangle(0, 256, 256, 512)
        tth = get_tth_of_peak(mid_frame, rect, delta, gamma, sdd, bc["y"],bc["x"])

        scan = sp.get_scan(run_n)
        sdf = scan.to_dataframe()
        thetas = sdf.iloc[:, 0].to_numpy()
        intensity = sdf["imroi1"].to_numpy()
        rel_thetas = thetas - theta
        
        # TODO: Check the scientific meaning
        new_tths = tth + 2 * rel_thetas
        run_tth_map[run_n] = {"tth": new_tths, "intensity": intensity}

    names = list(map(str, run_ns))
    tths = list(map(lambda n: run_tth_map[n]['tth'], run_ns))
    intensities = list(map(lambda n: run_tth_map[n]['intensity'], run_ns))

    plot_curve(names, tths, intensities)
    plt.show()


if __name__ == '__main__':
    main()
