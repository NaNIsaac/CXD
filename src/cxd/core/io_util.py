from pathlib import Path

import tifffile
from scipy.io import loadmat
import numpy as np


def load_cxd_mat(file: str | Path) -> np.ndarray:
    file: str = str(file)
    if 'data' not in loadmat(file):
        raise KeyError(f"'data' key not found in the .mat file: {file}")
    mat_imgs = loadmat(file)['data']

    return np.transpose(mat_imgs, (2, 0, 1))

def load_raw_tifs(tif_fils: list[Path | str]) -> np.ndarray:
    frames = []
    for tif_file in tif_fils:
        frame = tifffile.imread(tif_file)
        frames.append(frame)
    frames: np.ndarray = np.stack(frames)
    return frames
