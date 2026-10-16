from pathlib import Path

import numpy as np
from scipy.io import loadmat


def load_roi_mat(file: Path | str) -> np.ndarray:
    matarr = loadmat(file)["data"]
    return np.transpose(matarr, axes=(2, 0, 1))
