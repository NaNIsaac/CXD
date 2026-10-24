"""Detector frame loaders: MATLAB .mat stacks, TIFF frames, HDF5 datasets.

Part of the I/O layer (``cxd.io``). h5py and tifffile come with ``cxd[io]`` and are imported
inside the functions that use them.
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scipy.io import loadmat

__all__ = ["load_h5", "load_mat", "load_tif_stack"]


def load_mat(file: str | Path, key: str = "data") -> np.ndarray:
    """Read a (y, x, frame) array from a .mat file and return it as (frame, y, x)."""
    mat = loadmat(str(file))
    if key not in mat:
        raise KeyError(f"'{key}' key not found in the .mat file: {file}")
    return np.transpose(mat[key], (2, 0, 1))


def load_tif_stack(files: Iterable[str | Path]) -> np.ndarray:
    """Stack single-frame TIFF files into one (frame, y, x) array, in the given order."""
    import tifffile

    return np.stack([tifffile.imread(f) for f in files])


def load_h5(file: str | Path, dataset: str) -> np.ndarray:
    """Read one dataset from an HDF5 file.

    Import hdf5plugin first for bitshuffle/LZ4-compressed data (PAL-XFEL, Eiger).
    """
    import h5py

    with h5py.File(file, "r") as f:
        return f[dataset][()]
