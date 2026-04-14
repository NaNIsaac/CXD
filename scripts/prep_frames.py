from pathlib import Path

import numpy as np

from cxd.core.io_util import load_raw_tifs

def merge_raw_tif_to_npy(tif_dir: Path | str, save_fie: Path | str) -> None:
    tif_dir = Path(tif_dir)
    frames: np.ndarray = load_raw_tifs(sorted(tif_dir.glob("*.tif")))

    np.save(save_fie,frames)


if __name__ == "__main__":
    num: int = 3358
    tif_dir: Path = Path(rf"D:\05_Resources\Data\9C_260324\data\S{num:04d}")
    save_file: Path = Path(r"D:\05_Resources\Data\9C_260324\analysis\merged") / f"scan_{num:04d}.npy"
    merge_raw_tif_to_npy(tif_dir, save_file)