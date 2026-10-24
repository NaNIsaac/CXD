import argparse
from pathlib import Path

import numpy as np

from cxd.io.frames import load_tif_stack


def merge_raw_tif_to_npy(tif_dir: Path | str, save_file: Path | str) -> None:
    frames: np.ndarray = load_tif_stack(sorted(Path(tif_dir).glob("*.tif")))
    np.save(save_file, frames)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge one scan's TIFF frames into a single .npy (frame, y, x)."
    )
    parser.add_argument(
        "tif_dir", type=Path, help="folder with the scan's *.tif frames, e.g. examples/data/S0001"
    )
    parser.add_argument(
        "save_file", type=Path, help="output .npy, e.g. examples/data/scan_0001.npy"
    )
    args = parser.parse_args()
    merge_raw_tif_to_npy(args.tif_dir, args.save_file)


if __name__ == "__main__":
    main()
