# CXD

Analysis code for coherent X-ray diffraction (CXD) data: SPEC file parsing, Timepix TIFF frame loading, rocking-curve to q-space regridding, and 3D q-space visualization.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```
uv sync
```

## Usage

Read motor positions and scan data from a SPEC file:

```python
from cxd.core.spec import Spec

with Spec("path/to/specfile") as sp:
    scan = sp.get_scan(3105)
    scan.motors          # {"Theta": ..., "Delta": ..., "Gamma": ...}
    scan.sdd             # sample-detector distance from #UXML
    df = scan.to_dataframe()
```

Regrid a rocking scan into q-space and view it:

```python
from cxd.core.transform import correct_rocking
from cxd.core.visualize import visualize_q_space_vtk

q = correct_rocking(frames, beam_center, thetas, sdd, beam_energy, pixel_size)
visualize_q_space_vtk(q, mode="contour", log=True)
```

The scripts in `scripts/` and `src/cxd/gam_del.py` use hardcoded data paths in `main()`; edit them before running, e.g. `uv run python -m scripts.analyze_raw` from the repo root.

## Structure

- `src/cxd/core/spec.py` - SPEC file parser (`Spec`, `SpecScan`, `get_positions`)
- `src/cxd/core/io_util.py` - load `.mat` stacks and raw TIFF frames
- `src/cxd/core/transform.py` - rocking-angle correction and q-space regridding
- `src/cxd/core/visualize.py` - PyVista 3D q-space viewer
- `src/cxd/twin_finder.py` - split reconstructed amplitudes into true/twin groups (FFT cross-correlation + KMeans)
- `src/cxd/gam_del.py` - interactive gamma/delta vs. theoretical 2theta check
- `scripts/prep_frames.py` - merge a scan's TIFFs into one `.npy`
- `scripts/analyze_raw.py` - 2theta of a peak and intensity curves per run

## Tests

```
uv run pytest
```
