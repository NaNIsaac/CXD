# CXD

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Analysis code for coherent X-ray diffraction (CXD) data: SPEC file parsing, Timepix TIFF frame loading, rocking-curve to q-space regridding, and 3D q-space visualization.**

In a Bragg coherent X-ray diffraction (BCDI) measurement a small crystal is rocked through a Bragg reflection while an area detector records a coherent diffraction pattern at each angle; the stack of frames samples the 3D diffraction pattern around one reciprocal-lattice point. This package is for reducing such scans recorded with SPEC and a Timepix detector: it reads the scan metadata and frames, maps detector pixels and rocking angles to q, shows the result in 3D, finds facet/streak directions, and sorts reconstructed amplitudes into true and twin-image groups.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```
uv sync
```

## Usage

The SPEC parser, frame loaders and geometry live in [QoraGeom](https://github.com/NaNIsaac/QoraGeom) (`qorageom`), installed from git as a dependency. `cxd.core.spec` re-exports `Spec`, `SpecScan` and adds `get_positions`.

```python
from cxd.core.spec import Spec, get_positions

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

Scripts take their inputs as arguments (`--help` lists them). Run from the repo root:

```
uv run python -m scripts.prep_frames data/S3358 merged/scan_3358.npy
uv run python -m scripts.analyze_raw <specfile> <data_dir> <npy_folder> 3105 3089
uv run python -m scripts.rocking_curve <specfile> 1104 879 --save rc.png
uv run python -m scripts.fwhm_vs_q <specfile> data1.csv data2.csv --energy 5.55
uv run python -m scripts.streak_profiles scan_0255.mat --save-dir out --energy 5.55 --delta 69
uv run python -m scripts.orientation_map scan_0255.mat --energy 5.55 --tth 59 --sdd 5110
uv run python -m cxd.gam_del <specfile> 3884 3893 3913
```

## Structure

- `src/cxd/core/spec.py` - `get_positions`; `Spec`, `SpecScan` from `qorageom.spec`
- `src/cxd/core/transform.py` - rocking-angle correction and q-space regridding
- `src/cxd/core/visualize.py` - PyVista 3D q-space viewer
- `src/cxd/mapping/spherical.py` - (cos θ, φ) histogram around the CoM, DBSCAN streak directions
- `src/cxd/mapping/profiles.py` - thick line profiles along paired opposite directions
- `src/cxd/mapping/vtk_view.py` - VTK render of the pattern in reciprocal-space coordinates
- `src/cxd/mapping/qspace.py` - isotropic q resampling, central-peak mask, r²-weighted θ/φ orientation map, local-max peaks
- `src/cxd/mapping/orientation_view.py` - Lambert projection and PyVista 3D view of the orientations
- `src/cxd/twin_finder.py` - split reconstructed amplitudes into true/twin groups (FFT cross-correlation + KMeans)
- `src/cxd/gam_del.py` - interactive gamma/delta vs. theoretical 2theta check
- `scripts/` - `prep_frames`, `analyze_raw`, `rocking_curve`, `fwhm_vs_q`, `streak_profiles`, `orientation_map`

## Tests

```
uv run pytest
```

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
