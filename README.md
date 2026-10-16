# CXD

Analysis code for coherent X-ray diffraction (CXD) data: SPEC file parsing, Timepix TIFF frame loading, rocking-curve to q-space regridding, and 3D q-space visualization.

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
- `src/cxd/mapping/spherical.py` - (cos θ, φ) histogram around the CoM, DBSCAN streak directions (from cxd_map)
- `src/cxd/mapping/profiles.py` - thick line profiles along paired opposite directions (from cxd_map)
- `src/cxd/mapping/vtk_view.py` - VTK render of the pattern in reciprocal-space coordinates (from cxd_map)
- `src/cxd/mapping/qspace.py` - isotropic q resampling, central-peak mask, r²-weighted θ/φ orientation map, local-max peaks (from bcdi_mapping)
- `src/cxd/mapping/orientation_view.py` - Lambert projection and PyVista 3D view of the orientations (from bcdi_mapping)
- `src/cxd/twin_finder.py` - split reconstructed amplitudes into true/twin groups (FFT cross-correlation + KMeans)
- `src/cxd/gam_del.py` - interactive gamma/delta vs. theoretical 2theta check
- `scripts/` - `prep_frames`, `analyze_raw`, `rocking_curve` and `fwhm_vs_q` (from spec_manager), `streak_profiles` (cxd_map), `orientation_map` (bcdi_mapping)

## Tests

```
uv run pytest
```

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
