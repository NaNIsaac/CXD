# Design intent

## Purpose

CXD reduces Bragg coherent X-ray diffraction rocking scans recorded with SPEC and a Timepix
detector. It maps detector frames and rocking angles to q, shows the result in 3D, finds streak and
facet directions, and sorts reconstructed amplitudes into true and twin-image groups.

## Responsibilities

- `cxd.core`: rocking scan to q with the detector angles and bin edges (`rocking`), the
  detector-frame crop and mask for phase retrieval (`handoff`), centre of mass (`statistics`), the
  older `correct_rocking` (`transform`), and the `Spec` re-export plus `get_positions` (`spec`).
- `cxd.mapping`: the (cos θ, φ) histogram and DBSCAN streak directions (`spherical`, `analysis`),
  thick line profiles (`profiles`), isotropic q resampling and the orientation map (`qspace`).
- `cxd.twin_finder`: true and twin groups by FFT cross-correlation and KMeans.
- Viewers: `core.visualize`, `mapping.profiles_view`, `mapping.orientation_view`,
  `mapping.vtk_view`.
- `scripts/` and `examples/`: command-line workflows and the synthetic scan. They are not part of
  the package.

## Planned boundary

SPEC parsing and frame loading belong to CXD; QoraGeom keeps the diffractometer geometry only.
The target layout is `cxd.io` (the SPEC parser, `geometry_from_spec`, the frame loaders and
`positions`, with the extra `cxd[io]`) and `cxd.viz` (the pyvista, vtk and matplotlib views, with
the extra `cxd[viz]`). The core imports neither. No compatibility re-exports are planned for the
old paths. Until the move, the readers still come from `qorageom.spec` and `qorageom.io` (see
Known gaps).

## Non-goals

- Diffractometer geometry and q of a detector pixel: these are QoraGeom's.
- Phase retrieval: QoraPhase does it; CXD only prepares its input (`to_qoraphase`).
- Plots that need only a SPEC file: splotter (`cxd.gam_del` is a shim that points there).
- A beamline data-reduction pipeline: QoraFlow.
- Measured data in the repository.

## Dependencies

Allowed at runtime: numpy, scipy, scikit-learn and qorageom. pandas is used only by `core.spec`
and belongs with the readers. matplotlib and pyvista are allowed only in the viewer modules.

Never imported by the package:

- `qoraphase`. It is a dev dependency for `tests/test_handoff.py`, so CXD and QoraPhase can be
  installed and released apart.
- A viewer library (matplotlib, pyvista, vtk) in a numerical module, so the analysis runs headless
  and on a cluster.
- A file reader inside numerical code: numerical functions take arrays and plain metadata, and
  reading files stays in the reader layer and `scripts/`.

`tifffile` and `roi-rectangle` are used by `scripts/` and `examples/` only and sit in the
`scripts` dependency group.

`tests/test_import_boundary.py` checks these rules: no numerical module loads a viewer library,
none loads the SPEC parser, the package never loads qoraphase, and every runtime dependency is
imported by the package. Known gaps are marked `xfail(strict=True)`, so a fix turns the test red
until the mark goes.

## Boundaries to sibling libraries

- QoraGeom: q of a detector pixel, diffractometer conventions, wavelength and the scan geometry.
  CXD calls them and does not copy them. A new geometry goes there; a new file format goes into
  CXD's reader layer.
- cxd_map: its modules were moved into `cxd.mapping`, which is the home for that code. New
  mapping work goes into `cxd.mapping`.
- cdiutils: not used by CXD.
- QoraPhase: the handoff contract is the table in the README (axes, origin, values, mask). A change
  on either side updates that table and `tests/test_handoff.py`.

## Conventions

- Arrays are `(frame, y, x)`: rocking first, then detector rows and columns.
- Angles in degrees, sample-detector distance and pixel size in mm, q in 1/Å with 2π included.
- When a result changes, the old function stays with a `DeprecationWarning` (`correct_rocking`),
  so earlier analyses can be reproduced. The old import paths that still forward with a warning
  (`cxd.core.spec`, `cxd.gam_del`, the plot names once in `mapping.profiles`) go when the planned
  boundary is in place.
- Each function that takes or returns a mask states its polarity in its docstring until the
  polarity is unified.
- Test data is synthetic and seeded (`examples/synthetic_scan.py`).

## Criteria for a new feature

- It is scan analysis for coherent diffraction and does not belong to QoraGeom or QoraPhase.
- Numerical code takes and returns arrays; file reading and plotting live in separate modules.
- It has a test on synthetic data, with expected values from a closed form where one exists.
- A new runtime dependency is imported by the package and declared in `pyproject.toml`.
- A change to physics, units or fitting says so in the pull request and keeps the old behaviour
  reachable with a `DeprecationWarning`.

## Known gaps

- Readers not moved yet: the SPEC parser, `geometry_from_spec` and the frame loaders are still in
  `qorageom.spec` and `qorageom.io`; `cxd.core.spec` re-exports the parser, and `scripts/`, the
  notebooks and `tests/test_pipeline.py` import them from QoraGeom. There is no `cxd.io` or
  `cxd[io]` yet.
- Viewers not split yet: the views sit in `core.visualize` and `mapping/*_view.py`, and pyvista and
  matplotlib are runtime dependencies. There is no `cxd.viz` or `cxd[viz]` yet.

- Mask polarity: `core.handoff` uses `True` = unusable, `mapping.qspace` uses `True` = kept
  ([#25](https://github.com/NaNIsaac/CXD/issues/25)).
- `core.visualize` imports pyvista at import time inside the `core` package
  ([#26](https://github.com/NaNIsaac/CXD/issues/26)).
- `mapping.vtk_view` imports vtk, which is not declared and arrives through pyvista
  ([#27](https://github.com/NaNIsaac/CXD/issues/27)).
- `rocking`, `transform` and `qspace` load `qorageom.spec` through the `qorageom` package init
  (the strict xfails in `tests/test_import_boundary.py`).
