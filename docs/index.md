# CXD

CXD is analysis code for Bragg coherent X-ray diffraction (BCDI) scans recorded with SPEC and a
Timepix detector. It reads a scan's metadata and frames, maps detector pixels and rocking angles to
q, shows the result in 3D, finds facet and streak directions, and sorts reconstructed amplitudes
into true and twin-image groups. The SPEC parser, frame loaders and geometry come from
[QoraGeom](https://github.com/NaNIsaac/QoraGeom); phase retrieval is
[QoraPhase](https://github.com/NaNIsaac/QoraPhase), and CXD hands it a detector-frame crop
(`cxd.core.handoff.to_qoraphase`).

## Install

Python 3.13 and [uv](https://docs.astral.sh/uv/). CXD is not on PyPI yet; install it from a clone:

```sh
git clone https://github.com/NaNIsaac/CXD
cd CXD
uv sync
uv run python -c "import cxd, qorageom"
```

The first release will add `pip install cxd`, after QoraGeom is on PyPI.

## First run

No measured data ships with the repository. `examples/synthetic_scan.py` writes a synthetic
rocking scan (a SPEC file, 48 TIFF frames and a `.mat` ROI) to `examples/data/`:

```sh
uv run python examples/synthetic_scan.py
```

Read it back with the SPEC parser:

```python
from cxd.io.spec import Spec

with Spec("examples/data/synthetic.spec") as sp:
    scan = sp.get_scan(1)
    print(scan.motors)
    print(scan.sdd)
    print(scan.to_dataframe().shape)
```

Output, from CXD 0.1.0 on Python 3.13 (Windows):

```
{'Theta': 30.0, 'Delta': 60.0, 'Gamma': 0.0, 'Chi': 0.0}
1000.0
(48, 2)
```

## Where to go next

The README is the full guide. Its sections:

- [Usage](../README.md#usage): regridding a rocking scan with `map_rocking_scan`, the 3D viewer, the QoraPhase handoff and its axis and mask conventions, and the command-line scripts.
- [Validation](../README.md#validation): what the synthetic tests check and the known limits.
- [Structure](../README.md#structure): the modules.

On this site:

- [Profile of the 3D reciprocal-space mapping](perf.md): timing and memory of the regridding step on synthetic volumes.
- The Python API page lists the public functions and classes with their docstrings.
