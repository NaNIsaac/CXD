"""Rocking scan to q with the detector angles (CXD#10), against closed forms and QoraGeom.

Units: q in 1/Å with 2π, angles in degrees, lengths in mm.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from qorageom import get_diffractometer, wavelength
from qorageom.diffractometer import axis_vector

from cxd.core import transform
from cxd.core.rocking import map_rocking_scan, rocking_q

ENERGY = 9.0  # keV; not in the synthetic SPEC file
PIXEL = 0.055  # mm, Timepix
SDD = 1000.0  # mm
K = 2 * np.pi / float(wavelength(ENERGY))

_spec = importlib.util.spec_from_file_location(
    "synthetic_scan", Path(__file__).parents[1] / "examples" / "synthetic_scan.py"
)
synthetic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(synthetic)
THETAS = synthetic.THETA0 + synthetic.THETA_STEP * (
    np.arange(synthetic.N_FRAMES) - synthetic.N_FRAMES // 2
)
CENTER = (synthetic.N_FRAMES // 2, synthetic.ROI // 2, synthetic.ROI // 2)
SHAPE = (synthetic.N_FRAMES, synthetic.ROI, synthetic.ROI)


def q_of(delta, gamma):
    return rocking_q(SHAPE, THETAS, delta, gamma, CENTER, SDD, PIXEL, ENERGY)


@pytest.mark.parametrize(("delta", "gamma"), [(60.0, 0.0), (55.0, 12.0)])
def test_matches_the_closed_forms(delta, gamma):
    q = q_of(delta, gamma)
    f, y, x = CENTER
    centre = q[f, y, x]
    two_theta = np.arccos(np.cos(np.radians(delta)) * np.cos(np.radians(gamma)))
    assert np.linalg.norm(centre) == pytest.approx(2 * K * np.sin(two_theta / 2), rel=1e-12)

    axis = axis_vector(get_diffractometer("34idc").sample_axes[0])  # th, the rocked circle
    q_perp = np.linalg.norm(centre - (centre @ axis) * axis)
    step = np.linalg.norm(q[f + 1, y, x] - centre)
    assert step == pytest.approx(
        2 * q_perp * np.sin(np.radians(synthetic.THETA_STEP) / 2), rel=1e-9
    )
    # first order in p / L
    assert np.linalg.norm(q[f, y, x + 1] - centre) == pytest.approx(K * PIXEL / SDD, rel=1e-3)


def test_agrees_with_the_qorageom_rocking_basis():
    """QoraGeom's linear basis (no 2π): exact along the rocking step, first order per pixel."""
    delta, gamma = 55.0, 12.0
    q = q_of(delta, gamma)
    f, y, x = CENTER
    B = (
        2
        * np.pi
        * get_diffractometer("34idc").rocking_basis(
            float(wavelength(ENERGY)),
            SDD,
            PIXEL,
            [synthetic.THETA0, 90.0, 0.0],
            [delta, gamma],
            "th",
            synthetic.THETA_STEP,
        )
    )
    steps = {"column": q[f, y, x + 1], "row": q[f, y + 1, x], "rock": q[f + 1, y, x]}
    for (name, moved), column, tol in zip(steps.items(), B.T, (1e-4, 1e-4, 1e-10), strict=True):
        diff = moved - q[f, y, x]
        assert np.linalg.norm(diff - column) <= tol * np.linalg.norm(diff), name


def test_binning_conserves_counts_and_marks_empty_bins():
    frames = synthetic.make_volume().astype(float)
    m = map_rocking_scan(frames, THETAS, 60.0, 0.0, CENTER, SDD, PIXEL, ENERGY)
    assert (m.intensity * m.hits).sum() == pytest.approx(frames.sum(), rel=1e-12)
    assert m.hits.min() == 0 and (m.intensity[m.hits == 0] == 0).all()
    assert [len(e) - 1 for e in m.edges] == list(frames.shape)


def test_old_mapping_collapses_the_rocking_sweep(monkeypatch):
    """How far each pixel's q travels from the first frame to the last, old against new.

    The old correct_rocking places the detector on the direct beam, so it rotates the small q of
    the pixel offsets instead of the Bragg q: the sweep collapses.
    """
    frames = synthetic.make_volume().astype(float)
    captured = {}
    original = np.histogramdd

    def keep_points(sample, *args, **kwargs):
        captured["points"] = np.asarray(sample)
        return original(sample, *args, **kwargs)

    monkeypatch.setattr(transform.np, "histogramdd", keep_points)
    with pytest.warns(DeprecationWarning):
        transform.correct_rocking(frames, np.array(CENTER, float), THETAS, SDD, ENERGY, PIXEL)
    monkeypatch.undo()
    old = captured["points"].reshape(SHAPE[0], -1, 3)
    old_sweep = np.linalg.norm(old[-1] - old[0], axis=1)

    new = q_of(60.0, 0.0).reshape(SHAPE[0], -1, 3)
    new_sweep = np.linalg.norm(new[-1] - new[0], axis=1)
    # 47 steps of 0.01 deg about th, perpendicular to the Bragg q at gamma = 0
    expected = 2 * 2 * K * np.sin(np.radians(30)) * np.sin(np.radians(0.47) / 2)
    centre = np.ravel_multi_index(CENTER[1:], SHAPE[1:])
    assert new_sweep[centre] == pytest.approx(expected, rel=1e-9)
    assert old_sweep.max() < 0.01 * new_sweep.min()
