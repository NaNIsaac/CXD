"""End to end on the synthetic scan: SPEC + TIFF frames in, QoraPhase input and q sampling out.

The scan comes from examples/synthetic_scan.py (seeded, nothing measured). Quantities with
a closed form are checked against it; counts that depend on the Poisson draw are pinned,
so a change in the generator or a loader shows up here.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from cxd.core.handoff import to_qoraphase
from cxd.io.frames import load_tif_stack
from cxd.io.spec import Spec
from cxd.mapping.qspace import calculate_qspace_params

ENERGY_KEV = 9.0  # not in the synthetic SPEC file; chosen for the test
PIXEL_MM = 0.055  # Timepix pixel pitch, the detector the generator imitates
HC_KEV_A = 12.398_419_843  # h c in keV Angstrom


@pytest.fixture(scope="module")
def scan_dir(tmp_path_factory) -> Path:
    path = Path(__file__).parents[1] / "examples" / "synthetic_scan.py"
    spec = importlib.util.spec_from_file_location("synthetic_scan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path_factory.mktemp("scan")
    module.main(out)
    return out


@pytest.fixture(scope="module")
def scan(scan_dir):
    return Spec(scan_dir / "synthetic.spec").get_scan(1)


@pytest.fixture(scope="module")
def stack(scan_dir):
    return load_tif_stack(sorted((scan_dir / "S0001").glob("*.tif")))


def test_spec_geometry(scan):
    assert scan.motors == {"Theta": 30.0, "Delta": 60.0, "Gamma": 0.0, "Chi": 0.0}  # deg
    assert scan.sdd == 1000.0  # mm
    theta = scan.to_dataframe()["Theta"].to_numpy()  # deg
    assert len(theta) == 48
    np.testing.assert_allclose(theta[[0, -1]], [29.76, 30.23], atol=1e-9)
    np.testing.assert_allclose(np.diff(theta), 0.01, atol=1e-9)


def test_frames_load_as_frame_y_x(stack, scan):
    assert stack.shape == (48, 256, 256) and stack.dtype == np.uint16
    assert np.unravel_index(stack.argmax(), stack.shape) == (24, 128, 128)
    assert int(stack.sum()) == 663_184  # counts, seed 0
    # the SPEC ROI counter is the sum of the same pixels
    roi = stack[:, 96:160, 96:160].reshape(len(stack), -1).sum(axis=1)
    np.testing.assert_array_equal(scan.to_dataframe()["imroi1"].to_numpy(), roi)


def test_mat_holds_the_same_roi_as_y_x_frame(scan_dir, stack):
    data = loadmat(scan_dir / "scan_0001.mat")["data"]
    assert data.shape == (64, 64, 48)
    np.testing.assert_array_equal(data.transpose(2, 0, 1), stack[:, 96:160, 96:160])


def test_q_sampling_matches_the_closed_form(scan):
    two_theta = scan.motors["Delta"]
    dq_rock, dq_y, dq_x = calculate_qspace_params(
        ENERGY_KEV, two_theta, scan.sdd, PIXEL_MM, omega_step_deg=0.01
    )
    k = 2 * np.pi * ENERGY_KEV / HC_KEV_A  # 1/A
    q_bragg = 2 * k * np.sin(np.radians(two_theta / 2))  # 1/A, 2pi included
    assert dq_y == dq_x == pytest.approx(k * PIXEL_MM / scan.sdd, rel=1e-6)
    assert dq_rock == pytest.approx(q_bragg * np.radians(0.01), rel=1e-6)
    # regression values, 1/A per step
    assert dq_rock == pytest.approx(7.9604e-4, rel=1e-4)
    assert dq_y == pytest.approx(2.5085e-4, rel=1e-4)


def test_handoff_centres_the_peak(stack, scan):
    frames = len(scan.to_dataframe())
    intensity, mask, offset = to_qoraphase(stack, 32, frames=frames)
    assert offset == (8, 112, 112)
    assert intensity.shape == (32, 32, 32) and intensity.dtype == np.float64
    assert np.unravel_index(intensity.argmax(), intensity.shape) == (16, 16, 16)
    assert not mask.any()
    assert intensity.sum() == 649_686  # counts inside the crop, seed 0


# negative cases: inputs that look plausible but are wrong


def test_rejects_a_stack_in_mat_axis_order(scan_dir, scan):
    data = loadmat(scan_dir / "scan_0001.mat")["data"]  # (y, x, frame)
    with pytest.raises(ValueError, match="is it \\(y, x, frame\\)"):
        to_qoraphase(data, 16, frames=len(scan.to_dataframe()))


def test_rejects_an_inverted_mask(stack):
    kept = np.ones(stack.shape, bool)  # cxd.mapping.qspace polarity: True = kept
    kept[:, :, :8] = False
    with pytest.raises(ValueError, match="mask polarity"):
        to_qoraphase(stack, 32, bad=kept, center=(24, 128, 128))
