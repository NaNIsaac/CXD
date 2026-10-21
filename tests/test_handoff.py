import numpy as np
import pytest

from cxd.core.handoff import to_qoraphase


def make_object(n=16, m=6, seed=2):
    """A compact complex object (tetrahedral support, so it has no twin) centred in n^3."""
    rng = np.random.default_rng(seed)
    block = rng.uniform(0.5, 1.0, (m,) * 3) * np.exp(1j * rng.uniform(-0.5, 0.5, (m,) * 3))
    block *= np.indices((m,) * 3).sum(0) < m
    obj = np.zeros((n,) * 3, complex)
    obj[(slice(n // 2 - m // 2, n // 2 - m // 2 + m),) * 3] = block
    return obj


def diffraction(obj):
    """Centre-origin |FFT|^2, written independently of QoraPhase."""
    return np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj), norm="ortho"))) ** 2


def similarity(a, b):
    """Peak normalised cross-correlation of the magnitudes, allowing the twin."""
    def peak(x, y):
        return np.abs(np.fft.ifftn(np.fft.fftn(x) * np.conj(np.fft.fftn(y)))).max()

    a, b = np.abs(a), np.abs(b)
    twin = np.roll(a[::-1, ::-1, ::-1], 1, axis=(0, 1, 2))
    return max(peak(a, b), peak(twin, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


@pytest.fixture
def scan():
    """A (frame, y, x) stack with the 16^3 pattern off-centre and one dead detector row."""
    obj = make_object()
    intensity = diffraction(obj)
    stack = np.zeros((20, 24, 24))
    stack[1:17, 5:21, 3:19] = intensity
    bad = np.zeros(stack.shape, bool)
    bad[:, 12, :] = True
    stack[bad] = np.nan
    return obj, intensity, stack, bad


def test_puts_the_peak_at_the_centre_index(scan):
    _, intensity, stack, bad = scan
    out, mask, offset = to_qoraphase(stack, 16, bad)
    assert offset == (1, 5, 3)
    assert out.shape == mask.shape == (16, 16, 16)
    assert np.unravel_index(np.argmax(out), out.shape) == (8, 8, 8)
    np.testing.assert_array_equal(out[~mask], intensity[~mask])


def test_marks_unusable_voxels_true_and_zero_fills_them(scan):
    _, _, stack, bad = scan
    stack[10, 7, 7] = -3.0
    stack[9, 8, 8] = np.inf
    out, mask, offset = to_qoraphase(stack, 16, bad, center=(9, 13, 11))
    assert mask.dtype == bool
    assert mask[:, 12 - offset[1], :].all()  # the dead row, True = unusable
    assert mask[10 - offset[0], 7 - offset[1], 7 - offset[2]]  # negative
    assert mask[9 - offset[0], 8 - offset[1], 8 - offset[2]]  # inf
    assert np.isfinite(out).all() and (out[mask] == 0).all()
    assert not mask.all()


def test_refuses_a_crop_that_runs_off_the_stack(scan):
    _, _, stack, bad = scan
    with pytest.raises(ValueError, match="runs off"):
        to_qoraphase(stack, 20, bad)
    with pytest.raises(ValueError, match="frame, y, x"):
        to_qoraphase(stack[0], 8)


def test_qoraphase_reconstructs_the_handed_over_scan(scan):
    qp = pytest.importorskip("qoraphase")
    obj, _, stack, bad = scan
    with pytest.raises(ValueError):
        qp.CDI(stack[1:17, 5:21, 3:19], mask=bad[1:17, 5:21, 3:19])  # raw NaN is refused

    intensity, mask, _ = to_qoraphase(stack, 16, bad)
    support = np.abs(obj) > 0
    recipe = qp.Recipe(qp.RAAR(150, beta=0.9), qp.ER(100))
    ensemble = qp.reconstruct(intensity, support=support, mask=mask, recipe=recipe, starts=4)
    # the thresholds of QoraPhase's own 3D test (tests/test_shapes.py)
    assert ensemble.best.error < 1e-3
    assert similarity(ensemble.best.object, obj) > 0.95
