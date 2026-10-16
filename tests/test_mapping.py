import numpy as np
from scipy.ndimage import center_of_mass

from cxd.mapping.profiles import get_symmetric_profiles, get_thick_line_profile
from cxd.mapping.spherical import extract_peak_from_2d_map, map2spherical


def streak_volume(direction, n=41, length=18, blob=True):
    """Streak along +/-direction (z, y, x order), optionally with a 3x3x3 blob at the centre."""
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    c = (n - 1) / 2
    vol = np.zeros((n, n, n))
    for t in np.linspace(-length, length, 4 * length + 1):
        z, y, x = np.round(c + t * d).astype(int)
        vol[z, y, x] = 1.0
    if blob:
        vol[int(c) - 1 : int(c) + 2, int(c) - 1 : int(c) + 2, int(c) - 1 : int(c) + 2] = 1.0
    return vol, d


def test_map2spherical_weights_sum_and_unit_vectors():
    vol, _ = streak_volume([1, 0, 0])
    com = np.array(center_of_mass(vol))
    u_vecs, hist, u_edges, phi_edges = map2spherical(vol, com, bins=50)
    assert u_vecs.shape == (3, *vol.shape)
    norms = np.linalg.norm(u_vecs.reshape(3, -1), axis=0)
    assert np.all(np.isclose(norms, 1) | np.isclose(norms, 0))  # 0 only at the CoM voxel
    assert hist.sum() == np.float64(vol.sum())
    # a streak along z lands in the u = cos(theta) = +/-1 bins
    assert hist[0].sum() + hist[-1].sum() > 0.5 * hist.sum()


def test_extract_peaks_recovers_streak_direction():
    vol, d = streak_volume([1, 2, 2], blob=False)
    com = np.array(center_of_mass(vol))
    _, hist, u_edges, phi_edges = map2spherical(vol, com, bins=100)
    peaks = extract_peak_from_2d_map(hist, u_edges, phi_edges, eps=0.2, min_samples=1, threshold_rel=0.3)
    assert len(peaks) == 2
    cos = np.sort(peaks @ d)
    assert cos[0] < -0.98 and cos[1] > 0.98


def test_line_profile_and_symmetric_pairing():
    vol, d = streak_volume([0, 0, 1])
    com = np.array(center_of_mass(vol))
    along = get_thick_line_profile(vol, com, d, length=15, thickness=0)
    across = get_thick_line_profile(vol, com, np.array([0.0, 1.0, 0.0]), length=15, thickness=0)
    assert along[5:].min() > 0.9
    assert across[5:].max() == 0

    res = get_symmetric_profiles(vol, com, np.array([d, -d]), length=10, thickness=0)
    assert len(res) == 1
    assert res[0]["pair"] == (0, 1)
    assert res[0]["dot"] == -1.0
    assert len(res[0]["profile"]) == len(res[0]["dist"]) == 19
