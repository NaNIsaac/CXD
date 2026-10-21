import numpy as np
from scipy.fft import fftn, ifftn

from cxd.twin_finder import compute_normalized_xcorr, filter_twin_solutions


def _pairwise_reference(amplitudes: list[np.ndarray]) -> np.ndarray:
    """The similarity matrix as computed before the FFT cache: two fresh FFTs per pair."""
    n = len(amplitudes)
    sim = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = amplitudes[i], amplitudes[j]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            score = (
                0.0
                if na == 0 or nb == 0
                else np.max(np.real(ifftn(fftn(a) * np.conj(fftn(b))))) / (na * nb)
            )
            sim[i, j] = sim[j, i] = score
    return sim


def _solutions() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    obj = rng.random((12, 10, 8))
    twin = obj[::-1, ::-1, ::-1]  # the centrosymmetric twin of a phase-retrieval solution
    maps = [np.roll(obj, k, axis=0) + 0.01 * rng.random(obj.shape) for k in range(4)]
    maps += [np.roll(twin, k, axis=1) + 0.01 * rng.random(obj.shape) for k in range(4)]
    maps.append(np.zeros(obj.shape))
    return maps


def test_cached_similarity_matrix_matches_the_pairwise_computation():
    amplitudes = _solutions()
    _, _, sim = filter_twin_solutions(amplitudes)
    np.testing.assert_array_equal(sim, _pairwise_reference(amplitudes))


def test_true_and_twin_solutions_land_in_separate_groups():
    group_0, group_1, _ = filter_twin_solutions(_solutions()[:8])
    groups = {frozenset(group_0.tolist()), frozenset(group_1.tolist())}
    assert groups == {frozenset(range(4)), frozenset(range(4, 8))}


def test_xcorr_is_shift_invariant_and_zero_for_an_empty_map():
    a = np.random.default_rng(1).random((6, 6, 6))
    assert np.isclose(compute_normalized_xcorr(a, np.roll(a, 2, axis=2)), 1.0)
    assert compute_normalized_xcorr(a, np.zeros_like(a)) == 0.0
