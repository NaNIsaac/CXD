from typing import Any
import numpy as np
from scipy.fft import fftn, ifftn
from sklearn.cluster import KMeans


def _xcorr_peak(F1: np.ndarray, F2: np.ndarray, norm1: float, norm2: float) -> float:
    """Maximum cross-correlation of two spectra, normalized by the L2 norms (range [-1, 1])."""
    if norm1 == 0 or norm2 == 0:
        return 0.0
    xcorr: np.ndarray = np.real(ifftn(F1 * np.conj(F2)))
    return np.max(xcorr) / (norm1 * norm2)


def compute_normalized_xcorr(A1: np.ndarray, A2: np.ndarray) -> float:
    """Shift-invariant maximum normalized cross-correlation (similarity) via FFT."""
    norm1: np.floating[Any] = np.linalg.norm(A1)
    norm2: np.floating[Any] = np.linalg.norm(A2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return _xcorr_peak(fftn(A1), fftn(A2), norm1, norm2)


def filter_twin_solutions(amplitudes: list[np.ndarray]):
    """Split amplitude maps into two groups (true and twin solutions) by K-Means on similarity.

    Returns the two index arrays and the similarity matrix.
    """
    N: int = len(amplitudes)
    similarity_matrix: np.ndarray = np.zeros((N, N))

    # one FFT and norm per map; recomputing per pair would take N(N-1) FFTs
    norms: list[float] = [np.linalg.norm(a) for a in amplitudes]
    spectra: list[np.ndarray] = [fftn(a) for a in amplitudes]

    # the matrix is symmetric, so only the upper triangle is computed
    for i in range(N):
        similarity_matrix[i, i] = 1.0
        for j in range(i + 1, N):
            score: float = _xcorr_peak(spectra[i], spectra[j], norms[i], norms[j])
            similarity_matrix[i, j] = score
            similarity_matrix[j, i] = score

    distance_matrix: np.ndarray = 1.0 - similarity_matrix

    # each row of the distance matrix is the feature vector of one solution
    kmeans: KMeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels: np.ndarray = kmeans.fit_predict(distance_matrix)

    group_0: np.ndarray = np.where(labels == 0)[0]
    group_1: np.ndarray = np.where(labels == 1)[0]

    return group_0, group_1, similarity_matrix
