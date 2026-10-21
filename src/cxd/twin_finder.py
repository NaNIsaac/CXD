from typing import Any
import numpy as np
from scipy.fft import fftn, ifftn
from sklearn.cluster import KMeans


def compute_normalized_xcorr(A1: np.ndarray, A2: np.ndarray) -> float:
    """FFT를 이용해 이동 불변 최대 교차 상관계수(유사도)를 계산합니다."""
    # 정규화를 위한 분산 계산 (L2 norm)
    norm1: np.floating[Any] = np.linalg.norm(A1)
    norm2: np.floating[Any] = np.linalg.norm(A2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    # FFT 기반 Cross-correlation
    F1: np.ndarray = fftn(A1)
    F2: np.ndarray = fftn(A2)
    xcorr: np.ndarray = np.real(ifftn(F1 * np.conj(F2)))

    # 정렬이 최적화된 위치에서의 최대 상관계수 추출 및 정규화 [-1, 1]
    max_corr: float = np.max(xcorr) / (norm1 * norm2)
    return max_corr


def filter_twin_solutions(amplitudes: list[np.ndarray]):
    """수십 개의 진폭 맵 리스트를 받아 True/Twin 그룹을 분리합니다."""
    N: int = len(amplitudes)
    similarity_matrix: np.ndarray = np.zeros((N, N))

    # 1. N x N 유사도 행렬 계산 (대칭 행렬이므로 절반만 계산)
    for i in range(N):
        similarity_matrix[i, i] = 1.0
        for j in range(i + 1, N):
            score: float = compute_normalized_xcorr(amplitudes[i], amplitudes[j])
            similarity_matrix[i, j] = score
            similarity_matrix[j, i] = score

    # 2. 클러스터링을 위해 유사도를 거리(Distance) 개념으로 변환
    distance_matrix: np.ndarray = 1.0 - similarity_matrix

    # 3. K-Means 클러스터링 (두 개의 그룹으로 분리)
    # 거리 행렬의 각 행을 해당 솔루션의 Feature(N차원 공간의 좌표)로 사용
    kmeans: KMeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels: np.ndarray = kmeans.fit_predict(distance_matrix)

    group_0: np.ndarray = np.where(labels == 0)[0]
    group_1: np.ndarray = np.where(labels == 1)[0]

    return group_0, group_1, similarity_matrix
