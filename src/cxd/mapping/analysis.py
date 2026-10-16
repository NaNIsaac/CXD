import numpy as np
from sklearn.cluster import DBSCAN


def extract_peak_directions(
    unit_vectors: np.ndarray,  # shape (3, nz, ny, nx)
    weights: np.ndarray,
    eps: float = 0.1,  # DBSCAN eps parameter
    min_samples: int = 5,  # DBSCAN min_samples parameter
) -> np.ndarray:
    clustering: DBSCAN = DBSCAN(eps=eps, min_samples=min_samples).fit(
        unit_vectors.reshape(3, -1).T, sample_weight=weights.ravel()
    )
    labels = clustering.labels_

    peak_vectors: list[np.ndarray] = []
    unique_labels: set = set(labels) - {-1}

    for label in unique_labels:
        cluster_mask = labels == label
        w_cluster = weights[cluster_mask]
        v_cluster = unit_vectors[:, cluster_mask]

        avg_v = np.average(v_cluster, axis=1, weights=w_cluster)
        norm = np.linalg.norm(avg_v)
        if norm > 0:
            peak_vectors.append(avg_v / np.linalg.norm(avg_v))

    return np.array(peak_vectors)
