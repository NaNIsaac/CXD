import numpy as np
from sklearn.cluster import DBSCAN


def map2spherical(
    imgs: np.ndarray, com: np.ndarray, bins: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    3D 이미지를 구면 좌표계(u, phi) 히스토그램으로 매핑합니다.
    """
    # 1. 좌표 그리드 생성 (3, Z, Y, X)
    indices = np.indices(imgs.shape).astype(float)

    # 2. 중심점(CoM) 기준 상대 좌표 계산 (Broadcasting 수정)
    # com을 (3, 1, 1, 1)로 변환하여 뺄셈 수행
    coords = indices - com[:, None, None, None]

    # 3. 노름 계산 (axis=0이 좌표축임)
    r = np.linalg.norm(coords, axis=0, keepdims=True)
    r = np.where(r == 0, 1e-10, r)

    # 4. 단위 방향 벡터 (3, Z, Y, X)
    unit_vectors = coords / r

    # 5. u-phi 매핑 데이터 준비
    u = unit_vectors[0]  # nz (cos theta)
    phi = np.arctan2(unit_vectors[1], unit_vectors[2])  # ny, nx

    # 6. 히스토그램 생성
    hist, u_edges, phi_edges = np.histogram2d(
        u.ravel(),
        phi.ravel(),
        bins=[bins, bins],
        range=[[-1, 1], [-np.pi, np.pi]],
        weights=imgs.ravel(),
    )

    return unit_vectors, hist, u_edges, phi_edges


def extract_peak_from_2d_map(
    hist: np.ndarray,
    u_edges: np.ndarray,
    phi_edges: np.ndarray,
    eps: float = 0.1,
    min_samples: int = 10,
    threshold_rel: float = 0.2,
) -> np.ndarray:
    """
    2D 히스토그램에서 전처리된 강도를 바탕으로 3D 피크 방향 벡터를 추출합니다.
    """
    # 1. 빈(bin)의 중심 좌표 계산
    u_centers = (u_edges[:-1] + u_edges[1:]) / 2
    phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2
    uu, pp = np.meshgrid(u_centers, phi_centers, indexing="ij")

    # 2. 강도 기반 필터링 (전처리된 hist 사용)
    mask = hist > (np.max(hist) * threshold_rel)
    u_val = uu[mask]
    p_val = pp[mask]
    w_val = hist[mask]

    if len(u_val) == 0:
        return np.array([])

    # 3. (u, phi)를 다시 3D 단위 벡터로 역변환 (중요: map2spherical의 규칙 준수)
    # nz = u, ny = r_xy * sin(phi), nx = r_xy * cos(phi)
    r_xy = np.sqrt(np.maximum(0, 1 - u_val**2))
    nz = u_val
    ny = r_xy * np.sin(p_val)
    nx = r_xy * np.cos(p_val)

    v_pts = np.stack([nz, ny, nx], axis=1)  # (N_points, 3)

    # 4. 3D 벡터 공간에서 클러스터링 (주기성 자동 해결)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(
        v_pts, sample_weight=w_val
    )
    labels = clustering.labels_

    peak_vectors = []
    for label in set(labels) - {-1}:
        c_mask = labels == label
        # 가중 평균으로 정밀한 피크 위치 결정
        avg_v = np.average(v_pts[c_mask], axis=0, weights=w_val[c_mask])
        peak_vectors.append(avg_v / np.linalg.norm(avg_v))

    return np.array(peak_vectors)


def unit2spherical(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p_u = coords[:, 0]
    p_phi = np.arctan2(coords[:, 1], coords[:, 2])
    return p_u, p_phi
