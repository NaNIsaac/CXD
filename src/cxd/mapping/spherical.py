import numpy as np
from sklearn.cluster import DBSCAN


def map2spherical(
    imgs: np.ndarray, com: np.ndarray, bins: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Histogram a 3D image over the direction (u = cos theta, phi) around com, by intensity.

    Returns the unit vectors, the histogram and the u and phi bin edges.
    """
    # index grid, (3, Z, Y, X)
    indices = np.indices(imgs.shape).astype(float)

    # com broadcast as (3, 1, 1, 1)
    coords = indices - com[:, None, None, None]

    # axis 0 holds the coordinates
    r = np.linalg.norm(coords, axis=0, keepdims=True)
    r = np.where(r == 0, 1e-10, r)

    unit_vectors = coords / r

    u = unit_vectors[0]  # nz (cos theta)
    phi = np.arctan2(unit_vectors[1], unit_vectors[2])  # ny, nx

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
    """3D peak direction vectors from a preprocessed (u, phi) histogram.

    Bins above threshold_rel of the maximum are clustered with DBSCAN as unit vectors; each
    cluster gives its intensity-weighted mean direction.
    """
    u_centers = (u_edges[:-1] + u_edges[1:]) / 2
    phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2
    uu, pp = np.meshgrid(u_centers, phi_centers, indexing="ij")

    mask = hist > (np.max(hist) * threshold_rel)
    u_val = uu[mask]
    p_val = pp[mask]
    w_val = hist[mask]

    if len(u_val) == 0:
        return np.array([])

    # back to 3D unit vectors, following the map2spherical convention:
    # nz = u, ny = r_xy * sin(phi), nx = r_xy * cos(phi)
    r_xy = np.sqrt(np.maximum(0, 1 - u_val**2))
    nz = u_val
    ny = r_xy * np.sin(p_val)
    nx = r_xy * np.cos(p_val)

    v_pts = np.stack([nz, ny, nx], axis=1)  # (N_points, 3)

    # clustering in 3D vector space avoids the periodicity of phi
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(v_pts, sample_weight=w_val)
    labels = clustering.labels_

    peak_vectors = []
    for label in set(labels) - {-1}:
        c_mask = labels == label
        # the weighted mean gives a sub-bin peak position
        avg_v = np.average(v_pts[c_mask], axis=0, weights=w_val[c_mask])
        peak_vectors.append(avg_v / np.linalg.norm(avg_v))

    return np.array(peak_vectors)


def unit2spherical(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p_u = coords[:, 0]
    p_phi = np.arctan2(coords[:, 1], coords[:, 2])
    return p_u, p_phi
