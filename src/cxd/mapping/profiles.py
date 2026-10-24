import warnings

import numpy as np
from scipy.ndimage import map_coordinates


def get_thick_line_profile(
    imgs: np.ndarray,
    com: np.ndarray,
    direction: np.ndarray,
    length: int = 100,
    thickness: int = 2,
) -> np.ndarray:
    """Mean intensity along a line from com in a direction, averaged over +-thickness offsets."""
    t = np.arange(length)

    offsets = np.arange(-thickness, thickness + 1)
    oz, oy, ox = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    offsets_vec = np.stack([oz.ravel(), oy.ravel(), ox.ravel()], axis=1)  # (N_offsets, 3)

    # P(t) = CoM + t*n + offset
    line_points = com + t[:, None] * direction  # (length, 3)
    all_points = line_points[None, :, :] + offsets_vec[:, None, :]  # (N_offsets, length, 3)

    # map_coordinates wants shape (3, N_total_points)
    flat_points = all_points.reshape(-1, 3).T

    values = map_coordinates(imgs, flat_points, order=1, mode="constant", cval=0.0)

    values = values.reshape(len(offsets_vec), length)
    return np.mean(values, axis=0)


def get_symmetric_profiles(imgs, com, peak_vectors, length=50, thickness=2):
    """Pair each peak with the most opposite one and return one combined profile per pair."""
    num_peaks = len(peak_vectors)
    used = [False] * num_peaks
    combined_results = []

    for i in range(num_peaks):
        if used[i]:
            continue

        # the partner is the peak with the most negative dot product
        best_j = -1
        min_dot = 0  # only a negative dot product counts as opposite

        for j in range(num_peaks):
            if i == j or used[j]:
                continue
            dot = np.dot(peak_vectors[i], peak_vectors[j])
            if dot < min_dot:
                min_dot = dot
                best_j = j

        if best_j != -1:
            used[i] = used[best_j] = True

            pos_profile = get_thick_line_profile(imgs, com, peak_vectors[i], length, thickness)
            neg_profile = get_thick_line_profile(imgs, com, peak_vectors[best_j], length, thickness)

            # reverse neg_profile so the axis runs from -length to +length
            full_profile = np.concatenate([neg_profile[::-1], pos_profile[1:]])
            dist_axis = np.linspace(-length, length, len(full_profile))

            combined_results.append(
                {
                    "pair": (i, best_j),
                    "dist": dist_axis,
                    "profile": full_profile,
                    "dot": min_dot,
                }
            )

    return combined_results


_MOVED = {"plot_paired_lines_subplots", "plot_origin_style"}


def __getattr__(name):
    if name in _MOVED:
        warnings.warn(
            f"cxd.mapping.profiles.{name} moved to cxd.mapping.profiles_view",
            DeprecationWarning,
            stacklevel=2,
        )
        from cxd.mapping import profiles_view

        return getattr(profiles_view, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
