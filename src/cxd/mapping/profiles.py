import matplotlib.pyplot as plt
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


def plot_paired_lines_subplots(combined_results):
    num_axes = len(combined_results)
    # sharex keeps the distance axis the same so the profiles compare directly
    fig, axes = plt.subplots(num_axes, 1, figsize=(10, 4 * num_axes), sharex=True)

    # a single subplot comes back as an Axes, not an array
    if num_axes == 1:
        axes = [axes]

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, num_axes))

    for i, (ax, res) in enumerate(zip(axes, combined_results)):
        p1, p2 = res["pair"]

        ax.plot(
            res["dist"],
            res["profile"],
            color=colors[i],
            lw=2.5,
            label=f"Axis {i + 1}: Peaks {p1 + 1} & {p2 + 1}",
        )

        ax.axvline(0, color="white", linestyle="--", alpha=0.3)

        ax.set_title(f"Reciprocal Axis {i + 1} (Dot Product: {res['dot']:.4f})", fontsize=12)
        ax.set_ylabel("Intensity")
        ax.grid(True, which="both", alpha=0.1)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Distance along axis (pixels)")

    plt.tight_layout()
    plt.show()


def plot_origin_style(combined_results, show=True) -> plt.Figure:
    num_axes = len(combined_results)
    fig, axes = plt.subplots(num_axes, 1, figsize=(8, 4 * num_axes), sharex=True)

    if num_axes == 1:
        axes = [axes]

    # Tableau10 colors, for an Origin-like look
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, (ax, res) in enumerate(zip(axes, combined_results)):
        p1, p2 = res["pair"]

        ax.plot(
            res["dist"],
            res["profile"],
            color=colors[i % 3],
            lw=2,
            label=f"Axis {i + 1} (Peaks {p1 + 1}-{p2 + 1})",
        )

        ax.axvline(0, color="black", lw=1, ls="--", alpha=0.5)

        ax.tick_params(which="both", width=1.5, length=6)
        ax.tick_params(which="minor", length=3)

        ax.set_ylabel("Intensity (a.u.)", fontweight="bold")
        ax.legend(frameon=False, loc="upper right")
        ax.set_title(f"Reciprocal Axis {i + 1}", pad=10)

    axes[-1].set_xlabel("Distance from CoM (pixels)", fontweight="bold")

    plt.tight_layout()
    if show:
        plt.show()

    return fig
