"""Plots of the paired line profiles from ``cxd.mapping.profiles``."""

import matplotlib.pyplot as plt
import numpy as np


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
