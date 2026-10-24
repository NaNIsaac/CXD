import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv


def plot_full_lambert_projection(theta, phi, intensity_map, peaks=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="polar")

    # Lambert equal-area radius: R = sqrt(2) * sin(theta / 2)
    r_projection = np.sqrt(2) * np.sin(theta / 2)
    P, R_proj = np.meshgrid(phi, r_projection, indexing="ij")

    mesh = ax.pcolormesh(P, R_proj, intensity_map.T, cmap="magma", shading="auto")

    if peaks is not None:
        for i, peak in enumerate(peaks):
            p_theta = peak[0]
            p_phi = peak[1]

            p_r_proj = np.sqrt(2) * np.sin(p_theta / 2)

            ax.scatter(
                p_phi,
                p_r_proj,
                color="cyan",
                marker="x",
                s=100,
                linewidths=2.5,
                label="Detected Peaks" if i == 0 else "",
            )

            ax.text(
                p_phi,
                p_r_proj + 0.05,
                f"{i + 1}",
                color="white",
                fontsize=12,
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    tick_angles = np.array([30, 60, 90, 120, 150, 180]) * np.pi / 180
    tick_labels = ["30°", "60°", "90°(Eq)", "120°", "150°", "180°(S)"]
    ax.set_rticks(np.sqrt(2) * np.sin(tick_angles / 2))
    ax.set_yticklabels(tick_labels, fontsize=9, color="gray")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.colorbar(mesh, label="Max Integrated Intensity")
    plt.title("Hough-Parameter Space (Spherical Lambert Projection)", fontsize=13)
    plt.show()


def plot_3d_pattern_with_orientations(imgs, com, orientations, scale=1.0, threshold_value=None):
    print("\nStarting 3D visualization (IsoVolume mode)...")
    plotter = pv.Plotter(title="3D Pattern & Extracted Fringe Directions")
    pv.set_plot_theme("dark")

    # numpy (Z, Y, X) to PyVista (X, Y, Z), so that the vector (dx, dy, dz) matches the image
    # orientation
    imgs_xyz = np.transpose(imgs, (2, 1, 0))
    NX, NY, NZ = imgs_xyz.shape

    grid = pv.ImageData()
    grid.dimensions = (NX, NY, NZ)
    grid.point_data["values"] = imgs_xyz.flatten(order="F")

    # threshold (a volume) rather than contour (a shell) keeps the bulk of the pattern
    if threshold_value is None:
        # tune to the data brightness, e.g. the 95th to 99th percentile
        threshold_value = np.percentile(imgs[imgs > 0], 95)

    isovolume = grid.threshold(threshold_value)

    plotter.add_mesh(isovolume, opacity=0.3, cmap="viridis", show_scalar_bar=True, name="pattern")

    # COM reordered from (Z, Y, X) to (X, Y, Z)
    pv_com = np.array([com[2], com[1], com[0]])
    center_sphere = pv.Sphere(radius=max(imgs.shape) * 0.015, center=pv_com)
    plotter.add_mesh(center_sphere, color="yellow")

    arrow_length = max(imgs.shape) * scale * 0.4

    for i, (t, p, _) in enumerate(orientations):
        dx = np.sin(t) * np.cos(p)
        dy = np.sin(t) * np.sin(p)
        dz = np.cos(t)
        direction = np.array([dx, dy, dz])

        color = plt.get_cmap("hsv")(i / max(1, len(orientations)))[:3]

        # streaks are usually symmetric, so draw arrows in both directions through the center
        arrow_fwd = pv.Arrow(start=pv_com, direction=direction, scale=arrow_length)
        plotter.add_mesh(arrow_fwd, color=color, line_width=3)

        arrow_rev = pv.Arrow(start=pv_com, direction=-direction, scale=arrow_length)
        plotter.add_mesh(arrow_rev, color=color, line_width=3)

    plotter.add_axes()
    print("3D Visualization window will open.")
    plotter.show()
