from typing import Any, Literal

import numpy as np
import pyvista as pv


def visualize_q_space_vtk(
    q_data: np.ndarray,
    threshold_ratio: float = 0.1,
    show_axes: bool = True,
    *,
    mode: Literal["contour", "volume", "threshold"] = "contour",
    log: bool = False,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    show: bool = True,
) -> Any:
    """Plot a 3D Q-space intensity volume with PyVista.

    Args:
        q_data: intensity array of shape (n_omega, ny, nx); axes are read as (Qz, Qy, Qx).
        threshold_ratio: lower display limit as a fraction of the maximum intensity.
        show_axes: whether to show the orientation axes widget.
        mode: "contour" (isosurface), "volume" (volume rendering), or "threshold"
            (voxels above the limit).
        log: plot np.log1p(q_data) to compress the dynamic range.
        spacing: Q-unit step per axis (Qx, Qy, Qz).
        origin: origin in Q-space (Qx, Qy, Qz).
        show: if False, return the plotter without showing it.

    Returns:
        The result of plotter.show() (a trame widget in Jupyter) if show, else the pv.Plotter.
    """
    data = np.log1p(q_data) if log else q_data

    # VTK ImageData flat index is i + j*nx + k*nx*ny (x fastest), which a C-order flatten of
    # (nz, ny, nx) data matches.
    nz, ny, nx = data.shape
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = spacing
    grid.origin = origin
    grid.point_data["Intensity"] = data.flatten(order="C")

    val_max = float(np.nanmax(data))
    val_min = val_max * threshold_ratio

    plotter = pv.Plotter(title="Q-Space 3D Visualization")

    if mode == "contour":
        contours = grid.contour(
            isosurfaces=5,
            scalars="Intensity",
            rng=[val_min, val_max],
        )
        plotter.add_mesh(contours, opacity=0.5, cmap="viridis")
    elif mode == "volume":
        plotter.add_volume(grid, cmap="fire", opacity="linear", scalars="Intensity")
    else:  # "threshold"
        threshed = grid.threshold(val_min, scalars="Intensity")
        plotter.add_mesh(threshed, opacity=0.5, cmap="viridis")

    if show_axes:
        plotter.add_axes(xlabel="Qx", ylabel="Qy", zlabel="Qz")

    if show:
        return plotter.show()
    return plotter
