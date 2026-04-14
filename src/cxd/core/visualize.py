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
    """3D Q-space intensity를 PyVista로 시각화.

    Args:
        q_data: (n_omega, ny, nx) 형태의 intensity 배열. 축 순서는 (Qz, Qy, Qx)로 해석.
        threshold_ratio: 최대 강도 대비 시각화 하한 비율.
        show_axes: 방향 축 위젯 표시 여부.
        mode: "contour"(isosurface), "volume"(volume rendering), "threshold"(하한 이상 복셀 메시).
        log: True면 np.log1p(q_data)로 변환한 뒤 시각화 (동적 범위 압축).
        spacing: 각 축의 Q-unit 간격 (Qx, Qy, Qz).
        origin: Q-공간 원점 (Qx, Qy, Qz).
        show: False면 plotter를 띄우지 않고 반환만 함 (테스트/추가 커스터마이즈용).

    Returns:
        show=True면 plotter.show()의 결과(Jupyter에서는 trame 위젯). show=False면 pv.Plotter.
    """
    data = np.log1p(q_data) if log else q_data

    # VTK ImageData는 point(i,j,k) → flat index i + j*nx + k*nx*ny (x가 가장 빨리 변함).
    # q_data shape가 (nz, ny, nx)일 때 C-order flatten이 정확히 이 규약을 만족.
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
