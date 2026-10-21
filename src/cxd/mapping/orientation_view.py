import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv


def plot_full_lambert_projection(theta, phi, intensity_map, peaks=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="polar")

    # Lambert 투영 변환: R = sqrt(2)*sin(theta/2)
    r_projection = np.sqrt(2) * np.sin(theta / 2)
    P, R_proj = np.meshgrid(phi, r_projection, indexing="ij")

    # 배경 히트맵 그리기
    mesh = ax.pcolormesh(P, R_proj, intensity_map.T, cmap="magma", shading="auto")

    # 피크 표시 로직
    if peaks is not None:
        for i, peak in enumerate(peaks):
            p_theta = peak[0]
            p_phi = peak[1]

            # 피크의 Theta 값을 Lambert 반경(R)으로 변환
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

    # ==========================================
    # 1. 완벽한 좌표계 일치 (매우 중요)
    # Numpy의 (Z, Y, X) 배열을 PyVista가 기대하는 (X, Y, Z) 구조로 변환합니다.
    # 이렇게 해야 벡터의 (dx, dy, dz)와 이미지의 방향이 일치합니다.
    # ==========================================
    imgs_xyz = np.transpose(imgs, (2, 1, 0))
    NX, NY, NZ = imgs_xyz.shape

    grid = pv.ImageData()
    grid.dimensions = (NX, NY, NZ)
    grid.point_data["values"] = imgs_xyz.flatten(order="F")

    # ==========================================
    # 2. IsoVolume (ParaView 스타일) 렌더링
    # contour(껍질)가 아닌 threshold(부피)를 사용하여 덩어리감을 살립니다.
    # ==========================================
    if threshold_value is None:
        # 데이터의 밝기에 따라 이 값을 조절하세요 (예: 95~99 퍼센타일)
        threshold_value = np.percentile(imgs[imgs > 0], 95)

    # 설정한 임계값 이상의 데이터를 모두 포함하는 3D 볼륨 생성
    isovolume = grid.threshold(threshold_value)

    # cmap을 추가하여 강도에 따른 색상 변화를 주면 더 ParaView와 비슷해집니다.
    plotter.add_mesh(isovolume, opacity=0.3, cmap="viridis", show_scalar_bar=True, name="pattern")

    # ==========================================
    # 3. 중심점(COM) 및 방향 벡터(Arrows) 생성
    # ==========================================
    # COM 역배열 (Z, Y, X) -> (X, Y, Z)
    pv_com = np.array([com[2], com[1], com[0]])
    center_sphere = pv.Sphere(radius=max(imgs.shape) * 0.015, center=pv_com)
    plotter.add_mesh(center_sphere, color="yellow")

    arrow_length = max(imgs.shape) * scale * 0.4

    for i, (t, p, _) in enumerate(orientations):
        # 구면 좌표 -> 데카르트 좌표 (X, Y, Z)
        dx = np.sin(t) * np.cos(p)
        dy = np.sin(t) * np.sin(p)
        dz = np.cos(t)
        direction = np.array([dx, dy, dz])

        # 스트릭은 보통 대칭이므로, 중심을 관통하도록 양방향으로 화살표를 그립니다.
        color = plt.get_cmap("hsv")(i / max(1, len(orientations)))[:3]

        # 정방향 (Forward)
        arrow_fwd = pv.Arrow(start=pv_com, direction=direction, scale=arrow_length)
        plotter.add_mesh(arrow_fwd, color=color, line_width=3)

        # 역방향 (Backward) - 스트릭의 반대편도 관통하는지 확인하기 위함
        arrow_rev = pv.Arrow(start=pv_com, direction=-direction, scale=arrow_length)
        plotter.add_mesh(arrow_rev, color=color, line_width=3)

    # 축 설정 및 출력
    plotter.add_axes()
    print("3D Visualization window will open.")
    plotter.show()
