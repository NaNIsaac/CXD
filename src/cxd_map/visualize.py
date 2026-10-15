import numpy as np
import vtk
from vtk.util import numpy_support

def vtk_visualize_cxd(imgs, com, combined_results, peaks, B_recip, thickness=1.0, save_path="vtk_output.png", show=True):
    """
    BCDI/CXD 데이터를 3D로 시각화합니다.
    - 배경: 하양
    - 파티클: 파랑
    - 확대 및 중심 정렬 적용
    """
    # imgs가 True/False면 렌더링이 안 될 수 있으므로 float32로 변환
    imgs_float = imgs.astype(np.float32) 
    
    renderer = vtk.vtkRenderer()
    render_win = vtk.vtkRenderWindow()
    render_win.AddRenderer(renderer)
    render_win.SetSize(1000, 800)
    
    if not show:
        render_win.SetOffScreenRendering(1) 

    # --- 좌표계 스케일 및 중심 이동 ---
    # B_recip의 물리적 단위(m^-1) 시각화 스케일링 [cite: 238, 329]
    vis_scale = 3 * 1e-6 
    
    # 변환 행렬 생성: q = B_recip @ (n - com) [cite: 133, 183, 187]
    transform = vtk.vtkTransform()
    transform.Scale(vis_scale, vis_scale, vis_scale) 
    
    vtk_mat = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            vtk_mat.SetElement(i, j, B_recip[i, j])
    transform.Concatenate(vtk_mat) # 전단 보정 적용 [cite: 329]
    transform.Translate(-com[2], -com[1], -com[0]) # 원점(CoM) 이동 [cite: 187]

    # --- 볼륨 데이터 생성 ---
    vtk_data = numpy_support.numpy_to_vtk(num_array=imgs_float.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    volume_data = vtk.vtkImageData()
    volume_data.SetDimensions(imgs.shape[2], imgs.shape[1], imgs.shape[0])
    volume_data.GetPointData().SetScalars(vtk_data)

    volume_property = vtk.vtkVolumeProperty()
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()

    # 불투명도 설정
    opacity_func = vtk.vtkPiecewiseFunction()
    opacity_func.AddPoint(0, 0.0)
    opacity_func.AddPoint(0.5, 0.2)
    opacity_func.AddPoint(1.0, 0.7)
    volume_property.SetScalarOpacity(opacity_func)

    # 컬러 설정: 파란색 (배경 하양에 대비되도록 설정)
    color_func = vtk.vtkColorTransferFunction()
    color_func.AddRGBPoint(0, 1.0, 1.0, 1.0)  # 배경색과 동일 (투명화 처리됨)
    color_func.AddRGBPoint(1.0, 0.1, 0.4, 0.8) # 선명한 파란색
    volume_property.SetColor(color_func)

    volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
    volume_mapper.SetInputData(volume_data)

    volume_actor = vtk.vtkVolume()
    volume_actor.SetMapper(volume_mapper)
    volume_actor.SetProperty(volume_property)
    volume_actor.SetUserTransform(transform) 
    renderer.AddActor(volume_actor)

    # --- Q-space 대칭축 그리기 ---
    line_len = np.linalg.norm(B_recip @ np.array(imgs.shape)) * 0.4 * vis_scale
    for i, res in enumerate(combined_results):
        p_idx = res["pair"][0]
        dir_idx = np.array([peaks[p_idx][2], peaks[p_idx][1], peaks[p_idx][0]])
        dir_q = B_recip @ dir_idx # q 방향 변환 [cite: 133]
        dir_q /= np.linalg.norm(dir_q)

        line = vtk.vtkLineSource()
        line.SetPoint1(-(dir_q * line_len))
        line.SetPoint2(dir_q * line_len)
        
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(line.GetOutputPort())
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetLineWidth(3 * thickness)
        # 하얀 배경을 고려하여 대칭축 색상을 검은색(또는 어두운 회색)으로 변경
        a.GetProperty().SetColor(0.2, 0.2, 0.2) 
        renderer.AddActor(a)

    # --- Qx, Qy, Qz 위젯 설정 ---
    axes_actor = vtk.vtkAxesActor()
    axes_actor.SetXAxisLabelText("Qx")
    axes_actor.SetYAxisLabelText("Qy")
    axes_actor.SetZAxisLabelText("Qz")
    # 라벨 색상 설정 (배경이 하양이라 검은색 계열이 좋음)
    axes_actor.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0,0,0)
    axes_actor.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0,0,0)
    axes_actor.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0,0,0)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_win)

    marker = vtk.vtkOrientationMarkerWidget()
    marker.SetOrientationMarker(axes_actor)
    marker.SetInteractor(interactor)
    marker.SetViewport(0.0, 0.0, 0.2, 0.2)
    marker.SetEnabled(1)
    marker.InteractiveOff()

    # 배경색을 하양(1, 1, 1)으로 설정
    renderer.SetBackground(1.0, 1.0, 1.0)
    renderer.ResetCamera()
    
    # --- [확대 로직 추가] ---
    # ResetCamera() 이후에 줌을 적용해야 효과가 있습니다.
    # 2.0은 2배 확대를 의미합니다.
    renderer.GetActiveCamera().Zoom(2.5) 

    render_win.Render()

    # 이미지 저장 로직
    if save_path:
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(render_win)
        w2if.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(save_path))
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()
        print(f"이미지가 저장되었습니다: {save_path}")

    if show:
        interactor.Start()
    
    render_win.Finalize()
    del render_win