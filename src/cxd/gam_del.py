import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

from cxd.core.spec import Spec


# === 1. 기하학 이론 곡선 계산 함수 ===
def calculate_theory_delta(gamma_array, tth):
    gam_rad = np.radians(gamma_array)
    tth_rad = np.radians(tth)
    cos_val = np.cos(tth_rad) / np.cos(gam_rad)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return np.degrees(np.arccos(cos_val))


# === 2. SPEC 데이터 로더 함수 (Run 순서 유지) ===
def load_run_data(sp: Spec, run_ns: list[int]) -> pd.DataFrame:
    """
    주어진 run 번호 리스트에서 Gamma와 Delta 값을 추출하여 DataFrame으로 반환합니다.
    입력된 run_ns 리스트 순서가 그대로 유지됩니다.
    """
    data = []
    for run_n in run_ns:
        scan = sp.get_scan(run_n)
        gamma = scan.motors.get("Gamma", np.nan)
        delta = scan.motors.get("Delta", np.nan)
        data.append({"run_n": run_n, "gamma": gamma, "delta": delta})

    return pd.DataFrame(data).dropna()


# ==========================================
# [Main 실행부]
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Measured gamma/delta of runs against the theoretical arc of one 2theta."
    )
    parser.add_argument("spec_file", type=Path)
    parser.add_argument("runs", type=int, nargs="+", help="run numbers, plotted in the given order")
    args = parser.parse_args()
    spec_file, run_ns = args.spec_file, args.runs

    # --- 1. SPEC 데이터 로드 ---
    sp = Spec(spec_file)
    df = load_run_data(sp, run_ns)
    gamma_data = df["gamma"].values
    delta_data = df["delta"].values

    # --- 2. Matplotlib UI 및 그래프 초기화 ---
    fig, ax = plt.subplots(figsize=(11, 8))
    plt.subplots_adjust(bottom=0.25)

    # ⭐ 정렬 없이 run_ns 리스트 인덱스 순서대로 라인 연결
    (data_line,) = ax.plot(
        gamma_data,
        delta_data,
        color="black",
        lw=2.0,
        marker="o",
        markersize=6,
        alpha=0.8,
        label="Measured Run Data",
    )

    (theory_line,) = ax.plot(
        [], [], color="crimson", lw=2.5, linestyle="--", label="Theoretical Casing Arc"
    )

    ax.set_title(
        r"Diffractometer Space Check: Spec Data vs Pure Theoretical Curve",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel(r"Gamma ($\gamma$) [deg]", fontsize=11)
    ax.set_ylabel(r"Delta ($\delta$) [deg]", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)

    border_width = 1.5
    for side in ["top", "bottom", "left", "right"]:
        ax.spines[side].set_linewidth(border_width)
    ax.tick_params(direction="in", length=5, width=border_width, labelsize=10, top=True, right=True)

    # --- 3. 인터랙티브 위젯 배정 ---
    ax_tth = plt.axes([0.15, 0.11, 0.45, 0.03])
    ax_txt = plt.axes([0.72, 0.11, 0.15, 0.03])

    init_tth = 70.0
    STEP_SIZE = 0.001

    s_tth = Slider(
        ax_tth,
        r"True $2\theta$ (°)",
        30.0,
        120.0,
        valinit=init_tth,
        valstep=STEP_SIZE,
        valfmt="%.3f",
        color="teal",
    )
    t_tth = TextBox(ax_txt, r"Direct $2\theta$ ", initial=str(init_tth))

    # 이론 가이드라인은 부드러운 곡선 출력을 위해 Gamma 최솟값~최댓값 스페이스 유지
    g_range = np.linspace(gamma_data.min() - 4, gamma_data.max() + 4, 500)

    # --- 4. 실시간 가이드라인 업데이트 루틴 ---
    def update(val=None):
        current_tth = s_tth.val
        y_theory = calculate_theory_delta(g_range, current_tth)
        theory_line.set_data(g_range, y_theory)
        fig.suptitle(
            f"True $2\\theta$ (Adjusted): {current_tth:.3f}°\n(Use Mouse or Left/Right Arrow Keys)",
            fontsize=12,
            fontweight="bold",
            color="darkblue",
        )
        fig.canvas.draw_idle()

    def submit(text):
        try:
            val = float(text)
            s_tth.set_val(val)
        except ValueError:
            pass

    def on_key_press(event):
        if event.key == "right":
            new_val = min(s_tth.valmax, s_tth.val + STEP_SIZE)
            s_tth.set_val(new_val)
        elif event.key == "left":
            new_val = max(s_tth.valmin, s_tth.val - STEP_SIZE)
            s_tth.set_val(new_val)

    # 위젯 및 키보드 이벤트 바인딩
    s_tth.on_changed(update)
    t_tth.on_submit(submit)
    fig.canvas.mpl_connect("key_press_event", on_key_press)

    # 뷰포트 마진 세팅 및 범례 표시
    ax.set_xlim(gamma_data.min() - 2, gamma_data.max() + 2)
    ax.set_ylim(delta_data.min() - 1, delta_data.max() + 1)
    ax.legend(loc="upper right", frameon=True)

    # 첫 렌더링 구동
    update()
    plt.show()


if __name__ == "__main__":
    main()
