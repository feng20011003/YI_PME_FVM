# main_pme_m2.py
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")  # ⚠️ 一定要在 import pyplot 之前
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from pme_yi import run_pme_neumann_scheme, barenblatt_1d

if __name__ == "__main__":
    # ---- 基本參數 ----
    m = 2.0
    Gamma = 1.0
    X_domain_max = 10.0
    T_sim_max = 0.5
    t0 = 0.1                         # Barenblatt 初始化用的起始時間
    center = X_domain_max / 2.0       # 置中，避免一開始貼邊界

    common_params = {
        'N_CELLS': 401,
        'NT_frames': 121,
        'T_MAX': T_sim_max,
        'CFL_adv': 0.5,
        'CFL_diff': 0.3,
    }
    RHO_FLOOR_const = 1e-9

    # ---- 數值解 ----
    print("--- Running PME Neumann Scheme (I, Y), m=2 ---")
    x_grid, t_grid, history_rho = run_pme_neumann_scheme(
        common_params, m, Gamma, X_domain_max, RHO_FLOOR_const, t0=t0, center=center
    )

    # ---- 解析解（用 t0 + t 對照）----
    def exact_bar(x, t):
        return barenblatt_1d(x - center, t0 + t, m=m, Gamma=Gamma)

    # ---- 輸出資料夾 ----
    output_dir = "results_pme_m2"
    os.makedirs(output_dir, exist_ok=True)

    # ========= (A) 解對照動畫 =========
    print("\nCreating PME comparison animation...")
    fig_anim, ax_anim = plt.subplots(figsize=(10, 6))

    line_fvm, = ax_anim.plot([], [], 'r--', lw=2.5, label="FVM (I,Y) PME m=2")
    line_exact, = ax_anim.plot([], [], 'g-', lw=2.0, alpha=0.8, label="Barenblatt (exact)")
    time_text = ax_anim.text(0.05, 0.9, '', transform=ax_anim.transAxes)

    ax_anim.set_xlim(0.0, X_domain_max)

    # 粗估 y 範圍（避免被 0~1 固定死）
    sample_ts = np.linspace(0, T_sim_max, 5)
    ymax = 0.0
    for tt in sample_ts:
        ymax = max(ymax, np.max(exact_bar(x_grid, tt)))
    ymax = max(ymax, np.max(history_rho[0]))
    ax_anim.set_ylim(0.0, ymax * 1.2 + 1e-12)

    ax_anim.set_xlabel('x')
    ax_anim.set_ylabel(r'$\rho(x,t)$')
    ax_anim.set_title(r'1D PME: $\rho_t=(\rho^m)_{xx}$, $m=2$ (Neumann BC)')
    ax_anim.legend(loc='upper right')
    ax_anim.grid(True)

    def update_solution(frame):
        t = t_grid[frame]
        line_fvm.set_data(x_grid, history_rho[frame])
        line_exact.set_data(x_grid, exact_bar(x_grid, t))
        time_text.set_text(f't = {t:.4f} s')
        return line_fvm, line_exact, time_text

    anim = FuncAnimation(fig_anim, update_solution, frames=len(history_rho), interval=50, blit=True)
    writer = PillowWriter(fps=15)
    anim_path = f'{output_dir}/animation_pme_m2.gif'
    anim.save(anim_path, writer=writer)
    print(f"PME animation saved to {anim_path}")
    plt.close(fig_anim)

    # ========= (B) 誤差動畫 =========
    print("\nCreating error animation...")
    all_err_sq = []
    all_mse = []
    for i, t in enumerate(t_grid):
        num = history_rho[i]
        ref = exact_bar(x_grid, t)
        e2 = (num - ref)**2
        all_err_sq.append(e2)
        all_mse.append(np.mean(e2))

    max_e2 = float(np.max(all_err_sq))

    fig_err, ax_err = plt.subplots(figsize=(10, 6))
    line_err, = ax_err.plot([], [], 'b-', lw=2, label="Squared Error")
    txt = ax_err.text(0.05, 0.95, '', transform=ax_err.transAxes,
                      va='top', fontsize=12,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax_err.set_xlim(0.0, X_domain_max)
    ax_err.set_ylim(0, max_e2 * 1.1 + 1e-15)
    ax_err.set_xlabel('x')
    ax_err.set_ylabel('Squared Error')
    ax_err.set_title('Pointwise Squared Error Over Time')

    sf = ScalarFormatter(useMathText=True)
    sf.set_powerlimits((0, 0))              # 自動科學記號
    ax_err.yaxis.set_major_formatter(sf)
    ax_err.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_err.legend(loc='upper right')
    ax_err.grid(True)

    def update_error(frame):
        t = t_grid[frame]
        line_err.set_data(x_grid, all_err_sq[frame])
        txt.set_text(f't = {t:.4f} s\nMSE = {all_mse[frame]:.3e}')
        return line_err, txt

    anim_err = FuncAnimation(fig_err, update_error, frames=len(t_grid), interval=50, blit=True)
    err_path = f'{output_dir}/animation_error.gif'
    anim_err.save(err_path, writer=writer)
    print(f"Error animation saved to {err_path}")
    plt.close(fig_err)

    print("\nAll tasks complete!")
