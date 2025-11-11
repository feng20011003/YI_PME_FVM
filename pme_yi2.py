# fvm.py
import numpy as np

def gradient_fd(q, dx):
        dq = np.zeros_like(q)
        dq[1:-1] = (q[2:] - q[:-2]) / (2.0 * dx)
        dq[0] = (q[1] - q[0]) / dx
        dq[-1] = (q[-1] - q[-2]) / dx
        return dq

def build_face_velocity_noflux(u):
    """將 cell-centered u 轉 face，並在兩端強制 0（零通量）。"""
    N = len(u)
    u_face = np.zeros(N + 1)
    u_face[1:-1] = 0.5 * (u[:-1] + u[1:])
    # 關鍵：邊界設為 0 → ρu 的邊界通量為 0
    u_face[0] = 0.0
    u_face[-1] = 0.0
    return u_face

def upwind_flux_dirichlet(q, u_face, q_left_bc, q_right_bc):
    N = len(q)
    F = np.zeros(N + 1)
    # 左邊界面（u_face[0] 已為 0，這裡仍寫完整邏輯）
    if u_face[0] >= 0.0:
        F[0] = u_face[0] * q_left_bc
    else:
        F[0] = u_face[0] * q[0]
    # 內部面
    qL, qR, uf_int = q[:-1], q[1:], u_face[1:-1]
    F[1:-1] = np.where(uf_int >= 0.0, uf_int * qL, uf_int * qR)
    # 右邊界面
    if u_face[-1] >= 0.0:
        F[-1] = u_face[-1] * q[-1]
    else:
        F[-1] = u_face[-1] * q_right_bc
    return F

def update_fvm_dirichlet(q, u, dt, dx, source_term=None, q_left_bc=None, q_right_bc=None):
    # 如果未提供，採用「零梯度」型 Dirichlet：使用當前邊界 cell 值
    if q_left_bc is None:
        q_left_bc = q[0]
    if q_right_bc is None:
        q_right_bc = q[-1]
    u_face = build_face_velocity_noflux(u)
    F = upwind_flux_dirichlet(q, u_face, q_left_bc, q_right_bc)
    q_new = q - (dt / dx) * (F[1:] - F[:-1])
    if source_term is not None:
        q_new = q_new + dt * source_term
    return q_new








def gradient_fd_neumann(q, dx):
    """
    一維中央差分 + Neumann(零梯度)鬼點
    q: (N,) -> dq/dx: (N,)
    """
    N = len(q)
    q_ext = np.empty(N + 2, dtype=q.dtype)
    q_ext[1:-1] = q
    q_ext[0] = q[0]     # 左鬼點: 零梯度
    q_ext[-1] = q[-1]   # 右鬼點: 零梯度
    return (q_ext[2:] - q_ext[:-2]) / (2.0 * dx)

def build_face_velocity_neumann(u):
    """cell 中心速度 -> 面速度；邊界面速度強制 0（無通量）"""
    N = len(u)
    u_face = np.zeros(N + 1, dtype=u.dtype)
    u_face[1:-1] = 0.5 * (u[:-1] + u[1:])
    u_face[0] = 0.0
    u_face[-1] = 0.0
    return u_face

def upwind_flux_neumann(q, u_face):
    """一階 upwind 通量；邊界通量自動為 0（因 u_face=0）"""
    N = len(q)
    F = np.zeros(N + 1, dtype=q.dtype)
    uf = u_face[1:-1]
    qL, qR = q[:-1], q[1:]
    F[1:-1] = np.where(uf >= 0.0, uf * qL, uf * qR)
    # F[0] = F[-1] = 0 已由 u_face 構造保證
    return F

def update_fvm_neumann(q, u, dt, dx, source_term=None):
    """單步 FVM 更新：q_t + (u q)_x = source"""
    u_face = build_face_velocity_neumann(u)
    F = upwind_flux_neumann(q, u_face)
    q_new = q - (dt / dx) * (F[1:] - F[:-1])
    if source_term is not None:
        q_new = q_new + dt * source_term
    return q_new

# ===== Barenblatt (1D) =====
def barenblatt_1d(x, t, m=2.0, Gamma=1.0):
    """
    1D Barenblatt 解（整條實線），t>0
    ρ(x,t) = t^{-α} * [ Γ - (α(m-1)/(2m)) * x^2 / t^{2α} ]_+^{1/(m-1)}
    α = 1/(m+1)
    """
    if t <= 0:
        raise ValueError("Barenblatt 解需要 t > 0（避免 δ 初值）")
    alpha = 1.0 / (m + 1.0)
    inside = Gamma - (alpha * (m - 1.0) / (2.0 * m)) * (x**2) / (t ** (2.0 * alpha))
    inside = np.maximum(inside, 0.0)
    return (t ** (-alpha)) * (inside ** (1.0 / (m - 1.0)))

# ===== 主模擬：PME (m 任意，預設 m=2) + Neumann =====
def run_pme_neumann_scheme(common_params, m, Gamma, X_MAX, RHO_FLOOR, t0=1e-3, center=None):
    """
    解 1D PME:  ρ_t = (ρ^m)_{xx}，Neumann(無通量) 邊界。
    以連續方程 ρ_t + (ρ u)_x = 0、速度 u = -∇p, p = m/(m-1) ρ^{m-1} 寫成 (I,Y) 形式來離散。

    初始值：取 Barenblatt 在 t=t0 的切片 (避免 δ 初值)。
    對照解析解：t -> t0 + t。
    """
    N_CELLS   = common_params['N_CELLS']
    NT_frames = common_params['NT_frames']
    T_MAX     = common_params['T_MAX']
    CFL_adv   = common_params['CFL_adv']
    CFL_diff  = common_params['CFL_diff']

    dx = X_MAX / N_CELLS
    x_face = np.linspace(0.0, X_MAX, N_CELLS + 1)
    x = 0.5 * (x_face[:-1] + x_face[1:])

    xc = 0.5 * X_MAX if center is None else float(center)

    # 初始密度 = Barenblatt(t0)
    rho0 = barenblatt_1d(x - xc, t0, m=m, Gamma=Gamma)
    rho  = np.maximum(rho0, 0.0)

    # (I, Y)
    I = np.ones_like(x)
    Y = x.copy()

    frame_times  = np.linspace(0.0, T_MAX, NT_frames)
    history_rho  = [rho.copy()]
    history_time = [0.0]
    next_frame   = 1

    inv_mminus1 = 1.0 / (m - 1.0)
    coeff = m * inv_mminus1  # m/(m-1)

    # 供重建用：ρ0(Y)
    def rho0_of(y):
        return barenblatt_1d(y - xc, t0, m=m, Gamma=Gamma)

    t = 0.0
    print(f"Starting PME (m={m}) simulation with Neumann BC...")
    while t < T_MAX - 1e-15:
        rho_clip = np.clip(rho, RHO_FLOOR, None)

        # u = -∇p,  p = m/(m-1) * ρ^{m-1}
        power = np.power(rho_clip, (m - 1.0))
        grad_power = gradient_fd_neumann(power, dx)
        u = - coeff * grad_power

        # CFL：對流限制
        umax = np.max(np.abs(u))
        dt_adv  = np.inf if umax   < 1e-4 else CFL_adv  * dx*1.0 / umax

        # CFL：等效擴散 D_eff = m * ρ^{m-1}
        D_eff = m * power
        Dmax  = np.max(D_eff)
        dt_diff = np.inf if Dmax < 1e-4 else CFL_diff * dx*dx / Dmax

        dt = min(dt_adv, dt_diff, T_MAX - t)

        # I_t + (u I)_x = 0
        I_next = update_fvm_dirichlet(I, u, dt, dx, q_left_bc=I[0], q_right_bc=I[-1])
        #I_next = update_fvm_neumann(I, u, dt, dx)
        I_next = np.maximum(I_next, 0.0)

        # Y_t + u Y_x = Y u_x
        du_dx = gradient_fd_neumann(u, dx)
        Y_next = update_fvm_neumann(Y, u, dt, dx, source_term=Y * du_dx)

        # ρ = I * ρ0(Y)
        Yc = Y_next
        rho_next = I_next * rho0_of(Yc)
        rho_next = np.maximum(rho_next, 0.0)

        I, Y, rho = I_next, Y_next, rho_next
        t += dt

        # 存影格
        while next_frame < len(frame_times) and t >= frame_times[next_frame] - 1e-12:
            history_rho.append(rho.copy())
            history_time.append(frame_times[next_frame])
            next_frame += 1

    print("PME simulation complete.")
    return x, np.asarray(history_time), np.asarray(history_rho)
