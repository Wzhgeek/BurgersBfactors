# Author: Zihan Wang
# <wangzh011031@163.com>
"""
一维周期性格点上的 Burgers 方程 + 图耦合（RK4）。

支持两种耦合模式:
  - graph_diffusion:    du_i/dt = -u_i·du_i/dx + ν·d²u_i/dx² - ε·Σ_j L_ij·u_j
                        其中 L = D - A_adj 为标准图拉普拉斯矩阵（图扩散）。
  - coupled_laplacian:  du_i/dt = -u_i·du_i/dx + ν·d²u_i/dx² + ε·Σ_j A_ij·∂²u_j/∂x²
                        耦合通过空间二阶导，等价于 ε·A·D_xx·u。
"""

import numpy as np


def burgers_rhs(
    u: np.ndarray,
    A: np.ndarray,
    nu: float,
    epsilon: float,
    dx: float,
    coupling_mode: str = "graph_diffusion",
) -> np.ndarray:
    """
    计算 Burgers + 图耦合方程的右端。

    du/dt = -u * du/dx + nu * d²u/dx² + graph_coupling

    Args:
        u: 状态向量 (N,)
        A: Step1 生成的 Aij 矩阵 (N, N)，对角为负、行和为零。
        nu: 扩散系数
        epsilon: 图耦合强度
        dx: 格点间距
        coupling_mode:
            "graph_diffusion"   — graph_coupling = -epsilon * L @ u, L = D - A_adj
            "coupled_laplacian" — graph_coupling = epsilon * A @ laplacian_1d

    Returns:
        rhs: (N,) 右端向量
    """
    n = u.shape[0]
    im = (np.arange(n) - 1) % n  # i-1 (周期边界)
    ip = (np.arange(n) + 1) % n  # i+1 (周期边界)

    # 对流项: -u * du/dx
    dudx = (u[ip] - u[im]) / (2.0 * dx)
    convection = -u * dudx

    # 扩散项: nu * d²u/dx²
    laplacian_1d = (u[ip] - 2.0 * u + u[im]) / (dx * dx)
    diffusion = nu * laplacian_1d

    # 图耦合项
    if coupling_mode == "graph_diffusion":
        A_adj = A.copy()
        np.fill_diagonal(A_adj, 0.0)
        D = np.diag(np.sum(A_adj, axis=1))
        L = D - A_adj
        graph_coupling = -epsilon * (L @ u)
    elif coupling_mode == "coupled_laplacian":
        # ε * Σ_j A_ij * ∂²u_j/∂x² = ε * A @ laplacian_1d
        graph_coupling = epsilon * (A @ laplacian_1d)
    else:
        raise ValueError(f"未知 coupling_mode: {coupling_mode}")

    return convection + diffusion + graph_coupling


def rk4_step(
    u: np.ndarray,
    dt: float,
    A: np.ndarray,
    nu: float,
    epsilon: float,
    dx: float,
    coupling_mode: str = "graph_diffusion",
) -> np.ndarray:
    """四阶龙格-库塔单步推进。返回更新后的 u。"""
    k1 = burgers_rhs(u, A, nu, epsilon, dx, coupling_mode)
    k2 = burgers_rhs(u + 0.5 * dt * k1, A, nu, epsilon, dx, coupling_mode)
    k3 = burgers_rhs(u + 0.5 * dt * k2, A, nu, epsilon, dx, coupling_mode)
    k4 = burgers_rhs(u + dt * k3, A, nu, epsilon, dx, coupling_mode)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
