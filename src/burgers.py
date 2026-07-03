# Author: Zihan Wang
# <wangzh011031@163.com>
"""
一维周期性格点上的 Burgers 方程 + 图耦合（RK4）。

支持两种耦合模式:
  - graph_diffusion:    du_i/dt = -u_i·du_i/dx + ν·d²u_i/dx² - ε·Σ_j L_ij·u_j
                        其中 L = D - A_adj 为标准图拉普拉斯矩阵（图扩散）。
  - coupled_laplacian:  du_i/dt = -u_i·du_i/dx + ν·d²u_i/dx² + ε·Σ_j A_ij·∂²u_j/∂x²
                        耦合通过空间二阶导，等价于 ε·A·D_xx·u。

性能说明:
  - 推荐先调用 precompute_laplacian(A) 得到 L，再传入 rk4_step/burgers_rhs，
    避免每次 RHS 评估时重复构造 L（RK4 每步调用 4 次 RHS）。
  - 批量版本 rk4_step_batch / burgers_rhs_batch 接受 (batch_size, n) 状态矩阵，
    将 BLAS 矩阵-矩阵乘替代多次矩阵-向量乘，大幅提升脉冲响应并行演化效率。
  - _periodic_indices(n) 用 lru_cache 缓存周期边界索引，避免每次 RHS 调用重建。
"""

from functools import lru_cache

import numpy as np


# ─── 工具函数 ────────────────────────────────────────────────────────────────


@lru_cache(maxsize=16)
def _periodic_indices(n: int):
    """缓存长度为 n 的周期边界左右移索引（im, ip）。"""
    im = (np.arange(n) - 1) % n
    ip = (np.arange(n) + 1) % n
    return im, ip


def precompute_laplacian(A: np.ndarray) -> np.ndarray:
    """
    从 Aij 矩阵预计算图拉普拉斯矩阵 L = D - A_adj。

    Args:
        A: Aij 矩阵 (N, N)，对角为负、行和为零（来自 build_graph.py）。

    Returns:
        L: 图拉普拉斯矩阵 (N, N)，可缓存后反复传入 rk4_step / burgers_rhs。
    """
    A_adj = A.copy()
    np.fill_diagonal(A_adj, 0.0)
    D = np.diag(A_adj.sum(axis=1))
    return D - A_adj


# ─── 单向量版（单个脉冲响应） ─────────────────────────────────────────────────


def burgers_rhs(
    u: np.ndarray,
    A: np.ndarray,
    nu: float,
    epsilon: float,
    dx: float,
    coupling_mode: str = "graph_diffusion",
    L_cache: np.ndarray | None = None,
) -> np.ndarray:
    """
    计算 Burgers + 图耦合方程的右端（单向量版）。

    Args:
        u:            状态向量 (N,)
        A:            Aij 矩阵 (N, N)，对角为负、行和为零。
        nu:           扩散系数
        epsilon:      图耦合强度
        dx:           格点间距
        coupling_mode:
            "graph_diffusion"   — graph_coupling = -epsilon * L @ u
            "coupled_laplacian" — graph_coupling = epsilon * A @ laplacian_1d
        L_cache:      预计算的拉普拉斯矩阵（graph_diffusion 模式有效）。
                      传入后跳过 L 的重复计算，每次 RK4 节省 4 次 O(N²) 构造。

    Returns:
        rhs: (N,) 右端向量
    """
    n = u.shape[0]
    im, ip = _periodic_indices(n)

    dudx = (u[ip] - u[im]) / (2.0 * dx)
    convection = -u * dudx

    laplacian_1d = (u[ip] - 2.0 * u + u[im]) / (dx * dx)
    diffusion = nu * laplacian_1d

    if coupling_mode == "graph_diffusion":
        if L_cache is not None:
            L = L_cache
        else:
            A_adj = A.copy()
            np.fill_diagonal(A_adj, 0.0)
            D = np.diag(np.sum(A_adj, axis=1))
            L = D - A_adj
        graph_coupling = -epsilon * (L @ u)
    elif coupling_mode == "coupled_laplacian":
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
    L_cache: np.ndarray | None = None,
) -> np.ndarray:
    """
    四阶龙格-库塔单步推进（单向量版）。

    Args:
        L_cache: 预计算的图拉普拉斯（推荐传入以避免 4 次重复计算）。
    """
    k1 = burgers_rhs(u,                 A, nu, epsilon, dx, coupling_mode, L_cache)
    k2 = burgers_rhs(u + 0.5 * dt * k1, A, nu, epsilon, dx, coupling_mode, L_cache)
    k3 = burgers_rhs(u + 0.5 * dt * k2, A, nu, epsilon, dx, coupling_mode, L_cache)
    k4 = burgers_rhs(u + dt * k3,        A, nu, epsilon, dx, coupling_mode, L_cache)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ─── 批量版（多脉冲并行） ────────────────────────────────────────────────────


def burgers_rhs_batch(
    U: np.ndarray,
    A: np.ndarray,
    nu: float,
    epsilon: float,
    dx: float,
    coupling_mode: str = "graph_diffusion",
    L_cache: np.ndarray | None = None,
) -> np.ndarray:
    """
    计算 Burgers + 图耦合方程的右端（批量版）。

    将 batch_size 个独立脉冲响应并行计算：
      - 对流/扩散项为逐元素操作，行间无耦合。
      - 图耦合项用 BLAS dgemm（U @ L）替代 batch_size 次矩阵-向量乘，
        大规模时比串行快一个数量级。

    数学等价性（graph_diffusion）：
      - 对任意行 i：(U @ L)[i] = U[i] @ L = L @ U[i]（L 对称）
      - 即与单向量 L @ u_i 完全等价。

    Args:
        U:            状态矩阵 (batch_size, N)，每行是一个脉冲的状态向量。
        A:            Aij 矩阵 (N, N)。
        nu, epsilon, dx, coupling_mode: 同 burgers_rhs。
        L_cache:      预计算的图拉普拉斯（推荐传入）。

    Returns:
        rhs: (batch_size, N) 各行的右端向量。
    """
    n = U.shape[1]
    im, ip = _periodic_indices(n)

    U_ip = U[:, ip]   # (batch, n)
    U_im = U[:, im]   # (batch, n)

    dudx = (U_ip - U_im) / (2.0 * dx)
    convection = -U * dudx

    laplacian_1d = (U_ip - 2.0 * U + U_im) / (dx * dx)
    diffusion = nu * laplacian_1d

    if coupling_mode == "graph_diffusion":
        if L_cache is not None:
            L = L_cache
        else:
            A_adj = A.copy()
            np.fill_diagonal(A_adj, 0.0)
            D = np.diag(A_adj.sum(axis=1))
            L = D - A_adj
        # L 对称：(U @ L)[i] = L @ U[i]，用一次 dgemm 替代 batch_size 次 gemv
        graph_coupling = -epsilon * (U @ L)
    elif coupling_mode == "coupled_laplacian":
        # A 对称：(laplacian_1d @ A)[i] = A @ laplacian_1d[i]
        graph_coupling = epsilon * (laplacian_1d @ A)
    else:
        raise ValueError(f"未知 coupling_mode: {coupling_mode}")

    return convection + diffusion + graph_coupling


def rk4_step_batch(
    U: np.ndarray,
    dt: float,
    A: np.ndarray,
    nu: float,
    epsilon: float,
    dx: float,
    coupling_mode: str = "graph_diffusion",
    L_cache: np.ndarray | None = None,
) -> np.ndarray:
    """
    四阶龙格-库塔单步推进（批量版）。

    Args:
        U:       状态矩阵 (batch_size, N)。
        L_cache: 预计算的图拉普拉斯（推荐传入）。

    Returns:
        U_new: (batch_size, N) 更新后的状态矩阵。
    """
    k1 = burgers_rhs_batch(U,                 A, nu, epsilon, dx, coupling_mode, L_cache)
    k2 = burgers_rhs_batch(U + 0.5 * dt * k1, A, nu, epsilon, dx, coupling_mode, L_cache)
    k3 = burgers_rhs_batch(U + 0.5 * dt * k2, A, nu, epsilon, dx, coupling_mode, L_cache)
    k4 = burgers_rhs_batch(U + dt * k3,        A, nu, epsilon, dx, coupling_mode, L_cache)
    return U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
