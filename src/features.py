# Author: Zihan Wang
# <wangzh011031@163.com>
"""
特征提取：Burgers 脉冲响应轨迹的统计特征与轨迹模拟。
"""

from pathlib import Path

import numpy as np

from .burgers import rk4_step, precompute_laplacian


# ── 统计特征 ──────────────────────────────────────────────────────────────


def extract_stats_features(trj: np.ndarray) -> np.ndarray:
    """
    由轨迹矩阵 (n_atoms, n_times) 计算逐原子统计量。

    Args:
        trj: (n_atoms, n_times) 每个原子在各时间点的自身分量 u_i(t)。

    Returns:
        stats: (n_atoms, 6) 列依次为 [max, min, mean, var, median, std]。
    """
    return np.column_stack([
        trj.max(axis=1),
        trj.min(axis=1),
        trj.mean(axis=1),
        trj.var(axis=1),
        np.median(trj, axis=1),
        trj.std(axis=1),
    ])


# ── 轨迹模拟 ──────────────────────────────────────────────────────────────


def simulate_trajectories(
    A: np.ndarray,
    nu: float,
    epsilon: float,
    dt: float,
    dx: float,
    n_steps: int,
    n_atoms: int,
    n_sample_points: int = 100,
    coupling_mode: str = "graph_diffusion",
) -> tuple[np.ndarray, np.ndarray]:
    """
    对每个原子 i，以脉冲初值 u(0) = e_i 演化，记录 u_i(t) 轨迹并提取统计特征。

    Args:
        A: Aij 矩阵 (n_atoms, n_atoms)。
        nu: 扩散系数。
        epsilon: 图耦合强度。
        dt: 时间步长。
        dx: 格点间距。
        n_steps: 演化总步数。
        n_atoms: 原子数。
        n_sample_points: 从 n_steps 演化中等距采样的时间点数。
        coupling_mode: "graph_diffusion" 或 "coupled_laplacian"。

    Returns:
        trj: (n_atoms, n_sample_points) 采样轨迹。
        stats: (n_atoms, 6) 统计特征 [max, min, mean, var, median, std]。
    """
    # boolean 掩码替代 set 查找，O(1) 数组索引无 Python hash 开销
    sample_steps = np.linspace(0, n_steps, n_sample_points, dtype=int)
    sample_mask = np.zeros(n_steps + 1, dtype=bool)
    sample_mask[sample_steps] = True
    # 预计算图拉普拉斯：L 在整个模拟期间不变，避免每次 rk4_step 内重建
    L_cache = precompute_laplacian(A) if coupling_mode == "graph_diffusion" else None

    trj = np.zeros((n_atoms, n_sample_points), dtype=np.float64)

    for i in range(n_atoms):
        u = np.zeros(n_atoms, dtype=np.float64)
        u[i] = 1.0
        save_idx = 0

        for step in range(n_steps + 1):
            if sample_mask[step]:
                trj[i, save_idx] = u[i]
                save_idx += 1

            if step < n_steps:
                u = rk4_step(u, dt, A, nu, epsilon, dx, coupling_mode, L_cache=L_cache)

    return trj, extract_stats_features(trj)


# ── Aij 矩阵加载 ──────────────────────────────────────────────────────────


def load_aij_matrices(
    step1_dir: str | Path,
    pdb_id: str,
    num_levels: int = 10,
) -> list[np.ndarray]:
    """
    加载某个蛋白所有过滤层级的 Aij 矩阵。

    Args:
        step1_dir: Step1 结果根目录（含各蛋白子目录）。
        pdb_id: 蛋白 PDB ID。
        num_levels: 过滤层级数。

    Returns:
        aij_list: 长度为 num_levels 的 (n, n) ndarray 列表。
    """
    step1_dir = Path(step1_dir)
    aij_list = []
    for level in range(1, num_levels + 1):
        aij_path = step1_dir / pdb_id / "Aijandlabel" / f"{pdb_id}_Aij_0-{level}.npy"
        if not aij_path.exists():
            raise FileNotFoundError(f"Aij not found: {aij_path}")
        aij_list.append(np.load(aij_path))
    return aij_list


# ── 过滤阈值 ──────────────────────────────────────────────────────────────


def compute_thresholds(
    dist: np.ndarray,
    num_levels: int,
    pct_start: float = 5.0,
    pct_stop: float = 95.0,
) -> np.ndarray:
    """
    按距离分布的分位数生成 Rips 过滤阈值。

    在 [pct_start, pct_stop] 百分位区间取 num_levels 个等分位点，
    使每层包含大致相同的原子对增量。

    Args:
        dist: (N, N) 距离矩阵。
        num_levels: 过滤层级数。
        pct_start: 起始百分位（默认 5）。
        pct_stop: 终止百分位（默认 95）。

    Returns:
        thresholds: (num_levels,) 递增阈值。
    """
    n = dist.shape[0]
    if n < 2:
        return np.zeros(num_levels, dtype=np.float64)

    # 提取上三角非对角元素
    idx = np.triu_indices(n, k=1)
    off = dist[idx]

    if off.size == 0:
        return np.zeros(num_levels, dtype=np.float64)

    d_min = float(np.min(off))
    d_max = float(np.max(off))
    if d_min == d_max:
        return np.full(num_levels, d_max, dtype=np.float64)

    percentiles = np.linspace(pct_start, pct_stop, num_levels)
    thresholds = np.percentile(off, percentiles).astype(np.float64)
    return thresholds
