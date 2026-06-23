# Author: Zihan Wang
# <wangzh011031@163.com>
"""距离矩阵与 Rips 过滤阈值计算。"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def pairwise_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    计算所有原子对的欧氏距离矩阵。

    Args:
        coords: (N, 3)

    Returns:
        (N, N) 对称矩阵，对角线为 0
    """
    n = coords.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    dist_vec = pdist(coords, metric="euclidean")
    dist = squareform(dist_vec).astype(np.float64)
    np.fill_diagonal(dist, 0.0)
    return dist


def off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    """提取矩阵非对角元素（上三角，不含对角）。"""
    n = matrix.shape[0]
    if n < 2:
        return np.array([], dtype=np.float64)
    idx = np.triu_indices(n, k=1)
    return matrix[idx]


def compute_filtration_thresholds(
    dist: np.ndarray,
    num_levels: int = 10,
) -> tuple[np.ndarray, float, float]:
    """
    按距离分布的分位数生成 num_levels 个过滤半径。

    在 [p5, p95] 上取 num_levels 个等分位点，使每层包含大致相同的原子对增量。

    Returns:
        thresholds: (num_levels,) 递增阈值
        d_min, d_max: 非对角距离极值
    """
    off = off_diagonal_values(dist)
    if off.size == 0:
        return np.zeros(num_levels, dtype=np.float64), 0.0, 0.0

    d_min = float(np.min(off))
    d_max = float(np.max(off))
    if d_min == d_max:
        thresholds = np.full(num_levels, d_max, dtype=np.float64)
    else:
        percentiles = np.linspace(5, 95, num_levels)
        thresholds = np.percentile(off, percentiles).astype(np.float64)
    return thresholds, d_min, d_max


def rips_filtered_distance(dist: np.ndarray, threshold: float) -> np.ndarray:
    """
    Rips 单纯复形过滤：距离 <= 阈值则保留距离，否则置 0。

    对角线恒为 0。
    """
    n = dist.shape[0]
    filtered = np.zeros_like(dist, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = dist[i, j]
            if d <= threshold:
                filtered[i, j] = d
                filtered[j, i] = d
    return filtered


def to_binary_matrix(filtered_dist: np.ndarray) -> np.ndarray:
    """非零元素置 1，零置 0（含对角线）。"""
    return (filtered_dist != 0).astype(np.float64)
