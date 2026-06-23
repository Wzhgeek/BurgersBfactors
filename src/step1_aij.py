# Author: Zihan Wang
# <wangzh011031@163.com>
"""Aij 邻接权重矩阵计算。"""

import numpy as np


def _off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    if n < 2:
        return np.array([], dtype=np.float64)
    return matrix[np.triu_indices(n, k=1)]


def compute_aij_matrix(
    filtered_dist: np.ndarray,
    k: int = 1,
) -> np.ndarray:
    """
    按公式计算 Aij 矩阵。

    非对角 (i != j): A_ij = exp(-d_ij^k / (k * sigma^k))，仅当 d_ij > 0
    对角 (i = j):    A_ii = -sum_{j != i} A_ij

    sigma 取当前过滤距离矩阵中非零非对角元素的中位数；
    若不存在正值，则 sigma = 1.0 避免除零。
    """
    n = filtered_dist.shape[0]
    aij = np.zeros((n, n), dtype=np.float64)

    off_pos = _off_diagonal_values(filtered_dist)
    off_pos = off_pos[off_pos > 0]
    if off_pos.size == 0:
        return aij

    sigma = float(np.median(off_pos))
    if sigma <= 0:
        sigma = 1.0

    denom = k * (sigma**k)
    idx = np.where((filtered_dist > 0) & ~np.eye(n, dtype=bool))
    d_vals = filtered_dist[idx]
    aij[idx] = np.exp(-(d_vals**k) / denom)

    row_sum = np.sum(aij, axis=1) - np.diag(aij)
    np.fill_diagonal(aij, -row_sum)
    return aij
