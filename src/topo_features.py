# Author: Zihan Wang
# <wangzh011031@163.com>
"""V1 拓扑特征: 拉普拉斯谱 (10 level x 6 feature, 扰动实验)。"""

from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from scipy.spatial.distance import pdist, squareform

TOPO_FEATURE_NAMES = [
    "harmonic_multiplicity",
    "eig_min_nz",
    "eig_mean",
    "eig_std",
    "eig_max",
    "eig_sum",
]


# ── 拉普拉斯谱 ─────────────────────────────────────────

def _laplacian(mat, eps=1e-10):
    m = mat + eps * np.eye(mat.shape[0])
    m = 0.5 * (m + m.T)
    return np.linalg.eigvalsh(m)


def _spectral_features(lap, zero_tol=1e-8):
    evals = _laplacian(lap)
    scale = max(float(np.max(np.abs(evals))), 1.0) if evals.size else 1.0
    tol = zero_tol * scale
    harmonic = int(np.sum(np.abs(evals) <= tol))
    nonzero = evals[np.abs(evals) > tol]
    if nonzero.size == 0:
        return [harmonic, np.nan, np.nan, np.nan, np.nan, np.nan]
    return np.array([
        harmonic,
        float(np.min(np.abs(nonzero))),
        float(np.mean(nonzero)),
        float(np.std(nonzero)),
        float(np.max(nonzero)),
        float(np.sum(nonzero)),
    ])


def resolve_n_jobs(n_jobs: int | None = None) -> int:
    """并行 worker 数；默认 SLURM_CPUS_PER_TASK 或 cpu_count。"""
    if n_jobs is not None and n_jobs > 0:
        return int(n_jobs)
    env = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("OMP_NUM_THREADS")
    if env and str(env).isdigit() and int(env) > 0:
        return int(env)
    return int(os.cpu_count() or 1)


def _perturb_one_cell(aij_matrix: np.ndarray, base: np.ndarray, i: int):
    """单细胞扰动：去掉 i 后谱特征差分（供多进程调用）。"""
    n = aij_matrix.shape[0]
    keep = np.delete(np.arange(n, dtype=int), i)
    if keep.size < 2:
        return i, None
    sub = aij_matrix[np.ix_(keep, keep)]
    return i, np.abs(base - _spectral_features(sub))


def perturbation_from_aij(
    aij_matrix: np.ndarray,
    n_jobs: int | None = None,
) -> np.ndarray:
    """对已有拉普拉斯型 Aij 做逐节点扰动谱差分，返回 (n, 6)。

    复杂度约 O(n * n^3)（每层 n 次 eigvalsh）；n_jobs>1 时按细胞并行。
    """
    n = aij_matrix.shape[0]
    if n < 3:
        return np.zeros((n, len(TOPO_FEATURE_NAMES)), dtype=np.float64)
    base = _spectral_features(aij_matrix)
    features = np.zeros((n, len(TOPO_FEATURE_NAMES)), dtype=np.float64)
    workers = resolve_n_jobs(n_jobs)
    if workers <= 1 or n < 4:
        for i in range(n):
            _, row = _perturb_one_cell(aij_matrix, base, i)
            if row is not None:
                features[i, :] = row
        return np.nan_to_num(features, nan=0.0)

    worker = partial(_perturb_one_cell, aij_matrix, base)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in pool.map(
            worker,
            range(n),
            chunksize=max(1, n // (workers * 4)),
        ):
            if row is not None:
                features[i, :] = row
    return np.nan_to_num(features, nan=0.0)


def stack_level_features(level_features: list[np.ndarray]) -> np.ndarray:
    """拼接多层特征 → (n, num_levels * 6)。"""
    return np.hstack(level_features)


# ── 距离 & Aij ─────────────────────────────────────────

def _pairwise_distance(coords):
    n = coords.shape[0]
    if n < 2:
        return np.zeros((n, n))
    d = squareform(pdist(coords, "euclidean")).astype(np.float64)
    np.fill_diagonal(d, 0)
    return d


def _rips_filter(dist, threshold):
    n = dist.shape[0]
    f = np.zeros_like(dist)
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] <= threshold:
                f[i, j] = f[j, i] = dist[i, j]
    return f


def _compute_aij(filtered_dist, k=1):
    n = filtered_dist.shape[0]
    off = filtered_dist[np.triu_indices(n, k=1)]
    off_pos = off[off > 0]
    if off_pos.size == 0:
        return np.zeros((n, n))
    sigma = float(np.median(off_pos)) or 1.0
    denom = k * (sigma ** k)
    aij = np.zeros((n, n))
    idx = np.where((filtered_dist > 0) & ~np.eye(n, dtype=bool))
    aij[idx] = np.exp(-(filtered_dist[idx] ** k) / denom)
    np.fill_diagonal(aij, -np.sum(aij, axis=1) + np.diag(aij))
    return aij


# ── 扰动实验 ───────────────────────────────────────────

def _perturbation_experiment(filtered_dist, aij_k=1, n_jobs: int | None = None):
    n = filtered_dist.shape[0]
    if n < 3:
        return np.zeros((n, len(TOPO_FEATURE_NAMES)))
    lap_full = _compute_aij(filtered_dist, k=aij_k)
    return perturbation_from_aij(lap_full, n_jobs=n_jobs)


# ── 批量提取 ───────────────────────────────────────────

def _load_xyzb(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    rows.append([float(p) for p in parts[:4]])
                except ValueError:
                    continue
    if not rows:
        return np.zeros((0, 3)), np.zeros((0,))
    data = np.array(rows, dtype=np.float64)
    return data[:, :3], data[:, 3]


def _compute_thresholds(dist, num_levels=10):
    off = dist[np.triu_indices(dist.shape[0], k=1)]
    return np.percentile(off, np.linspace(5, 95, num_levels))


def extract_protein_topo(xyzb_path, num_levels=10, aij_k=1):
    coords, labels = _load_xyzb(xyzb_path)
    if coords.shape[0] < 3 or labels.shape[0] == 0:
        return None
    dist = _pairwise_distance(coords)
    thresholds = _compute_thresholds(dist, num_levels)
    parts = [_perturbation_experiment(_rips_filter(dist, thr), aij_k)
             for thr in thresholds]
    return np.hstack(parts)


def save_topo_csv(xyzb_path, output_path, num_levels=10):
    coords, labels = _load_xyzb(xyzb_path)
    pdb_id = xyzb_path.stem.replace("_ca", "")
    if coords.shape[0] < 3 or labels.shape[0] == 0:
        print(f"  [SKIP] {pdb_id}")
        return
    feat = extract_protein_topo(xyzb_path, num_levels)
    cols = [f"{name}_L{lv:02d}" for lv in range(1, num_levels + 1)
            for name in TOPO_FEATURE_NAMES]
    header = f"pdb_id,atom_index,{','.join(cols)},label"
    lines = [header]
    for i in range(feat.shape[0]):
        row = f"{pdb_id},{i}," + ",".join(f"{v:.6f}" for v in feat[i]) + f",{labels[i]:.4f}"
        lines.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"  {pdb_id}: {feat.shape[0]} atoms x {feat.shape[1]} features")


def extract_dataset(xyzb_dir, output_dir, num_levels=10):
    xyzb_files = sorted(xyzb_dir.glob("*_ca.xyzb"))
    print(f"提取 {len(xyzb_files)} proteins -> {output_dir}")
    for f in xyzb_files:
        out = output_dir / f"{f.stem.replace('_ca', '')}_topo_feature.csv"
        save_topo_csv(f, out, num_levels)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    base = Path("/Volumes/CodeandDataset/ProNAB/BfactorBT")
    extract_dataset(base / "code_data" / args.dataset,
                    base / "result" / args.dataset / "topo_features")
