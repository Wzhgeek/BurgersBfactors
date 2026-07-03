# Author: Zihan Wang
# <wangzh011031@163.com>
"""拓扑特征: Rips H0/H1 barcode + 拉普拉斯迹 (10 level x 6 feature)。"""

from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

TOPO_FEATURE_NAMES = [
    "H0_bar_count", "H1_birth_count", "H1_death_count",
    "trace_L", "eig_mean", "eig_std",
]

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False


def _barcode_counts(adj_matrix):
    """H0/H1 有限消亡条码数。"""
    if not HAS_RIPSER or adj_matrix.shape[0] < 3:
        return 0, 0
    A = adj_matrix.copy()
    np.fill_diagonal(A, 0)
    pos = A[A > 0]
    if len(pos) == 0:
        return 0, 0
    d_max, d_min = np.max(pos), np.min(pos)
    if d_max <= d_min:
        return 0, 0
    D = np.clip((d_max - A) / (d_max - d_min), 0, 1)
    np.fill_diagonal(D, 0)
    D = 0.5 * (D + D.T)
    try:
        dgms = ripser(D, maxdim=1, distance_matrix=True, thresh=np.inf)['dgms']
        h0 = sum(1 for b, d in dgms[0] if d != np.inf and d - b > 1e-5) if len(dgms) > 0 else 0
        h1 = sum(1 for b, d in dgms[1] if d != np.inf and d - b > 1e-5) if len(dgms) > 1 else 0
        return float(h0), float(h1)
    except Exception:
        return 0, 0


def _geom_features(adj_matrix):
    """迹公式: [trace_L, mean_lambda, std_lambda]。"""
    deg = np.abs(adj_matrix).sum(axis=1)
    tr = float(np.sum(deg))
    N = adj_matrix.shape[0]
    mu = tr / N if N > 0 else 0
    ms = (np.sum(deg**2) + np.sum(adj_matrix**2)) / N if N > 0 else 0
    var = max(ms - mu**2, 0)
    return np.array([tr, mu, np.sqrt(var)], dtype=np.float32)


def _extract_6d(adj_matrix):
    h0, h1 = _barcode_counts(adj_matrix)
    g = _geom_features(adj_matrix)
    return np.array([h0, h1, h1, g[0], g[1], g[2]], dtype=np.float32)


# ── 距离 & Aij ──────────────────────────────────────

def pairwise_distance(coords):
    n = coords.shape[0]
    if n < 2:
        return np.zeros((n, n))
    d = squareform(pdist(coords, "euclidean")).astype(np.float64)
    np.fill_diagonal(d, 0)
    return d


def rips_filter(dist, threshold):
    n = dist.shape[0]
    f = np.zeros_like(dist)
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] <= threshold:
                f[i, j] = f[j, i] = dist[i, j]
    return f


def compute_aij(filtered_dist, k=1):
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


# ── 扰动实验 ────────────────────────────────────────

def perturbation_experiment(filtered_dist, aij_k=1):
    n = filtered_dist.shape[0]
    if n < 3:
        return np.zeros((n, len(TOPO_FEATURE_NAMES)), dtype=np.float32)
    adj_full = compute_aij(filtered_dist, k=aij_k)
    base = _extract_6d(adj_full)
    features = np.zeros((n, len(TOPO_FEATURE_NAMES)), dtype=np.float32)
    for atom_idx in range(n):
        keep = [i for i in range(n) if i != atom_idx]
        if len(keep) < 2:
            continue
        sub_adj = compute_aij(filtered_dist[np.ix_(keep, keep)], k=aij_k)
        features[atom_idx, :] = np.abs(base - _extract_6d(sub_adj))
    return np.nan_to_num(features, nan=0.0)


# ── 批量提取 ────────────────────────────────────────

def load_xyzb(path):
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


def compute_thresholds(dist, num_levels=10):
    off = dist[np.triu_indices(dist.shape[0], k=1)]
    return np.percentile(off, np.linspace(5, 95, num_levels))


def save_topo_csv(xyzb_path, output_path, num_levels=10):
    coords, labels = load_xyzb(xyzb_path)
    pdb_id = xyzb_path.stem.replace("_ca", "")
    if coords.shape[0] < 3 or labels.shape[0] == 0:
        print(f"  [SKIP] {pdb_id}")
        return
    dist = pairwise_distance(coords)
    thresholds = compute_thresholds(dist, num_levels)
    parts = [perturbation_experiment(rips_filter(dist, thr)) for thr in thresholds]
    feat = np.hstack(parts)
    cols = [f"{name}_L{lv:02d}" for lv in range(1, num_levels + 1)
            for name in TOPO_FEATURE_NAMES]
    header = f"pdb_id,atom_index,{','.join(cols)},label"
    lines = [header]
    for i in range(feat.shape[0]):
        row = f"{pdb_id},{i}," + ",".join(f"{v:.4f}" for v in feat[i]) + f",{labels[i]:.4f}"
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
                    base / "result" / args.dataset / "topo_features_v2")
