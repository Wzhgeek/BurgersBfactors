#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
scRNA-seq 细胞图构建: 表达矩阵 → 欧氏距离 → 多尺度 Aij (10层分位数过滤)。

输出每个切片的:
  distance.npy          — 完整欧氏距离矩阵
  Aij_0-{level}.npy     — 10层 Rips 过滤后的 Aij 矩阵
  thresholds.json       — 10个分位数阈值

用法:
    python build_graph.py
    python build_graph.py --slice GSE84133_human1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.distance import pdist, squareform


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_expression(slice_dir: Path) -> np.ndarray:
    data = np.load(slice_dir / "expression.npz", allow_pickle=True)
    return data["matrix"].astype(np.float64)


def rips_filter(distance_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """Rips 过滤: 距离 > threshold → 0."""
    n = distance_matrix.shape[0]
    filtered = np.zeros_like(distance_matrix, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_matrix[i, j]
            if d <= threshold:
                filtered[i, j] = d
                filtered[j, i] = d
    return filtered


def compute_thresholds(distance_matrix: np.ndarray, num_levels: int,
                       pct_start: float = 5.0, pct_stop: float = 95.0) -> np.ndarray:
    """计算分位数过滤阈值。"""
    n = distance_matrix.shape[0]
    off = distance_matrix[np.triu_indices(n, k=1)]
    off_pos = off[off > 0]
    if off_pos.size == 0:
        return np.zeros(num_levels, dtype=np.float64)
    percentiles = np.linspace(pct_start, pct_stop, num_levels)
    return np.percentile(off_pos, percentiles).astype(np.float64)


def compute_aij(filtered_dist: np.ndarray, k: int = 1) -> np.ndarray:
    """
    A_ij = exp(-d_ij^k / (k * σ^k))   (i ≠ j)
    A_ii = -Σ_{j≠i} A_ij
    σ = 非零非对角距离的中位数
    """
    n = filtered_dist.shape[0]
    off = filtered_dist[np.triu_indices(n, k=1)]
    off_pos = off[off > 0]
    sigma = float(np.median(off_pos)) if off_pos.size > 0 else 1.0
    if sigma <= 0:
        sigma = 1.0

    aij = np.zeros((n, n), dtype=np.float64)
    denom = k * (sigma ** k)
    idx = np.where((filtered_dist > 0) & ~np.eye(n, dtype=bool))
    d_vals = filtered_dist[idx]
    aij[idx] = np.exp(-(d_vals ** k) / denom)

    row_sum = aij.sum(axis=1)
    np.fill_diagonal(aij, -row_sum)
    return aij


def process_slice(slice_name: str, input_dir: Path, output_dir: Path,
                  cfg: dict, dry_run: bool = False):
    slice_dir = input_dir / slice_name
    npz_path = slice_dir / "expression.npz"
    meta_path = slice_dir / "meta.json"

    if not npz_path.exists():
        print(f"  ✗ {slice_name}: expression.npz 不存在")
        return

    with open(meta_path) as f:
        meta = json.load(f)
    n_cells = meta["n_cells"]
    n_genes = meta["n_genes"]

    graph_cfg = cfg["graph"]
    num_levels = graph_cfg["num_levels"]
    pct_start = graph_cfg["percentile_start"]
    pct_stop = graph_cfg["percentile_stop"]
    aij_k = graph_cfg["aij_k"]

    if dry_run:
        mb = n_cells * n_cells * 8 / (1024 * 1024)
        print(f"  {slice_name}: {n_cells}c → dist ({mb:.1f}MB) × {num_levels} levels")
        return

    # 1. 加载表达矩阵，计算欧氏距离
    print(f"  {slice_name}: {n_cells}c×{n_genes}g", end=" ", flush=True)
    matrix = load_expression(slice_dir)
    dist = squareform(pdist(matrix, metric="euclidean"))

    # 2. 计算分位数阈值
    thresholds = compute_thresholds(dist, num_levels, pct_start, pct_stop)

    # 3. 保存完整距离矩阵
    out_dir = output_dir / slice_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "distance.npy", dist)

    # 4. 逐层过滤距离 → 计算 Aij
    for lvl in range(1, num_levels + 1):
        thr = thresholds[lvl - 1]
        filtered = rips_filter(dist, thr)
        aij = compute_aij(filtered, k=aij_k)

        aij_path = out_dir / f"{slice_name}_Aij_0-{lvl}.npy"
        np.save(aij_path, aij)

        # 统计边数
        aij_adj = aij.copy()
        np.fill_diagonal(aij_adj, 0.0)
        n_edges = int(np.sum(aij_adj > 1e-12))
        if lvl == num_levels:
            print(f"→ L{lvl:02d}:{n_edges}e", end=" ", flush=True)

    # 5. 保存元数据
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump({
            "n_cells": n_cells, "n_genes": n_genes,
            "num_levels": num_levels, "pct_range": [pct_start, pct_stop],
            "aij_k": aij_k,
            "thresholds": [round(float(t), 2) for t in thresholds],
            "max_distance": float(dist.max()),
            "min_distance": float(dist[dist > 0].min()),
        }, f, indent=2)

    # 统计最后一层的边数
    aij_last = np.load(out_dir / f"{slice_name}_Aij_0-{num_levels}.npy")
    aij_adj = aij_last.copy()
    np.fill_diagonal(aij_adj, 0.0)
    n_edges_full = int(np.sum(aij_adj > 1e-12))
    print(f"✓ L{num_levels:02d}:{n_edges_full}e")


def main():
    parser = argparse.ArgumentParser(description="scRNA multi-scale graph builder")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("config_graph.yaml"))
    parser.add_argument("--slice", type=str, help="Process single slice only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    base = Path(__file__).resolve().parent
    input_dir = base / cfg["paths"]["input_dir"]
    output_dir = base / cfg["paths"]["output_dir"]

    if not input_dir.exists():
        print(f"输入目录不存在: {input_dir}")
        return 1

    slices = cfg["slices"]
    if args.slice:
        slices = [args.slice]

    num_levels = cfg["graph"]["num_levels"]
    if args.dry_run:
        print(f"[DRY-RUN] {len(slices)} slices × {num_levels} levels → {output_dir}\n")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"图构建: {len(slices)} slices × {num_levels} levels → {output_dir}\n")

    for s in slices:
        process_slice(s, input_dir, output_dir, cfg, args.dry_run)

    print(f"\n完成. 输出: {output_dir}")


if __name__ == "__main__":
    main()
