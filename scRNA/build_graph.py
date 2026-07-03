#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
scRNA-seq 细胞图构建: 表达矩阵 → 多尺度 Aij (10 层分位数过滤)。

支持两种相似性度量:
  - euclidean: 欧氏距离 → Rips 过滤 → Aij
  - pearson:   Pearson 相关 → 连接强度 → 分位数过滤 → Aij

输出每个切片的:
  euclidean/ (默认根目录，兼容旧路径)
    distance.npy, thresholds.json, {slice}_Aij_0-{level}.npy
  pearson/
    pearson.npy, connectivity.npy, thresholds.json, {slice}_Aij_0-{level}.npy

用法:
    python build_graph.py
    python build_graph.py --slice GSE84133_human1
    python build_graph.py --graph-mode pearson
    python build_graph.py --graph-mode both
"""

import argparse
import json
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
    """Rips 过滤: 距离 > threshold → 0（向量化实现）。"""
    filtered = np.where(distance_matrix <= threshold, distance_matrix, 0.0).astype(np.float64)
    np.fill_diagonal(filtered, 0.0)
    return filtered


def compute_thresholds(matrix: np.ndarray, num_levels: int,
                       pct_start: float = 5.0, pct_stop: float = 95.0) -> np.ndarray:
    """按非零非对角元素的分位数生成过滤阈值。"""
    n = matrix.shape[0]
    off = matrix[np.triu_indices(n, k=1)]
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


def count_aij_edges(aij: np.ndarray) -> int:
    aij_adj = aij.copy()
    np.fill_diagonal(aij_adj, 0.0)
    return int(np.sum(aij_adj > 1e-12))


def save_level_aij(out_dir: Path, slice_name: str, num_levels: int,
                   matrix: np.ndarray, thresholds: np.ndarray,
                   filter_fn, aij_k: int) -> int:
    """逐层过滤 → 计算 Aij，返回最后一层边数。"""
    n_edges_last = 0
    for lvl in range(1, num_levels + 1):
        thr = thresholds[lvl - 1]
        filtered = filter_fn(matrix, thr)
        aij = compute_aij(filtered, k=aij_k)
        np.save(out_dir / f"{slice_name}_Aij_0-{lvl}.npy", aij)
        if lvl == num_levels:
            n_edges_last = count_aij_edges(aij)
    return n_edges_last


def compute_pearson_matrix(matrix: np.ndarray, clip_negative: bool = True) -> np.ndarray:
    """
    细胞间 Pearson 相关 (rowvar=1: 每行一个细胞，跨基因算相关)。

    负相关可置 0；对角置 0。
    """
    pearson = np.corrcoef(matrix, rowvar=1).astype(np.float64)
    np.fill_diagonal(pearson, 0.0)
    if clip_negative:
        pearson[pearson < 0] = 0.0
    return pearson


def compute_connectivity(pearson: np.ndarray, eta: float, kappa: float) -> np.ndarray:
    """
    连接强度: conn_ij = 1 - exp(-(r_ij / η)^κ)，仅非对角。

    与旧版 GSE 脚本一致；对角保持 0。
    """
    n = pearson.shape[0]
    conn = np.zeros((n, n), dtype=np.float64)
    idx = ~np.eye(n, dtype=bool)
    r_vals = pearson[idx]
    conn[idx] = 1.0 - np.exp(-((r_vals / eta) ** kappa))
    return conn


def connectivity_filter(conn: np.ndarray, min_conn: float) -> np.ndarray:
    """连接强度 < min_conn 的边置 0，保留强连接。"""
    filtered = np.where(conn >= min_conn, conn, 0.0).astype(np.float64)
    np.fill_diagonal(filtered, 0.0)
    return filtered


def connectivity_to_dissimilarity(filtered_conn: np.ndarray) -> np.ndarray:
    """将过滤后的连接强度转为 Aij 用的非相似度: d = 1 - conn。"""
    dissim = np.zeros_like(filtered_conn, dtype=np.float64)
    idx = (filtered_conn > 0) & ~np.eye(filtered_conn.shape[0], dtype=bool)
    dissim[idx] = 1.0 - filtered_conn[idx]
    return dissim


def build_euclidean_graph(slice_name: str, matrix: np.ndarray,
                          out_dir: Path, n_cells: int, n_genes: int,
                          graph_cfg: dict) -> int:
    """欧氏距离构图，写入 out_dir（切片根目录，兼容旧路径）。"""
    num_levels = graph_cfg["num_levels"]
    pct_start = graph_cfg["percentile_start"]
    pct_stop = graph_cfg["percentile_stop"]
    aij_k = graph_cfg["aij_k"]

    dist = squareform(pdist(matrix, metric="euclidean"))
    thresholds = compute_thresholds(dist, num_levels, pct_start, pct_stop)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "distance.npy", dist)

    n_edges_last = save_level_aij(
        out_dir, slice_name, num_levels, dist, thresholds,
        filter_fn=rips_filter, aij_k=aij_k,
    )

    with open(out_dir / "thresholds.json", "w") as f:
        json.dump({
            "graph_mode": "euclidean",
            "n_cells": n_cells,
            "n_genes": n_genes,
            "num_levels": num_levels,
            "pct_range": [pct_start, pct_stop],
            "aij_k": aij_k,
            "thresholds": [round(float(t), 6) for t in thresholds],
            "max_distance": float(dist.max()),
            "min_distance": float(dist[dist > 0].min()) if np.any(dist > 0) else 0.0,
        }, f, indent=2)

    return n_edges_last


def build_pearson_graph(slice_name: str, matrix: np.ndarray,
                        pearson_dir: Path, n_cells: int, n_genes: int,
                        graph_cfg: dict) -> int:
    """Pearson 相关 → 连接强度 → 分位数过滤 → Aij，写入 pearson/ 子目录。"""
    pearson_cfg = graph_cfg.get("pearson", {})
    num_levels = graph_cfg["num_levels"]
    pct_start = graph_cfg["percentile_start"]
    pct_stop = graph_cfg["percentile_stop"]
    aij_k = graph_cfg["aij_k"]
    eta = float(pearson_cfg.get("eta", 3.0))
    kappa = float(pearson_cfg.get("kappa", 1.0))
    clip_negative = bool(pearson_cfg.get("clip_negative", True))

    pearson = compute_pearson_matrix(matrix, clip_negative=clip_negative)
    conn = compute_connectivity(pearson, eta=eta, kappa=kappa)
    thresholds = compute_thresholds(conn, num_levels, pct_start, pct_stop)

    pearson_dir.mkdir(parents=True, exist_ok=True)
    np.save(pearson_dir / "pearson.npy", pearson)
    np.save(pearson_dir / "connectivity.npy", conn)

    def _filter_and_dissim(c: np.ndarray, thr: float) -> np.ndarray:
        filtered_conn = connectivity_filter(c, thr)
        return connectivity_to_dissimilarity(filtered_conn)

    n_edges_last = save_level_aij(
        pearson_dir, slice_name, num_levels, conn, thresholds,
        filter_fn=_filter_and_dissim, aij_k=aij_k,
    )

    off = conn[np.triu_indices(conn.shape[0], k=1)]
    off_pos = off[off > 0]
    with open(pearson_dir / "thresholds.json", "w") as f:
        json.dump({
            "graph_mode": "pearson",
            "n_cells": n_cells,
            "n_genes": n_genes,
            "num_levels": num_levels,
            "pct_range": [pct_start, pct_stop],
            "aij_k": aij_k,
            "eta": eta,
            "kappa": kappa,
            "clip_negative": clip_negative,
            "thresholds": [round(float(t), 6) for t in thresholds],
            "max_connectivity": float(off_pos.max()) if off_pos.size else 0.0,
            "min_connectivity": float(off_pos.min()) if off_pos.size else 0.0,
        }, f, indent=2)

    return n_edges_last


def resolve_graph_modes(cfg: dict, cli_mode: str | None) -> list[str]:
    """解析要构建的图模式列表。"""
    if cli_mode:
        mode = cli_mode.lower()
        if mode == "both":
            return ["euclidean", "pearson"]
        if mode in ("euclidean", "pearson"):
            return [mode]
        raise ValueError(f"未知 graph-mode: {cli_mode}")

    pearson_enabled = cfg.get("graph", {}).get("pearson", {}).get("enabled", False)
    default_mode = cfg.get("graph", {}).get("default_mode", "euclidean")
    if default_mode == "both":
        return ["euclidean", "pearson"]
    if default_mode == "pearson":
        return ["pearson"]
    if pearson_enabled:
        return ["euclidean", "pearson"]
    return ["euclidean"]


def process_slice(slice_name: str, input_dir: Path, output_dir: Path,
                  cfg: dict, modes: list[str], dry_run: bool = False):
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

    if dry_run:
        mb = n_cells * n_cells * 8 / (1024 * 1024)
        tags = "+".join(modes)
        print(f"  {slice_name}: {n_cells}c → [{tags}] ({mb:.1f}MB/matrix) × {num_levels} levels")
        return

    print(f"  {slice_name}: {n_cells}c×{n_genes}g", end=" ", flush=True)
    matrix = load_expression(slice_dir)
    slice_out = output_dir / slice_name
    parts: list[str] = []

    if "euclidean" in modes:
        n_e = build_euclidean_graph(
            slice_name, matrix, slice_out, n_cells, n_genes, graph_cfg)
        parts.append(f"euclid L{num_levels:02d}:{n_e}e")

    if "pearson" in modes:
        n_p = build_pearson_graph(
            slice_name, matrix, slice_out / "pearson", n_cells, n_genes, graph_cfg)
        parts.append(f"pearson L{num_levels:02d}:{n_p}e")

    print("✓ " + " | ".join(parts))


def main():
    parser = argparse.ArgumentParser(description="scRNA multi-scale graph builder")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("config_graph.yaml"))
    parser.add_argument("--slice", type=str, help="Process single slice only")
    parser.add_argument(
        "--graph-mode", type=str, default=None,
        choices=["euclidean", "pearson", "both"],
        help="构图模式；默认读 config graph.default_mode / pearson.enabled",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    base = Path(__file__).resolve().parent
    input_dir = base / cfg["paths"]["input_dir"]
    output_dir = base / cfg["paths"]["output_dir"]
    modes = resolve_graph_modes(cfg, args.graph_mode)

    if not input_dir.exists():
        print(f"输入目录不存在: {input_dir}")
        return 1

    slices = cfg["slices"]
    if args.slice:
        slices = [args.slice]

    num_levels = cfg["graph"]["num_levels"]
    mode_tag = "+".join(modes)
    if args.dry_run:
        print(f"[DRY-RUN] {len(slices)} slices × [{mode_tag}] × {num_levels} levels → {output_dir}\n")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"图构建: {len(slices)} slices × [{mode_tag}] × {num_levels} levels → {output_dir}\n")

    for s in slices:
        process_slice(s, input_dir, output_dir, cfg, modes, args.dry_run)

    print(f"\n完成. 输出: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
