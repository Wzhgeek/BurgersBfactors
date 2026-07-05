#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
scRNA-seq 细胞图构建: 表达矩阵 → 多尺度 Aij (10 层分位数过滤)。

标准流程（config graph.qc + graph.pca）:
  1. QC：过滤低质量细胞 / 低表达基因
  2. CPM/10k + log1p
  3. ~2000 HVG（Scanpy seurat flavor）
  4. HVG 矩阵按基因 z-score
  5. PCA → 30（或 50）维
  6. PC 空间 cell-cell 欧氏距离 → Rips 过滤 → 多尺度 Aij

亦支持 distance_space=hvg（不降维）或 pearson 相关图。

输出每个切片的:
  根目录（euclidean 模式，兼容旧路径）
    distance.npy, pca_coords.npy, hvg_indices.npy, thresholds.json, {slice}_Aij_0-{level}.npy
  pearson/
    pearson.npy, connectivity.npy, pca_coords.npy, hvg_indices.npy, thresholds.json, {slice}_Aij_0-{level}.npy

用法:
    python build_graph.py
    python build_graph.py --slice GSE84133_human1
    python build_graph.py --graph-mode pearson
    python build_graph.py --graph-mode both
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_expression(slice_dir: Path) -> np.ndarray:
    data = np.load(slice_dir / "expression.npz", allow_pickle=True)
    return data["matrix"].astype(np.float64)


def get_qc_cfg(graph_cfg: dict) -> dict:
    """读取 QC 配置（graph.qc）。"""
    return graph_cfg.get("qc") or {}


def apply_qc(matrix: np.ndarray, qc_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    过滤低质量细胞与低表达基因（在归一化之前对原始 count 矩阵操作）。

    Returns:
        matrix_qc: 过滤后的矩阵
        cell_keep: 保留的细胞在原矩阵中的行索引
        gene_keep: 保留的基因在原矩阵中的列索引
        meta: QC 统计信息
    """
    n_cells_in, n_genes_in = matrix.shape
    cell_keep = np.arange(n_cells_in, dtype=np.int64)
    gene_keep = np.arange(n_genes_in, dtype=np.int64)
    meta: dict = {
        "qc_enabled": bool(qc_cfg.get("enabled", False)),
        "n_cells_before_qc": int(n_cells_in),
        "n_genes_before_qc": int(n_genes_in),
    }
    if not qc_cfg.get("enabled", False):
        meta.update({
            "n_cells_after_qc": int(n_cells_in),
            "n_genes_after_qc": int(n_genes_in),
            "n_cells_removed": 0,
            "n_genes_removed": 0,
        })
        return matrix, cell_keep, gene_keep, meta

    cell_mask = np.ones(n_cells_in, dtype=bool)
    gene_mask = np.ones(n_genes_in, dtype=bool)

    min_genes = qc_cfg.get("min_genes")
    if min_genes is not None:
        n_genes_per_cell = np.count_nonzero(matrix > 0, axis=1)
        cell_mask &= n_genes_per_cell >= int(min_genes)

    min_counts = qc_cfg.get("min_counts")
    if min_counts is not None:
        lib_size = matrix.sum(axis=1)
        cell_mask &= lib_size >= float(min_counts)

    max_genes = qc_cfg.get("max_genes")
    if max_genes is not None:
        n_genes_per_cell = np.count_nonzero(matrix > 0, axis=1)
        cell_mask &= n_genes_per_cell <= int(max_genes)

    min_cells = qc_cfg.get("min_cells")
    if min_cells is not None:
        n_cells_per_gene = np.count_nonzero(matrix > 0, axis=0)
        gene_mask &= n_cells_per_gene >= int(min_cells)

    cell_keep = np.flatnonzero(cell_mask).astype(np.int64)
    gene_keep = np.flatnonzero(gene_mask).astype(np.int64)
    if cell_keep.size == 0 or gene_keep.size == 0:
        raise ValueError(
            f"QC 后无剩余细胞或基因: cells {n_cells_in}→{cell_keep.size}, "
            f"genes {n_genes_in}→{gene_keep.size}"
        )

    matrix_qc = matrix[np.ix_(cell_keep, gene_keep)]
    meta.update({
        "qc_min_genes": min_genes,
        "qc_min_cells": min_cells,
        "qc_min_counts": min_counts,
        "qc_max_genes": max_genes,
        "n_cells_after_qc": int(matrix_qc.shape[0]),
        "n_genes_after_qc": int(matrix_qc.shape[1]),
        "n_cells_removed": int(n_cells_in - matrix_qc.shape[0]),
        "n_genes_removed": int(n_genes_in - matrix_qc.shape[1]),
    })
    return matrix_qc, cell_keep, gene_keep, meta


def write_qc_artifacts(
    slice_dir: Path,
    out_dir: Path,
    cell_keep: np.ndarray,
    gene_keep: np.ndarray,
    qc_meta: dict,
) -> None:
    """保存 QC 索引与（若有过滤）子集 labels。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "qc_cell_indices.npy", cell_keep)
    np.save(out_dir / "qc_gene_indices.npy", gene_keep)

    labels_src = slice_dir / "labels.csv"
    labels_dst = out_dir / "labels.csv"
    if not labels_src.exists():
        return
    with open(labels_src) as f:
        rows = list(csv.DictReader(f))
    if len(rows) != int(qc_meta["n_cells_before_qc"]):
        print(
            f"    ⚠ labels 行数 {len(rows)} != QC 前细胞数 "
            f"{qc_meta['n_cells_before_qc']}，跳过 labels 子集"
        )
        return
    kept = [rows[i] for i in cell_keep.tolist()]
    with open(labels_dst, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cell_id", "cell_type", "label"])
        writer.writeheader()
        writer.writerows(kept)


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


def get_pca_cfg(graph_cfg: dict) -> dict:
    """欧氏 / Pearson 共用的 HVG+PCA 参数（优先 graph.pca，兼容 graph.euclidean）。"""
    return graph_cfg.get("pca") or graph_cfg.get("euclidean", {})


def use_hvg_distance_space(pca_cfg: dict) -> bool:
    """是否在 HVG 空间直接算距离（不做 PCA）。"""
    if str(pca_cfg.get("distance_space", "")).lower() == "hvg":
        return True
    if pca_cfg.get("use_pca") is False:
        return True
    return int(pca_cfg.get("n_pcs", 1)) <= 0


def similarity_method_tag(pca_meta: dict, graph_mode: str) -> str:
    """thresholds.json 用的构图方法标识。"""
    lib = pca_meta.get("preprocess_library", pca_meta.get("preprocess", "sklearn"))
    nh = pca_meta.get("n_hvg_requested", 2000)
    qc = "qc_" if pca_meta.get("qc_enabled") else ""
    if pca_meta.get("distance_space") == "hvg" or int(pca_meta.get("n_pcs_used", 1)) == 0:
        return f"{lib}_{qc}hvg{nh}_direct_{graph_mode}"
    np_ = pca_meta.get("n_pcs_requested", 30)
    return f"{lib}_{qc}hvg{nh}_pca{np_}_{graph_mode}"


def normalize_log_matrix(matrix: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """库大小归一化到 target_sum 后 log1p（细胞×基因）。"""
    lib = matrix.sum(axis=1, keepdims=True)
    lib = np.where(lib <= 0, 1.0, lib)
    return np.log1p(matrix / lib * target_sum)


def select_hvg_indices(log_matrix: np.ndarray, n_hvg: int) -> np.ndarray:
    """按 log 表达方差选取 top-n 高变基因索引。"""
    n_genes = log_matrix.shape[1]
    n_pick = min(int(n_hvg), n_genes)
    if n_pick <= 0 or n_pick >= n_genes:
        return np.arange(n_genes, dtype=np.int64)
    gene_var = np.var(log_matrix, axis=0)
    return np.argsort(gene_var)[-n_pick:].astype(np.int64)


def compute_pca_coords_sklearn(matrix: np.ndarray, pca_cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    sklearn HVG（方差 top-N）→ StandardScaler → PCA。

    Returns:
        coords: (n_cells, n_pcs)
        hvg_idx: 选用的基因列索引
        meta: 可写入 thresholds.json 的 PCA 参数
    """
    target_sum = float(pca_cfg.get("target_sum", 1e4))
    n_hvg = int(pca_cfg.get("n_hvg", 2000))
    n_pcs_req = int(pca_cfg.get("n_pcs", 3))
    scale = bool(pca_cfg.get("scale_before_pca", True))

    log_mat = normalize_log_matrix(matrix, target_sum=target_sum)
    hvg_idx = select_hvg_indices(log_mat, n_hvg)
    X = log_mat[:, hvg_idx]

    n_cells, n_hvg_used = X.shape
    n_pcs = min(n_pcs_req, n_cells, n_hvg_used)
    if scale and n_hvg_used > 1:
        X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_pcs, random_state=0)
    coords = pca.fit_transform(X).astype(np.float64)

    meta = {
        "preprocess": "sklearn",
        "preprocess_library": "sklearn",
        "n_hvg_requested": n_hvg,
        "n_hvg_used": int(n_hvg_used),
        "n_pcs_requested": n_pcs_req,
        "n_pcs_used": int(n_pcs),
        "target_sum": target_sum,
        "scale_before_pca": scale,
        "explained_variance_ratio": [round(float(v), 8) for v in pca.explained_variance_ratio_],
    }
    return coords, hvg_idx, meta


def _scanpy_hvg_scaled(matrix: np.ndarray, pca_cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Scanpy: normalize → log1p → HVG → scale，返回 (n_cells, n_hvg) 矩阵。"""
    import anndata as ad
    import scanpy as sc

    target_sum = float(pca_cfg.get("target_sum", 1e4))
    n_hvg = int(pca_cfg.get("n_hvg", 3000))
    scale = bool(pca_cfg.get("scale_before_pca", True))
    hvg_flavor = str(pca_cfg.get("hvg_flavor", "seurat"))

    adata = ad.AnnData(X=matrix.astype(np.float32))
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    n_top = min(n_hvg, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor=hvg_flavor)
    hvg_idx = np.flatnonzero(adata.var["highly_variable"].to_numpy()).astype(np.int64)
    adata = adata[:, adata.var["highly_variable"]].copy()
    n_cells, n_hvg_used = adata.n_obs, adata.n_vars
    X = adata.X.astype(np.float64)
    if scale and n_hvg_used > 1:
        X = StandardScaler().fit_transform(X)
    base_meta = {
        "preprocess_library": "scanpy",
        "hvg_flavor": hvg_flavor,
        "n_hvg_requested": n_hvg,
        "n_hvg_used": int(n_hvg_used),
        "target_sum": target_sum,
        "scale_before_pca": scale,
    }
    return X.astype(np.float64), hvg_idx, base_meta


def compute_coords_scanpy_hvg_direct(matrix: np.ndarray, pca_cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """HVG 标准化空间直接作为距离坐标（不降维）。"""
    X, hvg_idx, base = _scanpy_hvg_scaled(matrix, pca_cfg)
    meta = {
        **base,
        "preprocess": "scanpy_hvg_direct",
        "distance_space": "hvg",
        "n_pcs_requested": 0,
        "n_pcs_used": 0,
        "explained_variance_ratio": None,
    }
    return X, hvg_idx, meta


def compute_pca_coords_scanpy(matrix: np.ndarray, pca_cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Scanpy: normalize_total → log1p → highly_variable_genes → scale → pca。

    需在 conda eeg 环境安装 scanpy。
    """
    import anndata as ad
    import scanpy as sc

    if use_hvg_distance_space(pca_cfg):
        return compute_coords_scanpy_hvg_direct(matrix, pca_cfg)

    n_pcs_req = int(pca_cfg.get("n_pcs", 30))
    X, hvg_idx, base = _scanpy_hvg_scaled(matrix, pca_cfg)
    n_cells, n_hvg_used = X.shape
    n_pcs = min(n_pcs_req, n_cells, n_hvg_used)

    adata = ad.AnnData(X=X.astype(np.float32))
    svd_solver = "arpack" if n_pcs < min(n_cells, n_hvg_used) else "full"
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver=svd_solver, zero_center=True)
    coords = adata.obsm["X_pca"].astype(np.float64)
    vr = adata.uns["pca"]["variance_ratio"]

    meta = {
        **base,
        "preprocess": "scanpy",
        "distance_space": "pca",
        "n_pcs_requested": n_pcs_req,
        "n_pcs_used": int(n_pcs),
        "scale_max_value": float(pca_cfg.get("scale_max_value", 10)),
        "explained_variance_ratio": [round(float(v), 8) for v in vr],
    }
    return coords, hvg_idx, meta


def compute_pca_coords(matrix: np.ndarray, pca_cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """按 config graph.pca.preprocess 选择 scanpy 或 sklearn。"""
    method = str(pca_cfg.get("preprocess", "scanpy")).lower()
    if method == "scanpy":
        return compute_pca_coords_scanpy(matrix, pca_cfg)
    if method == "sklearn":
        return compute_pca_coords_sklearn(matrix, pca_cfg)
    raise ValueError(f"未知 preprocess: {method}（支持 scanpy | sklearn）")


def build_euclidean_graph(slice_name: str, matrix: np.ndarray,
                          out_dir: Path, n_cells: int, n_genes: int,
                          graph_cfg: dict, embed_meta: dict | None = None,
                          gene_keep: np.ndarray | None = None) -> int:
    """HVG 空间（或 PCA 空间）欧氏距离构图，写入 out_dir（切片根目录）。"""
    num_levels = graph_cfg["num_levels"]
    pct_start = graph_cfg["percentile_start"]
    pct_stop = graph_cfg["percentile_stop"]
    aij_k = graph_cfg["aij_k"]
    pca_cfg = get_pca_cfg(graph_cfg)
    embed_meta = embed_meta or {}

    coords, hvg_idx, pca_meta = compute_pca_coords(matrix, pca_cfg)
    pca_meta = {**embed_meta, **pca_meta}
    if gene_keep is not None and gene_keep.size:
        hvg_idx = gene_keep[hvg_idx]
    dist = squareform(pdist(coords, metric="euclidean"))
    thresholds = compute_thresholds(dist, num_levels, pct_start, pct_stop)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "distance.npy", dist)
    if pca_meta.get("distance_space") == "hvg":
        np.save(out_dir / "hvg_coords.npy", coords)
    else:
        np.save(out_dir / "pca_coords.npy", coords)
    np.save(out_dir / "hvg_indices.npy", hvg_idx)

    n_edges_last = save_level_aij(
        out_dir, slice_name, num_levels, dist, thresholds,
        filter_fn=rips_filter, aij_k=aij_k,
    )

    with open(out_dir / "thresholds.json", "w") as f:
        json.dump({
            "graph_mode": "euclidean",
            "similarity_method": similarity_method_tag(pca_meta, "euclidean"),
            "n_cells": n_cells,
            "n_genes": n_genes,
            "num_levels": num_levels,
            "pct_range": [pct_start, pct_stop],
            "aij_k": aij_k,
            **pca_meta,
            "thresholds": [round(float(t), 6) for t in thresholds],
            "max_distance": float(dist.max()),
            "min_distance": float(dist[dist > 0].min()) if np.any(dist > 0) else 0.0,
        }, f, indent=2)

    return n_edges_last


def build_pearson_graph(slice_name: str, matrix: np.ndarray,
                        pearson_dir: Path, n_cells: int, n_genes: int,
                        graph_cfg: dict, embed_meta: dict | None = None,
                        gene_keep: np.ndarray | None = None) -> int:
    """HVG+PCA PC 空间 Pearson 相关 → 连接强度 → Aij，写入 pearson/ 子目录。"""
    pearson_cfg = graph_cfg.get("pearson", {})
    num_levels = graph_cfg["num_levels"]
    pct_start = graph_cfg["percentile_start"]
    pct_stop = graph_cfg["percentile_stop"]
    aij_k = graph_cfg["aij_k"]
    eta = float(pearson_cfg.get("eta", 3.0))
    kappa = float(pearson_cfg.get("kappa", 1.0))
    clip_negative = bool(pearson_cfg.get("clip_negative", True))
    pca_cfg = get_pca_cfg(graph_cfg)
    embed_meta = embed_meta or {}

    coords, hvg_idx, pca_meta = compute_pca_coords(matrix, pca_cfg)
    pca_meta = {**embed_meta, **pca_meta}
    if gene_keep is not None and gene_keep.size:
        hvg_idx = gene_keep[hvg_idx]
    pearson = compute_pearson_matrix(coords, clip_negative=clip_negative)
    conn = compute_connectivity(pearson, eta=eta, kappa=kappa)
    thresholds = compute_thresholds(conn, num_levels, pct_start, pct_stop)

    pearson_dir.mkdir(parents=True, exist_ok=True)
    np.save(pearson_dir / "pearson.npy", pearson)
    np.save(pearson_dir / "connectivity.npy", conn)
    np.save(pearson_dir / "pca_coords.npy", coords)
    np.save(pearson_dir / "hvg_indices.npy", hvg_idx)

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
            "similarity_method": similarity_method_tag(pca_meta, "pearson"),
            "n_cells": n_cells,
            "n_genes": n_genes,
            "num_levels": num_levels,
            "pct_range": [pct_start, pct_stop],
            "aij_k": aij_k,
            "eta": eta,
            "kappa": kappa,
            "clip_negative": clip_negative,
            **pca_meta,
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
    n_cells_raw = meta["n_cells"]
    n_genes_raw = meta["n_genes"]
    graph_cfg = cfg["graph"]
    num_levels = graph_cfg["num_levels"]
    qc_cfg = get_qc_cfg(graph_cfg)

    if dry_run:
        mb = n_cells_raw * n_cells_raw * 8 / (1024 * 1024)
        tags = "+".join(modes)
        qc_tag = "QC+" if qc_cfg.get("enabled") else ""
        pca_cfg = get_pca_cfg(graph_cfg)
        pipe = (
            f"{qc_tag}CPM10k→log1p→HVG{pca_cfg.get('n_hvg', 2000)}"
            f"→zscore→PC{pca_cfg.get('n_pcs', 30)}"
        )
        print(
            f"  {slice_name}: {n_cells_raw}c → [{tags}] {pipe} "
            f"({mb:.1f}MB/matrix) × {num_levels} levels"
        )
        return

    print(f"  {slice_name}: {n_cells_raw}c×{n_genes_raw}g", end=" ", flush=True)
    matrix = load_expression(slice_dir)
    matrix, cell_keep, gene_keep, qc_meta = apply_qc(matrix, qc_cfg)
    n_cells = matrix.shape[0]
    n_genes = matrix.shape[1]
    embed_meta = qc_meta

    slice_out = output_dir / slice_name
    write_qc_artifacts(slice_dir, slice_out, cell_keep, gene_keep, qc_meta)
    if qc_meta.get("n_cells_removed", 0) or qc_meta.get("n_genes_removed", 0):
        print(
            f"[QC -{qc_meta['n_cells_removed']}c -{qc_meta['n_genes_removed']}g] ",
            end="",
            flush=True,
        )

    parts: list[str] = []

    if "euclidean" in modes:
        n_e = build_euclidean_graph(
            slice_name, matrix, slice_out, n_cells, n_genes, graph_cfg,
            embed_meta, gene_keep=gene_keep)
        parts.append(f"euclid L{num_levels:02d}:{n_e}e")

    if "pearson" in modes:
        n_p = build_pearson_graph(
            slice_name, matrix, slice_out / "pearson", n_cells, n_genes, graph_cfg,
            embed_meta, gene_keep=gene_keep)
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
