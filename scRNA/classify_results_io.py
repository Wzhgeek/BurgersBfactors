# Author: Zihan Wang
# <wangzh011031@163.com>
"""
分类结果落盘：对齐 Pcode 蛋白回归目录结构。

路径见 config_graph.yaml paths.classify_dir（scratch）:
  {classify_dir}/{slice}/all_score/{slice}_all_{metric}.csv
  {classify_dir}/{slice}/result.json

跨 slice 汇总见 paths.classify_summary_dir。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scRNA.classify_traj_stats import eps_tag, resolve_path


def legacy_classify_dir(cfg: dict, scrna_dir: Path, slice_name: str) -> Path:
    """单 slice 分类结果根目录：{classify_dir}/{slice}/。"""
    return resolve_path(scrna_dir, cfg["paths"]["classify_dir"]) / slice_name


def classify_summary_dir(cfg: dict, scrna_dir: Path) -> Path:
    """跨 slice 汇总 CSV 目录。"""
    return resolve_path(scrna_dir, cfg["paths"]["classify_summary_dir"])

# grid 字段 → all_score 文件名后缀
GRID_METRICS: dict[str, str] = {
    "balanced_accuracy_mean": "ba",
    "accuracy_mean": "acc",
    "precision_mean": "precision",
    "recall_mean": "recall",
    "f1_mean": "f1",
    "auc_mean": "auc",
    "kappa_mean": "kappa",
}


def pick_best_row(grid: list[dict]) -> dict:
    """按 BA 优先、acc 次之选取最优 (ε, 层)。"""
    return max(grid, key=lambda r: (r["balanced_accuracy_mean"], r["accuracy_mean"]))


def grid_eps_values(grid: list[dict]) -> list[float]:
    return sorted({round(float(r["epsilon"]), 1) for r in grid})


def grid_to_matrix(
    grid: list[dict],
    eps_values: list[float],
    num_levels: int,
    field: str,
    fill: float = np.nan,
) -> np.ndarray:
    """构建 shape (len(eps), num_levels) 矩阵。"""
    mat = np.full((len(eps_values), num_levels), fill, dtype=np.float64)
    eps_index = {e: i for i, e in enumerate(eps_values)}
    for row in grid:
        ei = eps_index.get(round(float(row["epsilon"]), 1))
        if ei is None:
            continue
        lvl = int(row["level"])
        if 1 <= lvl <= num_levels:
            val = row.get(field)
            if val is None:
                continue
            mat[ei, lvl - 1] = float(val)
    return mat


def level_header(num_levels: int) -> str:
    return ",".join(f"L{lvl:02d}" for lvl in range(1, num_levels + 1))


def save_metric_grid_csv(
    path: Path,
    grid: list[dict],
    eps_values: list[float],
    num_levels: int,
    field: str,
) -> None:
    mat = grid_to_matrix(grid, eps_values, num_levels, field)
    header = level_header(num_levels)
    np.savetxt(path, mat, delimiter=",", header=header, comments="", fmt="%.10g")


def save_all_score_dir(
    all_score_dir: Path,
    slice_name: str,
    grid: list[dict],
    num_levels: int = 10,
) -> list[Path]:
    """写入 all_score/*.csv，返回路径列表。"""
    all_score_dir.mkdir(parents=True, exist_ok=True)
    eps_values = grid_eps_values(grid)
    paths: list[Path] = []
    for field, short in GRID_METRICS.items():
        out = all_score_dir / f"{slice_name}_all_{short}.csv"
        save_metric_grid_csv(out, grid, eps_values, num_levels, field)
        paths.append(out)
    return paths


def row_to_level_entry(row: dict) -> dict:
    return {
        "epsilon": float(row["epsilon"]),
        "level": int(row["level"]),
        "balanced_accuracy": float(row["balanced_accuracy_mean"]),
        "balanced_accuracy_std": float(row.get("balanced_accuracy_std", 0)),
        "accuracy": float(row["accuracy_mean"]),
        "precision": float(row.get("precision_mean", 0) or 0),
        "recall": float(row.get("recall_mean", 0) or 0),
        "f1": float(row.get("f1_mean", 0) or 0),
        "auc": float(row.get("auc_mean", 0) or 0),
        "kappa": float(row.get("kappa_mean", 0) or 0),
    }


def build_per_level_best(grid: list[dict], num_levels: int = 10) -> dict[str, dict]:
    per_level: dict[str, dict] = {}
    for lvl in range(1, num_levels + 1):
        rows = [r for r in grid if int(r["level"]) == lvl]
        if not rows:
            continue
        per_level[f"L{lvl:02d}"] = row_to_level_entry(pick_best_row(rows))
    return per_level


def load_evolution_meta(
    sim_dir: Path,
    slice_name: str,
    level: int,
    epsilon: float,
    file_suffix: str = "",
) -> dict:
    """从 burgers partial level_meta.json 读取该 (ε, 层) 的演化步数信息。"""
    meta_path = sim_dir / slice_name / "partial" / f"L{level:02d}" / "level_meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        data = json.load(f)
    tag = eps_tag(epsilon)
    if file_suffix:
        tag = f"{tag}_{file_suffix}"
    run = data.get("epsilon_run_meta", {}).get(tag, {})
    if not run:
        return {
            "max_n_steps": data.get("n_steps"),
            "n_steps_mode": data.get("n_steps_mode"),
        }
    return {
        "actual_n_steps": run.get("actual_n_steps"),
        "max_n_steps": run.get("max_n_steps", data.get("n_steps")),
        "early_stopped": run.get("early_stopped"),
        "frac_at_stop": run.get("frac_at_stop"),
        "n_steps_mode": data.get("n_steps_mode"),
    }


def export_slice_results(
    *,
    legacy_dir: Path,
    slice_name: str,
    grid: list[dict],
    meta: dict,
    sim_dir: Path | None = None,
    file_suffix: str = "",
    n_cells: int | None = None,
    num_levels: int = 10,
) -> Path:
    """
    写入 legacy_classify/all_score/*.csv 与 legacy_classify/result.json。
    返回 result.json 路径。
    """
    if not grid:
        raise ValueError("grid 为空，无法导出")

    legacy_dir.mkdir(parents=True, exist_ok=True)
    all_score_dir = legacy_dir / "all_score"
    save_all_score_dir(all_score_dir, slice_name, grid, num_levels=num_levels)

    best = pick_best_row(grid)
    best_entry = row_to_level_entry(best)
    evo = {}
    if sim_dir is not None:
        evo = load_evolution_meta(
            sim_dir, slice_name, int(best["level"]), float(best["epsilon"]), file_suffix,
        )
        best_entry.update(evo)

    result = {
        "slice": slice_name,
        "n_cells": n_cells,
        "feature": meta.get("feature", "stats6"),
        "method": meta.get("method"),
        "strict_legacy": meta.get("strict_legacy", True),
        "file_suffix": file_suffix or None,
        "n_splits": meta.get("n_splits"),
        "n_repeats": meta.get("n_repeats"),
        "epsilon_values": grid_eps_values(grid),
        "selection": "balanced_accuracy",
        "best_by_ba": best_entry,
        "per_level": build_per_level_best(grid, num_levels=num_levels),
        "grid_size": len(grid),
        "partial_files": meta.get("partial_files"),
        "scan_tag": meta.get("scan_tag"),
    }
    if meta.get("trajectory_stride") is not None:
        result["trajectory_stride"] = meta["trajectory_stride"]

    result_path = legacy_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    return result_path


def load_slice_result(legacy_dir: Path) -> dict | None:
    path = legacy_dir / "result.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def find_latest_scan_json(legacy_dir: Path, slice_name: str) -> Path | None:
    scan_dir = legacy_dir / "scan"
    if not scan_dir.is_dir():
        return None
    cands = sorted(scan_dir.glob(f"{slice_name}_legacy_classify_*_strict_legacy.json"))
    return cands[-1] if cands else None
