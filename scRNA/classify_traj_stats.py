#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
从 Burgers 演化 partial 结果提取轨迹统计特征，并用 LR / SVM / RF / KNN 做交叉验证分类。

默认模式 best_level: 每个 slice 在固定 ε 下逐层 (L01–L10) 评估，按 balanced accuracy
选最优单层，仅用该层 6 维统计特征分类。

用法:
    python classify_traj_stats.py --slice GSE45719 --level 5   # 单层 CV（SLURM 单 job）
    python classify_traj_stats.py --aggregate                 # 汇总 partial → 选最优 level
    python classify_traj_stats.py --mode concat                 # 旧版：10 层拼接 60 维
    bash scRNA/classify_traj_stats.sh submit-all              # 13×10 SLURM jobs
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import extract_stats_features
from scRNA.burgers_sim import graph_data_dir

STAT_NAMES = ["max", "min", "mean", "var", "median", "std"]

ALL_SLICES = [
    "GSE45719",
    "GSE67835",
    "GSE75748_time",
    "GSE75748_cell",
    "GSE82187",
    "GSE89232",
    "GSE94820_discovery",
    "GSE59114_C57BL6",
    "GSE84133_mouse1",
    "GSE84133_mouse2",
    "GSE84133_human4",
    "GSE84133_human1",
    "GSE84133_human2",
]

MODELS = {
    "LR": lambda: LogisticRegression(),
    "SVM": lambda: SVC(),
    "RF": lambda: RandomForestClassifier(),
    "KNN": lambda: KNeighborsClassifier(),
}


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, path_value: str | Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else base / p


def eps_tag(eps: float) -> str:
    return f"{eps:.1f}".replace(".", "-")


def partial_level_dir(sim_dir: Path, slice_name: str, level: int) -> Path:
    return sim_dir / slice_name / "partial" / f"L{level:02d}"


def count_completed_levels(sim_dir: Path, slice_name: str, tag: str,
                           num_levels: int) -> int:
    n = 0
    for lvl in range(1, num_levels + 1):
        pdir = partial_level_dir(sim_dir, slice_name, lvl)
        if (pdir / f"stats_{tag}.npy").exists() and (pdir / "level_meta.json").exists():
            n += 1
    return n


def load_labels(
    label_dir: Path,
    slice_name: str,
    n_cells: int,
    aij_dir: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    scrna_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []
    if aij_dir is not None:
        candidates.extend([
            aij_dir / slice_name / "labels.csv",
            aij_dir / slice_name / f"{slice_name}_full_labels.csv",
        ])
    candidates.extend([
        label_dir / slice_name / "labels.csv",
        scrna_dir / "preprocessed" / slice_name / "labels.csv",
    ])
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError(
            f"{slice_name}: 未找到 labels，已尝试: {[str(p) for p in candidates]}"
        )

    def _parse_labels(path: Path) -> tuple[np.ndarray, list[str]]:
        labels: list[int] = []
        types: list[str] = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "label" in row:
                    labels.append(int(row["label"]))
                    types.append(row.get("cell_type", ""))
                elif "Label" in row:
                    labels.append(int(row["Label"]))
                    types.append(row.get("Cell type", row.get("cell_type", "")))
                else:
                    raise ValueError(f"{path}: 无法识别标签列 {reader.fieldnames}")
        return np.array(labels, dtype=np.int64), types

    for labels_path in existing:
        y, types = _parse_labels(labels_path)
        if y.shape[0] == n_cells:
            return y, types

    labels_path = existing[0]
    y, types = _parse_labels(labels_path)
    raise ValueError(
        f"{slice_name}: labels 行数 {y.shape[0]} != n_cells {n_cells} "
        f"({labels_path}); 已尝试: {[str(p) for p in existing]}"
    )


def load_n_cells(aij_dir: Path, slice_name: str, graph_mode: str) -> int:
    """从 pearson/ 等子目录读取 thresholds.json 中的 n_cells。"""
    sub = graph_data_dir(aij_dir, slice_name, graph_mode)
    meta_path = sub / "thresholds.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"缺少 thresholds.json: {meta_path}")
    with open(meta_path) as f:
        return int(json.load(f)["n_cells"])


def topo_features_root(cfg: dict, scrna_dir: Path) -> Path:
    """拓扑特征根目录（paths.topo_features_dir[/tag] 或 scratch/topo_features）。"""
    paths = cfg.get("paths", {})
    root = paths.get("topo_features_dir")
    if root:
        p = Path(root)
        base = p if p.is_absolute() else scrna_dir.parent / p
    else:
        base = resolve_path(scrna_dir, paths["scratch_root"]) / "topo_features"
    tag = paths.get("topo_features_tag")
    return base / tag if tag else base


def load_level_topo(
    topo_root: Path,
    slice_name: str,
    level: int,
) -> np.ndarray:
    """单层拓扑谱特征 (n_cells, 6)。"""
    from src.topo_features import TOPO_FEATURE_NAMES

    path = topo_root / slice_name / f"{slice_name}_topo_L{level:02d}.npy"
    if not path.exists():
        raise FileNotFoundError(f"缺少拓扑特征: {path}")
    topo = np.load(path)
    if topo.shape[1] != len(TOPO_FEATURE_NAMES):
        raise ValueError(
            f"{path}: 期望 {len(TOPO_FEATURE_NAMES)} 列, 实际 {topo.shape[1]}"
        )
    return sanitize_features(topo.astype(np.float64), f"{slice_name} L{level:02d} topo")


def load_alllevels_stats6(
    sim_dir: Path,
    slice_name: str,
    tag: str,
    num_levels: int = 10,
) -> np.ndarray:
    """L01..L10 各层 stats6 拼接 → (n_cells, 6 * num_levels)。"""
    blocks = [
        load_level_stats(sim_dir, slice_name, lvl, tag, recompute=False)
        for lvl in range(1, num_levels + 1)
    ]
    return np.hstack(blocks)


def load_pca_coords(aij_dir: Path, slice_name: str) -> np.ndarray:
    """读取构图阶段保存的 PCA 坐标 (n_cells, n_pcs)。"""
    candidates = [
        aij_dir / slice_name / "pca_coords.npy",
        aij_dir / slice_name / "pearson" / "pca_coords.npy",
    ]
    for path in candidates:
        if path.exists():
            return np.load(path).astype(np.float64)
    raise FileNotFoundError(
        f"{slice_name}: 缺少 pca_coords.npy（已检查 {[str(p) for p in candidates]}）"
    )


def load_alllevels_stats6_pca30(
    sim_dir: Path,
    aij_dir: Path,
    slice_name: str,
    tag: str,
    num_levels: int = 10,
) -> np.ndarray:
    """L01..L10 stats6 (60维) + PCA 坐标 → (n_cells, 6*num_levels + n_pcs)。"""
    stats = load_alllevels_stats6(sim_dir, slice_name, tag, num_levels=num_levels)
    pca = load_pca_coords(aij_dir, slice_name)
    if stats.shape[0] != pca.shape[0]:
        raise ValueError(
            f"{slice_name}: stats {stats.shape[0]} 行 != pca {pca.shape[0]} 行"
        )
    return np.hstack([stats, pca])


def load_alllevels_stats6_topo6_pca30(
    sim_dir: Path,
    topo_root: Path,
    aij_dir: Path,
    slice_name: str,
    tag: str,
    num_levels: int = 10,
) -> np.ndarray:
    """L01..L10 stats6+topo6 (120维) + PCA 坐标 → (n_cells, 12*num_levels + n_pcs)。"""
    feat = load_alllevels_stats6_topo6(
        sim_dir, topo_root, slice_name, tag, num_levels=num_levels,
    )
    pca = load_pca_coords(aij_dir, slice_name)
    if feat.shape[0] != pca.shape[0]:
        raise ValueError(
            f"{slice_name}: stats+topo {feat.shape[0]} 行 != pca {pca.shape[0]} 行"
        )
    return np.hstack([feat, pca])


def all_levels_stats_ready(
    sim_dir: Path,
    slice_name: str,
    tag: str,
    num_levels: int,
) -> bool:
    """检查 L01..L10 的 stats 是否齐全。"""
    for lvl in range(1, num_levels + 1):
        stats_path = partial_level_dir(sim_dir, slice_name, lvl) / f"stats_{tag}.npy"
        if not stats_path.exists():
            return False
    return True


def all_levels_trj_ready(
    sim_dir: Path,
    slice_name: str,
    tag: str,
    num_levels: int,
) -> bool:
    """检查 L01..L10 的 trj 是否齐全。"""
    for lvl in range(1, num_levels + 1):
        trj_path = partial_level_dir(sim_dir, slice_name, lvl) / f"trj_{tag}.npy"
        if not trj_path.exists():
            return False
    return True


def load_alllevels_traj100(
    sim_dir: Path,
    slice_name: str,
    tag: str,
    sample_idx: np.ndarray,
    num_levels: int = 10,
) -> np.ndarray:
    """L01..L10 各层 traj100 → (n_cells, num_levels, len(sample_idx))。"""
    from scRNA.analyze_traj_100pts import load_level_traj_features

    blocks: list[np.ndarray] = []
    for lvl in range(1, num_levels + 1):
        blocks.append(
            load_level_traj_features(sim_dir, slice_name, lvl, tag, sample_idx),
        )
    return np.stack(blocks, axis=1).astype(np.float64)


def load_level_traj_full(
    sim_dir: Path,
    slice_name: str,
    level: int,
    tag: str,
) -> np.ndarray:
    """单层完整保存轨迹 → (n_cells, n_traj_points)。"""
    trj_path = partial_level_dir(sim_dir, slice_name, level) / f"trj_{tag}.npy"
    return np.load(trj_path).astype(np.float64)


def load_alllevels_traj_full(
    sim_dir: Path,
    slice_name: str,
    tag: str,
    num_levels: int = 10,
) -> np.ndarray:
    """L01..L10 各层完整轨迹 → (n_cells, num_levels, n_traj_points)。"""
    blocks: list[np.ndarray] = []
    for lvl in range(1, num_levels + 1):
        blocks.append(load_level_traj_full(sim_dir, slice_name, lvl, tag))
    return np.stack(blocks, axis=1).astype(np.float64)


def load_alllevels_stats6_topo6(
    sim_dir: Path,
    topo_root: Path,
    slice_name: str,
    tag: str,
    num_levels: int = 10,
) -> np.ndarray:
    """L01..L10 各层 stats6+topo6 拼接 → (n_cells, 12 * num_levels)。"""
    blocks: list[np.ndarray] = []
    for lvl in range(1, num_levels + 1):
        blocks.append(load_level_stats(sim_dir, slice_name, lvl, tag, recompute=False))
        blocks.append(load_level_topo(topo_root, slice_name, lvl))
    return np.hstack(blocks)


def all_levels_stats_topo_ready(
    sim_dir: Path,
    topo_root: Path,
    slice_name: str,
    tag: str,
    num_levels: int,
) -> bool:
    """检查 L01..L10 的 stats 与 topo 是否齐全。"""
    for lvl in range(1, num_levels + 1):
        stats_path = partial_level_dir(sim_dir, slice_name, lvl) / f"stats_{tag}.npy"
        topo_path = topo_root / slice_name / f"{slice_name}_topo_L{lvl:02d}.npy"
        if not stats_path.exists() or not topo_path.exists():
            return False
    return True


def load_level_stats(
    sim_dir: Path,
    slice_name: str,
    level: int,
    tag: str,
    recompute: bool = False,
) -> np.ndarray:
    """单层统计特征 (n_cells, 6)."""
    pdir = partial_level_dir(sim_dir, slice_name, level)
    stats_path = pdir / f"stats_{tag}.npy"
    trj_path = pdir / f"trj_{tag}.npy"
    if recompute:
        if not trj_path.exists():
            raise FileNotFoundError(f"缺少轨迹: {trj_path}")
        stats = extract_stats_features(np.load(trj_path))
    else:
        if not stats_path.exists():
            raise FileNotFoundError(f"缺少统计特征: {stats_path}")
        stats = np.load(stats_path)
    if stats.shape[1] != len(STAT_NAMES):
        raise ValueError(
            f"{stats_path}: 期望 {len(STAT_NAMES)} 列, 实际 {stats.shape[1]}"
        )
    return stats.astype(np.float64)


def sanitize_features(X: np.ndarray, slice_name: str) -> np.ndarray:
    if np.isfinite(X).all():
        return X
    bad = int((~np.isfinite(X)).sum())
    print(f"[WARN] {slice_name}: 特征含 {bad} 个非有限值，已替换为 0")
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def extract_slice_features_concat(
    sim_dir: Path,
    slice_name: str,
    num_levels: int,
    tag: str,
    recompute: bool = False,
) -> np.ndarray:
    blocks = [
        load_level_stats(sim_dir, slice_name, lvl, tag, recompute)
        for lvl in range(1, num_levels + 1)
    ]
    stacked = np.stack(blocks, axis=1)
    return stacked.reshape(stacked.shape[0], -1)


def cv_n_splits(y: np.ndarray, requested: int = 10) -> int:
    counts = np.bincount(y)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0
    return max(2, min(requested, int(counts.min())))


def run_slice_classification(
    slice_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
) -> dict:
    n_splits = cv_n_splits(y, n_splits)
    if n_splits < 2:
        raise ValueError(f"{slice_name}: 有效类别过少，无法交叉验证")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    n_classes = len(np.unique(y))
    results: dict[str, dict] = {}

    for name, factory in MODELS.items():
        clf = factory()
        acc_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=1)
        bal_scores = cross_val_score(
            clf, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=1,
        )
        results[name] = {
            "accuracy_mean": float(acc_scores.mean()),
            "accuracy_std": float(acc_scores.std()),
            "accuracy_fold_scores": acc_scores.tolist(),
            "balanced_accuracy_mean": float(bal_scores.mean()),
            "balanced_accuracy_std": float(bal_scores.std()),
            "balanced_accuracy_fold_scores": bal_scores.tolist(),
            "fold_scores": acc_scores.tolist(),
            "n_splits": n_splits,
        }

    return {
        "slice": slice_name,
        "n_cells": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(n_classes),
        "stat_names": STAT_NAMES,
        "cv_folds_requested": 10,
        "cv_folds_used": n_splits,
        "models": results,
    }


def _level_score(models_result: dict) -> tuple[float, float, str]:
    """返回 (best_bal_acc, best_acc, best_model) 用于选 level."""
    best_bal = -1.0
    best_acc = -1.0
    best_model = ""
    for name, mres in models_result["models"].items():
        bal = mres["balanced_accuracy_mean"]
        acc = mres["accuracy_mean"]
        if bal > best_bal or (bal == best_bal and acc > best_acc):
            best_bal = bal
            best_acc = acc
            best_model = name
    return best_bal, best_acc, best_model


def select_best_level(
    sim_dir: Path,
    slice_name: str,
    y: np.ndarray,
    num_levels: int,
    tag: str,
    recompute: bool = False,
) -> tuple[int, dict, dict]:
    """
    逐层 CV，按各层 max(balanced_accuracy) 选最优 level。

    Returns:
        best_level, final_result_at_best_level, level_search_detail
    """
    level_rows: dict[int, dict] = {}
    best_level = 1
    best_bal = -1.0
    best_acc = -1.0

    for lvl in range(1, num_levels + 1):
        X = sanitize_features(
            load_level_stats(sim_dir, slice_name, lvl, tag, recompute),
            f"{slice_name} L{lvl:02d}",
        )
        lvl_result = run_slice_classification(f"{slice_name}_L{lvl:02d}", X, y)
        bal, acc, best_model = _level_score(lvl_result)
        level_rows[lvl] = {
            "level": lvl,
            "best_model": best_model,
            "best_balanced_accuracy": bal,
            "best_accuracy": acc,
            "models": lvl_result["models"],
        }
        if bal > best_bal or (bal == best_bal and acc > best_acc):
            best_bal = bal
            best_acc = acc
            best_level = lvl

    X_best = sanitize_features(
        load_level_stats(sim_dir, slice_name, best_level, tag, recompute),
        slice_name,
    )
    final = run_slice_classification(slice_name, X_best, y)
    final["best_level"] = best_level
    final["selection_metric"] = "max_balanced_accuracy"
    final["best_level_balanced_accuracy"] = best_bal
    final["best_level_accuracy"] = best_acc
    final["best_level_best_model"] = level_rows[best_level]["best_model"]

    search = {
        "selection_metric": "max_balanced_accuracy",
        "best_level": best_level,
        "levels": {str(lvl): level_rows[lvl] for lvl in sorted(level_rows)},
    }
    return best_level, final, search


def append_summary_rows(
    summary_rows: list[dict],
    slice_name: str,
    result: dict,
    best_level: int | None = None,
) -> None:
    for model, mres in result["models"].items():
        acc = mres["accuracy_mean"]
        acc_std = mres["accuracy_std"]
        bal = mres["balanced_accuracy_mean"]
        bal_std = mres["balanced_accuracy_std"]
        row = {
            "slice": slice_name,
            "best_level": best_level if best_level is not None else "",
            "n_cells": result["n_cells"],
            "n_classes": result["n_classes"],
            "n_features": result["n_features"],
            "model": model,
            "cv_folds": mres["n_splits"],
            "accuracy_mean": f"{acc:.6f}",
            "accuracy_std": f"{acc_std:.6f}",
            "balanced_accuracy_mean": f"{bal:.6f}",
            "balanced_accuracy_std": f"{bal_std:.6f}",
        }
        summary_rows.append(row)


def print_slice_result(
    slice_name: str,
    result: dict,
    best_level: int | None = None,
) -> None:
    lvl_tag = f", best_level=L{best_level:02d}" if best_level is not None else ""
    print(
        f"✓ {slice_name}: {result['n_cells']} cells, {result['n_classes']} classes, "
        f"CV={result['cv_folds_used']}-fold, feat={result['n_features']}d{lvl_tag}"
    )
    for model, mres in result["models"].items():
        acc = mres["accuracy_mean"]
        acc_std = mres["accuracy_std"]
        bal = mres["balanced_accuracy_mean"]
        bal_std = mres["balanced_accuracy_std"]
        print(
            f"    {model:3s}  acc={acc:.4f}±{acc_std:.4f}  "
            f"bal_acc={bal:.4f}±{bal_std:.4f}"
        )


def classify_partial_dir(out_root: Path, slice_name: str, level: int) -> Path:
    return out_root / slice_name / "partial" / f"L{level:02d}"


def default_out_root(scratch: Path, mode: str) -> Path:
    subdir = "classify_traj_stats" if mode == "concat" else "classify_traj_stats_best_level"
    return scratch / subdir


def save_summary_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "slice", "best_level", "n_cells", "n_classes", "n_features",
        "model", "cv_folds",
        "accuracy_mean", "accuracy_std",
        "balanced_accuracy_mean", "balanced_accuracy_std",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def process_level(
    slice_name: str,
    level: int,
    sim_dir: Path,
    label_dir: Path,
    aij_dir: Path,
    out_root: Path,
    tag: str,
    graph_mode: str,
    recompute: bool = False,
) -> dict:
    """单个 (slice, level) 分类，写入 partial/L{level}/cv_result.json。"""
    n_cells = load_n_cells(aij_dir, slice_name, graph_mode)

    y, _ = load_labels(label_dir, slice_name, n_cells, aij_dir=aij_dir)
    X = sanitize_features(
        load_level_stats(sim_dir, slice_name, level, tag, recompute),
        f"{slice_name} L{level:02d}",
    )
    result = run_slice_classification(slice_name, X, y)
    result["level"] = level
    result["epsilon_tag"] = tag

    pdir = classify_partial_dir(out_root, slice_name, level)
    pdir.mkdir(parents=True, exist_ok=True)
    with open(pdir / "cv_result.json", "w") as f:
        json.dump(result, f, indent=2)

    bal, acc, best_model = _level_score(result)
    print(
        f"  {slice_name} L{level:02d}: best={best_model} "
        f"acc={acc:.4f} bal_acc={bal:.4f} → {pdir / 'cv_result.json'}"
    )
    return result


def aggregate_best_levels(
    out_root: Path,
    slices: list[str],
    num_levels: int,
    eps: float,
    tag: str,
) -> tuple[list[dict], dict[str, dict]]:
    """读取各 slice partial 结果，按 balanced accuracy 选最优 level。"""
    summary_rows: list[dict] = []
    all_results: dict[str, dict] = {}

    for slice_name in slices:
        level_rows: dict[int, dict] = {}
        best_level = None
        best_bal = -1.0
        best_acc = -1.0
        missing: list[str] = []

        for lvl in range(1, num_levels + 1):
            p = classify_partial_dir(out_root, slice_name, lvl) / "cv_result.json"
            if not p.exists():
                missing.append(f"L{lvl:02d}")
                continue
            with open(p) as f:
                lvl_result = json.load(f)
            bal, acc, best_model = _level_score(lvl_result)
            level_rows[lvl] = {
                "level": lvl,
                "best_model": best_model,
                "best_balanced_accuracy": bal,
                "best_accuracy": acc,
                "models": lvl_result["models"],
            }
            if bal > best_bal or (bal == best_bal and acc > best_acc):
                best_bal = bal
                best_acc = acc
                best_level = lvl

        if best_level is None:
            print(f"[SKIP] {slice_name}: 无 partial 结果"
                  + (f" (缺 {missing})" if missing else ""))
            continue

        if missing:
            print(f"[WARN] {slice_name}: 缺 {len(missing)} 层 partial: {missing[:5]}")

        with open(
            classify_partial_dir(out_root, slice_name, best_level) / "cv_result.json"
        ) as f:
            best_raw = json.load(f)

        final = dict(best_raw)
        final["best_level"] = best_level
        final["selection_metric"] = "max_balanced_accuracy"
        final["best_level_balanced_accuracy"] = best_bal
        final["best_level_accuracy"] = best_acc
        final["best_level_best_model"] = level_rows[best_level]["best_model"]

        slice_out = out_root / slice_name
        slice_out.mkdir(parents=True, exist_ok=True)
        search = {
            "selection_metric": "max_balanced_accuracy",
            "best_level": best_level,
            "levels": {str(lvl): level_rows[lvl] for lvl in sorted(level_rows)},
        }
        with open(slice_out / "level_search.json", "w") as f:
            json.dump(search, f, indent=2)
        with open(slice_out / "cv_result.json", "w") as f:
            json.dump(final, f, indent=2)

        all_results[slice_name] = final
        append_summary_rows(summary_rows, slice_name, final, best_level)
        print_slice_result(slice_name, final, best_level)

    return summary_rows, all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Burgers 轨迹统计特征 → 细胞类型分类 (CV)"
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).with_name("config_graph.yaml"),
    )
    parser.add_argument(
        "--slices", nargs="*", default=None,
        help="指定 slice 列表；默认全部 13 个",
    )
    parser.add_argument("--slice", type=str, default=None, help="单个 slice")
    parser.add_argument("--level", type=int, default=None, help="单个 cutoff 层 (1-10)")
    parser.add_argument(
        "--aggregate", action="store_true",
        help="汇总 partial/ 结果，为每个 slice 选最优 level",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="结果目录（默认 scratch/classify_traj_stats_best_level）",
    )
    parser.add_argument(
        "--mode", choices=["best_level", "concat"], default="best_level",
        help="best_level: 逐层评估；concat: 10 层拼接 60 维",
    )
    parser.add_argument(
        "--recompute-stats", action="store_true",
        help="从 trj 重算统计量（默认直接读 stats_*.npy）",
    )
    parser.add_argument(
        "--min-levels", type=int, default=10,
        help="aggregate 时要求已完成的最少层数",
    )
    args = parser.parse_args()

    if args.level is not None and not args.slice:
        parser.error("--level 需要配合 --slice")
    if args.aggregate and args.level is not None:
        parser.error("--aggregate 与 --level 不能同时使用")

    cfg = load_yaml(args.config)
    base = args.config.resolve().parent
    sim_dir = resolve_path(base, cfg["paths"]["sim_dir"])
    label_dir = resolve_path(base, cfg["paths"]["input_dir"])
    aij_dir = resolve_path(base, cfg["paths"]["output_dir"])
    graph_mode = str(cfg.get("graph", {}).get("sim_mode", "pearson"))
    num_levels = int(cfg["graph"]["num_levels"])
    eps = float(cfg["epsilon"]["value"])
    tag = eps_tag(eps)
    scratch = Path(cfg["paths"]["scratch_root"])

    out_root = args.output_dir or default_out_root(scratch, args.mode)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.slice:
        slices = [args.slice]
    elif args.slices:
        slices = list(args.slices)
    else:
        slices = list(ALL_SLICES)

    t0 = time.time()

    # ── SLURM 单 job: 一个 slice × 一个 level ──
    if args.level is not None:
        if args.level < 1 or args.level > num_levels:
            parser.error(f"level 须在 1..{num_levels}")
        print(f"分类 L{args.level:02d}: {args.slice}  ε={eps} tag={tag}")
        process_level(
            args.slice, args.level, sim_dir, label_dir, aij_dir,
            out_root, tag, graph_mode, recompute=args.recompute_stats,
        )
        return

    # ── 汇总 partial → 最优 level ──
    if args.aggregate:
        print(f"汇总: {out_root}")
        print(f"ε={eps} tag={tag} 选 level 准则: max balanced accuracy\n")
        summary_rows, all_results = aggregate_best_levels(
            out_root, slices, num_levels, eps, tag,
        )
        save_summary_csv(summary_rows, out_root / "summary.csv")
        meta = {
            "mode": "best_level",
            "epsilon": eps,
            "eps_tag": tag,
            "level_selection": "max_balanced_accuracy",
            "slices_requested": slices,
            "slices_completed": list(all_results.keys()),
            "elapsed_s": int(time.time() - t0),
        }
        with open(out_root / "run_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        if all_results:
            print("\n=== 各 slice 最优 level 与最佳模型 ===")
            print(f"{'slice':25} {'L':>3} {'model':>4} {'acc':>8} {'bal_acc':>8}")
            print("-" * 55)
            for s, res in all_results.items():
                best_model = res["best_level_best_model"]
                mres = res["models"][best_model]
                print(
                    f"{s:25} L{res['best_level']:02d} {best_model:>4} "
                    f"{mres['accuracy_mean']:8.4f} {mres['balanced_accuracy_mean']:8.4f}"
                )
        print(f"\n完成 {len(all_results)}/{len(slices)} 个 slice")
        print(f"summary → {out_root / 'summary.csv'}")
        return

    # ── 本地批量（单节点跑全部，不推荐大任务） ──
    summary_rows: list[dict] = []
    all_results: dict[str, dict] = {}

    print(f"模拟目录: {sim_dir}")
    print(f"标签目录: {label_dir}")
    print(f"输出目录: {out_root}")
    print(f"模式:     {args.mode}")
    print(f"ε={eps} tag={tag}")
    print(f"待处理 slice: {len(slices)} 个\n")

    for slice_name in slices:
        done = count_completed_levels(sim_dir, slice_name, tag, num_levels)
        if done < args.min_levels:
            print(f"[SKIP] {slice_name}: burgers partial 仅 {done}/{num_levels} 层")
            continue

        aij_meta = graph_data_dir(aij_dir, slice_name, graph_mode) / "thresholds.json"
        if not aij_meta.exists():
            print(f"[SKIP] {slice_name}: 无 thresholds.json ({aij_meta})")
            continue

        try:
            n_cells = load_n_cells(aij_dir, slice_name, graph_mode)
            y, _ = load_labels(label_dir, slice_name, n_cells, aij_dir=aij_dir)
            slice_out = out_root / slice_name
            slice_out.mkdir(parents=True, exist_ok=True)

            if args.mode == "concat":
                X = sanitize_features(
                    extract_slice_features_concat(
                        sim_dir, slice_name, num_levels, tag,
                        recompute=args.recompute_stats,
                    ),
                    slice_name,
                )
                result = run_slice_classification(slice_name, X, y)
                best_level = None
            else:
                best_level, result, search = select_best_level(
                    sim_dir, slice_name, y, num_levels, tag,
                    recompute=args.recompute_stats,
                )
                with open(slice_out / "level_search.json", "w") as f:
                    json.dump(search, f, indent=2)

            all_results[slice_name] = result
            with open(slice_out / "cv_result.json", "w") as f:
                json.dump(result, f, indent=2)

            print_slice_result(slice_name, result, best_level)
            append_summary_rows(summary_rows, slice_name, result, best_level)

        except Exception as exc:
            print(f"[FAIL] {slice_name}: {exc}")

    save_summary_csv(summary_rows, out_root / "summary.csv")
    with open(out_root / "run_meta.json", "w") as f:
        json.dump({
            "mode": args.mode,
            "epsilon": eps,
            "eps_tag": tag,
            "slices_completed": list(all_results.keys()),
            "elapsed_s": int(time.time() - t0),
        }, f, indent=2)

    print(f"\n完成 {len(all_results)}/{len(slices)} 个 slice")
    print(f"summary → {out_root / 'summary.csv'}")


if __name__ == "__main__":
    main()

