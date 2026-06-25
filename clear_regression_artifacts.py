#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
清除指定数据集中回归相关产物，保留模拟轨迹、特征与基本元数据。

删除: all_score/, figures/, run_regression_only.log, 数据集级汇总 CSV
result.json: 去掉回归字段，仅保留 sim_only 风格元数据（或从 run.log 重建）

用法:
    python clear_regression_artifacts.py --dataset 36med
    python clear_regression_artifacts.py --dataset 35large 36med
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils import load_yaml, resolve_path, save_json

SIM_META_KEYS = (
    "pdb_id", "dataset", "n_atoms", "n_steps", "nu", "dt", "dx",
    "eps_range", "thresholds", "evaluation", "simulation_time_s",
)


def parse_args():
    p = argparse.ArgumentParser(description="Clear regression artifacts, keep sim metadata")
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    p.add_argument("--dataset", nargs="+", required=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _parse_run_log(log_path: Path) -> dict:
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors="replace")
    out = {}
    m = re.search(r"n_steps=(\d+)", text)
    if m:
        out["n_steps"] = int(m.group(1))
    m = re.search(r"Simulation done in (\d+)s", text)
    if m:
        out["simulation_time_s"] = int(m.group(1))
    return out


def _sim_meta_from_existing(data: dict, pdb_id: str, dataset: str) -> dict:
    meta = {k: data[k] for k in SIM_META_KEYS if k in data}
    meta["pdb_id"] = meta.get("pdb_id", pdb_id)
    meta["dataset"] = meta.get("dataset", dataset)
    meta["pipeline_mode"] = "sim_only"
    return meta


def _sim_meta_from_config(cfg: dict, pdb_id: str, dataset: str, log_hints: dict) -> dict:
    dyn = cfg["dynamics"]
    eps = cfg["epsilon"]
    graph = cfg["graph"]
    ev = cfg.get("evaluation", {})
    use_cv = bool(ev.get("use_cv", True))
    meta = {
        "pdb_id": pdb_id,
        "dataset": dataset,
        "pipeline_mode": "sim_only",
        "nu": dyn["nu"],
        "dt": dyn["dt"],
        "dx": dyn["dx"],
        "eps_range": [eps["start"], eps["stop"], eps["step"]],
        "evaluation": {
            "use_cv": use_cv,
            "cv_folds": int(ev.get("cv_folds", 10)) if use_cv else 0,
            "test_size": float(ev.get("test_size", 0.2)),
            "random_state": int(ev.get("random_state", 42)),
        },
        "simulation_time_s": int(log_hints.get("simulation_time_s", 0)),
        "n_steps": int(log_hints.get("n_steps", 0)),
    }
    if "num_levels" in graph:
        pass  # thresholds filled below if available
    return meta


def clean_protein(protein_dir: Path, dataset: str, dry_run: bool) -> dict:
    pdb_id = protein_dir.name
    stats = {"removed_dirs": 0, "result_rewritten": 0, "result_created": 0, "result_kept": 0}

    for name in ("all_score", "figures"):
        p = protein_dir / name
        if p.exists():
            stats["removed_dirs"] += 1
            if not dry_run:
                import shutil
                shutil.rmtree(p)

    reg_log = protein_dir / "run_regression_only.log"
    if reg_log.exists():
        stats["removed_dirs"] += 1
        if not dry_run:
            reg_log.unlink()

    result_path = protein_dir / "result.json"
    log_hints = _parse_run_log(protein_dir / "run.log")

    if result_path.exists():
        data = json.loads(result_path.read_text())
        if data.get("pipeline_mode") == "sim_only" and not any(
            k in data for k in ("per_level_stats", "best_combined_stats", "rf_time_stats_s")
        ):
            stats["result_kept"] += 1
        else:
            meta = _sim_meta_from_existing(data, pdb_id, dataset)
            if not meta.get("n_steps") and log_hints.get("n_steps"):
                meta["n_steps"] = log_hints["n_steps"]
            if not meta.get("simulation_time_s") and log_hints.get("simulation_time_s"):
                meta["simulation_time_s"] = log_hints["simulation_time_s"]
            stats["result_rewritten"] += 1
            if not dry_run:
                save_json(meta, result_path)
    else:
        cfg_path = protein_dir / "config.yaml"
        if cfg_path.exists() or log_hints:
            cfg = load_yaml(cfg_path) if cfg_path.exists() else load_yaml(
                Path(__file__).with_name("config.yaml"))
            meta = _sim_meta_from_config(cfg, pdb_id, dataset, log_hints)
            label = protein_dir / "Aijandlabel" / f"{pdb_id}_label.npy"
            if label.exists():
                import numpy as np
                meta["n_atoms"] = int(len(np.load(label)))
            stats["result_created"] += 1
            if not dry_run:
                save_json(meta, result_path)

    return stats


def clean_dataset(result_root: Path, dataset: str, dry_run: bool) -> None:
    base = result_root / dataset
    if not base.is_dir():
        print(f"SKIP {dataset}: not found {base}")
        return

    totals = {"removed_dirs": 0, "result_rewritten": 0, "result_created": 0, "result_kept": 0}
    n = 0
    for protein_dir in sorted(base.iterdir()):
        if not protein_dir.is_dir():
            continue
        n += 1
        s = clean_protein(protein_dir, dataset, dry_run)
        for k in totals:
            totals[k] += s[k]

    for csv_name in (f"{dataset}_result_stats.csv", f"{dataset}_result_trj.csv"):
        csv_path = base / csv_name
        if csv_path.exists():
            totals["removed_dirs"] += 1
            if not dry_run:
                csv_path.unlink()
            print(f"  removed {csv_path}")

    tag = "[dry-run] " if dry_run else ""
    print(f"{tag}{dataset}: {n} proteins | "
          f"rm_dirs={totals['removed_dirs']} "
          f"result_rewrite={totals['result_rewritten']} "
          f"result_create={totals['result_created']} "
          f"result_keep={totals['result_kept']}")


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    cfg = load_yaml(args.config)
    paths = cfg.get("paths", {})
    result_root = resolve_path(project_root, paths.get("output_root", project_root / "result"))

    for ds in args.dataset:
        clean_dataset(result_root, ds, args.dry_run)


if __name__ == "__main__":
    main()
