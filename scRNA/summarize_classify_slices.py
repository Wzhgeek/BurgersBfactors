#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
汇总 13 个 slice 分类结果 → 跨数据集最佳指标 CSV（按 BA 选取）。

读取各 slice 的 legacy_classify/result.json；若缺失则尝试 scan/*.json 并导出。

用法:
    python scRNA/summarize_classify_slices.py
    python scRNA/summarize_classify_slices.py --exclude GSE84133human1
    python scRNA/summarize_classify_slices.py --output scRNA/legacy_classify_summary/13slices_best_by_ba.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scRNA.classify_results_io import export_slice_results, load_slice_result
from scRNA.classify_traj_stats import load_n_cells, resolve_path
from scRNA.list_slices import list_slices


CSV_COLUMNS = [
    "slice",
    "n_cells",
    "epsilon",
    "level",
    "actual_n_steps",
    "max_n_steps",
    "early_stopped",
    "balanced_accuracy",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "kappa",
]


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def best_row_from_result(data: dict) -> dict:
    best = data.get("best_by_ba") or {}
    if not best and data.get("best_epsilon") is not None:
        best = {
            "epsilon": data["best_epsilon"],
            "level": data["best_level"],
            "balanced_accuracy": data.get("best_balanced_accuracy"),
            "accuracy": data.get("best_accuracy"),
            "precision": data.get("best_precision"),
            "recall": data.get("best_recall"),
            "f1": data.get("best_f1"),
            "auc": data.get("best_auc"),
            "kappa": data.get("best_kappa"),
        }
    return best


def collect_slice_row(
    slice_name: str,
    legacy_dir: Path,
    n_cells: int,
) -> dict | None:
    data = load_slice_result(legacy_dir)
    if data is None:
        scan_dir = legacy_dir / "scan"
        scans = sorted(scan_dir.glob(f"{slice_name}_legacy_classify_*_strict_legacy.json"))
        if not scans:
            return None
        with open(scans[-1]) as f:
            scan_data = json.load(f)
        grid = scan_data.get("grid", [])
        if not grid:
            return None
        meta = {k: scan_data[k] for k in (
            "feature", "method", "strict_legacy", "n_splits", "n_repeats",
        ) if k in scan_data}
        meta["partial_files"] = scan_data.get("partial_files")
        export_slice_results(
            legacy_dir=legacy_dir,
            slice_name=slice_name,
            grid=grid,
            meta=meta,
            n_cells=n_cells,
        )
        data = load_slice_result(legacy_dir)
        if data is None:
            return None

    best = best_row_from_result(data)
    if not best:
        return None

    return {
        "slice": slice_name,
        "n_cells": n_cells,
        "epsilon": best.get("epsilon"),
        "level": best.get("level"),
        "actual_n_steps": best.get("actual_n_steps"),
        "max_n_steps": best.get("max_n_steps"),
        "early_stopped": best.get("early_stopped"),
        "balanced_accuracy": best.get("balanced_accuracy"),
        "accuracy": best.get("accuracy"),
        "precision": best.get("precision"),
        "recall": best.get("recall"),
        "f1": best.get("f1"),
        "auc": best.get("auc"),
        "kappa": best.get("kappa"),
    }


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})

        if rows:
            avg = {"slice": "AVG"}
            for col in CSV_COLUMNS[1:]:
                vals = [r[col] for r in rows if r.get(col) not in (None, "")]
                if not vals:
                    avg[col] = ""
                    continue
                if col in ("n_cells", "level", "actual_n_steps", "max_n_steps"):
                    avg[col] = f"{sum(float(v) for v in vals) / len(vals):.2f}"
                elif col == "early_stopped":
                    avg[col] = ""
                else:
                    avg[col] = f"{sum(float(v) for v in vals) / len(vals):.4f}"
            writer.writerow(avg)


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总多 slice 分类最优指标 CSV")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config_graph.yaml"),
    )
    parser.add_argument("--exclude", default="", help="逗号分隔排除的 slice")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 CSV（默认 scRNA/legacy_classify_summary/13slices_best_by_ba.csv）",
    )
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    cfg = load_yaml(cfg_path)
    scrna_dir = cfg_path.parent
    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}
    slices = [name for _, name in list_slices(cfg, scrna_dir, exclude)]

    aij_dir = resolve_path(scrna_dir, cfg["paths"]["output_dir"])
    graph_mode = str(cfg.get("graph", {}).get("sim_mode", "pearson"))

    rows: list[dict] = []
    for slice_name in slices:
        n_cells = load_n_cells(aij_dir, slice_name, graph_mode) or 0
        legacy_dir = scrna_dir / slice_name / "legacy_classify"
        row = collect_slice_row(slice_name, legacy_dir, n_cells)
        if row is None:
            print(f"  SKIP {slice_name}: 无 result.json / scan 结果")
            continue
        rows.append(row)
        print(
            f"  {slice_name}: ε={row['epsilon']} L{row['level']:02d} "
            f"steps={row.get('actual_n_steps', '?')} "
            f"BA={float(row['balanced_accuracy']):.4f}"
        )

    if not rows:
        print("无可汇总 slice")
        return 1

    out_path = args.output
    if out_path is None:
        out_path = scrna_dir / "legacy_classify_summary" / "13slices_best_by_ba.csv"
    else:
        out_path = Path(out_path)
    write_summary_csv(out_path, rows)
    print(f"\n已写入 {len(rows)} 行 → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
