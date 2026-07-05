#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
权威标签源（按优先级）:
  1. scRNA/Single Cell RNA Sequencing/*_full_labels.csv
  2. data_aij/{slice}/{slice}_full_labels.csv（build_graph 阶段已落盘）

将原始 CSV 转为 preprocessed/{slice}/labels.csv（cell_id, cell_type, label），
同步到 scratch 与仓库 scRNA/preprocessed/。

用法:
    python sync_labels.py
    python sync_labels.py --dry-run
    python sync_labels.py --validate-only
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import yaml

LEGACY_RAW_NAMES: dict[str, str] = {
    "GSE75748time": "GSE75748time_full_labels.csv",
    "GSE75748cell": "GSE75748cell_full_labels.csv",
    "GSE84133mouse1": "GSE84133mouse1_full_labels.csv",
    "GSE84133mouse2": "GSE84133mouse2_full_labels.csv",
    "GSE84133human4": "GSE84133human4_full_labels.csv",
    "GSE84133human1": "GSE84133human1_full_labels.csv",
    "GSE84133human2": "GSE84133human2_full_labels.csv",
    "GSE94820": "GSE94820_full_labels.csv",
    "GSE59114": "GSE59114_full_labels.csv",
}


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, path_value: str | Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else base / p


def graph_data_dir(aij_root: Path, slice_name: str, graph_mode: str) -> Path:
    base = aij_root / slice_name
    sub = base / graph_mode
    if graph_mode in ("pearson", "euclidean") and sub.exists():
        return sub
    return base


def find_raw_labels(
    slice_name: str,
    raw_dir: Path,
    aij_root: Path,
) -> Path | None:
    candidates = [
        raw_dir / f"{slice_name}_full_labels.csv",
        aij_root / slice_name / f"{slice_name}_full_labels.csv",
    ]
    legacy = LEGACY_RAW_NAMES.get(slice_name)
    if legacy:
        candidates.insert(0, raw_dir / legacy)
    return next((p for p in candidates if p.exists()), None)


def convert_raw_to_labels(raw_path: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(raw_path) as fin, open(out_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["cell_id", "cell_type", "label"])
        for row in reader:
            if "Sample Name" in row:
                cell_id = row["Sample Name"].strip()
                cell_type = row["Cell type"].strip()
                label = int(row["Label"])
            elif "cell_id" in row:
                cell_id = row["cell_id"].strip()
                cell_type = row.get("cell_type", "").strip()
                label = int(row["label"])
            else:
                raise ValueError(f"{raw_path}: 无法识别列名 {reader.fieldnames}")
            writer.writerow([cell_id, cell_type, label])
            n += 1
    return n


def sync_all(
    slices: list[str],
    raw_dir: Path,
    aij_root: Path,
    out_root: Path,
    dry_run: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    for slice_name in slices:
        raw_path = find_raw_labels(slice_name, raw_dir, aij_root)
        out_path = out_root / slice_name / "labels.csv"
        if raw_path is None:
            rows.append({
                "slice": slice_name,
                "status": "MISSING_RAW",
                "tried": [
                    str(raw_dir / f"{slice_name}_full_labels.csv"),
                    str(aij_root / slice_name / f"{slice_name}_full_labels.csv"),
                ],
            })
            continue
        if dry_run:
            with open(raw_path) as f:
                n = sum(1 for _ in csv.DictReader(f))
            rows.append({
                "slice": slice_name,
                "status": "DRY_RUN",
                "n_labels": n,
                "raw": str(raw_path),
                "out": str(out_path),
            })
            continue
        n = convert_raw_to_labels(raw_path, out_path)
        rows.append({
            "slice": slice_name,
            "status": "OK",
            "n_labels": n,
            "raw": str(raw_path),
            "out": str(out_path),
        })
    return rows


def validate(
    slices: list[str],
    out_root: Path,
    aij_root: Path,
    graph_mode: str,
) -> list[dict]:
    report: list[dict] = []
    for slice_name in slices:
        labels_path = out_root / slice_name / "labels.csv"
        thr_path = graph_data_dir(aij_root, slice_name, graph_mode) / "thresholds.json"
        entry: dict = {"slice": slice_name}

        if not labels_path.exists():
            entry.update(status="NO_LABELS", labels_path=str(labels_path))
            report.append(entry)
            continue

        with open(labels_path) as f:
            rows = list(csv.DictReader(f))
        n_labels = len(rows)
        labels = [int(r["label"]) for r in rows]
        n_classes = len(set(labels))

        entry.update(n_labels=n_labels, n_classes=n_classes, labels_path=str(labels_path))

        if thr_path.exists():
            with open(thr_path) as f:
                n_cells = int(json.load(f)["n_cells"])
            entry["n_cells"] = n_cells
            entry["count_match"] = n_labels == n_cells
        else:
            entry["n_cells"] = None
            entry["count_match"] = None

        if entry.get("count_match") is True:
            entry["status"] = "OK"
        elif entry.get("count_match") is False:
            entry["status"] = "COUNT_MISMATCH"
        else:
            entry["status"] = "NO_THRESHOLDS"

        report.append(entry)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="同步 scRNA 标签到 preprocessed/")
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).with_name("config_graph.yaml"),
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=None,
        help="原始标签目录（默认 scRNA/Single Cell RNA Sequencing）",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    scrna_dir = Path(__file__).resolve().parent
    cfg = load_yaml(args.config)
    slices = list(cfg.get("slices", []))
    out_root = resolve_path(scrna_dir, cfg["paths"]["input_dir"])
    aij_root = resolve_path(scrna_dir, cfg["paths"]["output_dir"])
    graph_mode = str(cfg.get("graph", {}).get("sim_mode", "pearson"))
    raw_dir = args.raw_dir or (scrna_dir / "Single Cell RNA Sequencing")

    repo_preprocessed = scrna_dir / "preprocessed"

    if not args.validate_only:
        print(f"原始标签: {raw_dir}（不存在则回退 data_aij）")
        print(f"data_aij: {aij_root}")
        print(f"输出目录: {out_root}")
        sync_rows = sync_all(slices, raw_dir, aij_root, out_root, dry_run=args.dry_run)
        if not args.dry_run:
            repo_preprocessed.mkdir(parents=True, exist_ok=True)
            for slice_name in slices:
                src = out_root / slice_name / "labels.csv"
                if src.exists():
                    dst = repo_preprocessed / slice_name / "labels.csv"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
        for r in sync_rows:
            print(r)

    print("\n=== 校验 ===")
    report = validate(slices, out_root, aij_root, graph_mode)
    ok = mismatch = missing = 0
    for r in report:
        st = r.get("status", "?")
        if st == "OK":
            ok += 1
        elif st == "COUNT_MISMATCH":
            mismatch += 1
        else:
            missing += 1
        line = (
            f"{r['slice']:25} {st:16} "
            f"labels={r.get('n_labels', '-'):>5} "
            f"cells={r.get('n_cells', '-'):>5} "
            f"classes={r.get('n_classes', '-'):>3}"
        )
        if r.get("note"):
            line += f"  | {r['note']}"
        print(line)

    print(f"\n汇总: OK={ok} COUNT_MISMATCH={mismatch} 其他={missing} / {len(report)}")
    summary_path = out_root / "labels_sync_report.json"
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"报告 → {summary_path}")


if __name__ == "__main__":
    main()
