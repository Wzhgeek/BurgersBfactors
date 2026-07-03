#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
权威标签源: scRNA/Single Cell RNA Sequencing/*_full_labels.csv

将上述文件转为 preprocessed/{slice}/labels.csv（cell_id, cell_type, label），
并同步到 scratch 与仓库 preprocessed/。分类与下游一律以此为准。

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

# slice 名 → 原始标签文件名
SLICE_TO_RAW: dict[str, str] = {
    "GSE45719": "GSE45719_full_labels.csv",
    "GSE67835": "GSE67835_full_labels.csv",
    "GSE75748_time": "GSE75748time_full_labels.csv",
    "GSE75748_cell": "GSE75748cell_full_labels.csv",
    "GSE84133_mouse1": "GSE84133mouse1_full_labels.csv",
    "GSE84133_mouse2": "GSE84133mouse2_full_labels.csv",
    "GSE84133_human4": "GSE84133human4_full_labels.csv",
    "GSE84133_human1": "GSE84133human1_full_labels.csv",
    "GSE84133_human2": "GSE84133human2_full_labels.csv",
    "GSE82187": "GSE82187_full_labels.csv",
    "GSE89232": "GSE89232_full_labels.csv",
    "GSE94820_discovery": "GSE94820_full_labels.csv",
    "GSE59114_C57BL6": "GSE59114_full_labels.csv",
}

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, path_value: str | Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else base / p


def convert_raw_to_labels(raw_path: Path, out_path: Path) -> int:
    """Sample Name,Cell type,Label → cell_id,cell_type,label"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(raw_path) as fin, open(out_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["cell_id", "cell_type", "label"])
        for row in reader:
            writer.writerow([
                row["Sample Name"].strip(),
                row["Cell type"].strip(),
                int(row["Label"]),
            ])
            n += 1
    return n


def sync_all(raw_dir: Path, out_root: Path, dry_run: bool = False) -> list[dict]:
    rows: list[dict] = []
    for slice_name, raw_name in SLICE_TO_RAW.items():
        raw_path = raw_dir / raw_name
        out_path = out_root / slice_name / "labels.csv"
        if not raw_path.exists():
            rows.append({"slice": slice_name, "status": "MISSING_RAW", "raw": str(raw_path)})
            continue
        if dry_run:
            with open(raw_path) as f:
                n = sum(1 for _ in csv.DictReader(f))
            rows.append({"slice": slice_name, "status": "DRY_RUN", "n_labels": n, "out": str(out_path)})
            continue
        n = convert_raw_to_labels(raw_path, out_path)
        rows.append({"slice": slice_name, "status": "OK", "n_labels": n, "out": str(out_path)})
    return rows


def validate(out_root: Path, aij_root: Path) -> list[dict]:
    report: list[dict] = []
    for slice_name in SLICE_TO_RAW:
        labels_path = out_root / slice_name / "labels.csv"
        thr_path = aij_root / slice_name / "thresholds.json"
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

        entry["label_source"] = SLICE_TO_RAW.get(slice_name, "")

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
    out_root = resolve_path(scrna_dir, cfg["paths"]["input_dir"])
    aij_root = resolve_path(scrna_dir, cfg["paths"]["output_dir"])
    raw_dir = args.raw_dir or (scrna_dir / "Single Cell RNA Sequencing")

    # 仓库内也留一份 fallback（与 classify_traj_stats.py 第二候选路径一致）
    repo_preprocessed = scrna_dir / "preprocessed"

    if not args.validate_only:
        print(f"原始标签: {raw_dir}")
        print(f"输出目录: {out_root}")
        sync_rows = sync_all(raw_dir, out_root, dry_run=args.dry_run)
        if not args.dry_run:
            repo_preprocessed.mkdir(parents=True, exist_ok=True)
            for slice_name in SLICE_TO_RAW:
                src = out_root / slice_name / "labels.csv"
                if src.exists():
                    dst = repo_preprocessed / slice_name / "labels.csv"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
        for r in sync_rows:
            print(r)

    print("\n=== 校验 ===")
    report = validate(out_root, aij_root)
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
