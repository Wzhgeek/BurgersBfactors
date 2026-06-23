#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
跨蛋白结果汇总：读取各蛋白 result.json，生成数据集级 CSV。

用法:
    python summarize.py --dataset 33small
    python summarize.py --all
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Cross-protein result summary")
    p.add_argument("--dataset", type=str, default="33small")
    p.add_argument("--all", action="store_true", help="Process all datasets")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override output dir (default: Pcode/result/<dataset>)")
    return p.parse_args()


def summarize_dataset(dataset: str, output_dir: Path = None):
    project_root = Path(__file__).resolve().parent
    result_base = project_root / "result" / dataset

    if output_dir is None:
        output_dir = result_base

    rows_stats = []  # for stats features
    rows_trj = []    # for trajectory features

    for protein_dir in sorted(result_base.iterdir()):
        if not protein_dir.is_dir():
            continue
        pdb = protein_dir.name
        result_path = protein_dir / "result.json"
        if not result_path.exists():
            print(f"  SKIP {pdb}: no result.json")
            continue

        with open(result_path) as f:
            data = json.load(f)

        sim_time = data.get("simulation_time_s", 0)

        # Best per-level stats — pick the single best
        best_lvl, best_eps, best_pcc = "", "", -999
        per_lvl_s = data.get("per_level_stats", {})
        for lvl_key, info in per_lvl_s.items():
            if isinstance(info, dict):
                pcc = info.get("pcc", 0)
                if pcc > best_pcc:
                    best_pcc = pcc
                    best_lvl = lvl_key
                    best_eps = info.get("eps", "")
            elif isinstance(info, (int, float)):
                if info > best_pcc:
                    best_pcc = info
                    best_lvl = lvl_key

        rows_stats.append({
            "Protein": pdb, "BestLevel": best_lvl, "BestEpsilon": best_eps,
            "BestPCC": round(best_pcc, 4),
            "Runtime_s": data.get("rf_time_stats_s", sim_time),
        })

        # Best per-level trajectory — single best
        best_lvl_t, best_eps_t, best_pcc_t = "", "", -999
        per_lvl_t = data.get("per_level_trj", {})
        for lvl_key, info in per_lvl_t.items():
            if isinstance(info, dict):
                pcc = info.get("pcc", 0)
                if pcc > best_pcc_t:
                    best_pcc_t = pcc
                    best_lvl_t = lvl_key
                    best_eps_t = info.get("eps", "")
            elif isinstance(info, (int, float)):
                if info > best_pcc_t:
                    best_pcc_t = info
                    best_lvl_t = lvl_key

        rows_trj.append({
            "Protein": pdb, "BestLevel": best_lvl_t, "BestEpsilon": best_eps_t,
            "BestPCC": round(best_pcc_t, 4),
            "Runtime_s": data.get("rf_time_trj_s", sim_time),
        })

    # Write CSV for stats features
    if rows_stats:
        write_csv(output_dir / f"{dataset}_result_stats.csv", rows_stats)

    # Write CSV for trajectory features
    if rows_trj:
        write_csv(output_dir / f"{dataset}_result_trj.csv", rows_trj)

    print(f"{dataset}: {len(set(r['Protein'] for r in rows_stats))} proteins -> "
          f"{output_dir / f'{dataset}_result_stats.csv'}")
    return len(rows_stats)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]

    # Compute averages for PCC
    pcc_key = "BestPCC" if "BestPCC" in rows[0] else "PCC"
    pcc_vals = [r[pcc_key] for r in rows if isinstance(r[pcc_key], (int, float))]
    rt_vals = [r["Runtime_s"] for r in rows if isinstance(r["Runtime_s"], (int, float))]
    avg_pcc = np.mean(pcc_vals) if pcc_vals else 0
    avg_rt = np.mean(rt_vals) if rt_vals else 0

    for row in rows:
        lines.append(",".join(str(row[k]) for k in keys))

    # Average row
    avg_row = ["AVG"] + [""] * (len(keys) - 3) + [f"{avg_pcc:.4f}", f"{avg_rt:.1f}"]
    lines.append(",".join(avg_row))

    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.all:
        for ds in ["33small", "35large", "36med"]:
            summarize_dataset(ds, args.output_dir)
    else:
        summarize_dataset(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
