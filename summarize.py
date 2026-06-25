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
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils import load_yaml, resolve_path


def parse_args():
    p = argparse.ArgumentParser(description="Cross-protein result summary")
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    p.add_argument("--dataset", type=str, default="33small")
    p.add_argument("--all", action="store_true", help="Process all datasets")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override output dir (default: output_root/<dataset> from config)")
    return p.parse_args()


def _infer_use_cv(eval_info: dict) -> bool:
    if "use_cv" in eval_info:
        return bool(eval_info["use_cv"])
    return int(eval_info.get("cv_folds", 10)) >= 2


def _entry_score(info: dict, use_cv: bool) -> float:
    if isinstance(info, dict):
        if use_cv:
            return float(info.get("oof_pcc", info.get("pcc", -999)))
        return float(info.get("single_pcc", 0))
    if isinstance(info, (int, float)):
        return float(info)
    return -999.0


def _best_row(pdb: str, per_level: dict, runtime, eval_info: dict) -> dict:
    """从 per_level_* 中按主指标选全局最优，同时输出 OOF / 折 / single 指标。"""
    use_cv = _infer_use_cv(eval_info)
    best_lvl, best_eps, best_model = "", "", "RF"
    best_oof, best_fold, best_fold_pcc, best_mean, best_single = 0, 0, 0, 0, 0
    best_score = -999.0

    for lvl_key, info in per_level.items():
        if not isinstance(info, dict):
            if isinstance(info, (int, float)) and use_cv and float(info) > best_score:
                best_score = float(info)
                best_oof = float(info)
                best_lvl = lvl_key
            continue
        score = _entry_score(info, use_cv)
        if score > best_score:
            best_score = score
            best_lvl = lvl_key
            best_eps = info.get("eps", "")
            best_model = info.get("model", "RF")
            best_oof = float(info.get("oof_pcc", info.get("pcc", 0)))
            best_single = float(info.get("single_pcc", 0))
            best_fold = int(info.get("best_fold", 0))
            best_fold_pcc = float(info.get("best_fold_pcc", 0))
            best_mean = float(info.get("mean_fold_pcc", 0))

    return {
        "Protein": pdb,
        "Level": best_lvl,
        "Epsilon": best_eps,
        "OOF_PCC": round(best_oof, 4) if use_cv else 0.0,
        "BestFold": best_fold,
        "BestFold_PCC": round(best_fold_pcc, 4) if use_cv else 0.0,
        "MeanFold_PCC": round(best_mean, 4) if use_cv else 0.0,
        "Single_PCC": round(best_single, 4) if not use_cv else 0.0,
        "Runtime": runtime,
        "Model": best_model,
    }


def _best_row(pdb: str, per_level: dict, runtime) -> dict:
    """从 per_level_* 中按 OOF 选全局最优，同时输出三种 PCC。"""
    best_lvl, best_eps, best_model = "", "", "RF"
    best_oof, best_fold, best_fold_pcc, best_mean = -999, 1, -999, -999
    for lvl_key, info in per_level.items():
        if isinstance(info, dict):
            oof = float(info.get("oof_pcc", info.get("pcc", 0)))
            if oof > best_oof:
                best_oof = oof
                best_lvl = lvl_key
                best_eps = info.get("eps", "")
                best_model = info.get("model", "RF")
                best_fold = int(info.get("best_fold", 1))
                best_fold_pcc = float(info.get("best_fold_pcc", oof))
                best_mean = float(info.get("mean_fold_pcc", oof))
        elif isinstance(info, (int, float)):
            if info > best_oof:
                best_oof = info
                best_lvl = lvl_key

    return {
        "Protein": pdb,
        "Level": best_lvl,
        "Epsilon": best_eps,
        "OOF_PCC": round(best_oof, 4),
        "BestFold": best_fold,
        "BestFold_PCC": round(best_fold_pcc, 4) if best_fold_pcc != -999 else "",
        "MeanFold_PCC": round(best_mean, 4) if best_mean != -999 else "",
        "Runtime": runtime,
        "Model": best_model,
    }


def summarize_dataset(dataset: str, result_root: Path, output_dir: Path = None):
    result_base = result_root / dataset

    if output_dir is None:
        output_dir = result_base
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    rows_stats = []
    rows_trj = []

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

        if data.get("pipeline_mode") == "sim_only":
            print(f"  SKIP {pdb}: sim_only (no regression results)")
            continue

        sim_time = data.get("simulation_time_s", 0)
        eval_info = data.get("evaluation", {})

        rows_stats.append(_best_row(
            pdb, data.get("per_level_stats", {}),
            data.get("rf_time_stats_s", sim_time),
            eval_info,
        ))

        rows_trj.append(_best_row(
            pdb, data.get("per_level_trj", {}),
            data.get("rf_time_trj_s", sim_time),
            eval_info,
        ))

    if rows_stats:
        write_csv(output_dir / f"{dataset}_result_stats.csv", rows_stats)

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

    oof_vals = [r["OOF_PCC"] for r in rows if isinstance(r.get("OOF_PCC"), (int, float))]
    bf_vals = [r["BestFold_PCC"] for r in rows if isinstance(r.get("BestFold_PCC"), (int, float))]
    mf_vals = [r["MeanFold_PCC"] for r in rows if isinstance(r.get("MeanFold_PCC"), (int, float))]
    sp_vals = [r["Single_PCC"] for r in rows if isinstance(r.get("Single_PCC"), (int, float))]
    rt_vals = [r["Runtime"] for r in rows if isinstance(r.get("Runtime"), (int, float))]
    avg_oof = np.mean(oof_vals) if oof_vals else 0
    avg_bf = np.mean(bf_vals) if bf_vals else 0
    avg_mf = np.mean(mf_vals) if mf_vals else 0
    avg_sp = np.mean([v for v in sp_vals if v > 0]) if any(v > 0 for v in sp_vals) else 0
    avg_rt = np.mean(rt_vals) if rt_vals else 0

    for row in rows:
        lines.append(",".join(str(row[k]) for k in keys))

    avg_row = {k: "" for k in keys}
    avg_row["Protein"] = "AVG"
    avg_row["OOF_PCC"] = f"{avg_oof:.4f}"
    avg_row["BestFold_PCC"] = f"{avg_bf:.4f}"
    avg_row["MeanFold_PCC"] = f"{avg_mf:.4f}"
    avg_row["Single_PCC"] = f"{avg_sp:.4f}"
    avg_row["Runtime"] = f"{avg_rt:.1f}"
    lines.append(",".join(str(avg_row[k]) for k in keys))

    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    cfg = load_yaml(args.config)
    paths = cfg.get("paths", {})
    result_root = resolve_path(project_root, paths.get("output_root", project_root / "result"))
    out_override = Path(args.output_dir) if args.output_dir else None
    datasets = cfg.get("datasets", ["33small", "35large", "36med"])

    if args.all:
        for ds in datasets:
            summarize_dataset(ds, result_root, out_override)
    else:
        summarize_dataset(args.dataset, result_root, out_override)


if __name__ == "__main__":
    main()
