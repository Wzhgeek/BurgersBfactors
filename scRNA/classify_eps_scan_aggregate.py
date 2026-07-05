#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
汇总并行 ε 分类 partial JSON → 完整 eps_scan 结果。

用法:
    python classify_eps_scan_aggregate.py --slice GSE84133human1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scRNA.classify_results_io import export_slice_results, legacy_classify_dir
from scRNA.classify_traj_stats import eps_tag, load_n_cells, load_yaml, resolve_path


def parse_eps_list(raw: str | None) -> list[float] | None:
    if not raw:
        return None
    text = raw.replace("#", ",")
    return [round(float(x.strip()), 1) for x in text.split(",") if x.strip()]


def feature_partial_prefix(
    slice_name: str, feature: str, classifier: str = "rf", level: int | None = None,
) -> str:
    prefix = "lstm_classify" if classifier == "lstm" else "legacy_classify"
    if feature == "traj100":
        return f"{slice_name}_{prefix}_traj100_eps"
    if feature == "stats6_traj100":
        return f"{slice_name}_{prefix}_stats6_traj100_eps"
    if feature == "stats6_topo6":
        return f"{slice_name}_{prefix}_stats6_topo6_eps"
    if feature == "stats6_l10":
        return f"{slice_name}_{prefix}_stats6_l10_eps"
    if feature == "stats6_topo6_l10":
        return f"{slice_name}_{prefix}_stats6_topo6_l10_eps"
    if feature == "stats6_l10_pca30":
        return f"{slice_name}_{prefix}_stats6_l10_pca30_eps"
    if feature == "stats6_topo6_l10_pca30":
        return f"{slice_name}_{prefix}_stats6_topo6_l10_pca30_eps"
    if feature == "traj100_l10":
        return f"{slice_name}_{prefix}_traj100_l10_eps"
    if feature == "trajfull_l10":
        return f"{slice_name}_{prefix}_trajfull_l10_eps"
    if feature == "trajfull":
        if level is None:
            raise ValueError("trajfull aggregate 须 --level")
        return f"{slice_name}_{prefix}_trajfull_L{level:02d}_eps"
    return f"{slice_name}_{prefix}_eps"


def feature_scan_name(feature: str) -> str:
    return feature


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", default="GSE84133human1")
    parser.add_argument("--level", type=int, default=None, help="trajfull 单层汇总时指定 L01-L10")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--eps-list", default=None, help="仅汇总指定 ε，逗号或 # 分隔")
    parser.add_argument("--scan-tag", default=None, help="输出文件名标签，如 eps5_10")
    parser.add_argument("--file-suffix", default="", help="partial 文件名后缀，如 ns6000")
    parser.add_argument(
        "--experiment-tag", default=None,
        help="结果目录 classify_dir/{slice}/{tag}/",
    )
    parser.add_argument(
        "--feature",
        choices=(
            "stats6", "traj100", "stats6_traj100", "stats6_topo6", "stats6_l10",
            "stats6_topo6_l10", "stats6_l10_pca30", "stats6_topo6_l10_pca30",
            "traj100_l10", "trajfull_l10", "trajfull",
        ),
        default="stats6",
    )
    parser.add_argument(
        "--classifier",
        choices=("rf", "lstm"),
        default="rf",
        help="partial 文件名前缀：rf=legacy_classify; lstm=lstm_classify",
    )
    parser.add_argument("--strict-legacy", action="store_true", default=True)
    parser.add_argument(
        "--config", default="scRNA/config_graph.yaml",
        help="config_graph.yaml 或 config_graph_pc3.yaml",
    )
    args = parser.parse_args()

    scrna_dir = Path(__file__).resolve().parent
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = scrna_dir.parent / cfg_path
    cfg = load_yaml(cfg_path)
    slice_dir = legacy_classify_dir(cfg, scrna_dir, args.slice)
    if args.experiment_tag:
        slice_dir = slice_dir / args.experiment_tag
    partial_dir = Path(args.out_dir) if args.out_dir else slice_dir / "partial"
    scan_dir = slice_dir / "scan"
    scan_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_strict_legacy" if args.strict_legacy else ""
    eps_filter = parse_eps_list(args.eps_list)
    fs = f"_{args.file_suffix}" if args.file_suffix else ""
    feat_prefix = feature_partial_prefix(
        args.slice, args.feature, args.classifier, level=args.level,
    )

    if eps_filter:
        partials = [
            partial_dir / f"{feat_prefix}{eps_tag(e)}{fs}_partial{suffix}.json"
            for e in eps_filter
        ]
        partials = [p for p in partials if p.exists()]
    else:
        pattern = f"{feat_prefix}*{fs}_partial{suffix}.json"
        partials = sorted(partial_dir.glob(pattern))
        if not partials and args.experiment_tag:
            partials = sorted(
                (resolve_path(scrna_dir, cfg["paths"]["classify_dir"])
                 / args.slice / args.experiment_tag / "partial").glob(pattern)
            )

    if not partials:
        hint = args.eps_list or f"{args.slice}_legacy_classify_eps*{fs}_partial{suffix}.json"
        print(f"未找到 partial 文件: {hint}")
        sys.exit(1)

    grid: list[dict] = []
    eps_values: list[float] = []
    meta: dict = {}
    for path in partials:
        data = json.loads(path.read_text())
        grid.extend(data["grid"])
        eps_values.extend(data.get("epsilon_values", []))
        meta = {k: data[k] for k in (
            "slice", "feature", "method", "strict_legacy",
            "n_splits", "n_repeats",
        ) if k in data}

    eps_values = sorted(set(round(float(e), 1) for e in eps_values))
    best = max(grid, key=lambda r: (r["balanced_accuracy_mean"], r["accuracy_mean"]))
    summary = {
        **meta,
        "slice": args.slice,
        "epsilon_values": eps_values,
        "best_epsilon": best["epsilon"],
        "best_level": best["level"],
        "best_balanced_accuracy": best["balanced_accuracy_mean"],
        "best_accuracy": best["accuracy_mean"],
        "best_precision": best.get("precision_mean"),
        "best_recall": best.get("recall_mean"),
        "best_f1": best.get("f1_mean"),
        "best_auc": best.get("auc_mean"),
        "best_kappa": best.get("kappa_mean"),
        "grid": grid,
        "partial_files": [p.name for p in partials],
    }

    scan_tag = f"_{args.scan_tag}" if args.scan_tag else ""
    exp_tag = f"_{args.experiment_tag}" if args.experiment_tag and not args.scan_tag else ""
    feat_scan = feature_scan_name(args.feature)
    out_path = scan_dir / (
        f"{args.slice}_legacy_classify_{feat_scan}_eps_scan{scan_tag}{exp_tag}{suffix}.json"
    )
    out_path.write_text(json.dumps(summary, indent=2))

    # 对齐蛋白回归：all_score 网格 + result.json
    sim_dir = resolve_path(scrna_dir, cfg["paths"]["sim_dir"])
    aij_dir = resolve_path(scrna_dir, cfg["paths"]["output_dir"])
    graph_mode = str(cfg.get("graph", {}).get("sim_mode", "pearson"))
    n_cells = load_n_cells(aij_dir, args.slice, graph_mode)
    meta_export = {
        **meta,
        "feature": args.feature,
        "partial_files": summary.get("partial_files"),
        "scan_tag": args.scan_tag,
    }
    result_json = export_slice_results(
        legacy_dir=slice_dir,
        slice_name=args.slice,
        grid=grid,
        meta=meta_export,
        sim_dir=sim_dir,
        file_suffix=args.file_suffix,
        n_cells=n_cells,
        num_levels=int(cfg["graph"]["num_levels"]),
    )

    print(f"汇总 {len(partials)} 个 partial → {len(grid)} 格点")
    print(f"最优 ε={best['epsilon']:.1f} L{best['level']:02d}: "
          f"acc={best['accuracy_mean']*100:.2f}% BA={best['balanced_accuracy_mean']*100:.2f}% "
          f"P={best.get('precision_mean', 0):.4f} R={best.get('recall_mean', 0):.4f} "
          f"F1={best.get('f1_mean', 0):.4f} AUC={best.get('auc_mean', 0):.4f} "
          f"κ={best.get('kappa_mean', 0):.4f}")
    print(f"结果 → {out_path}")
    print(f"all_score/ + result.json → {result_json}")


if __name__ == "__main__":
    main()
