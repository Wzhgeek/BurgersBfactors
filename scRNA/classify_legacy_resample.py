#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
参考旧版 GSE 分类脚本：折内 adjust_train_test 上采样 + StandardScaler + RF，
5-fold CV 重复多次，评估分类指标。

主指标（与旧版一致）: balanced accuracy (BA), accuracy (acc)
扩展指标（macro 多分类平均）: Precision (P), Recall (R), F1, AUC (OvR), Cohen's Kappa

特征：Burgers partial 的 6 维统计量 (max/min/mean/var/median/std)。

用法:
    python classify_legacy_resample.py --slice GSE84133human1
    python classify_legacy_resample.py --slice GSE84133human1 --strict-legacy
    python classify_legacy_resample.py --slice GSE84133human1 --feature traj100 --strict-legacy
    python classify_legacy_resample.py --slice GSE84133human1 --level 7
    python classify_legacy_resample.py --slice GSE84133human1 --eps-list 1.0,2.0,3.0,4.0,5.0 --strict-legacy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scRNA.classify_traj_stats import (
    eps_tag,
    load_labels,
    load_level_stats,
    load_n_cells,
    load_yaml,
    partial_level_dir,
    resolve_path,
    sanitize_features,
)
from scRNA.analyze_traj_100pts import (
    load_level_traj_features,
    trajectory_sample_indices,
)
from scRNA.classify_results_io import export_slice_results


def adjust_train_test(
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    rng: np.random.Generator | None = None,
    min_train: int = 5,
    min_test: int = 3,
    multiplier: int = 5,
    strict_legacy: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    折内调整训练/测试集。

    strict_legacy=False（默认修正版）:
      - 有放回上采样，保留重复样本（不做 set 去重）

    strict_legacy=True（与旧 GSE 脚本一致）:
      - np.random.seed(1) 固定随机种子
      - choice 后 list(set(...)) 去重，训练集无重复行
    """
    if strict_legacy:
        np.random.seed(1)

    unique_labels_temp = np.intersect1d(np.unique(y_train), np.unique(y_test))
    unique_labels: list = []
    counter: list[int] = []
    test_pos_parts: list[np.ndarray] = []

    for label in unique_labels_temp:
        l_train = np.where(y_train == label)[0]
        l_test = np.where(y_test == label)[0]
        if l_train.shape[0] > min_train and l_test.shape[0] > min_test:
            unique_labels.append(label)
            test_pos_parts.append(l_test)
            counter.append(int(l_train.shape[0]))

    if not unique_labels:
        return None

    test_pos = np.concatenate(test_pos_parts)
    test_pos.sort()
    new_test_index = test_index[test_pos]
    new_y_test = y_test[test_pos]

    avg_count = int(np.ceil(np.mean(counter)))
    target = multiplier * avg_count

    train_pos_parts: list[np.ndarray] = []
    for label in unique_labels:
        l_train = np.where(y_train == label)[0]
        if strict_legacy:
            index = np.random.choice(l_train, target)
            train_pos_parts.append(index)
        else:
            chosen = rng.choice(l_train, size=target, replace=True)
            train_pos_parts.append(chosen)

    if strict_legacy:
        # 旧代码: list(set(np.concatenate(...))) — 去重后训练集样本数远小于 5×avgCount
        train_pos = np.array(sorted(set(np.concatenate(train_pos_parts))), dtype=int)
    else:
        train_pos = np.concatenate(train_pos_parts)

    new_train_index = train_index[train_pos]
    new_y_train = y_train[train_pos]

    return new_y_train, new_y_test, new_train_index, new_test_index


def compute_rf(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """训练 RF，返回 (y_pred, y_proba, clf.classes_)。"""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = RandomForestClassifier(random_state=1)
    clf.fit(X_train_s, y_train)
    return clf.predict(X_test_s), clf.predict_proba(X_test_s), clf.classes_


def _safe_macro_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> float:
    """Macro AUC (OvR)；测试集不足 2 类或标签不匹配时返回 nan。"""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(
            roc_auc_score(
                y_true, y_proba, labels=classes,
                multi_class="ovr", average="macro",
            )
        )
    except ValueError:
        return float("nan")


def fold_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    """单折 macro P/R/F1、AUC、Kappa。"""
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc": _safe_macro_auc(y_true, y_proba, classes),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def _repeat_metric_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def compute_kfold_classification(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed_base: int = 0,
    strict_legacy: bool = False,
) -> dict:
    """5-fold × n_repeats，折内 adjust_train_test + RF，返回 BA/acc/P/R/F1/AUC/Kappa 统计。"""
    metric_names = ("ba", "acc", "precision", "recall", "f1", "auc", "kappa")
    repeat_scores: dict[str, list[float]] = {k: [] for k in metric_names}
    fold_details: list[dict] = []

    for rep in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed_base + rep)
        fold_vals: dict[str, np.ndarray] = {
            k: np.full(n_splits, np.nan) for k in metric_names
        }
        rng = None if strict_legacy else np.random.default_rng(seed_base + rep)

        for fold_i, (train_index, test_index) in enumerate(kf.split(X)):
            y_train = y[train_index]
            y_test = y[test_index]

            adjusted = adjust_train_test(
                y_train, y_test, train_index, test_index,
                rng=rng, strict_legacy=strict_legacy,
            )
            if adjusted is None:
                continue

            y_tr, y_te, tr_idx, te_idx = adjusted
            X_tr = X[tr_idx]
            X_te = X[te_idx]

            y_pred, y_proba, classes = compute_rf(X_tr, X_te, y_tr, y_te)
            y_eval = y[te_idx]
            m = fold_classification_metrics(y_eval, y_pred, y_proba, classes)

            fold_vals["ba"][fold_i] = m["balanced_accuracy"]
            fold_vals["acc"][fold_i] = m["accuracy"]
            fold_vals["precision"][fold_i] = m["precision"]
            fold_vals["recall"][fold_i] = m["recall"]
            fold_vals["f1"][fold_i] = m["f1"]
            fold_vals["auc"][fold_i] = m["auc"]
            fold_vals["kappa"][fold_i] = m["kappa"]

            fold_details.append({
                "repeat": rep + 1,
                "fold": fold_i + 1,
                "n_train": int(len(y_tr)),
                "n_test": int(len(y_te)),
                "n_classes_test": int(len(np.unique(y_te))),
                **m,
            })

        for key in metric_names:
            repeat_scores[key].append(float(np.nanmean(fold_vals[key])))

    out: dict = {
        "balanced_accuracy_mean": _repeat_metric_summary(repeat_scores["ba"])["mean"],
        "balanced_accuracy_std": _repeat_metric_summary(repeat_scores["ba"])["std"],
        "accuracy_mean": _repeat_metric_summary(repeat_scores["acc"])["mean"],
        "accuracy_std": _repeat_metric_summary(repeat_scores["acc"])["std"],
        "precision_mean": _repeat_metric_summary(repeat_scores["precision"])["mean"],
        "precision_std": _repeat_metric_summary(repeat_scores["precision"])["std"],
        "recall_mean": _repeat_metric_summary(repeat_scores["recall"])["mean"],
        "recall_std": _repeat_metric_summary(repeat_scores["recall"])["std"],
        "f1_mean": _repeat_metric_summary(repeat_scores["f1"])["mean"],
        "f1_std": _repeat_metric_summary(repeat_scores["f1"])["std"],
        "auc_mean": _repeat_metric_summary(repeat_scores["auc"])["mean"],
        "auc_std": _repeat_metric_summary(repeat_scores["auc"])["std"],
        "kappa_mean": _repeat_metric_summary(repeat_scores["kappa"])["mean"],
        "kappa_std": _repeat_metric_summary(repeat_scores["kappa"])["std"],
        "repeat_scores_ba": repeat_scores["ba"],
        "repeat_scores_acc": repeat_scores["acc"],
        "repeat_scores_precision": repeat_scores["precision"],
        "repeat_scores_recall": repeat_scores["recall"],
        "repeat_scores_f1": repeat_scores["f1"],
        "repeat_scores_auc": repeat_scores["auc"],
        "repeat_scores_kappa": repeat_scores["kappa"],
        "fold_details": fold_details,
        "metrics_note": (
            "precision/recall/f1: macro average; "
            "auc: macro OvR; kappa: Cohen's kappa"
        ),
    }
    return out


def load_level_features(
    feature: str,
    sim_dir: Path,
    slice_name: str,
    level: int,
    tag: str,
    sample_idx: np.ndarray | None = None,
) -> np.ndarray:
    if feature == "stats6":
        return load_level_stats(sim_dir, slice_name, level, tag, recompute=False)
    if feature == "traj100":
        if sample_idx is None:
            raise ValueError("traj100 需要 sample_idx")
        X = load_level_traj_features(sim_dir, slice_name, level, tag, sample_idx)
        return sanitize_features(X, f"{slice_name} L{level:02d}")
    raise ValueError(f"未知特征: {feature}")


def main() -> None:
    parser = argparse.ArgumentParser(description="旧版折内上采样 + RF 分类")
    parser.add_argument("--config", default="scRNA/config_graph.yaml")
    parser.add_argument("--slice", default="GSE84133human1")
    parser.add_argument("--level", type=int, default=None, help="指定层；默认跑 L01-L10")
    parser.add_argument(
        "--feature", choices=("stats6", "traj100"), default="stats6",
        help="stats6=6维统计量; traj100=轨迹每10步采样100点",
    )
    parser.add_argument(
        "--epsilon", type=float, default=None,
        help="指定单个耦合强度 ε（默认读 config epsilon.value）",
    )
    parser.add_argument(
        "--eps-list", type=str, default=None,
        help="扫描多个 ε，逗号分隔，如 1.0,2.0,3.0,4.0,5.0",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--strict-legacy", action="store_true",
        help="与旧 GSE 脚本一致: np.random.seed(1) + set() 去重上采样",
    )
    parser.add_argument(
        "--partial-out", action="store_true",
        help="并行分片模式：单 ε 结果写入 partial JSON，供 aggregate 汇总",
    )
    parser.add_argument(
        "--file-suffix", type=str, default="",
        help="读取 stats/trj 文件名后缀，如 ns2000 → stats_40-0_ns2000.npy",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parent.parent / cfg_path
    cfg = load_yaml(cfg_path)
    scrna_dir = Path(__file__).resolve().parent
    slice_name = args.slice

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        sub = "partial" if args.partial_out else "scan"
        out_dir = scrna_dir / slice_name / "legacy_classify" / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_dir = resolve_path(scrna_dir, cfg["paths"]["sim_dir"])
    aij_dir = resolve_path(scrna_dir, cfg["paths"]["output_dir"])
    label_dir = resolve_path(scrna_dir, cfg["paths"]["input_dir"])
    graph_mode = str(cfg.get("graph", {}).get("sim_mode", "pearson"))
    num_levels = int(cfg["graph"]["num_levels"])
    n_steps = int(cfg["dynamics"]["n_steps"])
    stride = 10
    sample_idx = trajectory_sample_indices(n_steps, stride) if args.feature == "traj100" else None

    if args.eps_list:
        eps_raw = args.eps_list.replace("#", ",")
        eps_values = [round(float(x.strip()), 1) for x in eps_raw.split(",") if x.strip()]
    elif args.epsilon is not None:
        eps_values = [round(float(args.epsilon), 1)]
    else:
        eps_values = [round(float(cfg["epsilon"]["value"]), 1)]

    slice_name = args.slice
    n_cells = load_n_cells(aij_dir, slice_name, graph_mode)
    y, _ = load_labels(label_dir, slice_name, n_cells, aij_dir=aij_dir)

    levels = [args.level] if args.level else list(range(1, num_levels + 1))

    mode = "strict_legacy_set_dedup" if args.strict_legacy else "legacy_resample_no_dedup"
    feat_label = "6维统计" if args.feature == "stats6" else f"轨迹100点(stride={stride})"
    print(f"=== 旧版训练流程: {slice_name} {feat_label} ({mode}) ===")
    print(f"ε 列表: {eps_values}")
    print(f"CV: {args.n_splits}-fold × {args.n_repeats} repeats, RF + StandardScaler")
    if args.strict_legacy:
        print("adjust_train_test: train>5, test>3, choice(5×avgCount) + set() 去重, seed=1\n")
    else:
        print("adjust_train_test: train>5, test>3, 上采样 5×avgCount (有放回, 不去重)\n")

    grid_results: list[dict] = []
    for eps in eps_values:
        tag = eps_tag(eps)
        if args.file_suffix:
            tag = f"{tag}_{args.file_suffix}"
        for lvl in levels:
            data_name = f"stats_{tag}.npy" if args.feature == "stats6" else f"trj_{tag}.npy"
            data_path = partial_level_dir(sim_dir, slice_name, lvl) / data_name
            if not data_path.exists():
                print(f"[SKIP] ε={eps:.1f} L{lvl:02d}: 无 {data_path.name}")
                continue
            X = load_level_features(
                args.feature, sim_dir, slice_name, lvl, tag, sample_idx=sample_idx,
            )
            print(f"ε={eps:.1f} L{lvl:02d}: X{X.shape} ...", flush=True)
            res = compute_kfold_classification(
                X, y, n_splits=args.n_splits, n_repeats=args.n_repeats,
                strict_legacy=args.strict_legacy,
            )
            res["epsilon"] = eps
            res["level"] = lvl
            grid_results.append(res)
            print(
                f"  BA={res['balanced_accuracy_mean']:.4f}±{res['balanced_accuracy_std']:.4f}  "
                f"acc={res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f}  "
                f"P={res['precision_mean']:.4f} R={res['recall_mean']:.4f} "
                f"F1={res['f1_mean']:.4f} AUC={res['auc_mean']:.4f} κ={res['kappa_mean']:.4f}",
                flush=True,
            )

    if not grid_results:
        print("无可用结果")
        sys.exit(1)

    best = max(
        grid_results,
        key=lambda r: (r["balanced_accuracy_mean"], r["accuracy_mean"]),
    )
    summary = {
        "slice": slice_name,
        "feature": args.feature,
        "method": mode,
        "strict_legacy": args.strict_legacy,
        "epsilon_values": eps_values,
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "best_epsilon": best["epsilon"],
        "best_level": best["level"],
        "best_balanced_accuracy": best["balanced_accuracy_mean"],
        "best_accuracy": best["accuracy_mean"],
        "best_precision": best["precision_mean"],
        "best_recall": best["recall_mean"],
        "best_f1": best["f1_mean"],
        "best_auc": best["auc_mean"],
        "best_kappa": best["kappa_mean"],
        "grid": grid_results,
    }
    if args.feature == "traj100":
        summary["trajectory_stride"] = stride
        summary["n_trajectory_points"] = int(sample_idx.size)

    suffix = "_strict_legacy" if args.strict_legacy else ""
    if args.partial_out and len(eps_values) == 1:
        tag = eps_tag(eps_values[0])
        if args.file_suffix:
            tag = f"{tag}_{args.file_suffix}"
        out_stem = (
            f"{slice_name}_legacy_classify_eps{tag}_partial{suffix}"
            if args.feature == "stats6"
            else f"{slice_name}_legacy_classify_traj100_eps{tag}_partial{suffix}"
        )
    else:
        scan_tag = "_eps_scan" if len(eps_values) > 1 or args.eps_list else ""
        out_stem = (
            f"{slice_name}_legacy_classify_stats6{scan_tag}{suffix}"
            if args.feature == "stats6"
            else f"{slice_name}_legacy_classify_traj100{scan_tag}{suffix}"
        )
    out_path = out_dir / f"{out_stem}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # 完整扫描时写出 all_score/ + result.json（partial 单片由 aggregate 汇总后再写）
    if not (args.partial_out and len(eps_values) == 1):
        legacy_dir = scrna_dir / slice_name / "legacy_classify"
        export_meta = {
            "feature": args.feature,
            "method": mode,
            "strict_legacy": args.strict_legacy,
            "n_splits": args.n_splits,
            "n_repeats": args.n_repeats,
        }
        result_json = export_slice_results(
            legacy_dir=legacy_dir,
            slice_name=slice_name,
            grid=grid_results,
            meta=export_meta,
            sim_dir=sim_dir,
            file_suffix=args.file_suffix,
            n_cells=n_cells,
            num_levels=num_levels,
        )
        print(f"all_score/ + result.json → {result_json.parent}")

    print("\n--- 网格结果 (ε × 层) ---")
    print(f"{'ε':>5} {'层':>4} {'BA':>8} {'acc':>8} {'P':>8} {'R':>8} {'F1':>8} {'AUC':>8} {'κ':>8}")
    for r in sorted(grid_results, key=lambda x: (x["epsilon"], x["level"])):
        print(
            f"{r['epsilon']:5.1f} L{r['level']:02d}  "
            f"{r['balanced_accuracy_mean']:8.4f} {r['accuracy_mean']:8.4f} "
            f"{r['precision_mean']:8.4f} {r['recall_mean']:8.4f} "
            f"{r['f1_mean']:8.4f} {r['auc_mean']:8.4f} {r['kappa_mean']:8.4f}"
        )

    print(
        f"\n>>> 最优 ε={best['epsilon']:.1f} L{best['level']:02d}: "
        f"BA={best['balanced_accuracy_mean']:.4f} ({best['balanced_accuracy_mean']*100:.2f}%)  "
        f"acc={best['accuracy_mean']:.4f}  "
        f"P={best['precision_mean']:.4f} R={best['recall_mean']:.4f} "
        f"F1={best['f1_mean']:.4f} AUC={best['auc_mean']:.4f} κ={best['kappa_mean']:.4f}"
    )
    print(f"结果 → {out_path}")


if __name__ == "__main__":
    main()
