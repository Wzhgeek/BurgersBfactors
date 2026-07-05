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
    python classify_legacy_resample.py --slice GSE84133human1 --feature stats6_traj100 --strict-legacy
    python classify_legacy_resample.py --slice GSE59114 --feature stats6_topo6 --strict-legacy
    python classify_legacy_resample.py --slice GSE59114 --feature stats6_l10 --strict-legacy
    python classify_legacy_resample.py --slice GSE59114 --feature stats6_topo6_l10 --strict-legacy
    python classify_legacy_resample.py --slice GSE59114 --feature stats6_l10_pca30 --strict-legacy
    python classify_legacy_resample.py --slice GSE59114 --feature stats6_topo6_l10_pca30 --strict-legacy
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

from scRNA.classify_lstm import compute_lstm, compute_traj_lstm, resolve_lstm_device
from scRNA.classify_traj_stats import (
    all_levels_stats_ready,
    all_levels_stats_topo_ready,
    all_levels_trj_ready,
    eps_tag,
    load_alllevels_stats6,
    load_alllevels_stats6_pca30,
    load_alllevels_stats6_topo6,
    load_alllevels_stats6_topo6_pca30,
    load_alllevels_traj100,
    load_alllevels_traj_full,
    load_level_traj_full,
    load_labels,
    load_level_stats,
    load_pca_coords,
    load_level_topo,
    load_n_cells,
    load_yaml,
    partial_level_dir,
    resolve_path,
    sanitize_features,
    topo_features_root,
)
from scRNA.classify_results_io import export_slice_results, legacy_classify_dir


def _traj100_helpers():
    from scRNA.analyze_traj_100pts import (
        load_level_traj_features,
        trajectory_sample_indices,
    )
    return load_level_traj_features, trajectory_sample_indices


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


def _sanitize_features(X: np.ndarray) -> np.ndarray:
    """Burgers 高 ε 可能产生 nan/极大值，分类前清理以保证 StandardScaler/RF 可运行。"""
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -1e6, 1e6)


def compute_rf(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """训练 RF，返回 (y_pred, y_proba, clf.classes_)。"""
    X_train = _sanitize_features(X_train)
    X_test = _sanitize_features(X_test)
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
    classifier: str = "rf",
    lstm_params: dict | None = None,
) -> dict:
    """5-fold × n_repeats，折内 adjust_train_test + RF/LSTM，返回 BA/acc/P/R/F1/AUC/Kappa 统计。"""
    lstm_params = lstm_params or {}
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

            if classifier == "lstm":
                lstm_mode = str(lstm_params.get("mode", "flat150"))
                common = dict(
                    hidden_size=int(lstm_params.get("hidden_size", 64)),
                    num_layers=int(lstm_params.get("num_layers", 2)),
                    epochs=int(lstm_params.get("epochs", 100)),
                    lr=float(lstm_params.get("lr", 0.001)),
                    batch_size=int(lstm_params.get("batch_size", 64)),
                    device=str(lstm_params.get("device", "cpu")),
                    seed=int(lstm_params.get("seed", 1)),
                )
                if lstm_mode == "traj":
                    y_pred, y_proba, classes = compute_traj_lstm(
                        _sanitize_features(X_tr),
                        _sanitize_features(X_te),
                        y_tr, y_te,
                        **common,
                    )
                else:
                    y_pred, y_proba, classes = compute_lstm(
                        _sanitize_features(X_tr),
                        _sanitize_features(X_te),
                        y_tr, y_te,
                        **common,
                    )
            else:
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


def feature_partial_stem(
    feature: str, slice_name: str, tag: str, suffix: str,
    classifier: str = "rf", level: int | None = None,
) -> str:
    """partial JSON 文件名（无 .json）。"""
    prefix = "lstm_classify" if classifier == "lstm" else "legacy_classify"
    if feature == "stats6":
        return f"{slice_name}_{prefix}_eps{tag}_partial{suffix}"
    if feature == "traj100":
        return f"{slice_name}_{prefix}_traj100_eps{tag}_partial{suffix}"
    if feature == "stats6_traj100":
        return f"{slice_name}_{prefix}_stats6_traj100_eps{tag}_partial{suffix}"
    if feature == "stats6_topo6":
        return f"{slice_name}_{prefix}_stats6_topo6_eps{tag}_partial{suffix}"
    if feature == "stats6_l10":
        return f"{slice_name}_{prefix}_stats6_l10_eps{tag}_partial{suffix}"
    if feature == "stats6_topo6_l10":
        return f"{slice_name}_{prefix}_stats6_topo6_l10_eps{tag}_partial{suffix}"
    if feature == "stats6_l10_pca30":
        return f"{slice_name}_{prefix}_stats6_l10_pca30_eps{tag}_partial{suffix}"
    if feature == "stats6_topo6_l10_pca30":
        return f"{slice_name}_{prefix}_stats6_topo6_l10_pca30_eps{tag}_partial{suffix}"
    if feature == "traj100_l10":
        return f"{slice_name}_{prefix}_traj100_l10_eps{tag}_partial{suffix}"
    if feature == "trajfull_l10":
        return f"{slice_name}_{prefix}_trajfull_l10_eps{tag}_partial{suffix}"
    if feature == "trajfull":
        if level is None:
            raise ValueError("trajfull partial 须指定 level")
        return f"{slice_name}_{prefix}_trajfull_L{level:02d}_eps{tag}_partial{suffix}"
    raise ValueError(f"未知特征: {feature}")


def load_level_features(
    feature: str,
    sim_dir: Path,
    slice_name: str,
    level: int,
    tag: str,
    sample_idx: np.ndarray | None = None,
    topo_root: Path | None = None,
) -> np.ndarray:
    if feature == "stats6":
        return load_level_stats(sim_dir, slice_name, level, tag, recompute=False)
    if feature == "traj100":
        if sample_idx is None:
            raise ValueError("traj100 需要 sample_idx")
        load_level_traj_features, _ = _traj100_helpers()
        X = load_level_traj_features(sim_dir, slice_name, level, tag, sample_idx)
        return sanitize_features(X, f"{slice_name} L{level:02d}")
    if feature == "stats6_traj100":
        if sample_idx is None:
            raise ValueError("stats6_traj100 需要 sample_idx")
        stats = load_level_stats(sim_dir, slice_name, level, tag, recompute=False)
        load_level_traj_features, _ = _traj100_helpers()
        traj = load_level_traj_features(sim_dir, slice_name, level, tag, sample_idx)
        traj = sanitize_features(traj, f"{slice_name} L{level:02d} traj")
        return np.hstack([stats, traj])
    if feature == "stats6_topo6":
        if topo_root is None:
            raise ValueError("stats6_topo6 需要 topo_root")
        stats = load_level_stats(sim_dir, slice_name, level, tag, recompute=False)
        topo = load_level_topo(topo_root, slice_name, level)
        if stats.shape[0] != topo.shape[0]:
            raise ValueError(
                f"{slice_name} L{level:02d}: stats {stats.shape[0]} != topo {topo.shape[0]}"
            )
        return np.hstack([stats, topo])
    raise ValueError(f"未知特征: {feature}")


def main() -> None:
    parser = argparse.ArgumentParser(description="旧版折内上采样 + RF 分类")
    parser.add_argument("--config", default="scRNA/config_graph.yaml")
    parser.add_argument("--slice", default="GSE84133human1")
    parser.add_argument("--level", type=int, default=None, help="指定层；默认跑 L01-L10")
    parser.add_argument(
        "--feature",
        choices=(
            "stats6", "traj100", "stats6_traj100", "stats6_topo6",
            "stats6_l10", "stats6_topo6_l10", "stats6_l10_pca30",
            "stats6_topo6_l10_pca30", "traj100_l10",
            "trajfull_l10", "trajfull",
        ),
        default="stats6",
        help="stats6=6维单层; stats6_topo6_l10=120维; stats6_l10_pca30=60+PCA; stats6_topo6_l10_pca30=120+PCA",
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
        "--classifier",
        choices=("rf", "lstm"),
        default="rf",
        help="分类器：rf=RandomForest; lstm=L01-L10序列LSTM+PCA30",
    )
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--lstm-epochs", type=int, default=100)
    parser.add_argument("--lstm-lr", type=float, default=0.001)
    parser.add_argument("--lstm-batch-size", type=int, default=64)
    parser.add_argument(
        "--lstm-device", default="auto",
        help="LSTM 设备: auto|cuda|cpu（默认 auto=有 GPU 则用 cuda）",
    )
    parser.add_argument(
        "--partial-out", action="store_true",
        help="并行分片模式：单 ε 结果写入 partial JSON，供 aggregate 汇总",
    )
    parser.add_argument(
        "--file-suffix", type=str, default="",
        help="读取 stats/trj 文件名后缀，如 ns2000 → stats_40-0_ns2000.npy",
    )
    parser.add_argument(
        "--experiment-tag", type=str, default=None,
        help="结果写入 classify_dir/{slice}/{tag}/，避免覆盖主扫描",
    )
    parser.add_argument(
        "--topo-root", type=str, default=None,
        help="拓扑特征根目录（默认 paths.topo_features_dir/{slice}/）",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parent.parent / cfg_path
    cfg = load_yaml(cfg_path)
    scrna_dir = Path(__file__).resolve().parent
    slice_name = args.slice

    slice_result_dir = legacy_classify_dir(cfg, scrna_dir, slice_name)
    if args.experiment_tag:
        slice_result_dir = slice_result_dir / args.experiment_tag
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        sub = "partial" if args.partial_out else "scan"
        out_dir = slice_result_dir / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_dir = resolve_path(scrna_dir, cfg["paths"]["sim_dir"])
    if args.feature in ("stats6_topo6", "stats6_topo6_l10", "stats6_topo6_l10_pca30"):
        topo_root = (
            Path(args.topo_root)
            if args.topo_root
            else topo_features_root(cfg, scrna_dir)
        )
    else:
        topo_root = None
    aij_dir = resolve_path(scrna_dir, cfg["paths"]["output_dir"])
    label_dir = resolve_path(scrna_dir, cfg["paths"]["input_dir"])
    graph_mode = str(cfg.get("graph", {}).get("sim_mode", "pearson"))
    num_levels = int(cfg["graph"]["num_levels"])
    n_steps = int(cfg["dynamics"]["n_steps"])
    stride = 10
    if args.feature in ("traj100", "stats6_traj100", "traj100_l10"):
        _, trajectory_sample_indices = _traj100_helpers()
        sample_idx = trajectory_sample_indices(n_steps, stride)
    else:
        sample_idx = None

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

    lstm_features = (
        "stats6_topo6_l10_pca30", "traj100_l10", "trajfull_l10", "trajfull",
    )
    if args.classifier == "lstm" and args.feature not in lstm_features:
        parser.error(f"LSTM 分类器当前仅支持 --feature {' / '.join(lstm_features)}")
    if args.feature == "trajfull" and args.level is None:
        parser.error("trajfull 须 --level 指定单层（L01-L10）")
    if args.feature == "trajfull_l10" and args.level is not None:
        parser.error("trajfull_l10 为 10 层混合特征，勿指定 --level")

    mode = "strict_legacy_set_dedup" if args.strict_legacy else "legacy_resample_no_dedup"
    if args.classifier == "lstm":
        mode = f"lstm_{mode}"
    if args.feature == "trajfull_l10":
        clf_label = "LSTM(L01-L10 trajfull)"
    elif args.feature == "trajfull":
        clf_label = f"LSTM(L{args.level:02d} trajfull)"
    elif args.feature == "traj100_l10":
        clf_label = "LSTM(L01-L10 traj100)"
    elif args.classifier == "lstm":
        clf_label = "LSTM(L01-L10 seq + PCA30)"
    else:
        clf_label = "RF + StandardScaler"
    lstm_params = {
        "hidden_size": args.lstm_hidden,
        "num_layers": args.lstm_layers,
        "epochs": args.lstm_epochs,
        "lr": args.lstm_lr,
        "batch_size": args.lstm_batch_size,
        "device": resolve_lstm_device(args.lstm_device),
        "seed": 1,
        "mode": (
            "traj" if args.feature in ("traj100_l10", "trajfull_l10", "trajfull")
            else "flat150"
        ),
    }
    cv_kwargs = dict(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        strict_legacy=args.strict_legacy,
        classifier=args.classifier,
        lstm_params=lstm_params,
    )
    feat_labels = {
        "stats6": "6维统计",
        "traj100": f"轨迹100点(stride={stride})",
        "stats6_traj100": f"stats6+traj100拼接106维(stride={stride})",
        "stats6_topo6": "stats6+topo6拼接12维(同层动力学+拓扑)",
        "stats6_l10": "L01-L10各层stats6拼接60维(无拓扑)",
        "stats6_topo6_l10": "L01-L10各层stats6+topo6拼接120维",
        "stats6_l10_pca30": "L01-L10 stats6(60维)+PCA坐标拼接",
        "stats6_topo6_l10_pca30": "L01-L10 stats6+topo6(120维)+PCA坐标拼接",
        "traj100_l10": "L01-L10 Burgers轨迹100点(10×100)",
        "trajfull_l10": "L01-L10 Burgers完整保存轨迹(10×T)",
    }
    if args.feature == "trajfull":
        feat_label = f"L{args.level:02d} Burgers完整保存轨迹(1×T)"
    else:
        feat_label = feat_labels[args.feature]
    print(f"=== 旧版训练流程: {slice_name} {feat_label} ({mode}) ===")
    print(f"ε 列表: {eps_values}")
    print(f"CV: {args.n_splits}-fold × {args.n_repeats} repeats, {clf_label}")
    if args.classifier == "lstm":
        print(f"LSTM device: {lstm_params['device']}, batch={lstm_params['batch_size']}, "
              f"hidden={lstm_params['hidden_size']}, layers={lstm_params['num_layers']}, "
              f"epochs={lstm_params['epochs']}, lr={lstm_params['lr']}")
    if args.strict_legacy:
        print("adjust_train_test: train>5, test>3, choice(5×avgCount) + set() 去重, seed=1\n")
    else:
        print("adjust_train_test: train>5, test>3, 上采样 5×avgCount (有放回, 不去重)\n")

    grid_results: list[dict] = []
    for eps in eps_values:
        tag = eps_tag(eps)
        if args.file_suffix:
            tag = f"{tag}_{args.file_suffix}"
        if args.feature == "stats6_topo6_l10":
            if not all_levels_stats_topo_ready(
                sim_dir, topo_root, slice_name, tag, num_levels,
            ):
                print(f"[SKIP] ε={eps:.1f}: 缺少 L01-L10 stats 或 topo")
                continue
            X = load_alllevels_stats6_topo6(
                sim_dir, topo_root, slice_name, tag, num_levels=num_levels,
            )
            print(f"ε={eps:.1f} L01-L10: X{X.shape} ...", flush=True)
            res = compute_kfold_classification(
                X, y, **cv_kwargs,
            )
            res["epsilon"] = eps
            res["level"] = 0
            grid_results.append(res)
            print(
                f"  BA={res['balanced_accuracy_mean']:.4f}±{res['balanced_accuracy_std']:.4f}  "
                f"acc={res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f}  "
                f"P={res['precision_mean']:.4f} R={res['recall_mean']:.4f} "
                f"F1={res['f1_mean']:.4f} AUC={res['auc_mean']:.4f} κ={res['kappa_mean']:.4f}",
                flush=True,
            )
            continue
        if args.feature == "stats6_l10":
            if not all_levels_stats_ready(sim_dir, slice_name, tag, num_levels):
                print(f"[SKIP] ε={eps:.1f}: 缺少 L01-L10 stats")
                continue
            X = load_alllevels_stats6(
                sim_dir, slice_name, tag, num_levels=num_levels,
            )
            print(f"ε={eps:.1f} L01-L10 stats6: X{X.shape} ...", flush=True)
            res = compute_kfold_classification(
                X, y, **cv_kwargs,
            )
            res["epsilon"] = eps
            res["level"] = 0
            grid_results.append(res)
            print(
                f"  BA={res['balanced_accuracy_mean']:.4f}±{res['balanced_accuracy_std']:.4f}  "
                f"acc={res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f}  "
                f"P={res['precision_mean']:.4f} R={res['recall_mean']:.4f} "
                f"F1={res['f1_mean']:.4f} AUC={res['auc_mean']:.4f} κ={res['kappa_mean']:.4f}",
                flush=True,
            )
            continue
        if args.feature == "traj100_l10":
            if sample_idx is None:
                raise RuntimeError("traj100_l10 需要 sample_idx")
            if not all_levels_trj_ready(sim_dir, slice_name, tag, num_levels):
                print(f"[SKIP] ε={eps:.1f}: 缺少 L01-L10 trj")
                continue
            X = load_alllevels_traj100(
                sim_dir, slice_name, tag, sample_idx, num_levels=num_levels,
            )
            print(f"ε={eps:.1f} L01-L10 traj100: X{X.shape} ...", flush=True)
            res = compute_kfold_classification(X, y, **cv_kwargs)
            res["epsilon"] = eps
            res["level"] = 0
            grid_results.append(res)
            print(
                f"  BA={res['balanced_accuracy_mean']:.4f}±{res['balanced_accuracy_std']:.4f}  "
                f"acc={res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f}  "
                f"P={res['precision_mean']:.4f} R={res['recall_mean']:.4f} "
                f"F1={res['f1_mean']:.4f} AUC={res['auc_mean']:.4f} κ={res['kappa_mean']:.4f}",
                flush=True,
            )
            continue
        if args.feature == "trajfull_l10":
            if not all_levels_trj_ready(sim_dir, slice_name, tag, num_levels):
                print(f"[SKIP] ε={eps:.1f}: 缺少 L01-L10 trj")
                continue
            X = load_alllevels_traj_full(
                sim_dir, slice_name, tag, num_levels=num_levels,
            )
            print(f"ε={eps:.1f} L01-L10 trajfull: X{X.shape} ...", flush=True)
            res = compute_kfold_classification(X, y, **cv_kwargs)
            res["epsilon"] = eps
            res["level"] = 0
            grid_results.append(res)
            print(
                f"  BA={res['balanced_accuracy_mean']:.4f}±{res['balanced_accuracy_std']:.4f}  "
                f"acc={res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f}  "
                f"P={res['precision_mean']:.4f} R={res['recall_mean']:.4f} "
                f"F1={res['f1_mean']:.4f} AUC={res['auc_mean']:.4f} κ={res['kappa_mean']:.4f}",
                flush=True,
            )
            continue
        if args.feature == "trajfull":
            trj_path = partial_level_dir(sim_dir, slice_name, args.level) / f"trj_{tag}.npy"
            if not trj_path.exists():
                print(f"[SKIP] ε={eps:.1f} L{args.level:02d}: 无 {trj_path.name}")
                continue
            X = load_level_traj_full(
                sim_dir, slice_name, args.level, tag,
            )[:, np.newaxis, :]
            print(f"ε={eps:.1f} L{args.level:02d} trajfull: X{X.shape} ...", flush=True)
            res = compute_kfold_classification(X, y, **cv_kwargs)
            res["epsilon"] = eps
            res["level"] = args.level
            grid_results.append(res)
            print(
                f"  BA={res['balanced_accuracy_mean']:.4f}±{res['balanced_accuracy_std']:.4f}  "
                f"acc={res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f}  "
                f"P={res['precision_mean']:.4f} R={res['recall_mean']:.4f} "
                f"F1={res['f1_mean']:.4f} AUC={res['auc_mean']:.4f} κ={res['kappa_mean']:.4f}",
                flush=True,
            )
            continue
        if args.feature == "stats6_topo6_l10_pca30":
            if not all_levels_stats_topo_ready(
                sim_dir, topo_root, slice_name, tag, num_levels,
            ):
                print(f"[SKIP] ε={eps:.1f}: 缺少 L01-L10 stats 或 topo")
                continue
            X = load_alllevels_stats6_topo6_pca30(
                sim_dir, topo_root, aij_dir, slice_name, tag, num_levels=num_levels,
            )
            print(f"ε={eps:.1f} L01-L10 stats6+topo6+PCA: X{X.shape} ...", flush=True)
            res = compute_kfold_classification(
                X, y, **cv_kwargs,
            )
            res["epsilon"] = eps
            res["level"] = 0
            grid_results.append(res)
            print(
                f"  BA={res['balanced_accuracy_mean']:.4f}±{res['balanced_accuracy_std']:.4f}  "
                f"acc={res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f}  "
                f"P={res['precision_mean']:.4f} R={res['recall_mean']:.4f} "
                f"F1={res['f1_mean']:.4f} AUC={res['auc_mean']:.4f} κ={res['kappa_mean']:.4f}",
                flush=True,
            )
            continue
        if args.feature == "stats6_l10_pca30":
            if not all_levels_stats_ready(sim_dir, slice_name, tag, num_levels):
                print(f"[SKIP] ε={eps:.1f}: 缺少 L01-L10 stats")
                continue
            X = load_alllevels_stats6_pca30(
                sim_dir, aij_dir, slice_name, tag, num_levels=num_levels,
            )
            print(f"ε={eps:.1f} L01-L10 stats6+PCA: X{X.shape} ...", flush=True)
            res = compute_kfold_classification(
                X, y, **cv_kwargs,
            )
            res["epsilon"] = eps
            res["level"] = 0
            grid_results.append(res)
            print(
                f"  BA={res['balanced_accuracy_mean']:.4f}±{res['balanced_accuracy_std']:.4f}  "
                f"acc={res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f}  "
                f"P={res['precision_mean']:.4f} R={res['recall_mean']:.4f} "
                f"F1={res['f1_mean']:.4f} AUC={res['auc_mean']:.4f} κ={res['kappa_mean']:.4f}",
                flush=True,
            )
            continue
        for lvl in levels:
            if args.feature == "stats6_traj100":
                stats_path = partial_level_dir(sim_dir, slice_name, lvl) / f"stats_{tag}.npy"
                trj_path = partial_level_dir(sim_dir, slice_name, lvl) / f"trj_{tag}.npy"
                if not stats_path.exists() or not trj_path.exists():
                    missing = trj_path.name if not trj_path.exists() else stats_path.name
                    print(f"[SKIP] ε={eps:.1f} L{lvl:02d}: 无 {missing}")
                    continue
            elif args.feature == "stats6_topo6":
                stats_path = partial_level_dir(sim_dir, slice_name, lvl) / f"stats_{tag}.npy"
                topo_path = topo_root / slice_name / f"{slice_name}_topo_L{lvl:02d}.npy"
                if not stats_path.exists() or not topo_path.exists():
                    missing = topo_path.name if not topo_path.exists() else stats_path.name
                    print(f"[SKIP] ε={eps:.1f} L{lvl:02d}: 无 {missing}")
                    continue
            else:
                data_name = f"stats_{tag}.npy" if args.feature == "stats6" else f"trj_{tag}.npy"
                data_path = partial_level_dir(sim_dir, slice_name, lvl) / data_name
                if not data_path.exists():
                    print(f"[SKIP] ε={eps:.1f} L{lvl:02d}: 无 {data_path.name}")
                    continue
            X = load_level_features(
                args.feature, sim_dir, slice_name, lvl, tag,
                sample_idx=sample_idx, topo_root=topo_root,
            )
            print(f"ε={eps:.1f} L{lvl:02d}: X{X.shape} ...", flush=True)
            res = compute_kfold_classification(
                X, y, **cv_kwargs,
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
        "classifier": args.classifier,
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
    if args.feature in ("traj100", "stats6_traj100"):
        summary["trajectory_stride"] = stride
        summary["n_trajectory_points"] = int(sample_idx.size)
    if args.feature == "stats6_traj100":
        summary["n_feature_dims"] = 106
    if args.feature == "stats6_topo6":
        summary["n_feature_dims"] = 12
        summary["topo_features_dir"] = str(topo_root)
    if args.feature == "stats6_l10":
        summary["n_feature_dims"] = 6 * num_levels
        summary["level_mode"] = "L01-L10_stats6_concat"
    if args.feature == "stats6_topo6_l10":
        summary["n_feature_dims"] = 12 * num_levels
        summary["level_mode"] = "L01-L10_concat"
        summary["topo_features_dir"] = str(topo_root)
    if args.feature == "stats6_l10_pca30":
        pca_dim = int(load_pca_coords(aij_dir, slice_name).shape[1])
        summary["n_feature_dims"] = 6 * num_levels + pca_dim
        summary["level_mode"] = "L01-L10_stats6_plus_pca"
        summary["pca_dims"] = pca_dim
    if args.feature == "stats6_topo6_l10_pca30":
        pca_dim = int(load_pca_coords(aij_dir, slice_name).shape[1])
        summary["n_feature_dims"] = 12 * num_levels + pca_dim
        summary["level_mode"] = "L01-L10_stats6_topo6_plus_pca"
        summary["pca_dims"] = pca_dim
        summary["topo_features_dir"] = str(topo_root)
    if args.feature == "traj100_l10":
        summary["n_feature_dims"] = num_levels * int(sample_idx.size)
        summary["level_mode"] = "L01-L10_traj100"
        summary["trajectory_stride"] = stride
        summary["n_trajectory_points"] = int(sample_idx.size)
    if args.feature == "trajfull_l10":
        summary["level_mode"] = "L01-L10_trajfull"
        summary["trajectory_mode"] = "saved_full_uniform"
    if args.feature == "trajfull":
        summary["level_mode"] = f"L{args.level:02d}_trajfull"
        summary["trajectory_mode"] = "saved_full_uniform"

    suffix = "_strict_legacy" if args.strict_legacy else ""
    if args.partial_out and len(eps_values) == 1:
        tag = eps_tag(eps_values[0])
        if args.file_suffix:
            tag = f"{tag}_{args.file_suffix}"
        stem_level = args.level if args.feature == "trajfull" else None
        out_stem = feature_partial_stem(
            args.feature, slice_name, tag, suffix,
            classifier=args.classifier, level=stem_level,
        )
    else:
        scan_tag = "_eps_scan" if len(eps_values) > 1 or args.eps_list else ""
        exp_tag = f"_{args.experiment_tag}" if args.experiment_tag else ""
        feat_scan = {
            "stats6": "stats6",
            "traj100": "traj100",
            "stats6_traj100": "stats6_traj100",
            "stats6_topo6": "stats6_topo6",
            "stats6_l10": "stats6_l10",
            "stats6_topo6_l10": "stats6_topo6_l10",
            "stats6_l10_pca30": "stats6_l10_pca30",
            "stats6_topo6_l10_pca30": "stats6_topo6_l10_pca30",
            "traj100_l10": "traj100_l10",
            "trajfull_l10": "trajfull_l10",
            "trajfull": "trajfull",
        }[args.feature]
        cls_prefix = "lstm_classify" if args.classifier == "lstm" else "legacy_classify"
        out_stem = f"{slice_name}_{cls_prefix}_{feat_scan}{scan_tag}{exp_tag}{suffix}"
    out_path = out_dir / f"{out_stem}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # 完整扫描时写出 all_score/ + result.json（partial 单片由 aggregate 汇总后再写）
    if not (args.partial_out and len(eps_values) == 1):
        legacy_dir = slice_result_dir
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
