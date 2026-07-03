#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
scRNA 细胞类型分类 — 基于 Burgers 轨迹特征。

用法:
    python classify.py --slice GSE45719
    python classify.py --slice GSE45719 --feature-type trj
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_data(slice_name: str, data_dir: Path, sim_dir: Path, feature_type: str):
    """加载特征和标签."""
    # 特征
    feat_dir = sim_dir / slice_name / "features"
    feat_files = sorted(feat_dir.glob("*.npy"))
    if not feat_files:
        raise FileNotFoundError(f"无特征文件: {feat_dir}")
    fpath = feat_files[0]
    features = np.load(fpath)  # (n_levels, n_cells, n_features)
    n_levels, n_cells, n_feats = features.shape

    if feature_type == "trj":
        traj_dir = sim_dir / slice_name / "trajectory"
        traj_files = sorted(traj_dir.glob("*.npy"))
        trj = np.load(traj_files[0])  # (n_levels, n_cells, n_pts)
        features = trj

    # 标签
    label_path = data_dir / slice_name / "label.npy"
    labels = np.load(label_path).astype(int)

    return features, labels, n_levels, n_cells


def get_classifiers(random_state: int = 42) -> dict:
    return {
        "RF": RandomForestClassifier(
            n_estimators=200, max_depth=5, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVC": SVC(kernel="rbf", C=1.0, random_state=random_state),
        "LR": LogisticRegression(
            max_iter=1000, random_state=random_state),
    }


def evaluate(features: np.ndarray, labels: np.ndarray, n_levels: int):
    """逐层 + 全层组合评估."""
    clfs = get_classifiers()

    print(f"\n{'='*65}")
    print(f"  Level          " + "".join(f"{name:>8}" for name in clfs))
    print(f"{'-'*65}")

    for lvl in range(n_levels):
        X = features[lvl]       # (n_cells, n_feats)
        scores = {}
        for name, clf in clfs.items():
            s = cross_val_score(clf, X, labels, cv=5, scoring="accuracy")
            scores[name] = s
        print(f"  L{lvl+1:02d}             " +
              "".join(f"{scores[n].mean():>8.4f}" for n in clfs))

    # 组合所有层
    X_all = features.reshape(features.shape[1], -1)  # (n_cells, n_levels * n_feats)
    print(f"{'-'*65}")
    print(f"  All({n_levels}lvl)      " +
          "".join(f"{cross_val_score(clf, X_all, labels, cv=5, scoring='accuracy').mean():>8.4f}"
                  for clf in clfs.values()))

    # 最佳单层
    best_lvl = 0
    best_score = 0
    for lvl in range(n_levels):
        s = cross_val_score(
            RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
            features[lvl], labels, cv=5, scoring="accuracy").mean()
        if s > best_score:
            best_score = s
            best_lvl = lvl + 1
    print(f"\n  最佳单层: L{best_lvl:02d} (RF, acc={best_score:.4f})")


def main():
    parser = argparse.ArgumentParser(description="scRNA cell type classifier")
    parser.add_argument("--slice", type=str, required=True)
    parser.add_argument("--feature-type", type=str, default="stats",
                        choices=["stats", "trj"])
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    data_dir = base / "data_aij"
    sim_dir = base / "burgers_sim"

    features, labels, n_levels, n_cells = load_data(
        args.slice, data_dir, sim_dir, args.feature_type)

    print(f"{args.slice}: {n_cells} cells, {len(set(labels))} types, "
          f"{n_levels} levels, feature={args.feature_type}")
    print(f"  feature shape: {features.shape}")

    evaluate(features, labels, n_levels)


if __name__ == "__main__":
    main()
