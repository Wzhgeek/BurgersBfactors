#!/usr/bin/env python3
"""
跨蛋白质盲测 (Leave-One-Protein-Out) + 网格搜索。

特征: 100 ε × 10 level × 6 stat = 6000 维/原子
标签: B-factor (来自 .xyzb 文件)
模型: GBDT + RF, 支持 GroupKFold 网格搜索

用法:
    python blind_loo.py
    python blind_loo.py --config config.yaml
    python blind_loo.py --dataset 33small
"""

import argparse
import itertools
import re
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler


# ── 工具函数 ────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def pearson_scorer(y_true, y_pred):
    """GridSearchCV 用的 PCC scorer。"""
    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        return 0.0
    return pearsonr(y_true, y_pred)[0]


def parse_eps(filename: str) -> float:
    m = re.search(r"feature_(\d+)-(\d+)\.csv", filename)
    if not m:
        raise ValueError(f"无法解析 eps: {filename}")
    return float(f"{m.group(1)}.{m.group(2)}")


# ── 数据加载 ────────────────────────────────────────────

def load_labels(xyzb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = []
    valid_mask = []
    with open(xyzb_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    values.append(float(parts[3]))
                    valid_mask.append(True)
                except ValueError:
                    valid_mask.append(False)
    return np.array(values, dtype=np.float64), np.array(valid_mask, dtype=bool)


def load_protein_features(pdb_id: str, dataset: str,
                          exp_res: Path, code_data: Path,
                          eps_agg_methods: list[str] | None = None
                          ) -> tuple[np.ndarray, np.ndarray] | None:
    stats_dir = exp_res / dataset / pdb_id / "features" / "stats"
    xyzb_path = code_data / dataset / f"{pdb_id}_ca.xyzb"

    if not stats_dir.exists():
        print(f"  [SKIP] {pdb_id}: stats 目录不存在")
        return None
    if not xyzb_path.exists():
        print(f"  [SKIP] {pdb_id}: 标签文件不存在")
        return None

    y, valid_mask = load_labels(xyzb_path)
    if len(y) == 0:
        print(f"  [SKIP] {pdb_id}: 无有效标签 (全部 {len(valid_mask)} 行)")
        return None

    n_dropped = len(valid_mask) - len(y)
    extra = f" (丢弃 {n_dropped} 个无效标签)" if n_dropped > 0 else ""

    files = sorted(stats_dir.glob("*.csv"), key=lambda f: parse_eps(f.name))
    parts = []
    for f in files:
        feat = np.loadtxt(f, delimiter=",", skiprows=1, dtype=np.float64)
        if feat.shape[0] == len(y):
            feat_valid = feat
        elif feat.shape[0] == len(valid_mask):
            feat_valid = feat[valid_mask]
        else:
            print(f"  [SKIP] {pdb_id}: {f.name} 行数 {feat.shape[0]} "
                  f"与标签数 {len(y)} 或原始行数 {len(valid_mask)} 都不匹配")
            return None
        parts.append(feat_valid)

    n_eps = len(files)
    n_per_eps = parts[0].shape[1]

    if eps_agg_methods:
        X = _aggregate_across_eps(parts, eps_agg_methods)
        n_features = X.shape[1]
        print(f"  {pdb_id}: {len(y)} atoms × {n_features} features "
              f"(eps_agg={eps_agg_methods}, {n_eps} eps × {n_per_eps} cols){extra}")
    else:
        X = np.hstack(parts)
        n_features = X.shape[1]
        print(f"  {pdb_id}: {len(y)} atoms × {n_features} features "
              f"({n_eps} eps × {n_per_eps} cols){extra}")
    return X, y


def _aggregate_across_eps(parts: list[np.ndarray],
                          agg_methods: list[str]) -> np.ndarray:
    stacked = np.stack(parts, axis=0)
    agg_funcs = {
        "mean": np.mean, "max": np.max, "min": np.min,
        "std": np.std, "median": np.median,
    }
    results = []
    for method in agg_methods:
        results.append(agg_funcs[method](stacked, axis=0))
    return np.hstack(results)


def load_dataset(dataset: str, exp_res: Path, code_data: Path,
                 exclude: list[str] | None = None,
                 eps_agg_methods: list[str] | None = None
                 ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ds_dir = exp_res / dataset
    if not ds_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在: {ds_dir}")

    if exclude is None:
        exclude = []

    proteins = [
        p.name for p in sorted(ds_dir.iterdir())
        if p.is_dir() and (p / "features" / "stats").exists()
        and p.name not in exclude
    ]
    if exclude:
        removed = [p for p in exclude if (ds_dir / p).exists()]
        print(f"\n加载 {dataset} ({len(proteins)} proteins, 排除 {len(removed)}: {removed}):")
    else:
        print(f"\n加载 {dataset} ({len(proteins)} proteins):")

    data = {}
    n_skipped = 0
    for pdb in proteins:
        result = load_protein_features(pdb, dataset, exp_res, code_data, eps_agg_methods)
        if result is not None:
            data[pdb] = result
        else:
            n_skipped += 1

    print(f"  成功: {len(data)}, 跳过: {n_skipped}")
    return data


# ── 模型构建 ────────────────────────────────────────────

def build_gbdt(params: dict, random_state: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params.get("min_samples_split", 5),
        learning_rate=params["learning_rate"],
        subsample=params.get("subsample", 0.8),
        max_features=params.get("max_features", "sqrt"),
        random_state=random_state,
    )


def build_rf(params: dict, random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params.get("min_samples_split", 4),
        min_samples_leaf=params.get("min_samples_leaf", 2),
        random_state=random_state,
        n_jobs=-1,
    )


BUILDERS = {"GBDT": build_gbdt, "RF": build_rf}


def expand_param_grid(param_grid: dict) -> list[dict]:
    """展开网格为参数组合列表。"""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# ── 网格搜索 ────────────────────────────────────────────

def grid_search_best_params(
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    model_name: str,
    param_grid: dict,
    model_cfg: dict,
    grid_cfg: dict,
    use_scale: bool,
) -> dict:
    """
    用 GroupKFold（蛋白为组）在所有数据上搜索最优参数。
    返回最优参数字典。
    """
    pdb_ids = sorted(data.keys())
    X_all = np.vstack([data[p][0] for p in pdb_ids])
    y_all = np.concatenate([data[p][1] for p in pdb_ids])
    groups = np.concatenate([
        np.full(data[p][0].shape[0], i) for i, p in enumerate(pdb_ids)
    ])

    if use_scale:
        X_all = StandardScaler().fit_transform(X_all)

    cv = GroupKFold(n_splits=grid_cfg["cv_folds"])
    random_state = grid_cfg.get("random_state", 42)

    param_list = expand_param_grid(param_grid)
    print(f"\n[{model_name}] 网格搜索: {len(param_list)} 组参数 × {grid_cfg['cv_folds']} 折")

    builder = BUILDERS[model_name]
    best_pcc = -999
    best_params = None

    for params in param_list:
        fold_pccs = []
        for train_idx, val_idx in cv.split(X_all, y_all, groups):
            model = builder({**model_cfg, **params}, random_state)
            model.fit(X_all[train_idx], y_all[train_idx])
            y_pred = model.predict(X_all[val_idx])
            pcc, _ = pearsonr(y_pred, y_all[val_idx])
            fold_pccs.append(pcc)

        mean_pcc = np.mean(fold_pccs)
        print(f"  {params} → PCC={mean_pcc:.4f} (±{np.std(fold_pccs):.4f})")

        if mean_pcc > best_pcc:
            best_pcc = mean_pcc
            best_params = dict(params)

    print(f"  最优: {best_params} (PCC={best_pcc:.4f})")
    return best_params


# ── LOO 评估 ────────────────────────────────────────────

def loo_evaluate(data: dict, dataset: str, model_name: str,
                 model_params: dict, output_dir: Path,
                 random_state: int, use_scale: bool,
                 pca_cfg: dict | None = None) -> dict:
    """用固定参数跑 LOO 盲测。"""
    pdb_ids = sorted(data.keys())
    n_proteins = len(pdb_ids)

    use_pca = pca_cfg and pca_cfg.get("enabled", False)
    pca_n = pca_cfg.get("n_components", 0.95) if use_pca else None

    results = []
    all_y_true = []
    all_y_pred = []

    tag = f" + PCA({pca_n})" if use_pca else ""
    print(f"\n[{model_name}{tag}] LOO 评估 ({n_proteins} proteins)")

    for i, test_pdb in enumerate(pdb_ids):
        t0 = time.time()

        X_test, y_test = data[test_pdb]
        train_parts_X = [data[p][0] for p in pdb_ids if p != test_pdb]
        train_parts_y = [data[p][1] for p in pdb_ids if p != test_pdb]
        X_train = np.vstack(train_parts_X)
        y_train = np.concatenate(train_parts_y)

        if use_scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if use_pca:
            pca = PCA(n_components=pca_n, random_state=random_state)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            if i == 0:
                print(f"  [PCA] {X_train.shape[1]} components (from {pca.n_features_in_})")

        builder = BUILDERS[model_name]
        model = builder(model_params, random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        pcc, _ = pearsonr(y_pred, y_test)
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

        elapsed = time.time() - t0
        print(f"  [{i+1}/{n_proteins}] {test_pdb}: PCC={pcc:.4f} RMSE={rmse:.4f} "
              f"({elapsed:.1f}s)")

        results.append({
            "pdb_id": test_pdb,
            "n_atoms": len(y_test),
            "pcc": pcc,
            "rmse": rmse,
        })
        all_y_true.append(y_test)
        all_y_pred.append(y_pred)

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    overall_pcc, _ = pearsonr(all_y_pred, all_y_true)

    pccs = [r["pcc"] for r in results]
    mean_pcc = np.mean(pccs)
    std_pcc = np.std(pccs)

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "per_protein.csv", "w") as f:
        f.write("pdb_id,n_atoms,pcc,rmse\n")
        for r in results:
            f.write(f"{r['pdb_id']},{r['n_atoms']},{r['pcc']:.6f},{r['rmse']:.6f}\n")

    with open(output_dir / "summary.csv", "w") as f:
        f.write("dataset,n_proteins,model,mean_pcc,std_pcc,min_pcc,max_pcc,overall_pcc\n")
        f.write(f"{dataset},{n_proteins},{model_name},{mean_pcc:.6f},{std_pcc:.6f},"
                f"{min(pccs):.6f},{max(pccs):.6f},{overall_pcc:.6f}\n")

    # 保存最优参数
    with open(output_dir / "best_params.txt", "w") as f:
        for k, v in model_params.items():
            f.write(f"{k}: {v}\n")

    print(f"\n[{model_name}] {dataset} 汇总: mean_PCC={mean_pcc:.4f} ± {std_pcc:.4f} "
          f"[{min(pccs):.4f}, {max(pccs):.4f}]  overall={overall_pcc:.4f}")

    return {
        "dataset": dataset, "model": model_name,
        "n_proteins": n_proteins, "mean_pcc": mean_pcc, "std_pcc": std_pcc,
        "min_pcc": min(pccs), "max_pcc": max(pccs), "overall_pcc": overall_pcc,
    }


# ── 主流程 ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="跨蛋白质盲测 LOO + 网格搜索")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--dataset", type=str,
                        help="覆盖 config 中的 datasets")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    paths = cfg["paths"]
    exp_res = Path(paths["exp_res"])
    code_data = Path(paths["code_data"])
    output_root = Path(__file__).resolve().parent / "results"

    if args.dataset:
        datasets = [args.dataset] if args.dataset != "all" else cfg["datasets"]
    else:
        datasets = cfg["datasets"]

    feat_cfg = cfg.get("features", {})
    eps_agg_methods = feat_cfg.get("eps_aggregation", None)
    if isinstance(eps_agg_methods, list) and len(eps_agg_methods) == 0:
        eps_agg_methods = None

    pre_cfg = cfg.get("preprocessing", {})
    use_scale = pre_cfg.get("scale", True)
    pca_cfg = pre_cfg.get("pca", None)

    eval_cfg = cfg["evaluation"]
    random_state = eval_cfg.get("random_state", 42)

    grid_cfg = cfg.get("grid_search", {})
    do_grid = grid_cfg.get("enabled", False)

    all_summaries = []
    exclude_cfg = cfg.get("exclude", {})

    for ds in datasets:
        exclude_list = exclude_cfg.get(ds, [])
        data = load_dataset(ds, exp_res, code_data, exclude_list, eps_agg_methods)
        if len(data) < 2:
            print(f"[SKIP] {ds}: 蛋白数不足 ({len(data)})")
            continue

        # 确定输出路径后缀
        use_pca = pca_cfg and pca_cfg.get("enabled", False)
        use_agg = bool(eps_agg_methods)
        if use_agg:
            suffix = f"Agg_{''.join(eps_agg_methods)}"
        elif use_pca:
            suffix = f"PCA{pca_cfg['n_components']}"
        elif do_grid:
            suffix = "GridSearch"
        else:
            suffix = "Full"

        for model_name, model_cfg in cfg["models"].items():
            if not model_cfg.get("enabled", True):
                continue

            # 网格搜索
            if do_grid and model_cfg.get("param_grid"):
                best_params = grid_search_best_params(
                    data, model_name,
                    model_cfg["param_grid"], model_cfg,
                    grid_cfg, use_scale,
                )
            else:
                best_params = model_cfg

            summary = loo_evaluate(data, ds, model_name, best_params,
                                   output_root / ds / suffix / model_name,
                                   random_state, use_scale, pca_cfg)
            all_summaries.append(summary)

    # 模型对比
    if len(all_summaries) > 1:
        print(f"\n{'='*60}")
        print(f"模型对比 (mean_PCC)")
        print(f"{'='*60}")
        print(f"{'Dataset':<12} {'GBDT':>8} {'RF':>8}  {'Δ(RF-GBDT)':>12}")
        print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*12}")
        by_ds = {}
        for s in all_summaries:
            by_ds.setdefault(s["dataset"], {})[s["model"]] = s
        for ds_name, models in by_ds.items():
            gbdt = models.get("GBDT", {}).get("mean_pcc", 0)
            rf = models.get("RF", {}).get("mean_pcc", 0)
            delta = rf - gbdt
            print(f"{ds_name:<12} {gbdt:>8.4f} {rf:>8.4f} {delta:>+12.4f}")

    print(f"\n结果保存至: {output_root}")


if __name__ == "__main__":
    main()
