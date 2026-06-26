#!/usr/bin/env python3
"""
跨蛋白质盲测 (Leave-One-Protein-Out)。

特征: 100 ε × 10 level × 6 stat = 6000 维/原子
标签: B-factor (来自 .xyzb 文件)
模型: GBDT + RF

用法:
    python blind_loo.py
    python blind_loo.py --config config.yaml
    python blind_loo.py --dataset 33small
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_labels(xyzb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """从 .xyzb 文件加载 B-factor 标签。"""
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


def parse_eps(filename: str) -> float:
    m = re.search(r"feature_(\d+)-(\d+)\.csv", filename)
    if not m:
        raise ValueError(f"无法解析 eps: {filename}")
    return float(f"{m.group(1)}.{m.group(2)}")


def load_protein_features(pdb_id: str, dataset: str,
                          exp_res: Path, code_data: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """加载一个蛋白的全部 stats 特征和标签。"""
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

    X = np.hstack(parts)
    n_eps = len(files)
    n_features = X.shape[1]
    print(f"  {pdb_id}: {len(y)} atoms × {n_features} features "
          f"({n_eps} eps × {n_features // n_eps} cols){extra}")
    return X, y


def load_dataset(dataset: str, exp_res: Path, code_data: Path,
                 exclude: list[str] | None = None
                 ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """加载整个数据集。"""
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
        result = load_protein_features(pdb, dataset, exp_res, code_data)
        if result is not None:
            data[pdb] = result
        else:
            n_skipped += 1

    print(f"  成功: {len(data)}, 跳过: {n_skipped}")
    return data


def build_model(model_name: str, model_cfg: dict, random_state: int):
    """根据配置构造模型。"""
    if model_name == "GBDT":
        return GradientBoostingRegressor(
            n_estimators=model_cfg.get("n_estimators", 1000),
            max_depth=model_cfg.get("max_depth", 7),
            min_samples_split=model_cfg.get("min_samples_split", 5),
            learning_rate=model_cfg.get("learning_rate", 0.002),
            subsample=model_cfg.get("subsample", 0.8),
            max_features=model_cfg.get("max_features", "sqrt"),
            random_state=random_state,
        )
    elif model_name == "RF":
        return RandomForestRegressor(
            n_estimators=model_cfg.get("n_estimators", 1000),
            max_depth=model_cfg.get("max_depth", 8),
            min_samples_split=model_cfg.get("min_samples_split", 4),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 2),
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"未知模型: {model_name}")


def run_one_model(data: dict, dataset: str, model_name: str,
                  model_cfg: dict, output_dir: Path,
                  random_state: int, use_scale: bool) -> dict:
    """用指定模型跑 LOO 盲测。"""
    pdb_ids = sorted(data.keys())
    n_proteins = len(pdb_ids)

    results = []
    all_y_true = []
    all_y_pred = []

    print(f"\n[{model_name}] LOO 评估 ({n_proteins} proteins):")

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

        model = build_model(model_name, model_cfg, random_state)
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

    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "per_protein.csv", "w") as f:
        f.write("pdb_id,n_atoms,pcc,rmse\n")
        for r in results:
            f.write(f"{r['pdb_id']},{r['n_atoms']},{r['pcc']:.6f},{r['rmse']:.6f}\n")

    with open(output_dir / "summary.csv", "w") as f:
        f.write("dataset,n_proteins,model,mean_pcc,std_pcc,min_pcc,max_pcc,overall_pcc\n")
        f.write(f"{dataset},{n_proteins},{model_name},{mean_pcc:.6f},{std_pcc:.6f},"
                f"{min(pccs):.6f},{max(pccs):.6f},{overall_pcc:.6f}\n")

    print(f"\n[{model_name}] {dataset} 汇总: mean_PCC={mean_pcc:.4f} ± {std_pcc:.4f} "
          f"[{min(pccs):.4f}, {max(pccs):.4f}]  overall={overall_pcc:.4f}")

    return {
        "dataset": dataset, "model": model_name,
        "n_proteins": n_proteins, "mean_pcc": mean_pcc, "std_pcc": std_pcc,
        "min_pcc": min(pccs), "max_pcc": max(pccs), "overall_pcc": overall_pcc,
    }


def main():
    parser = argparse.ArgumentParser(description="跨蛋白质盲测 LOO")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--dataset", type=str,
                        help="覆盖 config 中的 datasets")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # 路径
    paths = cfg["paths"]
    exp_res = Path(paths["exp_res"])
    code_data = Path(paths["code_data"])
    output_root = Path(__file__).resolve().parent / "results"

    # 数据集
    if args.dataset:
        datasets = [args.dataset] if args.dataset != "all" else cfg["datasets"]
    else:
        datasets = cfg["datasets"]

    # 预处理
    pre_cfg = cfg.get("preprocessing", {})
    use_scale = pre_cfg.get("scale", True)

    # 评估
    eval_cfg = cfg["evaluation"]
    random_state = eval_cfg.get("random_state", 42)

    all_summaries = []
    exclude_cfg = cfg.get("exclude", {})

    for ds in datasets:
        exclude_list = exclude_cfg.get(ds, [])
        data = load_dataset(ds, exp_res, code_data, exclude_list)
        if len(data) < 2:
            print(f"[SKIP] {ds}: 蛋白数不足 ({len(data)})")
            continue

        for model_name, model_cfg in cfg["models"].items():
            if not model_cfg.get("enabled", True):
                continue
            summary = run_one_model(data, ds, model_name, model_cfg,
                                    output_root / ds / model_name,
                                    random_state, use_scale)
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
