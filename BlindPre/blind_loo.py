#!/usr/bin/env python3
"""
跨蛋白质盲测 Leave-One-Protein-Out (与 B-factor 盲测一致)。

特征: 100 ε × 10 level × 6 stat = 6000 维/原子
标签: B-factor (来自 .xyzb 文件)
模型: GBDT + RF, 无 StandardScaler, 默认参数

用法:
    python blind_loo.py
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


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_eps(filename: str) -> float:
    m = re.search(r"feature_(\d+)-(\d+)\.csv", filename)
    if not m:
        raise ValueError(f"无法解析 eps: {filename}")
    return float(f"{m.group(1)}.{m.group(2)}")


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
                          exp_res: Path, code_data: Path
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
        print(f"  [SKIP] {pdb_id}: 无有效标签")
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
            print(f"  [SKIP] {pdb_id}: 特征/标签行数不匹配")
            return None
        parts.append(feat_valid)

    X = np.hstack(parts)
    print(f"  {pdb_id}: {len(y)} atoms × {X.shape[1]} features{extra}")
    return X, y


def load_dataset(dataset: str, exp_res: Path, code_data: Path,
                 exclude: list[str] | None = None
                 ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ds_dir = exp_res / dataset
    if exclude is None:
        exclude = []
    proteins = [p.name for p in sorted(ds_dir.iterdir())
                if p.is_dir() and (p / "features" / "stats").exists()
                and p.name not in exclude]
    print(f"\n加载 {dataset} ({len(proteins)} proteins" +
          (f", 排除 {len(exclude)}" if exclude else "") + "):")

    data = {}
    for pdb in proteins:
        result = load_protein_features(pdb, dataset, exp_res, code_data)
        if result is not None:
            data[pdb] = result
    print(f"  成功: {len(data)}")
    return data


def build_model(model_name: str, cfg: dict, rs: int):
    if model_name == "GBDT":
        return GradientBoostingRegressor(
            n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_split"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"], max_features=cfg["max_features"],
            random_state=rs,
        )
    else:
        return RandomForestRegressor(
            n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_split"],
            min_samples_leaf=cfg["min_samples_leaf"],
            random_state=rs, n_jobs=-1,
        )


def loo_evaluate(data: dict, dataset: str, model_name: str,
                 model_cfg: dict, output_dir: Path, rs: int) -> dict:
    pdb_ids = sorted(data.keys())
    n = len(pdb_ids)

    results = []
    all_y_true, all_y_pred = [], []

    print(f"\n[{model_name}] LOO ({n} proteins):")

    for i, test_pdb in enumerate(pdb_ids):
        t0 = time.time()

        X_test, y_test = data[test_pdb]
        X_train = np.vstack([data[p][0] for p in pdb_ids if p != test_pdb])
        y_train = np.concatenate([data[p][1] for p in pdb_ids if p != test_pdb])

        model = build_model(model_name, model_cfg, rs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        pcc, _ = pearsonr(y_pred, y_test)
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

        print(f"  [{i+1}/{n}] {test_pdb}: PCC={pcc:.4f} RMSE={rmse:.4f} "
              f"({time.time()-t0:.1f}s)")

        results.append({"pdb_id": test_pdb, "n_atoms": len(y_test),
                        "pcc": pcc, "rmse": rmse})
        all_y_true.append(y_test)
        all_y_pred.append(y_pred)

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    overall_pcc, _ = pearsonr(all_y_pred, all_y_true)
    pccs = [r["pcc"] for r in results]

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "per_protein.csv", "w") as f:
        f.write("pdb_id,n_atoms,pcc,rmse\n")
        for r in results:
            f.write(f"{r['pdb_id']},{r['n_atoms']},{r['pcc']:.6f},{r['rmse']:.6f}\n")
    with open(output_dir / "summary.csv", "w") as f:
        f.write("dataset,n_proteins,model,mean_pcc,std_pcc,min_pcc,max_pcc,overall_pcc\n")
        f.write(f"{dataset},{n},{model_name},{np.mean(pccs):.6f},{np.std(pccs):.6f},"
                f"{min(pccs):.6f},{max(pccs):.6f},{overall_pcc:.6f}\n")

    print(f"\n[{model_name}] {dataset}: mean_PCC={np.mean(pccs):.4f} ± {np.std(pccs):.4f} "
          f"[{min(pccs):.4f}, {max(pccs):.4f}]  overall={overall_pcc:.4f}")
    return {"dataset": dataset, "model": model_name, "mean_pcc": np.mean(pccs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg["paths"]
    exp_res = Path(paths["exp_res"])
    code_data = Path(paths["code_data"])
    output_root = Path(__file__).resolve().parent / "results"
    rs = cfg["evaluation"]["random_state"]

    datasets = [args.dataset] if args.dataset else cfg["datasets"]
    exclude_cfg = cfg.get("exclude", {})

    for ds in datasets:
        data = load_dataset(ds, exp_res, code_data, exclude_cfg.get(ds, []))
        if len(data) < 2:
            print(f"[SKIP] {ds}: 蛋白数不足")
            continue

        for name, mcfg in cfg["models"].items():
            if not mcfg.get("enabled", True):
                continue
            loo_evaluate(data, ds, name, mcfg, output_root / ds / name, rs)

    print(f"\n结果保存至: {output_root}")


if __name__ == "__main__":
    main()
