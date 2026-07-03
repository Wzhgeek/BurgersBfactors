#!/usr/bin/env python3
"""
跨蛋白质盲测 Leave-One-Protein-Out (与 B-factor 盲测一致)。

特征: 100 ε × 10 level × 6 stat = 6000 维/原子
标签: B-factor (来自 .xyzb 文件)
模型: GBDT + RF, 与 B-factor 盲测一致

用法:
    python blind_loo.py
    python blind_loo.py --dataset 33small
"""

import json
import argparse
import itertools
import re
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


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
                          exp_res: Path, code_data: Path,
                          per_level_eps: dict | None = None,
                          feature_type: str = "stats",
                          holdout_dir: Path | None = None,
                          use_topo: bool = False,
                          result_dir: Path | None = None,
                          agg_across: bool = False,
                          eps_sensitivity: bool = False,
                          prune_mask: np.ndarray | None = None,
                          ) -> tuple[np.ndarray, np.ndarray] | None:
    feat_dir = exp_res / dataset / pdb_id / "features" / feature_type
    xyzb_path = code_data / dataset / f"{pdb_id}_ca.xyzb"

    if not feat_dir.exists():
        print(f"  [SKIP] {pdb_id}: {feature_type} 目录不存在")
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

    single_level = None
    # 如果 holdout_dir 存在，读取该蛋白自己的最优 eps
    if holdout_dir and per_level_eps is not None:
        rj = holdout_dir / dataset / pdb_id / "result.json"
        if rj.exists():
            d = json.loads(rj.read_text())
            if "__from_holdout_single__" in per_level_eps:
                # 只取最优 level（OOF 最高的那个 level）
                best_lv = max(d.get("per_level_stats", {}).items(),
                              key=lambda kv: kv[1]["oof_pcc"])
                per_level_eps = {best_lv[0]: float(best_lv[1]["eps"])}
                single_level = int(best_lv[0][1:])  # L01 -> 1
            else:
                per_level_eps = {}
                for lv_key, v in sorted(d.get("per_level_stats", {}).items()):
                    per_level_eps[lv_key] = float(v["eps"])

    cols_per_level = 100 if feature_type == "trj" else 6

    if per_level_eps:
        # 每 level 可选多个 eps → Σ(level_eps_count) × cols_per_level 维
        parts = []
        for lv_key, eps_vals in sorted(per_level_eps.items()):
            lv = int(lv_key[1:])  # L01 → 1
            if not isinstance(eps_vals, list):
                eps_vals = [eps_vals]
            for eps_val in eps_vals:
                eps_tag = f"{eps_val:.1f}".replace(".", "-")
                f = feat_dir / f"{pdb_id}_dyn_trj_feature_{eps_tag}.csv"
                if not f.exists():
                    print(f"  [SKIP] {pdb_id}: 缺失 {f.name}")
                    return None
                feat = np.loadtxt(f, delimiter=",", skiprows=1, dtype=np.float64)
                # 提取该 level 的列
                col_start = (lv - 1) * cols_per_level
                feat_lv = feat[:, col_start:col_start + cols_per_level]
            if feat_lv.shape[0] == len(y):
                feat_valid = feat_lv
            elif feat_lv.shape[0] == len(valid_mask):
                feat_valid = feat_lv[valid_mask]
            else:
                print(f"  [SKIP] {pdb_id}: 特征/标签行数不匹配")
                return None
            parts.append(feat_valid)
        X = np.hstack(parts)
        if agg_across:
            # (n_atoms, 10 levels, 6 stats) → aggregate across levels → (n_atoms, 4*6=24)
            X_3d = X.reshape(X.shape[0], 10, 6)
            agg_parts = [np.mean(X_3d, axis=1), np.max(X_3d, axis=1),
                         np.min(X_3d, axis=1), np.std(X_3d, axis=1)]
            X = np.hstack(agg_parts)
        if single_level:
            tag = f"best_level=L{single_level:02d}"
        else:
            tag = f"{len(per_level_eps)} levels × 6 stats"
        print(f"  {pdb_id}: {len(y)} atoms × {X.shape[1]} features ({tag}){extra}")
    else:
        files = sorted(feat_dir.glob("*.csv"), key=lambda f: parse_eps(f.name))
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
    # 跨 eps 敏感度: 每个 (level, stat) 在 100 eps 上的 std
    if eps_sensitivity:
        all_files = sorted(feat_dir.glob("*.csv"), key=lambda f: parse_eps(f.name))
        all_parts = []
        for f in all_files:
            feat = np.loadtxt(f, delimiter=",", skiprows=1, dtype=np.float64)
            if feat.shape[0] == len(y):
                all_parts.append(feat)
            elif feat.shape[0] == len(valid_mask):
                all_parts.append(feat[valid_mask])
            else:
                break
        if len(all_parts) == len(all_files):
            stacked = np.stack(all_parts, axis=0)  # (100, n_atoms, 60)
            sens = np.std(stacked, axis=0)         # (n_atoms, 60) 跨 eps 标准差
            X = np.hstack([X, sens])
            extra += f" +sens({sens.shape[1]}d)"
        else:
            print(f"  [WARN] {pdb_id}: eps sensitivity 加载失败, 跳过")

    # 拼接拓扑特征 (V1)
    if use_topo:
        topo_path = result_dir / dataset / "topo_features" / f"{pdb_id}_topo_feature.csv"
        if topo_path.exists():
            topo = np.loadtxt(topo_path, delimiter=",", skiprows=1, dtype=np.float64,
                              usecols=range(2, 62))
            topo_feat = np.nan_to_num(topo, nan=0.0)
            if topo_feat.shape[0] == len(y):
                X = np.hstack([X, topo_feat])
            elif topo_feat.shape[0] == len(valid_mask):
                X = np.hstack([X, topo_feat[valid_mask]])
            else:
                print(f"  [WARN] {pdb_id}: topo 行数不匹配, 跳过")
            n_atoms = len(y)
            if n_atoms > 0 and topo_feat.shape[1] == 60:
                # 新增: isolated_ratio + mean_degree (每 level 2d, 共 20d)
                extra_feat = np.zeros((topo_feat.shape[0], 20))
                for lv in range(10):
                    harm_idx = lv * 6       # harmonic_multiplicity
                    trace_idx = lv * 6 + 5  # eig_sum
                    extra_feat[:, lv*2]     = topo_feat[:, harm_idx] / n_atoms
                    extra_feat[:, lv*2 + 1] = topo_feat[:, trace_idx] / (n_atoms * (n_atoms - 1) + 1)
                topo_feat = np.hstack([topo_feat, extra_feat])
            X = np.hstack([X, topo_feat])
            extra += f" +topo({topo_feat.shape[1]}d)"
    # 剪枝
    if prune_mask is not None:
        if X.shape[1] == len(prune_mask):
            X = X[:, prune_mask]
            extra += f" -> {X.shape[1]}d"
        else:
            print(f"  [WARN] {pdb_id}: prune mask ({len(prune_mask)}) != features ({X.shape[1]}), skip")
    return X, y


def load_dataset(dataset: str, exp_res: Path, code_data: Path,
                 exclude: list[str] | None = None,
                 per_level_eps: dict | None = None,
                 feature_type: str = "stats",
                 holdout_dir: Path | None = None,
                 use_topo: bool = False,
                 result_dir: Path | None = None,
                 agg_across: bool = False,
                 eps_sensitivity: bool = False,
                 prune_mask: np.ndarray | None = None,
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
        result = load_protein_features(pdb, dataset, exp_res, code_data,
                                        per_level_eps, feature_type, holdout_dir,
                                        use_topo, result_dir, agg_across, eps_sensitivity, prune_mask)
        if result is not None:
            data[pdb] = result
    print(f"  成功: {len(data)}")
    return data


def build_model(model_name: str, cfg: dict, rs: int):
    if model_name == "GBDT":
        return GradientBoostingRegressor(
            n_estimators=cfg.get("n_estimators", 100),
            max_depth=cfg.get("max_depth", 3),
            min_samples_split=cfg.get("min_samples_split", 2),
            learning_rate=cfg.get("learning_rate", 0.1),
            subsample=cfg.get("subsample", 1.0),
            max_features=cfg.get("max_features", None),
            random_state=rs,
        )
    else:
        return RandomForestRegressor(
            n_estimators=cfg.get("n_estimators", 100),
            max_depth=cfg.get("max_depth", None),
            min_samples_split=cfg.get("min_samples_split", 2),
            min_samples_leaf=cfg.get("min_samples_leaf", 1),
            random_state=rs, n_jobs=-1,
        )


def grid_search_best_params(data: dict, model_name: str,
                            model_cfg: dict, grid_cfg: dict,
                            use_scale: bool, rs: int) -> dict:
    """3-fold GroupKFold 网格搜索最优参数。"""
    param_grid = model_cfg.get("param_grid", {})
    if not param_grid:
        return model_cfg

    pdb_ids = sorted(data.keys())
    X_all = np.vstack([data[p][0] for p in pdb_ids])
    y_all = np.concatenate([data[p][1] for p in pdb_ids])
    groups = np.concatenate([np.full(data[p][0].shape[0], i)
                             for i, p in enumerate(pdb_ids)])

    if use_scale:
        X_all = StandardScaler().fit_transform(X_all)

    cv = GroupKFold(n_splits=grid_cfg.get("cv_folds", 3))
    keys, values = list(param_grid.keys()), list(param_grid.values())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"\n[{model_name}] Grid Search ({len(combos)} combos × "
          f"{grid_cfg['cv_folds']} folds):")

    best_pcc, best_params = -999, None
    for params in combos:
        fold_pccs = []
        for train_idx, val_idx in cv.split(X_all, y_all, groups):
            merged = {**model_cfg, **params}
            model = build_model(model_name, merged, rs)
            model.fit(X_all[train_idx], y_all[train_idx])
            y_pred = model.predict(X_all[val_idx])
            fold_pccs.append(pearsonr(y_pred, y_all[val_idx])[0])

        mean_pcc = np.mean(fold_pccs)
        if mean_pcc > best_pcc:
            best_pcc = mean_pcc
            best_params = dict(params)

    merged = {**model_cfg, **best_params}
    print(f"  最优: {best_params}  PCC={best_pcc:.4f}")
    return merged


def loo_evaluate(data: dict, dataset: str, model_name: str,
                 model_cfg: dict, output_dir: Path, rs: int,
                 use_scale: bool) -> dict:
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

        if use_scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

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
    with open(output_dir / "best_params.txt", "w") as f:
        for k, v in model_cfg.items():
            if k not in ("param_grid", "enabled"):
                f.write(f"{k}: {v}\n")
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
    eval_cfg = cfg["evaluation"]
    rs = eval_cfg["random_state"]
    n_cycle = eval_cfg.get("n_cycle", 1)
    use_scale = cfg.get("preprocessing", {}).get("scale", True)

    if args.dataset:
        datasets = cfg["datasets"] if args.dataset == "all" else [args.dataset]
    else:
        datasets = cfg["datasets"]
    exclude_cfg = cfg.get("exclude", {})
    grid_cfg = cfg.get("grid_search", {})
    do_grid = grid_cfg.get("enabled", False)
    per_level_eps_cfg = cfg.get("per_level_eps", None)
    feature_type = cfg.get("feature_type", "stats")
    use_per_protein = cfg.get("per_protein_eps", False)
    best_level_only = cfg.get("best_level_only", False)

    agg_across = cfg.get("agg_across_levels", False)
    eps_sensitivity = cfg.get("eps_sensitivity", False)
    use_topo = cfg.get("use_topo", False)
    result_dir = exp_res.parent / "result" if use_topo else None
    holdout_dir = None
    if use_per_protein:
        holdout_dir = exp_res / "Bfactor_result_holdout"
        per_level_eps = {"__from_holdout__": True}  # 非空占位
    if best_level_only:
        holdout_dir = exp_res / "Bfactor_result_holdout"
        per_level_eps = {"__from_holdout_single__": True}

    if best_level_only:
        cols_per = 100 if feature_type == "trj" else 6
        suffix = f"BestLevel_{feature_type}_{cols_per}d"
    elif use_per_protein:
        cols_per = 100 if feature_type == "trj" else 6
        suffix = f"PerProtein_{feature_type}_{cols_per*10}d"
    elif per_level_eps:
        cols_per = 100 if feature_type == "trj" else 6
        if agg_across:
            suffix = f"AggLevel24d"
        else:
            suffix = f"PerLevel_{feature_type}_{cols_per*10}d"
    else:
        suffix = ""
    if eps_sensitivity:
        suffix += "+Sens"
    feature_prune = cfg.get("feature_prune", False)
    if feature_prune:
        suffix += "+Prune"

    if use_topo:
        suffix += "+Topo"

    for ds in datasets:
        prune_mask = None
        if feature_prune:
            # 预计算的 importance >= 1% 特征索引 (120d: 60 dyn + 60 topo)
            prune_mask = np.zeros(120, dtype=bool)
            prune_mask[[3,5,7,10,13,16,19,22,25,28,29,31,32,34,37,38,40,41,43,44,45,46,47,49,50,51,52,53,55,56,57,58,59,71,77,80,83,92,95,98,101,104,110,119]] = True

        data = load_dataset(ds, exp_res, code_data, exclude_cfg.get(ds, []),
                            per_level_eps, feature_type, holdout_dir,
                            use_topo, result_dir, agg_across, eps_sensitivity, prune_mask)
        if len(data) < 2:
            print(f"[SKIP] {ds}: 蛋白数不足")
            continue

        for name, mcfg in cfg["models"].items():
            if not mcfg.get("enabled", True):
                continue
            if do_grid:
                mcfg = grid_search_best_params(data, name, mcfg, grid_cfg,
                                               use_scale, rs)

            # 多 cycle 平均 (与 B-factor 一致)
            pdb_ids = sorted(data.keys())
            all_cycle_pccs = np.zeros((n_cycle, len(pdb_ids)))
            for cyc in range(n_cycle):
                cycle_rs = rs + cyc
                out_dir = output_root / ds / suffix / name / f"cycle{cyc}"
                s = loo_evaluate(data, ds, name, mcfg, out_dir, cycle_rs, use_scale)
                # 从 per_protein.csv 读取 PCC
                pp = np.loadtxt(out_dir / "per_protein.csv", delimiter=",",
                                skiprows=1, dtype=str)
                if pp.shape[0] == len(pdb_ids):
                    all_cycle_pccs[cyc] = pp[:, 2].astype(float)

            mean_pccs = np.nanmean(all_cycle_pccs, axis=0)
            out_dir = output_root / ds / suffix / name
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "per_protein.csv", "w") as f:
                f.write("pdb_id,n_atoms,pcc_mean,pcc_std,cycles\n")
                for j, pdb in enumerate(pdb_ids):
                    n_a = data[pdb][0].shape[0]
                    f.write(f"{pdb},{n_a},{mean_pccs[j]:.6f},{np.nanstd(all_cycle_pccs[:,j]):.6f},{n_cycle}\n")
            with open(out_dir / "summary.csv", "w") as f:
                f.write("dataset,n_proteins,model,mean_pcc,std_pcc,cycles\n")
                f.write(f"{ds},{len(pdb_ids)},{name},{np.mean(mean_pccs):.6f},{np.std(mean_pccs):.6f},{n_cycle}\n")
            print(f"\n[{name}] {ds} ({n_cycle} cycles): mean_PCC={np.mean(mean_pccs):.4f} ± {np.std(mean_pccs):.4f}")

    print(f"\n结果保存至: {output_root}")


if __name__ == "__main__":
    main()
