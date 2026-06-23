#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
Pcode 主入口：Burgers 动力学模拟 → 特征提取 → 回归预测。

用法:
    python run.py --dataset 33small --protein 1Q9B
    python run.py --config config.yaml --dataset 33small --protein 1Q9B
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.burgers import rk4_step
from src.features import extract_stats_features, simulate_trajectories, load_aij_matrices
from src.regression import evaluate_regressor
from src.plot import plot_ux, plot_ut
from src.utils import load_yaml, save_json, setup_logging, pcc_10digit, resolve_path


def parse_args():
    p = argparse.ArgumentParser(description="Pcode Burgers B-factor prediction")
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--protein", type=str, required=True)
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: run single eps+level only")
    p.add_argument("--eps", type=float, default=0.5,
                   help="Epsilon for smoke test (default 0.5)")
    p.add_argument("--level", type=int, default=5,
                   help="Level for smoke test (default 5)")
    return p.parse_args()


def find_n_steps(A, nu, dt, dx, eps_ref, n_atoms, decay_threshold=0.1, max_steps=50000):
    """找出 eps_ref 下最慢原子衰减到 threshold 所需的步数。"""
    max_s = 0
    for i in range(n_atoms):
        u = np.zeros(n_atoms); u[i] = 1.0
        for step in range(1, max_steps + 1):
            u = rk4_step(u, dt, A, nu, eps_ref, dx, "graph_diffusion")
            if u[i] <= decay_threshold:
                if step > max_s: max_s = step
                break
    return max_s


def simulate_one_task(args):
    """单个 (lvl, eps) 模拟任务。"""
    lvl, eps, A, nu, dt, dx, n_steps, n_atoms, n_pts, coupling_mode = args
    trj, stats = simulate_trajectories(A, nu, eps, dt, dx, n_steps, n_atoms,
                                        n_sample_points=n_pts, coupling_mode=coupling_mode)
    return lvl, eps, trj, stats


def _save_model_csv(path: Path, model_grid: np.ndarray, header: str) -> None:
    """保存与 all_score 同维度的模型名称 CSV。"""
    lines = [header]
    for row in model_grid:
        lines.append(",".join(str(v) for v in row))
    path.write_text("\n".join(lines) + "\n")


def _oof_pcc(result: dict) -> float:
    return float(result.get("oof_pcc", result["test_pcc"]))


def _level_result_entry(result: dict, model: str, eps_str: str) -> dict:
    """构造 per_level / result.json 记录（同时含 OOF、最优折、折平均）。"""
    return {
        "eps": eps_str,
        "oof_pcc": pcc_10digit(result.get("oof_pcc", result["test_pcc"])),
        "best_fold_pcc": pcc_10digit(result.get("best_fold_pcc", result["test_pcc"])),
        "mean_fold_pcc": pcc_10digit(result.get("mean_fold_pcc", result["test_pcc"])),
        "best_fold": int(result.get("best_fold_idx", 1)),
        "model": model,
        "fold_val_pccs": result.get("fold_val_pccs", []),
    }


def _pick_best_regressor(reg_cfg, X, labels, test_size, random_state, cv_folds):
    """按 OOF PCC 选取最优回归器。"""
    best_oof, best_model, best_result = -999.0, "RF", None
    for reg_name, reg_params in reg_cfg.items():
        if not reg_params.get("enabled", True):
            continue
        r = evaluate_regressor(reg_name, X, labels, reg_params,
                               test_size, random_state, cv_folds)
        oof = _oof_pcc(r)
        if oof > best_oof:
            best_oof = oof
            best_model = reg_name.upper()
            best_result = r
    return best_oof, best_model, best_result


def _log_cv_metrics(prefix: str, r: dict, log) -> None:
    log.info(
        f"{prefix} OOF={r.get('oof_pcc', r['test_pcc']):.4f} "
        f"best_fold={r.get('best_fold_idx', 1)}"
        f"({r.get('best_fold_pcc', r['test_pcc']):.4f}) "
        f"mean_fold={r.get('mean_fold_pcc', r['test_pcc']):.4f}"
    )


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    dataset = args.dataset
    pdb_id = args.protein.strip().upper()

    # ── 路径 ──
    project_root = Path(__file__).resolve().parent
    paths = cfg.get("paths", {})
    result_root = resolve_path(project_root, paths.get("output_root", project_root / "result"))
    step1_root = resolve_path(project_root, paths.get("step1_result", result_root))
    code_data_dir = resolve_path(project_root, paths.get("code_data", project_root / "code_data")) / dataset
    step1_dir = step1_root / dataset
    protein_out = result_root / dataset / pdb_id
    protein_out.mkdir(parents=True, exist_ok=True)

    # ── 日志 ──
    log = setup_logging(protein_out / "run.log", level=getattr(logging, cfg["logging"]["level"]))

    # ── 参数 ──
    dyn = cfg["dynamics"]
    dt, dx, nu = dyn["dt"], dyn["dx"], dyn["nu"]
    coupling_mode = dyn["coupling_mode"]
    eps_ref = dyn["n_steps_ref_eps"]
    decay_thr = dyn["decay_threshold"]

    eps_start = cfg["epsilon"]["start"]
    eps_stop = cfg["epsilon"]["stop"]
    eps_step = cfg["epsilon"]["step"]
    eps_list = np.arange(eps_start, eps_stop + eps_step / 2, eps_step)

    graph_cfg = cfg["graph"]
    num_levels = graph_cfg["num_levels"]

    feat_cfg = cfg["features"]
    n_pts = feat_cfg["trajectory"]["n_points"]

    eval_cfg = cfg["evaluation"]
    test_size = eval_cfg["test_size"]
    random_state = eval_cfg["random_state"]
    cv_folds = eval_cfg.get("cv_folds", 10)

    n_jobs = cfg["parallel"]["n_jobs"]
    if n_jobs <= 0:
        n_jobs = cpu_count()

    # ── 前置检查：step1 数据是否存在 ──
    aij_dir = step1_dir / pdb_id / "Aijandlabel"
    if not aij_dir.exists():
        log.error(f"Step1 data not found: {aij_dir}")
        log.error(f"Run step1 first: python run_step1.py --dataset {dataset} --protein {pdb_id}")
        return 1
    label_path = aij_dir / f"{pdb_id}_label.npy"
    first_aij = aij_dir / f"{pdb_id}_Aij_0-1.npy"
    if not label_path.exists() or not first_aij.exists():
        log.error(f"Aij/label files missing in: {aij_dir}")
        log.error(f"Run: python run_step1.py --dataset {dataset} --protein {pdb_id}")
        return 1

    # ── 加载标签 ──
    labels = np.load(label_path).astype(np.float64)
    n_atoms = len(labels)

    # ── 分位数阈值 ──
    dist = np.load(step1_dir / pdb_id / "distance" / f"{pdb_id}_dist.npy")
    off = dist[dist > 0]
    thresholds = np.percentile(off, np.linspace(
        graph_cfg["percentile_start"], graph_cfg["percentile_stop"], num_levels)).tolist()

    log.info(f"Protein={pdb_id} dataset={dataset} atoms={n_atoms} levels={num_levels}")
    log.info(f"Eps range: [{eps_list[0]:.1f}, {eps_list[-1]:.1f}] n_jobs={n_jobs}")
    log.info(f"Thresholds: {[round(t,1) for t in thresholds]}")
    log.info(f"Evaluation: {cv_folds}-fold CV" if cv_folds >= 2 else f"Evaluation: hold-out test_size={test_size}")

    # ── 确定演化步数（基于 eps_ref, 中间层 L05） ──
    ref_lvl = min(5, num_levels)
    ref_A_path = step1_dir / pdb_id / "Aijandlabel" / f"{pdb_id}_Aij_0-{ref_lvl}.npy"
    ref_A = np.load(ref_A_path)
    n_steps = find_n_steps(ref_A, nu, dt, dx, eps_ref, n_atoms, decay_thr)
    log.info(f"n_steps={n_steps} (from eps_ref={eps_ref} at L{ref_lvl:02d})")

    # ── 加载全部 Aij 矩阵 ──
    aij_matrices = load_aij_matrices(step1_dir, pdb_id, num_levels)

    # ── 保存各层边数 ──
    edge_lines = []
    for lvl in range(1, num_levels + 1):
        A_adj = aij_matrices[lvl - 1].copy()
        np.fill_diagonal(A_adj, 0)
        n_edges = int(np.sum(A_adj > 1e-10))
        edge_lines.append(f"L{lvl:02d}:{n_edges}")
    (protein_out / "eps_edge.txt").write_text("\n".join(edge_lines) + "\n")
    log.info(f"Edge counts saved: {protein_out / 'eps_edge.txt'}")

    # ── 烟雾测试模式 ──
    if args.smoke:
        smoke_eps = args.eps
        smoke_lvl = args.level
        log.info(f"SMOKE TEST: level={smoke_lvl}, eps={smoke_eps}")
        A_smoke = aij_matrices[smoke_lvl - 1]
        trj_smoke, stats_smoke = simulate_trajectories(
            A_smoke, nu, smoke_eps, dt, dx, n_steps, n_atoms,
            n_sample_points=n_pts, coupling_mode=coupling_mode)

        # Save trajectory
        traj_dir = protein_out / "trajectory"; traj_dir.mkdir(exist_ok=True)
        eps_tag = f"{smoke_eps:.1f}".replace(".", "-")
        np.save(traj_dir / f"{pdb_id}_L{smoke_lvl:02d}_eps{eps_tag}_trj.npy", trj_smoke)

        # Quick RF test (trj + stats)
        r_trj = evaluate_regressor("rf", trj_smoke, labels, cfg["regressors"].get("rf", {}),
                                   test_size, random_state, cv_folds)
        r_stats = evaluate_regressor("rf", stats_smoke, labels, cfg["regressors"].get("rf", {}),
                                    test_size, random_state, cv_folds)
        _log_cv_metrics("RF trj ", r_trj, log)
        _log_cv_metrics("RF stats", r_stats, log)

        eps_str = f"{smoke_eps:.1f}"
        lvl_key = f"L{smoke_lvl:02d}"
        per_trj = _level_result_entry(r_trj, "RF", eps_str)
        per_stats = _level_result_entry(r_stats, "RF", eps_str)
        smoke_result = {
            "pdb_id": pdb_id,
            "dataset": dataset,
            "smoke": True,
            "n_atoms": n_atoms,
            "n_steps": n_steps,
            "nu": nu,
            "dt": dt,
            "dx": dx,
            "smoke_config": {"level": smoke_lvl, "eps": smoke_eps},
            "evaluation": {"cv_folds": cv_folds, "random_state": random_state},
            "best_combined_trj": {k: v for k, v in per_trj.items() if k != "fold_val_pccs"},
            "best_combined_stats": {k: v for k, v in per_stats.items() if k != "fold_val_pccs"},
            "per_level_trj": {lvl_key: per_trj},
            "per_level_stats": {lvl_key: per_stats},
            "smoke_trj": r_trj,
            "smoke_stats": r_stats,
        }
        save_json(smoke_result, protein_out / "result.json")
        log.info(f"Result saved: {protein_out / 'result.json'}")

        # Plot
        fig_dir = protein_out / "figures"; fig_dir.mkdir(exist_ok=True)
        pulse_idx = n_atoms // 2
        plot_ux(A_smoke, nu, smoke_eps, pdb_id, smoke_lvl, pulse_idx, dt, dx, n_steps,
                fig_dir / f"{pdb_id}_smoke_ux.png", model="RF")
        plot_ut(trj_smoke, nu, smoke_eps, pdb_id, smoke_lvl, dt, max(1, n_steps // n_pts),
                fig_dir / f"{pdb_id}_smoke_ut.png", model="RF")
        log.info(f"Smoke test done. Plots: {fig_dir}")
        return 0

    # ── 目录 ──
    traj_dir = protein_out / "trajectory"; traj_dir.mkdir(exist_ok=True)
    feat_stats_dir = protein_out / "features" / "stats"; feat_stats_dir.mkdir(parents=True, exist_ok=True)
    feat_trj_dir = protein_out / "features" / "trj"; feat_trj_dir.mkdir(parents=True, exist_ok=True)
    all_score_dir = protein_out / "all_score"; all_score_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = protein_out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)

    # ── 备份 config ──
    import shutil; shutil.copy(args.config, protein_out / "config.yaml")

    # ── 并行模拟所有 (level, eps) ──
    log.info(f"Starting simulation: {num_levels} levels × {len(eps_list)} eps")
    t0 = time.time()
    tasks = [(lvl, round(float(eps), 1), aij_matrices[lvl - 1], nu, dt, dx, n_steps, n_atoms, n_pts, coupling_mode)
             for eps in eps_list for lvl in range(1, num_levels + 1)]

    all_trj = {}
    all_stats_features = {}
    with Pool(n_jobs) as pool:
        for i, (lvl, eps, trj, stats) in enumerate(pool.imap_unordered(simulate_one_task, tasks)):
            eps_str = f"{eps:.1f}"
            key = (lvl, eps_str)
            all_trj[key] = trj
            all_stats_features[key] = stats
            if (i + 1) % 100 == 0:
                pct = 100 * (i + 1) / len(tasks)
                log.info(f"Sim progress: {i+1}/{len(tasks)} ({pct:.0f}%) elapsed={time.time()-t0:.0f}s")
    sim_time = time.time() - t0
    log.info(f"Simulation done in {sim_time:.0f}s")

    # ── 保存轨迹和特征 ──
    log.info("Saving trajectories and features...")
    for eps in eps_list:
        eps_str = f"{eps:.1f}"
        eps_tag = eps_str.replace(".", "-")

        # 轨迹: (10, n_pts, n_atoms) → 存为 (num_levels, n_atoms, n_pts) 或压平
        trj_levels = np.stack([all_trj[(lvl, eps_str)] for lvl in range(1, num_levels + 1)])
        np.save(traj_dir / f"{pdb_id}_dyn_trj_{eps_tag}.npy", trj_levels)

        # 统计量特征 CSV
        stats_levels = np.stack([all_stats_features[(lvl, eps_str)] for lvl in range(1, num_levels + 1)])
        # stats_levels: (10, n_atoms, 6)
        stats_2d = stats_levels.reshape(num_levels, -1)
        cols = [f"L{lvl:02d}_{s}" for lvl in range(1, num_levels + 1)
                for s in feat_cfg["stats"]["names"]]
        np.savetxt(feat_stats_dir / f"{pdb_id}_dyn_trj_feature_{eps_tag}.csv",
                   stats_2d.T, delimiter=",", header=",".join(cols), comments="", fmt="%.10g")

        # 采样轨迹特征 CSV
        trj_2d = trj_levels.reshape(num_levels, -1)
        trj_cols = [f"L{lvl:02d}_t{t}" for lvl in range(1, num_levels + 1) for t in range(n_pts)]
        np.savetxt(feat_trj_dir / f"{pdb_id}_dyn_trj_feature_{eps_tag}.csv",
                   trj_2d.T, delimiter=",", header=",".join(trj_cols), comments="", fmt="%.10g")

    # ── 回归评估 ──
    reg_cfg = cfg["regressors"]
    log.info("Starting regression evaluation...")

    # 网格 CSV：同时记录 OOF / 最优折 PCC / 折平均 PCC / 最优折号
    all_oof_stats = np.zeros((len(eps_list), num_levels))
    all_bestfold_pcc_stats = np.zeros((len(eps_list), num_levels))
    all_meanfold_stats = np.zeros((len(eps_list), num_levels))
    all_bestfold_stats = np.zeros((len(eps_list), num_levels), dtype=int)
    all_model_stats = np.empty((len(eps_list), num_levels), dtype=object)
    all_oof_trj = np.zeros((len(eps_list), num_levels))
    all_bestfold_pcc_trj = np.zeros((len(eps_list), num_levels))
    all_meanfold_trj = np.zeros((len(eps_list), num_levels))
    all_bestfold_trj = np.zeros((len(eps_list), num_levels), dtype=int)
    all_model_trj = np.empty((len(eps_list), num_levels), dtype=object)
    fold_detail_stats = {}
    fold_detail_trj = {}
    for ei in range(len(eps_list)):
        for lvl in range(num_levels):
            all_model_stats[ei, lvl] = "RF"
            all_model_trj[ei, lvl] = "RF"
            all_bestfold_stats[ei, lvl] = 1
            all_bestfold_trj[ei, lvl] = 1

    # per-level best tracking（按 OOF 选最优 ε）
    best_per_level_stats = {}
    best_per_level_trj = {}
    for lvl in range(1, num_levels + 1):
        best_per_level_stats[lvl] = {"eps": 0, "oof_pcc": -999, "model": "RF", "best_fold": 1}
        best_per_level_trj[lvl] = {"eps": 0, "oof_pcc": -999, "model": "RF", "best_fold": 1}

    best_combined_stats = {"eps": 0, "oof_pcc": -999, "model": "RF", "best_fold": 1}
    best_combined_trj = {"eps": 0, "oof_pcc": -999, "model": "RF", "best_fold": 1}

    # results for all regressors
    all_reg_results = {}

    rf_time_stats_total = 0
    rf_time_trj_total = 0

    for ei, eps in enumerate(eps_list):
        eps_str = f"{eps:.1f}"

        # ── Combined (10 levels) ──
        if feat_cfg["stats"]["enabled"]:
            t1 = time.time()
            parts_s = [all_stats_features[(lvl, eps_str)] for lvl in range(1, num_levels + 1)]
            X_s = np.hstack(parts_s)
            for reg_name, reg_params in reg_cfg.items():
                if not reg_params.get("enabled", True):
                    continue
                result = evaluate_regressor(reg_name, X_s, labels, reg_params,
                                           test_size, random_state, cv_folds)
                key = f"{reg_name}_stats_eps{eps_str}"
                all_reg_results[key] = result
            rf_time_stats_total += time.time() - t1

            # Track combined best across all regressors
            for reg_name, reg_params in reg_cfg.items():
                if not reg_params.get("enabled", True):
                    continue
                key = f"{reg_name}_stats_eps{eps_str}"
                if key not in all_reg_results:
                    continue
                r = all_reg_results[key]
                oof = _oof_pcc(r)
                if oof > best_combined_stats["oof_pcc"]:
                    best_combined_stats = _level_result_entry(r, reg_name.upper(), eps_str)

        if feat_cfg["trajectory"]["enabled"]:
            t1 = time.time()
            parts_t = [all_trj[(lvl, eps_str)] for lvl in range(1, num_levels + 1)]
            X_t = np.hstack(parts_t)
            for reg_name, reg_params in reg_cfg.items():
                if not reg_params.get("enabled", True):
                    continue
                result = evaluate_regressor(reg_name, X_t, labels, reg_params,
                                           test_size, random_state, cv_folds)
                key = f"{reg_name}_trj_eps{eps_str}"
                all_reg_results[key] = result
            rf_time_trj_total += time.time() - t1

            for reg_name, reg_params in reg_cfg.items():
                if not reg_params.get("enabled", True):
                    continue
                key = f"{reg_name}_trj_eps{eps_str}"
                if key not in all_reg_results:
                    continue
                r = all_reg_results[key]
                oof = _oof_pcc(r)
                if oof > best_combined_trj["oof_pcc"]:
                    best_combined_trj = _level_result_entry(r, reg_name.upper(), eps_str)

        # ── Per-level ──
        for lvl in range(1, num_levels + 1):
            # Stats
            if feat_cfg["stats"]["enabled"]:
                s = all_stats_features[(lvl, eps_str)]
                _, best_cell_model, best_r = _pick_best_regressor(
                    reg_cfg, s, labels, test_size, random_state, cv_folds)
                if best_r is None:
                    continue
                all_oof_stats[ei, lvl - 1] = pcc_10digit(_oof_pcc(best_r))
                all_bestfold_pcc_stats[ei, lvl - 1] = pcc_10digit(best_r.get("best_fold_pcc", best_r["test_pcc"]))
                all_meanfold_stats[ei, lvl - 1] = pcc_10digit(best_r.get("mean_fold_pcc", best_r["test_pcc"]))
                all_bestfold_stats[ei, lvl - 1] = int(best_r.get("best_fold_idx", 1))
                all_model_stats[ei, lvl - 1] = best_cell_model
                fold_detail_stats.setdefault(eps_str, {})[f"L{lvl:02d}"] = {
                    "model": best_cell_model,
                    "oof_pcc": best_r.get("oof_pcc", best_r["test_pcc"]),
                    "best_fold_pcc": best_r.get("best_fold_pcc", best_r["test_pcc"]),
                    "mean_fold_pcc": best_r.get("mean_fold_pcc", best_r["test_pcc"]),
                    "best_fold": int(best_r.get("best_fold_idx", 1)),
                    "fold_val_pccs": best_r.get("fold_val_pccs", []),
                }
                entry = _level_result_entry(best_r, best_cell_model, eps_str)
                if _oof_pcc(best_r) > best_per_level_stats[lvl]["oof_pcc"]:
                    best_per_level_stats[lvl] = entry

            if feat_cfg["trajectory"]["enabled"]:
                t = all_trj[(lvl, eps_str)]
                _, best_cell_model, best_r = _pick_best_regressor(
                    reg_cfg, t, labels, test_size, random_state, cv_folds)
                if best_r is None:
                    continue
                all_oof_trj[ei, lvl - 1] = pcc_10digit(_oof_pcc(best_r))
                all_bestfold_pcc_trj[ei, lvl - 1] = pcc_10digit(best_r.get("best_fold_pcc", best_r["test_pcc"]))
                all_meanfold_trj[ei, lvl - 1] = pcc_10digit(best_r.get("mean_fold_pcc", best_r["test_pcc"]))
                all_bestfold_trj[ei, lvl - 1] = int(best_r.get("best_fold_idx", 1))
                all_model_trj[ei, lvl - 1] = best_cell_model
                fold_detail_trj.setdefault(eps_str, {})[f"L{lvl:02d}"] = {
                    "model": best_cell_model,
                    "oof_pcc": best_r.get("oof_pcc", best_r["test_pcc"]),
                    "best_fold_pcc": best_r.get("best_fold_pcc", best_r["test_pcc"]),
                    "mean_fold_pcc": best_r.get("mean_fold_pcc", best_r["test_pcc"]),
                    "best_fold": int(best_r.get("best_fold_idx", 1)),
                    "fold_val_pccs": best_r.get("fold_val_pccs", []),
                }
                entry = _level_result_entry(best_r, best_cell_model, eps_str)
                if _oof_pcc(best_r) > best_per_level_trj[lvl]["oof_pcc"]:
                    best_per_level_trj[lvl] = entry

        lvl6 = best_per_level_stats.get(6, {})
        log.info(
            f"eps={eps_str} done: stats "
            f"OOF={lvl6.get('oof_pcc', 0):.4f} "
            f"best_fold={lvl6.get('best_fold', 1)}"
            f"({lvl6.get('best_fold_pcc', 0):.4f}) "
            f"mean_fold={lvl6.get('mean_fold_pcc', 0):.4f} "
            f"model={lvl6.get('model', 'RF')}"
        )

    # ── 保存 OOF / 最优折 / 折平均 / 折号 / 模型 CSV ──
    header = ",".join([f"L{lvl:02d}" for lvl in range(1, num_levels + 1)])
    np.savetxt(all_score_dir / f"{pdb_id}_all_oof_stats.csv", all_oof_stats,
               delimiter=",", header=header, comments="", fmt="%.10g")
    np.savetxt(all_score_dir / f"{pdb_id}_all_oof_trj.csv", all_oof_trj,
               delimiter=",", header=header, comments="", fmt="%.10g")
    np.savetxt(all_score_dir / f"{pdb_id}_all_bestfold_pcc_stats.csv", all_bestfold_pcc_stats,
               delimiter=",", header=header, comments="", fmt="%.10g")
    np.savetxt(all_score_dir / f"{pdb_id}_all_bestfold_pcc_trj.csv", all_bestfold_pcc_trj,
               delimiter=",", header=header, comments="", fmt="%.10g")
    np.savetxt(all_score_dir / f"{pdb_id}_all_meanfold_stats.csv", all_meanfold_stats,
               delimiter=",", header=header, comments="", fmt="%.10g")
    np.savetxt(all_score_dir / f"{pdb_id}_all_meanfold_trj.csv", all_meanfold_trj,
               delimiter=",", header=header, comments="", fmt="%.10g")
    np.savetxt(all_score_dir / f"{pdb_id}_all_bestfold_stats.csv", all_bestfold_stats,
               delimiter=",", header=header, comments="", fmt="%d")
    np.savetxt(all_score_dir / f"{pdb_id}_all_bestfold_trj.csv", all_bestfold_trj,
               delimiter=",", header=header, comments="", fmt="%d")
    _save_model_csv(all_score_dir / f"{pdb_id}_all_model_stats.csv", all_model_stats, header)
    _save_model_csv(all_score_dir / f"{pdb_id}_all_model_trj.csv", all_model_trj, header)
    save_json({
        "cv_folds": cv_folds,
        "stats": fold_detail_stats,
        "trj": fold_detail_trj,
    }, all_score_dir / f"{pdb_id}_fold_detail.json")
    log.info(f"Saved all_oof / all_bestfold_pcc / all_meanfold / fold_detail: {all_score_dir}")

    # ── 保存 result.json ──
    result = {
        "pdb_id": pdb_id, "dataset": dataset, "n_atoms": n_atoms,
        "n_steps": n_steps, "nu": nu, "dt": dt, "dx": dx,
        "eps_range": [eps_start, eps_stop, eps_step],
        "thresholds": [round(t, 1) for t in thresholds],
        "evaluation": {"cv_folds": cv_folds, "random_state": random_state},
        "simulation_time_s": int(sim_time),
        "rf_time_stats_s": round(rf_time_stats_total, 1),
        "rf_time_trj_s": round(rf_time_trj_total, 1),
        "best_combined_stats": best_combined_stats,
        "best_combined_trj": best_combined_trj,
        "per_level_stats": {f"L{lvl:02d}": best_per_level_stats[lvl] for lvl in range(1, num_levels + 1)},
        "per_level_trj": {f"L{lvl:02d}": best_per_level_trj[lvl] for lvl in range(1, num_levels + 1)},
    }
    save_json(result, protein_out / "result.json")

    # ── 绘图（最优 eps, 统计量特征的最佳层） ──
    best_eps_str = best_combined_stats["eps"]
    best_eps = float(best_eps_str)
    best_lvl = max(range(1, num_levels + 1),
                   key=lambda l: best_per_level_stats[l].get("oof_pcc", -999))
    best_plot_model = best_per_level_stats[best_lvl].get("model", "RF")
    best_ei = next((i for i, e in enumerate(eps_list) if f"{e:.1f}" == best_eps_str), 0)
    if feat_cfg["stats"]["enabled"]:
        best_plot_model = str(all_model_stats[best_ei, best_lvl - 1])

    pulse_idx = n_atoms // 2

    A_best = aij_matrices[best_lvl - 1]
    trj_best = all_trj[(best_lvl, best_eps_str)]

    plot_ux(A_best, nu, best_eps, pdb_id, best_lvl, pulse_idx, dt, dx, n_steps,
            fig_dir / "ux.png", model=best_plot_model)
    plot_ut(trj_best, nu, best_eps, pdb_id, best_lvl, dt, max(1, n_steps // n_pts),
            fig_dir / "ut.png", model=best_plot_model)
    log.info(
        f"Plots saved: ux.png, ut.png "
        f"(level={best_lvl}, eps={best_eps}, model={best_plot_model})"
    )

    bc = best_combined_stats
    bt = best_combined_trj
    log.info(
        f"Done. Best stats: OOF={bc.get('oof_pcc', 0):.4f} "
        f"best_fold={bc.get('best_fold', 1)}({bc.get('best_fold_pcc', 0):.4f}) "
        f"mean_fold={bc.get('mean_fold_pcc', 0):.4f} "
        f"eps={bc.get('eps')} model={bc.get('model', 'RF')}"
    )
    log.info(
        f"Best trj: OOF={bt.get('oof_pcc', 0):.4f} "
        f"best_fold={bt.get('best_fold', 1)}({bt.get('best_fold_pcc', 0):.4f}) "
        f"mean_fold={bt.get('mean_fold_pcc', 0):.4f} "
        f"eps={bt.get('eps')} model={bt.get('model', 'RF')}"
    )
    log.info(f"Result saved: {protein_out / 'result.json'}")
    log.info(f"Result dir: {protein_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
