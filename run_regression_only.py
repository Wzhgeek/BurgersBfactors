#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
临时脚本：从已保存的模拟结果补足回归阶段（跳过 Burgers 模拟）。

适用场景：run.py 在模拟完成后、回归阶段失败（如 2OLX 的 KNN 小样本错误）。

前提（蛋白输出目录下已有）：
  - trajectory/{PDB}_dyn_trj_{eps}.npy  （100 个 eps × 10 levels）
  - Aijandlabel/{PDB}_label.npy
  - config.yaml（可选，默认用项目 config.yaml）

用法:
    python run_regression_only.py --dataset 33small --protein 2OLX
    python run_regression_only.py --dataset 33small --protein 2OLX --sim-time 788
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import (
    _eval_meta,
    _level_result_entry,
    _oof_pcc,
    _pick_best_regressor,
    _primary_score,
    _save_model_csv,
    find_n_steps,
)
from src.features import extract_stats_features, load_aij_matrices
from src.plot import plot_ux, plot_ut
from src.regression import evaluate_regressor
from src.utils import load_yaml, pcc_10digit, resolve_path, save_json, setup_logging


def parse_args():
    p = argparse.ArgumentParser(description="Pcode regression-only resume")
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--protein", type=str, required=True)
    p.add_argument("--sim-time", type=float, default=None,
                   help="写入 result.json 的 simulation_time_s（默认从 run.log 解析）")
    return p.parse_args()


def _parse_sim_time_from_log(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    m = re.search(r"Simulation done in (\d+)s", text)
    return int(m.group(1)) if m else None


def load_saved_simulation(
    traj_dir: Path,
    pdb_id: str,
    eps_list: np.ndarray,
    num_levels: int,
) -> tuple[dict, dict]:
    """从 trajectory/*.npy 重建 run.py 回归阶段所需的特征字典。"""
    all_trj = {}
    all_stats_features = {}
    for eps in eps_list:
        eps_str = f"{eps:.1f}"
        eps_tag = eps_str.replace(".", "-")
        npy_path = traj_dir / f"{pdb_id}_dyn_trj_{eps_tag}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Missing trajectory file: {npy_path}")
        trj_levels = np.load(npy_path)
        if trj_levels.shape[0] != num_levels:
            raise ValueError(
                f"Bad shape in {npy_path}: expected {num_levels} levels, got {trj_levels.shape[0]}"
            )
        for lvl in range(1, num_levels + 1):
            trj = trj_levels[lvl - 1]
            all_trj[(lvl, eps_str)] = trj
            all_stats_features[(lvl, eps_str)] = extract_stats_features(trj)
    return all_trj, all_stats_features


def run_regression_phase(
    *,
    protein_out: Path,
    pdb_id: str,
    dataset: str,
    labels: np.ndarray,
    n_atoms: int,
    n_steps: int,
    nu: float,
    dt: float,
    dx: float,
    eps_list: np.ndarray,
    eps_start: float,
    eps_stop: float,
    eps_step: float,
    num_levels: int,
    feat_cfg: dict,
    reg_cfg: dict,
    use_cv: bool,
    test_size: float,
    random_state: int,
    cv_folds: int,
    thresholds: list,
    aij_matrices: list,
    all_trj: dict,
    all_stats_features: dict,
    sim_time: int,
    log: logging.Logger,
) -> int:
    """与 run.py 回归阶段逻辑一致，写出 all_score / result.json / figures。"""
    all_score_dir = protein_out / "all_score"
    fig_dir = protein_out / "figures"
    all_score_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting regression evaluation (resume)...")

    all_oof_stats = np.zeros((len(eps_list), num_levels))
    all_bestfold_pcc_stats = np.zeros((len(eps_list), num_levels))
    all_meanfold_stats = np.zeros((len(eps_list), num_levels))
    all_bestfold_stats = np.zeros((len(eps_list), num_levels), dtype=int)
    all_model_stats = np.empty((len(eps_list), num_levels), dtype=object)
    all_oof_trj = np.zeros((len(eps_list), num_levels))
    all_bestfold_pcc_trj = np.zeros((len(eps_list), num_levels))
    all_meanfold_trj = np.zeros((len(eps_list), num_levels))
    all_single_pcc_stats = np.zeros((len(eps_list), num_levels))
    all_single_pcc_trj = np.zeros((len(eps_list), num_levels))
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

    best_per_level_stats = {
        lvl: {"eps": 0, "oof_pcc": -999, "single_pcc": -999, "model": "RF", "best_fold": 1}
        for lvl in range(1, num_levels + 1)
    }
    best_per_level_trj = {
        lvl: {"eps": 0, "oof_pcc": -999, "single_pcc": -999, "model": "RF", "best_fold": 1}
        for lvl in range(1, num_levels + 1)
    }
    best_combined_stats = {"eps": 0, "oof_pcc": -999, "single_pcc": -999, "model": "RF", "best_fold": 1}
    best_combined_trj = {"eps": 0, "oof_pcc": -999, "single_pcc": -999, "model": "RF", "best_fold": 1}

    rf_time_stats_total = 0.0
    rf_time_trj_total = 0.0

    for ei, eps in enumerate(eps_list):
        eps_str = f"{eps:.1f}"

        if feat_cfg["stats"]["enabled"]:
            t1 = time.time()
            parts_s = [all_stats_features[(lvl, eps_str)] for lvl in range(1, num_levels + 1)]
            X_s = np.hstack(parts_s)
            for reg_name, reg_params in reg_cfg.items():
                if not reg_params.get("enabled", True):
                    continue
                result = evaluate_regressor(
                    reg_name, X_s, labels, reg_params,
                    test_size, random_state, cv_folds, use_cv=use_cv)
                if _primary_score(result, use_cv) > _primary_score(best_combined_stats, use_cv):
                    best_combined_stats = _level_result_entry(result, reg_name.upper(), eps_str, use_cv)
            rf_time_stats_total += time.time() - t1

        if feat_cfg["trajectory"]["enabled"]:
            t1 = time.time()
            parts_t = [all_trj[(lvl, eps_str)] for lvl in range(1, num_levels + 1)]
            X_t = np.hstack(parts_t)
            for reg_name, reg_params in reg_cfg.items():
                if not reg_params.get("enabled", True):
                    continue
                result = evaluate_regressor(
                    reg_name, X_t, labels, reg_params,
                    test_size, random_state, cv_folds, use_cv=use_cv)
                if _primary_score(result, use_cv) > _primary_score(best_combined_trj, use_cv):
                    best_combined_trj = _level_result_entry(result, reg_name.upper(), eps_str, use_cv)
            rf_time_trj_total += time.time() - t1

        for lvl in range(1, num_levels + 1):
            if feat_cfg["stats"]["enabled"]:
                s = all_stats_features[(lvl, eps_str)]
                _, best_cell_model, best_r = _pick_best_regressor(
                    reg_cfg, s, labels, test_size, random_state, cv_folds, use_cv)
                if best_r is None:
                    continue
                if use_cv:
                    all_oof_stats[ei, lvl - 1] = pcc_10digit(_oof_pcc(best_r))
                    all_bestfold_pcc_stats[ei, lvl - 1] = pcc_10digit(
                        best_r.get("best_fold_pcc", best_r["test_pcc"]))
                    all_meanfold_stats[ei, lvl - 1] = pcc_10digit(
                        best_r.get("mean_fold_pcc", best_r["test_pcc"]))
                    all_bestfold_stats[ei, lvl - 1] = int(best_r.get("best_fold_idx", 1))
                else:
                    all_single_pcc_stats[ei, lvl - 1] = pcc_10digit(_primary_score(best_r, use_cv))
                all_model_stats[ei, lvl - 1] = best_cell_model
                fold_detail_stats.setdefault(eps_str, {})[f"L{lvl:02d}"] = {
                    "model": best_cell_model,
                    "oof_pcc": best_r.get("oof_pcc", 0) if use_cv else 0,
                    "best_fold_pcc": best_r.get("best_fold_pcc", 0) if use_cv else 0,
                    "mean_fold_pcc": best_r.get("mean_fold_pcc", 0) if use_cv else 0,
                    "single_pcc": best_r.get("single_pcc", 0) if not use_cv else 0,
                    "best_fold": int(best_r.get("best_fold_idx", 0)),
                    "fold_val_pccs": best_r.get("fold_val_pccs", []),
                }
                entry = _level_result_entry(best_r, best_cell_model, eps_str, use_cv)
                if _primary_score(best_r, use_cv) > _primary_score(best_per_level_stats[lvl], use_cv):
                    best_per_level_stats[lvl] = entry

            if feat_cfg["trajectory"]["enabled"]:
                t = all_trj[(lvl, eps_str)]
                _, best_cell_model, best_r = _pick_best_regressor(
                    reg_cfg, t, labels, test_size, random_state, cv_folds, use_cv)
                if best_r is None:
                    continue
                if use_cv:
                    all_oof_trj[ei, lvl - 1] = pcc_10digit(_oof_pcc(best_r))
                    all_bestfold_pcc_trj[ei, lvl - 1] = pcc_10digit(
                        best_r.get("best_fold_pcc", best_r["test_pcc"]))
                    all_meanfold_trj[ei, lvl - 1] = pcc_10digit(
                        best_r.get("mean_fold_pcc", best_r["test_pcc"]))
                    all_bestfold_trj[ei, lvl - 1] = int(best_r.get("best_fold_idx", 1))
                else:
                    all_single_pcc_trj[ei, lvl - 1] = pcc_10digit(_primary_score(best_r, use_cv))
                all_model_trj[ei, lvl - 1] = best_cell_model
                fold_detail_trj.setdefault(eps_str, {})[f"L{lvl:02d}"] = {
                    "model": best_cell_model,
                    "oof_pcc": best_r.get("oof_pcc", 0) if use_cv else 0,
                    "best_fold_pcc": best_r.get("best_fold_pcc", 0) if use_cv else 0,
                    "mean_fold_pcc": best_r.get("mean_fold_pcc", 0) if use_cv else 0,
                    "single_pcc": best_r.get("single_pcc", 0) if not use_cv else 0,
                    "best_fold": int(best_r.get("best_fold_idx", 0)),
                    "fold_val_pccs": best_r.get("fold_val_pccs", []),
                }
                entry = _level_result_entry(best_r, best_cell_model, eps_str, use_cv)
                if _primary_score(best_r, use_cv) > _primary_score(best_per_level_trj[lvl], use_cv):
                    best_per_level_trj[lvl] = entry

        lvl6 = best_per_level_stats.get(6, {})
        if use_cv:
            log.info(
                f"eps={eps_str} done: stats "
                f"OOF={lvl6.get('oof_pcc', 0):.4f} "
                f"best_fold={lvl6.get('best_fold', 1)}"
                f"({lvl6.get('best_fold_pcc', 0):.4f}) "
                f"mean_fold={lvl6.get('mean_fold_pcc', 0):.4f} "
                f"model={lvl6.get('model', 'RF')}"
            )
        else:
            log.info(
                f"eps={eps_str} done: stats "
                f"single_pcc={lvl6.get('single_pcc', 0):.4f} "
                f"model={lvl6.get('model', 'RF')}"
            )

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
    if not use_cv:
        np.savetxt(all_score_dir / f"{pdb_id}_all_single_pcc_stats.csv", all_single_pcc_stats,
                   delimiter=",", header=header, comments="", fmt="%.10g")
        np.savetxt(all_score_dir / f"{pdb_id}_all_single_pcc_trj.csv", all_single_pcc_trj,
                   delimiter=",", header=header, comments="", fmt="%.10g")
    save_json({
        "use_cv": use_cv,
        "cv_folds": cv_folds,
        "stats": fold_detail_stats,
        "trj": fold_detail_trj,
    }, all_score_dir / f"{pdb_id}_fold_detail.json")
    log.info(f"Saved all_oof / all_bestfold_pcc / all_meanfold / fold_detail: {all_score_dir}")

    result = {
        "pdb_id": pdb_id,
        "dataset": dataset,
        "n_atoms": n_atoms,
        "n_steps": n_steps,
        "nu": nu,
        "dt": dt,
        "dx": dx,
        "eps_range": [eps_start, eps_stop, eps_step],
        "thresholds": [round(t, 1) for t in thresholds],
        "pipeline_mode": "regression_only",
        "evaluation": {
            "use_cv": use_cv,
            "cv_folds": cv_folds,
            "test_size": test_size,
            "random_state": random_state,
        },
        "simulation_time_s": int(sim_time),
        "regression_only": True,
        "rf_time_stats_s": round(rf_time_stats_total, 1),
        "rf_time_trj_s": round(rf_time_trj_total, 1),
        "best_combined_stats": best_combined_stats,
        "best_combined_trj": best_combined_trj,
        "per_level_stats": {
            f"L{lvl:02d}": best_per_level_stats[lvl] for lvl in range(1, num_levels + 1)
        },
        "per_level_trj": {
            f"L{lvl:02d}": best_per_level_trj[lvl] for lvl in range(1, num_levels + 1)
        },
    }
    save_json(result, protein_out / "result.json")

    n_pts = feat_cfg["trajectory"]["n_points"]
    best_eps_str = best_combined_stats["eps"]
    best_eps = float(best_eps_str)
    best_lvl = max(
        range(1, num_levels + 1),
        key=lambda l: _primary_score(best_per_level_stats[l], use_cv),
    )
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
    log.info(f"Result saved: {protein_out / 'result.json'}")
    return 0


def main():
    args = parse_args()
    saved_cfg = None
    pdb_id = args.protein.strip().upper()
    dataset = args.dataset

    project_root = Path(__file__).resolve().parent
    cfg = load_yaml(args.config)
    paths = cfg.get("paths", {})
    result_root = resolve_path(project_root, paths.get("output_root", project_root / "result"))
    step1_root = resolve_path(project_root, paths.get("step1_result", result_root))
    protein_out = result_root / dataset / pdb_id
    if not protein_out.exists():
        print(f"ERROR: output dir not found: {protein_out}", file=sys.stderr)
        return 1

    saved_cfg_path = protein_out / "config.yaml"
    if saved_cfg_path.exists():
        saved_cfg = load_yaml(saved_cfg_path)
        cfg = saved_cfg

    log = setup_logging(
        protein_out / "run_regression_only.log",
        level=getattr(logging, cfg["logging"]["level"]),
    )

    dyn = cfg["dynamics"]
    dt, dx, nu = dyn["dt"], dyn["dx"], dyn["nu"]
    eps_ref = dyn["n_steps_ref_eps"]
    decay_thr = dyn["decay_threshold"]
    eps_start = cfg["epsilon"]["start"]
    eps_stop = cfg["epsilon"]["stop"]
    eps_step = cfg["epsilon"]["step"]
    eps_list = np.arange(eps_start, eps_stop + eps_step / 2, eps_step)
    graph_cfg = cfg["graph"]
    num_levels = graph_cfg["num_levels"]
    feat_cfg = cfg["features"]
    eval_cfg = cfg["evaluation"]
    use_cv, cv_folds, test_size, random_state = _eval_meta(eval_cfg)
    reg_cfg = cfg["regressors"]

    step1_dir = step1_root / dataset
    label_path = step1_dir / pdb_id / "Aijandlabel" / f"{pdb_id}_label.npy"
    if not label_path.exists():
        label_path = protein_out / "Aijandlabel" / f"{pdb_id}_label.npy"
    if not label_path.exists():
        log.error(f"Label not found: {label_path}")
        return 1

    labels = np.load(label_path).astype(np.float64)
    n_atoms = len(labels)

    dist_path = step1_dir / pdb_id / "distance" / f"{pdb_id}_dist.npy"
    if not dist_path.exists():
        dist_path = protein_out / "distance" / f"{pdb_id}_dist.npy"
    dist = np.load(dist_path)
    off = dist[dist > 0]
    thresholds = np.percentile(
        off,
        np.linspace(graph_cfg["percentile_start"], graph_cfg["percentile_stop"], num_levels),
    ).tolist()

    ref_lvl = min(5, num_levels)
    ref_A_path = step1_dir / pdb_id / "Aijandlabel" / f"{pdb_id}_Aij_0-{ref_lvl}.npy"
    if not ref_A_path.exists():
        ref_A_path = protein_out / "Aijandlabel" / f"{pdb_id}_Aij_0-{ref_lvl}.npy"
    ref_A = np.load(ref_A_path)
    n_steps = find_n_steps(ref_A, nu, dt, dx, eps_ref, n_atoms, decay_thr)

    try:
        aij_matrices = load_aij_matrices(step1_dir, pdb_id, num_levels)
    except FileNotFoundError:
        local_aij = protein_out / "Aijandlabel"
        aij_matrices = []
        for level in range(1, num_levels + 1):
            p = local_aij / f"{pdb_id}_Aij_0-{level}.npy"
            if not p.exists():
                log.error(f"Aij not found: {p}")
                return 1
            aij_matrices.append(np.load(p))

    traj_dir = protein_out / "trajectory"
    log.info(f"Loading saved trajectories from {traj_dir}")
    all_trj, all_stats_features = load_saved_simulation(traj_dir, pdb_id, eps_list, num_levels)
    log.info(f"Loaded {len(eps_list)} eps × {num_levels} levels for {pdb_id} (atoms={n_atoms})")

    sim_time = args.sim_time
    if sim_time is None:
        sim_time = _parse_sim_time_from_log(protein_out / "run.log") or 0

    return run_regression_phase(
        protein_out=protein_out,
        pdb_id=pdb_id,
        dataset=dataset,
        labels=labels,
        n_atoms=n_atoms,
        n_steps=n_steps,
        nu=nu,
        dt=dt,
        dx=dx,
        eps_list=eps_list,
        eps_start=eps_start,
        eps_stop=eps_stop,
        eps_step=eps_step,
        num_levels=num_levels,
        feat_cfg=feat_cfg,
        reg_cfg=reg_cfg,
        use_cv=use_cv,
        test_size=test_size,
        random_state=random_state,
        cv_folds=cv_folds,
        thresholds=thresholds,
        aij_matrices=aij_matrices,
        all_trj=all_trj,
        all_stats_features=all_stats_features,
        sim_time=int(sim_time),
        log=log,
    )


if __name__ == "__main__":
    sys.exit(main())
