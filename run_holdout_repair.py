#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
小样本蛋白 hold-out 修补：重算 BestFold_PCC / MeanFold_PCC，保留 OOF_PCC 不变。

适用：33small 等数据集中 CV 折内样本过少导致 fold PCC 为 0 或负值。

前提：蛋白目录下已有 trajectory/*.npy 与 result.json。

用法:
    python run_holdout_repair.py --dataset 33small --protein 2OLX
    python run_holdout_repair.py --dataset 33small --auto
    python run_holdout_repair.py --dataset 33small --protein 1XY2 1YJO 2OLX
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.features import extract_stats_features
from src.regression import evaluate_regressor
from src.utils import load_yaml, pcc_10digit, resolve_path, save_json, setup_logging

# 33small 已知异常蛋白（可被 --auto 扫描扩展）
DEFAULT_REPAIR_33SMALL = (
    "1ETM", "1ETN", "1NOT", "1PEF", "1XY2", "1YJO", "2OL9", "2OLX",
)


def parse_args():
    p = argparse.ArgumentParser(description="Hold-out repair for small-sample fold PCC")
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--protein", type=str, nargs="*", default=[],
                   help="One or more PDB IDs; omit when using --auto")
    p.add_argument("--auto", action="store_true",
                   help="Scan dataset and repair all proteins with unreliable fold PCC")
    p.add_argument("--default-list", action="store_true",
                   help="Use built-in 33small repair list (with --dataset 33small)")
    p.add_argument("--list-only", action="store_true",
                   help="Print protein IDs to repair and exit (for batch submit)")
    return p.parse_args()


def effective_holdout_test_size(n_samples: int, test_size: float = 0.2) -> float:
    """保证测试集至少 2 个样本；样本够多时优先 3 个测试点以减少 ±1 极端值。"""
    if n_samples < 3:
        raise ValueError(f"Need at least 3 samples for hold-out repair, got {n_samples}")
    n_test = max(2, int(round(n_samples * test_size)))
    if n_samples >= 9:
        n_test = max(3, n_test)
    elif n_samples >= 6:
        n_test = max(3, n_test)
    n_test = min(n_test, n_samples - 2)  # 训练集至少 2 个
    n_test = max(2, n_test)
    return n_test / n_samples


def entry_needs_repair(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    folds = entry.get("fold_val_pccs") or []
    if not folds:
        return True
    if all(float(f) == 0.0 for f in folds):
        return True
    if float(entry.get("mean_fold_pcc", 0)) < 0:
        return True
    if any(float(f) < 0 for f in folds):
        return True
    return False


def result_needs_repair(data: dict) -> bool:
    for key in ("per_level_stats", "per_level_trj", "best_combined_stats", "best_combined_trj"):
        block = data.get(key)
        if not isinstance(block, dict):
            continue
        if key.startswith("per_level"):
            if any(entry_needs_repair(v) for v in block.values()):
                return True
        elif entry_needs_repair(block):
            return True
    return False


def _model_reg_params(reg_cfg: dict, model_name: str) -> tuple[str, dict]:
    key = model_name.strip().lower()
    if key not in reg_cfg:
        raise KeyError(f"Unknown model '{model_name}' in reg_cfg")
    params = dict(reg_cfg[key])
    if not params.get("enabled", True):
        raise ValueError(f"Regressor '{key}' is disabled in config")
    return key, params


def _parse_level(lvl_key: str) -> int:
    return int(lvl_key.replace("L", "").lstrip("0") or "0")


def load_level_features(
    traj_dir: Path,
    pdb_id: str,
    eps_str: str,
    lvl: int,
    cache: dict,
) -> tuple[np.ndarray, np.ndarray]:
    key = (eps_str, lvl)
    if key in cache:
        return cache[key]
    eps_tag = eps_str.replace(".", "-")
    npy_path = traj_dir / f"{pdb_id}_dyn_trj_{eps_tag}.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Missing trajectory: {npy_path}")
    trj_levels = np.load(npy_path)
    trj = trj_levels[lvl - 1]
    stats = extract_stats_features(trj)
    cache[key] = (trj, stats)
    return trj, stats


def _holdout_pcc(
    feature: str,
    reg_cfg: dict,
    model_name: str,
    trj: np.ndarray,
    stats: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    random_state: int,
) -> float:
    X = stats if feature == "stats" else trj
    reg_key, params = _model_reg_params(reg_cfg, model_name)
    result = evaluate_regressor(
        reg_key, X, labels, params,
        test_size=test_size,
        random_state=random_state,
        cv_folds=0,
    )
    return float(result["test_pcc"])


def _patch_entry(entry: dict, holdout_pcc: float) -> dict:
    pcc = pcc_10digit(holdout_pcc)
    entry["best_fold_pcc"] = pcc
    entry["mean_fold_pcc"] = pcc
    entry["best_fold"] = 1
    entry["fold_val_pccs"] = [pcc]
    entry["holdout_repair_pcc"] = pcc
    #  deliberately keep oof_pcc / eps / model unchanged
    return entry


def repair_level_block(
    block: dict,
    feature: str,
    traj_dir: Path,
    pdb_id: str,
    labels: np.ndarray,
    reg_cfg: dict,
    test_size: float,
    random_state: int,
    log: logging.Logger,
    cache: dict,
) -> int:
    n_fixed = 0
    for lvl_key, entry in block.items():
        if not isinstance(entry, dict):
            continue
        if not entry_needs_repair(entry):
            continue
        lvl = _parse_level(lvl_key)
        eps_str = str(entry.get("eps", ""))
        model = str(entry.get("model", "RF"))
        trj, stats = load_level_features(traj_dir, pdb_id, eps_str, lvl, cache)
        pcc = _holdout_pcc(
            feature, reg_cfg, model, trj, stats, labels,
            test_size, random_state,
        )
        _patch_entry(entry, pcc)
        n_fixed += 1
        log.info(
            f"  {feature} {lvl_key} eps={eps_str} model={model}: "
            f"holdout_pcc={pcc:.4f} (OOF kept {entry.get('oof_pcc')})"
        )
    return n_fixed


def repair_combined_block(
    entry: dict,
    feature: str,
    traj_dir: Path,
    pdb_id: str,
    num_levels: int,
    labels: np.ndarray,
    reg_cfg: dict,
    test_size: float,
    random_state: int,
    log: logging.Logger,
    cache: dict,
) -> bool:
    if not entry_needs_repair(entry):
        return False
    eps_str = str(entry.get("eps", ""))
    model = str(entry.get("model", "RF"))
    parts = []
    for lvl in range(1, num_levels + 1):
        trj, stats = load_level_features(traj_dir, pdb_id, eps_str, lvl, cache)
        parts.append(stats if feature == "stats" else trj)
    X = np.hstack(parts)
    reg_key, params = _model_reg_params(reg_cfg, model)
    result = evaluate_regressor(
        reg_key, X, labels, params,
        test_size=test_size,
        random_state=random_state,
        cv_folds=0,
    )
    pcc = float(result["test_pcc"])
    _patch_entry(entry, pcc)
    log.info(f"  combined {feature} eps={eps_str} model={model}: holdout_pcc={pcc:.4f}")
    return True


def repair_protein(
    dataset: str,
    pdb_id: str,
    result_root: Path,
    cfg: dict,
    log: logging.Logger,
) -> int:
    pdb_id = pdb_id.strip().upper()
    protein_out = result_root / dataset / pdb_id
    result_path = protein_out / "result.json"
    if not result_path.exists():
        log.error(f"No result.json: {result_path}")
        return 1

    traj_dir = protein_out / "trajectory"
    if not traj_dir.is_dir():
        log.error(f"No trajectory dir: {traj_dir}")
        return 1

    step1_root = resolve_path(
        Path(__file__).resolve().parent,
        cfg.get("paths", {}).get("step1_result", cfg.get("paths", {}).get("output_root")),
    )
    label_path = step1_root / dataset / pdb_id / "Aijandlabel" / f"{pdb_id}_label.npy"
    if not label_path.exists():
        label_path = protein_out / "Aijandlabel" / f"{pdb_id}_label.npy"
    labels = np.load(label_path).astype(np.float64)
    n_atoms = len(labels)

    eval_cfg = cfg["evaluation"]
    random_state = int(eval_cfg["random_state"])
    base_test = float(eval_cfg.get("test_size", 0.2))
    test_size = effective_holdout_test_size(n_atoms, base_test)
    num_levels = int(cfg["graph"]["num_levels"])
    reg_cfg = cfg["regressors"]

    with open(result_path) as f:
        data = json.load(f)

    if not result_needs_repair(data):
        log.info(f"{pdb_id}: no repair needed, skip")
        return 0

    log.info(f"{pdb_id}: n_atoms={n_atoms} holdout test_size={test_size:.3f}")
    cache: dict = {}
    n_fixed = 0
    n_fixed += repair_level_block(
        data.get("per_level_stats", {}), "stats",
        traj_dir, pdb_id, labels, reg_cfg, test_size, random_state, log, cache,
    )
    n_fixed += repair_level_block(
        data.get("per_level_trj", {}), "trj",
        traj_dir, pdb_id, labels, reg_cfg, test_size, random_state, log, cache,
    )
    if isinstance(data.get("best_combined_stats"), dict):
        if repair_combined_block(
            data["best_combined_stats"], "stats",
            traj_dir, pdb_id, num_levels, labels, reg_cfg,
            test_size, random_state, log, cache,
        ):
            n_fixed += 1
    if isinstance(data.get("best_combined_trj"), dict):
        if repair_combined_block(
            data["best_combined_trj"], "trj",
            traj_dir, pdb_id, num_levels, labels, reg_cfg,
            test_size, random_state, log, cache,
        ):
            n_fixed += 1

    data["holdout_repair"] = {
        "applied": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_entries_patched": n_fixed,
        "holdout_test_size": test_size,
        "note": "BestFold_PCC and MeanFold_PCC from single hold-out; OOF_PCC unchanged",
    }
    save_json(data, result_path)
    log.info(f"{pdb_id}: patched {n_fixed} entries -> {result_path}")
    return 0


def list_auto_repair(dataset: str, result_root: Path) -> list[str]:
    base = result_root / dataset
    found = []
    for protein_dir in sorted(base.iterdir()):
        if not protein_dir.is_dir():
            continue
        rp = protein_dir / "result.json"
        if not rp.exists():
            continue
        with open(rp) as f:
            data = json.load(f)
        if result_needs_repair(data):
            found.append(protein_dir.name)
    return found


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    cfg = load_yaml(args.config)
    paths = cfg.get("paths", {})
    result_root = resolve_path(project_root, paths.get("output_root", project_root / "result"))

    proteins: list[str] = [p.upper() for p in args.protein]
    if args.auto:
        proteins = list_auto_repair(args.dataset, result_root)
        if not args.list_only:
            print(f"Auto-detected {len(proteins)} protein(s): {', '.join(proteins)}")
    elif args.default_list and args.dataset == "33small" and not proteins:
        proteins = list(DEFAULT_REPAIR_33SMALL)
        if not args.list_only:
            print(f"Using default 33small list: {', '.join(proteins)}")

    if args.list_only:
        for p in proteins:
            print(p)
        return 0 if proteins else 1

    if not proteins:
        print("ERROR: no proteins to repair (use --protein, --auto, or --default-list)", file=sys.stderr)
        return 1

    log = setup_logging(
        result_root / args.dataset / "holdout_repair.log",
        level=getattr(logging, cfg["logging"]["level"]),
    )

    rc = 0
    for pdb in proteins:
        log.info(f"=== Repair {args.dataset}/{pdb} ===")
        if repair_protein(args.dataset, pdb, result_root, cfg, log) != 0:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
