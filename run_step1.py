#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
Step1 入口：从 xyzb（或 PDB）文件生成 Aij、distance、binary 矩阵。

用法:
    # 从 xyzb 文件
    python run_step1.py --dataset 33small

    # 从 PDB 文件（自动提取 Cα 到 xyzb）
    python run_step1.py --dataset mydata --from-pdb

    # 自定义新数据集
    python run_step1.py --dataset new_set --data-dir /path/to/xyzb_files
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils import load_yaml, setup_logging, resolve_path
from src.step1_distance import (
    compute_filtration_thresholds,
    pairwise_distance_matrix,
    rips_filtered_distance,
    to_binary_matrix,
)
from src.step1_aij import compute_aij_matrix
from src.step1_io import load_xyzb


def parse_args():
    p = argparse.ArgumentParser(description="Step1: xyzb/PDB -> Aij/distance/binary")
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    p.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. 33small)")
    p.add_argument("--from-pdb", action="store_true",
                   help="Extract Cα atoms from protein/*.pdb instead of using xyzb files")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Override xyzb directory (default: code_data/<dataset>)")
    p.add_argument("--protein", type=str, default=None,
                   help="Process single protein only")
    return p.parse_args()


def extract_ca_from_pdb(pdb_path: Path) -> np.ndarray:
    """Extract Cα atoms from PDB, return (N, 4) array [x, y, z, bfactor]."""
    ca = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                b = float(line[60:66]) if len(line) > 66 else 0.0
                ca.append([x, y, z, b])
    return np.array(ca)


def process_protein(xyzb: np.ndarray, pdb_id: str, output_dir: Path, num_levels: int, aij_k: int,
                    pct_start: float, pct_stop: float):
    """处理单个蛋白。"""
    coords = xyzb[:, :3]
    labels = xyzb[:, 3]

    dist = pairwise_distance_matrix(coords)
    off = dist[dist > 0]
    thresholds = np.percentile(off, np.linspace(pct_start, pct_stop, num_levels))

    # 目录
    aij_dir = output_dir / pdb_id / "Aijandlabel"; aij_dir.mkdir(parents=True, exist_ok=True)
    dist_dir = output_dir / pdb_id / "distance"; dist_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = output_dir / pdb_id / "binary"; bin_dir.mkdir(parents=True, exist_ok=True)
    for d in ["all_score", "trajectory", "features/stats", "features/trj", "figures"]:
        (output_dir / pdb_id / d).mkdir(parents=True, exist_ok=True)

    np.save(dist_dir / f"{pdb_id}_dist.npy", dist)
    np.save(aij_dir / f"{pdb_id}_label.npy", labels)

    for level in range(1, num_levels + 1):
        thr = thresholds[level - 1]
        filtered = rips_filtered_distance(dist, thr)
        binary = to_binary_matrix(filtered)
        aij = compute_aij_matrix(filtered, k=aij_k)

        np.save(dist_dir / f"{pdb_id}_dist_0-{level}.npy", filtered)
        np.save(bin_dir / f"{pdb_id}_binary_0-{level}.npy", binary)
        np.save(aij_dir / f"{pdb_id}_Aij_0-{level}.npy", aij)

    print(f"  {pdb_id}: {len(labels)} atoms, thresholds=[{thresholds[0]:.1f}..{thresholds[-1]:.1f}]")


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    project_root = Path(__file__).resolve().parent
    dataset = args.dataset
    paths = cfg.get("paths", {})
    result_root = resolve_path(project_root, paths.get("output_root", project_root / "result"))
    pdb_root = resolve_path(project_root, paths.get("pdb_dir", project_root / "protein"))
    xyz_root = resolve_path(project_root, paths.get("code_data", project_root / "code_data"))
    num_levels = cfg["graph"]["num_levels"]
    aij_k = cfg["graph"]["aij_k"]
    pct_start = cfg["graph"]["percentile_start"]
    pct_stop = cfg["graph"]["percentile_stop"]

    output_dir = result_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_pdb:
        pdb_dir = pdb_root
        for pdb_file in sorted(pdb_dir.glob("*.pdb")):
            pdb_id = pdb_file.stem
            if args.protein and pdb_id.upper() != args.protein.upper():
                continue
            xyzb = extract_ca_from_pdb(pdb_file)
            if len(xyzb) == 0:
                print(f"  SKIP {pdb_id}: no Cα atoms")
                continue
            process_protein(xyzb, pdb_id, output_dir, num_levels, aij_k, pct_start, pct_stop)
    else:
        xyz_dir = args.data_dir or (xyz_root / dataset)
        if not xyz_dir.exists():
            print(f"ERROR: data dir not found: {xyz_dir}")
            return 1
        for f in sorted(xyz_dir.glob("*_ca.xyzb")):
            pdb_id = f.name.replace("_ca.xyzb", "")
            if args.protein and pdb_id.upper() != args.protein.upper():
                continue
            coords, labels = load_xyzb(f)
            xyzb_4col = np.column_stack([coords, labels])
            process_protein(xyzb_4col, pdb_id, output_dir, num_levels, aij_k, pct_start, pct_stop)

    print(f"\nDone. Output: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
