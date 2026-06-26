#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
scRNA 细胞图 Burgers 动力学模拟: 脉冲响应 → 轨迹 → 特征。

对每个 (level, epsilon) 组合:
  - 每个细胞施加单位脉冲 u(0)=e_i
  - RK4 演化 Burgers 方程, 记录 u_i(t) 衰减轨迹
  - 提取统计特征 (max, min, mean, var, median, std)

用法:
    python burgers_sim.py
    python burgers_sim.py --slice GSE84133_human1
    python burgers_sim.py --slice GSE84133_human1 --smoke
"""

import sys, os
from pathlib import Path
# 复用 Pcode 核心模块（必须在其他 src import 之前）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import yaml

from src.burgers import rk4_step
from src.features import extract_stats_features


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_aij_matrices(aij_dir: Path, slice_name: str, num_levels: int) -> list[np.ndarray]:
    """加载所有层级的 Aij 矩阵."""
    aij_list = []
    for lvl in range(1, num_levels + 1):
        path = aij_dir / f"{slice_name}_Aij_0-{lvl}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Aij not found: {path}")
        aij_list.append(np.load(path).astype(np.float64))
    return aij_list


def find_n_steps(A: np.ndarray, nu: float, dt: float, dx: float,
                 eps_ref: float, n_cells: int, decay_threshold=0.1,
                 max_steps=50000) -> int:
    """找出 eps_ref 下最慢原子衰减到 threshold 所需的步数."""
    max_s = 0
    for i in range(n_cells):
        u = np.zeros(n_cells, dtype=np.float64)
        u[i] = 1.0
        for step in range(1, max_steps + 1):
            u = rk4_step(u, dt, A, nu, eps_ref, dx, "graph_diffusion")
            if u[i] <= decay_threshold:
                if step > max_s:
                    max_s = step
                break
    return max_s


def simulate_trajectories(A: np.ndarray, nu: float, epsilon: float,
                          dt: float, dx: float, n_steps: int,
                          n_cells: int, n_sample_points: int = 100,
                          coupling_mode: str = "graph_diffusion",
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    对每个细胞 i, 以脉冲初值 u(0)=e_i 演化, 记录 u_i(t).
    返回 (trj, stats).
    """
    sample_steps = np.linspace(0, n_steps, n_sample_points, dtype=int)
    sample_set = set(sample_steps.tolist())

    trj = np.zeros((n_cells, n_sample_points), dtype=np.float64)

    for i in range(n_cells):
        u = np.zeros(n_cells, dtype=np.float64)
        u[i] = 1.0
        save_idx = 0

        for step in range(n_steps + 1):
            if step in sample_set:
                trj[i, save_idx] = u[i]
                save_idx += 1
            if step < n_steps:
                u = rk4_step(u, dt, A, nu, epsilon, dx, coupling_mode)

    return trj, extract_stats_features(trj)


def simulate_one_task(args_tuple):
    """单个 (lvl, eps) 模拟任务 (供 multiprocessing 使用)."""
    lvl, eps, A, nu, dt, dx, n_steps, n_cells, n_pts, coupling_mode = args_tuple
    trj, stats = simulate_trajectories(
        A, nu, eps, dt, dx, n_steps, n_cells, n_pts, coupling_mode)
    return lvl, eps, trj, stats


def process_slice(slice_name: str, data_dir: Path, output_dir: Path,
                  cfg: dict, smoke: bool = False):
    """处理单个数据片."""
    aij_dir = data_dir / slice_name
    thresholds_path = aij_dir / "thresholds.json"

    if not thresholds_path.exists():
        print(f"  ✗ {slice_name}: thresholds.json 不存在, 请先运行 build_graph.py")
        return

    with open(thresholds_path) as f:
        info = json.load(f)
    n_cells = info["n_cells"]
    num_levels = info["num_levels"]

    dyn = cfg["dynamics"]
    dt, dx, nu = dyn["dt"], dyn["dx"], dyn["nu"]
    eps_ref = dyn["n_steps_ref_eps"]
    decay_thr = dyn["decay_threshold"]
    coupling_mode = dyn["coupling_mode"]

    eps_start = cfg["epsilon"]["start"]
    eps_stop = cfg["epsilon"]["stop"]
    eps_step = cfg["epsilon"]["step"]
    eps_list = np.arange(eps_start, eps_stop + eps_step / 2, eps_step)

    n_pts = cfg["features"]["trajectory"]["n_points"]
    n_jobs = cfg["parallel"]["n_jobs"]
    if n_jobs <= 0:
        n_jobs = cpu_count()

    out_dir = output_dir / slice_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载所有 Aij
    aij_matrices = load_aij_matrices(aij_dir, slice_name, num_levels)

    # 确定演化步数 (基于中间层 L05 和 eps_ref)
    ref_lvl = min(5, num_levels)
    ref_A = aij_matrices[ref_lvl - 1]
    n_steps = find_n_steps(ref_A, nu, dt, dx, eps_ref, n_cells, decay_thr)

    if smoke:
        # 烟雾测试: 单层 × 单 epsilon
        smoke_eps = cfg.get("smoke", {}).get("eps", 0.5)
        smoke_lvl = min(cfg.get("smoke", {}).get("level", 5), num_levels)
        A = aij_matrices[smoke_lvl - 1]
        print(f"  SMOKE: L{smoke_lvl:02d} eps={smoke_eps} n_steps={n_steps}")

        trj, stats = simulate_trajectories(
            A, nu, smoke_eps, dt, dx, n_steps, n_cells, n_pts, coupling_mode)

        traj_dir = out_dir / "trajectory"
        traj_dir.mkdir(exist_ok=True)
        eps_tag = f"{smoke_eps:.1f}".replace(".", "-")
        np.save(traj_dir / f"{slice_name}_L{smoke_lvl:02d}_eps{eps_tag}_trj.npy", trj)
        np.save(traj_dir / f"{slice_name}_L{smoke_lvl:02d}_eps{eps_tag}_stats.npy", stats)
        print(f"    trj: {trj.shape}, stats: {stats.shape}")
        return

    # 完整扫描
    tasks = [(lvl, round(float(eps), 1), aij_matrices[lvl - 1],
              nu, dt, dx, n_steps, n_cells, n_pts, coupling_mode)
             for eps in eps_list for lvl in range(1, num_levels + 1)]
    n_tasks = len(tasks)

    print(f"    n_cells={n_cells} n_steps={n_steps} eps={eps_list[0]:.1f}..{eps_list[-1]:.1f} "
          f"({len(eps_list)} values) n_jobs={n_jobs}")

    all_trj = {}
    all_stats = {}
    t0 = time.time()

    with Pool(n_jobs) as pool:
        for i, (lvl, eps, trj, stats) in enumerate(pool.imap_unordered(simulate_one_task, tasks)):
            eps_str = f"{eps:.1f}"
            all_trj[(lvl, eps_str)] = trj
            all_stats[(lvl, eps_str)] = stats
            if (i + 1) % max(1, n_tasks // 10) == 0:
                pct = 100 * (i + 1) / n_tasks
                print(f"    {i+1}/{n_tasks} ({pct:.0f}%) {time.time()-t0:.0f}s", flush=True)

    sim_time = time.time() - t0

    # 保存轨迹和特征
    traj_dir = out_dir / "trajectory"
    feat_dir = out_dir / "features"
    traj_dir.mkdir(exist_ok=True)
    feat_dir.mkdir(exist_ok=True)

    for eps in eps_list:
        eps_str = f"{eps:.1f}"
        eps_tag = eps_str.replace(".", "-")

        trj_levels = np.stack([all_trj[(lvl, eps_str)]
                               for lvl in range(1, num_levels + 1)])
        np.save(traj_dir / f"{slice_name}_dyn_trj_{eps_tag}.npy", trj_levels)

        stats_levels = np.stack([all_stats[(lvl, eps_str)]
                                 for lvl in range(1, num_levels + 1)])
        np.save(feat_dir / f"{slice_name}_dyn_stats_{eps_tag}.npy", stats_levels)

    # 保存元数据
    with open(out_dir / "sim_meta.json", "w") as f:
        json.dump({
            "slice": slice_name, "n_cells": n_cells, "n_levels": num_levels,
            "n_eps": len(eps_list), "eps_range": [eps_start, eps_stop, eps_step],
            "n_steps": n_steps, "dt": dt, "dx": dx, "nu": nu,
            "coupling_mode": coupling_mode, "n_pts": n_pts,
            "simulation_time_s": int(sim_time),
        }, f, indent=2)

    print(f"    ✓ {sim_time:.0f}s, {len(eps_list)} eps × {num_levels} levels")


def main():
    parser = argparse.ArgumentParser(description="Burgers simulation on scRNA graphs")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("config_graph.yaml"))
    parser.add_argument("--slice", type=str, help="Process single slice only")
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    base = Path(__file__).resolve().parent
    data_dir = base / cfg["paths"]["output_dir"]  # data_aij
    sim_dir = base / cfg["paths"].get("sim_dir", "burgers_sim")

    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        return 1

    slices = cfg["slices"]
    if args.slice:
        slices = [args.slice]

    sim_dir.mkdir(parents=True, exist_ok=True)
    smoke_tag = " [SMOKE]" if args.smoke else ""
    print(f"Burgers 模拟{smoke_tag}: {len(slices)} slices → {sim_dir}\n")

    for s in slices:
        aij_dir = data_dir / s
        if not aij_dir.exists():
            print(f"  ✗ {s}: 无 Aij 数据")
            continue
        print(f"  {s}:", end=" ", flush=True)
        process_slice(s, data_dir, sim_dir, cfg, args.smoke)

    print(f"\n完成. 输出: {sim_dir}")


if __name__ == "__main__":
    main()
