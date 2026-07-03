#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
scRNA 细胞图 Burgers 动力学模拟: 脉冲响应 → 轨迹 → 特征。

对每个 level，在固定耦合强度 ε 下:
  - 每个细胞施加单位脉冲 u(0)=e_i
  - RK4 演化 Burgers 方程, 记录 u_i(t) 衰减轨迹
  - n_steps 默认由 decay_threshold 自适应确定（上限 max_steps）
  - config slice_overrides 可 per-slice 固定 n_steps，跳过探测
  - 提取统计特征 (max, min, mean, var, median, std)

用法:
    python burgers_sim.py --slice GSE45719 --level 5          # 单层（SLURM 单 job）
    python burgers_sim.py --slice GSE45719 --aggregate       # 汇总 partial/ → trajectory/
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

from src.burgers import rk4_step, rk4_step_batch, precompute_laplacian
from src.features import extract_stats_features


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, path_value: str | Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else base / p


def eps_tag(eps: float) -> str:
    return f"{eps:.1f}".replace(".", "-")


def resolve_eps_list(cfg: dict) -> list[float]:
    """解析 ε 列表：优先 epsilon.value，兼容旧版 start/stop/step 扫描。"""
    eps_cfg = cfg.get("epsilon", {})
    if "value" in eps_cfg:
        return [round(float(eps_cfg["value"]), 1)]
    if "values" in eps_cfg:
        return [round(float(x), 1) for x in eps_cfg["values"]]
    start = float(eps_cfg["start"])
    stop = float(eps_cfg["stop"])
    step = float(eps_cfg["step"])
    return [round(float(x), 1) for x in np.arange(start, stop + step / 2, step)]


def resolve_graph_mode(cfg: dict) -> str:
    """Burgers 演化使用的 Aij 来源：pearson | euclidean | root。"""
    return str(cfg.get("graph", {}).get("sim_mode", "pearson"))


def graph_data_dir(data_dir: Path, slice_name: str, graph_mode: str) -> Path:
    """解析 slice 的 Aij/thresholds 所在子目录。"""
    base = data_dir / slice_name
    if graph_mode in ("pearson", "euclidean"):
        sub = base / graph_mode
        if sub.exists():
            return sub
    return base


def load_aij_matrix(data_dir: Path, slice_name: str, level: int,
                    graph_mode: str | None = None) -> np.ndarray:
    """加载 Aij 矩阵，自动解析 pearson/ 等子目录。"""
    if graph_mode is None:
        graph_mode = "root"
    aij_dir = graph_data_dir(data_dir, slice_name, graph_mode)
    path = aij_dir / f"{slice_name}_Aij_0-{level}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Aij not found: {path}")
    return np.load(path).astype(np.float64)


def load_aij_matrices(data_dir: Path, slice_name: str, num_levels: int,
                      graph_mode: str | None = None) -> list[np.ndarray]:
    return [load_aij_matrix(data_dir, slice_name, lvl, graph_mode)
            for lvl in range(1, num_levels + 1)]


def load_thresholds(data_dir: Path, slice_name: str,
                    graph_mode: str | None = None) -> tuple[Path, dict]:
    """加载 thresholds.json，返回 (路径, 内容)。"""
    if graph_mode is None:
        graph_mode = "root"
    aij_dir = graph_data_dir(data_dir, slice_name, graph_mode)
    thresholds_path = aij_dir / "thresholds.json"
    if not thresholds_path.exists():
        raise FileNotFoundError(f"thresholds.json not found: {thresholds_path}")
    with open(thresholds_path) as f:
        return thresholds_path, json.load(f)


def find_n_steps(A: np.ndarray, nu: float, dt: float, dx: float,
                 eps_ref: float, n_cells: int, decay_threshold=0.1,
                 max_steps=50000) -> int:
    """
    找出 eps_ref 下最慢细胞衰减到 threshold 所需的步数。

    预计算图拉普拉斯 L 并在所有细胞间复用，
    避免每次 rk4_step 内部重建（原始版本的主要瓶颈）。
    """
    L = precompute_laplacian(A)
    max_s = 0
    for i in range(n_cells):
        u = np.zeros(n_cells, dtype=np.float64)
        u[i] = 1.0
        for step in range(1, max_steps + 1):
            u = rk4_step(u, dt, A, nu, eps_ref, dx, "graph_diffusion", L_cache=L)
            if u[i] <= decay_threshold:
                if step > max_s:
                    max_s = step
                break
    return max_s


def resolve_slice_dynamics(cfg: dict, slice_name: str) -> dict:
    """合并全局 dynamics 与 slice_overrides（后者优先，不含 level_overrides）。"""
    dyn = dict(cfg.get("dynamics", {}))
    dyn.pop("level_overrides", None)
    overrides = cfg.get("slice_overrides", {}) or {}
    if slice_name in overrides:
        slice_dyn = dict(overrides[slice_name])
        slice_dyn.pop("level_overrides", None)
        dyn.update(slice_dyn)
    return dyn


def normalize_level_key(level: int | str) -> str:
    """统一层键：1 / '1' / 'L01' / 'l01' → '1'。"""
    if isinstance(level, str):
        s = level.strip().upper()
        if s.startswith("L"):
            return str(int(s[1:]))
        return str(int(s))
    return str(int(level))


def get_level_override(level_overrides: dict | None, level: int) -> dict:
    """从 level_overrides 中读取指定层的覆盖参数。"""
    if not level_overrides:
        return {}
    target = normalize_level_key(level)
    for key, value in level_overrides.items():
        if normalize_level_key(key) == target:
            return dict(value) if value else {}
    return {}


def resolve_level_dynamics(cfg: dict, slice_name: str, level: int) -> dict:
    """
    合并 slice 级动力学 + 层覆盖（后者优先）。

    层覆盖来源（优先级从低到高）:
      1. dynamics.level_overrides[Lxx]
      2. slice_overrides[slice].level_overrides[Lxx]
    """
    dyn = resolve_slice_dynamics(cfg, slice_name)
    dyn.update(get_level_override(cfg.get("dynamics", {}).get("level_overrides"), level))
    slice_cfg = (cfg.get("slice_overrides", {}) or {}).get(slice_name, {})
    dyn.update(get_level_override(slice_cfg.get("level_overrides"), level))
    return dyn


def write_n_steps_cache(out_dir: Path, n_steps: int, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "n_steps.json", "w") as f:
        json.dump({"n_steps": n_steps, **meta}, f, indent=2)


def resolve_n_steps(out_dir: Path, slice_name: str, ref_A: np.ndarray,
                    dyn: dict, n_cells: int, level: int | None = None) -> tuple[int, dict]:
    """
    确定演化步数。

    优先级:
      1. dyn.n_steps → 固定步数（可作 early_stop 上限），跳过 find_n_steps
      2. partial/L{level}/n_steps.json 或 slice/n_steps.json 缓存
      3. find_n_steps 自适应探测（原方法）
    """
    dt, dx, nu = dyn["dt"], dyn["dx"], dyn["nu"]
    eps_ref = float(dyn["n_steps_ref_eps"])
    decay_thr = float(dyn["decay_threshold"])
    max_steps = int(dyn.get("max_steps", 50000))

    cache_dir = partial_level_dir(out_dir, level) if level is not None else out_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    if dyn.get("n_steps") is not None:
        n_steps = int(dyn["n_steps"])
        meta = {
            "mode": "fixed",
            "slice": slice_name,
            "level": level,
            "max_steps": max_steps,
            "eps_ref": eps_ref,
            "decay_threshold": decay_thr,
        }
        write_n_steps_cache(cache_dir, n_steps, meta)
        return n_steps, meta

    cache = cache_dir / "n_steps.json"
    if cache.exists():
        with open(cache) as f:
            cached = json.load(f)
        if cached.get("mode") == "fixed":
            return int(cached["n_steps"]), cached
        if (cached.get("max_steps") == max_steps
                and cached.get("eps_ref") == eps_ref
                and cached.get("decay_threshold") == decay_thr):
            return int(cached["n_steps"]), cached

    n_steps = find_n_steps(ref_A, nu, dt, dx, eps_ref, n_cells, decay_thr,
                           max_steps=max_steps)
    n_steps = min(n_steps, max_steps)
    meta = {
        "mode": "adaptive",
        "slice": slice_name,
        "level": level,
        "max_steps": max_steps,
        "eps_ref": eps_ref,
        "decay_threshold": decay_thr,
    }
    write_n_steps_cache(cache_dir, n_steps, meta)
    return n_steps, meta


def get_or_compute_n_steps(out_dir: Path, ref_A: np.ndarray, nu: float,
                           dt: float, dx: float, eps_ref: float,
                           n_cells: int, decay_thr: float,
                           max_steps: int = 50000) -> int:
    """兼容旧调用：仅自适应探测（不含 slice_overrides）。"""
    cache = out_dir / "n_steps.json"
    if cache.exists():
        with open(cache) as f:
            cached = json.load(f)
        if cached.get("mode") == "fixed":
            return int(cached["n_steps"])
        if (cached.get("max_steps") == max_steps
                and cached.get("eps_ref") == eps_ref
                and cached.get("decay_threshold") == decay_thr):
            return int(cached["n_steps"])
    n_steps = find_n_steps(ref_A, nu, dt, dx, eps_ref, n_cells, decay_thr,
                           max_steps=max_steps)
    n_steps = min(n_steps, max_steps)
    write_n_steps_cache(out_dir, n_steps, {
        "mode": "adaptive",
        "max_steps": max_steps,
        "eps_ref": eps_ref,
        "decay_threshold": decay_thr,
    })
    return n_steps


def resample_traj_time(trj: np.ndarray, n_saved: int, n_target: int,
                       last_step: int) -> np.ndarray:
    """将提前停止后的不等长轨迹重采样为固定 n_target 个时间点。"""
    if n_saved <= 0:
        raise ValueError("n_saved must be positive")
    if n_saved == n_target:
        return trj[:, :n_saved].copy()
    src_x = np.linspace(0, last_step, n_saved)
    dst_x = np.linspace(0, last_step, n_target)
    out = np.empty((trj.shape[0], n_target), dtype=np.float64)
    for i in range(trj.shape[0]):
        out[i] = np.interp(dst_x, src_x, trj[i, :n_saved])
    return out


def pulse_self_responses(U: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """批量脉冲矩阵中各细胞对自身分量的响应 u_i(t)。"""
    return U[rows, cols]


def simulate_trajectories(A: np.ndarray, nu: float, epsilon: float,
                          dt: float, dx: float, n_steps: int,
                          n_cells: int, n_sample_points: int = 100,
                          coupling_mode: str = "graph_diffusion",
                          batch_size: int | None = None,
                          decay_threshold: float | None = None,
                          decay_fraction: float | None = None,
                          ) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    对每个细胞施加脉冲 u(0)=e_i，记录 u_i(t) 衰减轨迹，提取统计特征。

    优化策略（批量演化）：
      - 将若干细胞的脉冲初值堆成矩阵 U（每行一个 e_i），用 rk4_step_batch 一次演化，
        图耦合项 L @ u（n_cells 次 gemv）合并为一次 U @ L（gemm/dgemm），
        BLAS 效率远高于逐细胞串行，规模 n≈2000 时整体加速约一个数量级。
      - L 对称时 (U @ L)[i] == L @ U[i]，与单向量版数学等价（见 src/burgers.py）。
      - batch_size 控制每次同时演化的细胞数，用于在超大 slice 上限制内存
        （None/<=0 表示一次演化全部细胞）。
      - 预计算 L_cache：图拉普拉斯仅构建一次，所有 batch 和 RK4 子步复用。
      - boolean 采样掩码：O(1) 数组索引，避免 set.__contains__ 的 Python hash 开销。
      - early_stop_fraction 非空时：每步检查全部细胞脉冲自响应 u_i(t)<=decay_threshold
        的比例，达到阈值则提前停止并重采样轨迹。
    """
    early_stop = (
        decay_fraction is not None
        and decay_threshold is not None
    )
    if early_stop:
        decay_fraction = float(decay_fraction)
        decay_threshold = float(decay_threshold)
        if not (0.0 < decay_fraction <= 1.0):
            raise ValueError("early_stop_fraction 须在 (0, 1] 内")
        if decay_threshold <= 0:
            raise ValueError("decay_threshold 须 > 0")

    sample_steps = np.linspace(0, n_steps, n_sample_points, dtype=int)
    sample_mask = np.zeros(n_steps + 1, dtype=bool)
    sample_mask[sample_steps] = True
    L_cache = precompute_laplacian(A) if coupling_mode == "graph_diffusion" else None

    trj_buf = np.zeros((n_cells, n_sample_points), dtype=np.float64)

    if batch_size is None or batch_size <= 0:
        batch_size = n_cells

    batches: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for c0 in range(0, n_cells, batch_size):
        c1 = min(c0 + batch_size, n_cells)
        chunk = c1 - c0
        U = np.zeros((chunk, n_cells), dtype=np.float64)
        rows = np.arange(chunk)
        cols = np.arange(c0, c1)
        U[rows, cols] = 1.0
        batches.append((c0, c1, U, rows, cols))

    save_idx = 0
    actual_steps = n_steps
    frac_at_stop = 0.0
    stopped_early = False

    for step in range(n_steps + 1):
        if sample_mask[step]:
            for c0, c1, U, rows, cols in batches:
                trj_buf[c0:c1, save_idx] = pulse_self_responses(U, rows, cols)
            save_idx += 1

        if early_stop and step > 0:
            parts = [
                pulse_self_responses(U, rows, cols)
                for _, _, U, rows, cols in batches
            ]
            self_vals = np.concatenate(parts)
            frac_at_stop = float(np.mean(self_vals <= decay_threshold))
            if frac_at_stop >= decay_fraction:
                actual_steps = step
                stopped_early = True
                break

        if step < n_steps:
            new_batches = []
            for c0, c1, U, rows, cols in batches:
                U = rk4_step_batch(
                    U, dt, A, nu, epsilon, dx, coupling_mode, L_cache=L_cache,
                )
                new_batches.append((c0, c1, U, rows, cols))
            batches = new_batches

    if save_idx < n_sample_points:
        trj = resample_traj_time(trj_buf, save_idx, n_sample_points, actual_steps)
    else:
        trj = trj_buf

    sim_meta = {
        "actual_n_steps": int(actual_steps),
        "max_n_steps": int(n_steps),
        "early_stopped": stopped_early,
        "frac_at_stop": frac_at_stop,
        "decay_threshold": decay_threshold,
        "early_stop_fraction": decay_fraction,
    }
    return trj, extract_stats_features(trj), sim_meta


def resolve_early_stop(dyn: dict) -> tuple[float | None, float | None]:
    """解析演化中提前停止参数；early_stop_fraction 为 null 时关闭。"""
    thr = dyn.get("decay_threshold")
    frac = dyn.get("early_stop_fraction")
    if frac is None:
        return None, None
    if thr is None:
        raise ValueError("启用 early_stop_fraction 时必须设置 decay_threshold")
    return float(thr), float(frac)


def simulate_one_eps(args_tuple):
    (eps, A, nu, dt, dx, n_steps, n_cells, n_pts, coupling_mode,
     decay_threshold, decay_fraction) = args_tuple
    trj, stats, sim_meta = simulate_trajectories(
        A, nu, eps, dt, dx, n_steps, n_cells, n_pts, coupling_mode,
        decay_threshold=decay_threshold, decay_fraction=decay_fraction,
    )
    return eps, trj, stats, sim_meta


def simulate_one_task(args_tuple):
    """兼容旧接口: (lvl, eps, A, ...)."""
    lvl, eps, A, nu, dt, dx, n_steps, n_cells, n_pts, coupling_mode = args_tuple[:10]
    decay_threshold = args_tuple[10] if len(args_tuple) > 10 else None
    decay_fraction = args_tuple[11] if len(args_tuple) > 11 else None
    _, trj, stats, _ = simulate_one_eps(
        (eps, A, nu, dt, dx, n_steps, n_cells, n_pts, coupling_mode,
         decay_threshold, decay_fraction))
    return lvl, eps, trj, stats


def partial_level_dir(out_dir: Path, level: int) -> Path:
    return out_dir / "partial" / f"L{level:02d}"


def data_file_tag(eps: float, file_suffix: str = "") -> str:
    tag = eps_tag(eps)
    return f"{tag}_{file_suffix}" if file_suffix else tag


def process_level(slice_name: str, level: int, data_dir: Path,
                  output_dir: Path, cfg: dict,
                  epsilon: float | None = None,
                  n_steps_override: int | None = None,
                  file_suffix: str = "") -> None:
    """单层 Aij：固定 ε 演化，写入 partial/L{level}/."""
    graph_mode = resolve_graph_mode(cfg)
    try:
        thresholds_path, info = load_thresholds(data_dir, slice_name, graph_mode)
    except FileNotFoundError:
        print(f"  ✗ {slice_name}: thresholds.json 不存在 (graph_mode={graph_mode})")
        return

    n_cells = info["n_cells"]
    num_levels = info["num_levels"]
    if level < 1 or level > num_levels:
        raise ValueError(f"level 须在 1..{num_levels}, 得到 {level}")

    dyn = resolve_level_dynamics(cfg, slice_name, level)
    dt, dx, nu = dyn["dt"], dyn["dx"], dyn["nu"]
    decay_thr = float(dyn["decay_threshold"])
    decay_threshold, decay_fraction = resolve_early_stop(dyn)
    coupling_mode = dyn["coupling_mode"]
    eps_list = [round(float(epsilon), 1)] if epsilon is not None else resolve_eps_list(cfg)

    n_pts = cfg["features"]["trajectory"]["n_points"]
    n_jobs = cfg["parallel"]["n_jobs"]
    if n_jobs <= 0:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        n_jobs = int(slurm_cpus) if slurm_cpus else cpu_count()

    out_dir = output_dir / slice_name
    out_dir.mkdir(parents=True, exist_ok=True)

    A = load_aij_matrix(data_dir, slice_name, level, graph_mode)
    if n_steps_override is not None:
        n_steps = int(n_steps_override)
        steps_meta = {"mode": "override", "n_steps": n_steps}
    elif dyn.get("n_steps") is not None:
        n_steps, steps_meta = resolve_n_steps(
            out_dir, slice_name, A, dyn, n_cells, level=level)
    else:
        ref_lvl = min(5, num_levels)
        ref_A = load_aij_matrix(data_dir, slice_name, ref_lvl, graph_mode)
        n_steps, steps_meta = resolve_n_steps(
            out_dir, slice_name, ref_A, dyn, n_cells, level=level)

    pdir = partial_level_dir(out_dir, level)
    pdir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (eps, A, nu, dt, dx, n_steps, n_cells, n_pts, coupling_mode,
         decay_threshold, decay_fraction)
        for eps in eps_list
    ]
    n_tasks = len(tasks)
    eps_str = ",".join(f"{e:.1f}" for e in eps_list)
    mode_tag = steps_meta.get("mode", "adaptive")
    early_tag = (
        f" early_stop={decay_fraction}@{decay_threshold}"
        if decay_fraction is not None else ""
    )

    suffix_tag = f" suffix={file_suffix}" if file_suffix else ""
    print(f"  {slice_name} L{level:02d}: graph={graph_mode} n_cells={n_cells} "
          f"n_steps={n_steps} ({mode_tag}) eps=[{eps_str}] n_jobs={n_jobs}"
          f"{early_tag}{suffix_tag}",
          flush=True)

    t0 = time.time()
    eps_run_meta: dict[str, dict] = {}
    if n_tasks == 1:
        eps, trj, stats, sim_meta = simulate_one_eps(tasks[0])
        tag = data_file_tag(eps, file_suffix)
        np.save(pdir / f"trj_{tag}.npy", trj)
        np.save(pdir / f"stats_{tag}.npy", stats)
        eps_run_meta[tag] = sim_meta
        stop_msg = ""
        if sim_meta.get("early_stopped"):
            stop_msg = (
                f" early_stop@{sim_meta['actual_n_steps']}"
                f" frac={sim_meta['frac_at_stop']:.4f}"
            )
        print(f"    1/1 (100%) {time.time()-t0:.0f}s{stop_msg}", flush=True)
    else:
        with Pool(n_jobs) as pool:
            for i, (eps, trj, stats, sim_meta) in enumerate(
                    pool.imap_unordered(simulate_one_eps, tasks)):
                tag = data_file_tag(eps, file_suffix)
                np.save(pdir / f"trj_{tag}.npy", trj)
                np.save(pdir / f"stats_{tag}.npy", stats)
                eps_run_meta[tag] = sim_meta
                if (i + 1) % max(1, n_tasks // 5) == 0:
                    print(f"    {i+1}/{n_tasks} ({100*(i+1)/n_tasks:.0f}%) "
                          f"{time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    with open(pdir / "level_meta.json", "w") as f:
        json.dump({
            "slice": slice_name, "level": level, "n_cells": n_cells,
            "graph_mode": graph_mode,
            "n_eps": n_tasks, "epsilon": eps_list,
            "n_steps": n_steps, "n_steps_mode": steps_meta.get("mode", "adaptive"),
            "file_suffix": file_suffix or None,
            "decay_threshold": decay_thr,
            "early_stop_fraction": decay_fraction,
            "level_dynamics": {
                "decay_threshold": decay_thr,
                "early_stop_fraction": decay_fraction,
                "n_steps": n_steps,
                "max_steps": dyn.get("max_steps"),
            },
            "epsilon_run_meta": eps_run_meta,
            "elapsed_s": int(elapsed),
        }, f, indent=2)
    print(f"    ✓ L{level:02d} {elapsed:.0f}s → {pdir}")


def aggregate_slice(slice_name: str, output_dir: Path, cfg: dict) -> bool:
    """将 partial/L{lvl}/ 汇总为 trajectory/ 与 features/ 最终格式."""
    out_dir = output_dir / slice_name
    graph_mode = resolve_graph_mode(cfg)
    data_dir = Path(cfg["paths"]["output_dir"])
    try:
        _, info = load_thresholds(data_dir, slice_name, graph_mode)
    except FileNotFoundError:
        print(f"[SKIP] {slice_name}: 无 thresholds.json (graph_mode={graph_mode})")
        return False
    num_levels = info["num_levels"]
    n_cells = info["n_cells"]

    eps_list = resolve_eps_list(cfg)

    missing = []
    for eps in eps_list:
        tag = eps_tag(eps)
        for lvl in range(1, num_levels + 1):
            pdir = partial_level_dir(out_dir, lvl)
            if not (pdir / f"trj_{tag}.npy").exists():
                missing.append(f"L{lvl:02d}/trj_{tag}")

    if missing:
        print(f"[WARN] {slice_name}: 缺少 {len(missing)} 个 partial 文件"
              f" (例: {missing[:3]})")
        return False

    traj_dir = out_dir / "trajectory"
    feat_dir = out_dir / "features"
    traj_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    for eps in eps_list:
        tag = eps_tag(eps)
        trj_levels = np.stack([
            np.load(partial_level_dir(out_dir, lvl) / f"trj_{tag}.npy")
            for lvl in range(1, num_levels + 1)
        ])
        stats_levels = np.stack([
            np.load(partial_level_dir(out_dir, lvl) / f"stats_{tag}.npy")
            for lvl in range(1, num_levels + 1)
        ])
        np.save(traj_dir / f"{slice_name}_dyn_trj_{tag}.npy", trj_levels)
        np.save(feat_dir / f"{slice_name}_dyn_stats_{tag}.npy", stats_levels)

    n_steps = 0
    n_steps_mode = "adaptive"
    nsteps_file = out_dir / "n_steps.json"
    if nsteps_file.exists():
        with open(nsteps_file) as f:
            ns_data = json.load(f)
            n_steps = ns_data.get("n_steps", 0)
            n_steps_mode = ns_data.get("mode", "adaptive")

    dyn = resolve_slice_dynamics(cfg, slice_name)
    with open(out_dir / "sim_meta.json", "w") as f:
        json.dump({
            "slice": slice_name, "n_cells": n_cells, "n_levels": num_levels,
            "n_eps": len(eps_list),
            "epsilon": eps_list,
            "n_steps": n_steps,
            "n_steps_mode": n_steps_mode,
            "decay_threshold": dyn.get("decay_threshold", 0.1),
            "max_steps": dyn.get("max_steps", 50000),
            "dt": dyn["dt"], "dx": dyn["dx"],
            "nu": dyn["nu"], "coupling_mode": dyn["coupling_mode"],
            "n_pts": cfg["features"]["trajectory"]["n_points"],
            "layout": "partial_per_level",
        }, f, indent=2)

    print(f"  ✓ {slice_name}: {len(eps_list)} eps × {num_levels} levels → "
          f"{traj_dir.name}/ & {feat_dir.name}/")
    return True


def process_slice(slice_name: str, data_dir: Path, output_dir: Path,
                  cfg: dict, smoke: bool = False, level: int | None = None,
                  epsilon: float | None = None,
                  n_steps_override: int | None = None,
                  file_suffix: str = ""):
    """处理单个数据片（可指定单层）。"""
    if level is not None:
        process_level(
            slice_name, level, data_dir, output_dir, cfg, epsilon=epsilon,
            n_steps_override=n_steps_override, file_suffix=file_suffix,
        )
        return

    graph_mode = resolve_graph_mode(cfg)
    try:
        _, info = load_thresholds(data_dir, slice_name, graph_mode)
    except FileNotFoundError:
        print(f"  ✗ {slice_name}: thresholds.json 不存在 (graph_mode={graph_mode})")
        return

    n_cells = info["n_cells"]
    num_levels = info["num_levels"]

    dyn = resolve_slice_dynamics(cfg, slice_name)
    dt, dx, nu = dyn["dt"], dyn["dx"], dyn["nu"]
    decay_thr = dyn["decay_threshold"]
    coupling_mode = dyn["coupling_mode"]
    n_pts = cfg["features"]["trajectory"]["n_points"]

    out_dir = output_dir / slice_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        smoke_eps = cfg.get("smoke", {}).get("eps", resolve_eps_list(cfg)[0])
        smoke_lvl = min(cfg.get("smoke", {}).get("level", 5), num_levels)
        aij_matrices = load_aij_matrices(data_dir, slice_name, num_levels, graph_mode)
        smoke_dyn = resolve_level_dynamics(cfg, slice_name, smoke_lvl)
        n_steps, steps_meta = resolve_n_steps(
            out_dir, slice_name, aij_matrices[smoke_lvl - 1], smoke_dyn, n_cells,
            level=smoke_lvl)
        A = aij_matrices[smoke_lvl - 1]
        decay_threshold, decay_fraction = resolve_early_stop(smoke_dyn)
        print(f"  SMOKE: L{smoke_lvl:02d} eps={smoke_eps} n_steps={n_steps} "
              f"({steps_meta.get('mode', 'adaptive')})")
        trj, stats, sim_meta = simulate_trajectories(
            A, nu, smoke_eps, dt, dx, n_steps, n_cells, n_pts, coupling_mode,
            decay_threshold=decay_threshold, decay_fraction=decay_fraction,
        )
        traj_dir = out_dir / "trajectory"
        traj_dir.mkdir(exist_ok=True)
        tag = eps_tag(smoke_eps)
        np.save(traj_dir / f"{slice_name}_L{smoke_lvl:02d}_eps{tag}_trj.npy", trj)
        np.save(traj_dir / f"{slice_name}_L{smoke_lvl:02d}_eps{tag}_stats.npy", stats)
        print(f"    trj: {trj.shape}, stats: {stats.shape}, meta: {sim_meta}")
        return

    for lvl in range(1, num_levels + 1):
        process_level(slice_name, lvl, data_dir, output_dir, cfg)
    aggregate_slice(slice_name, output_dir, cfg)


def main():
    parser = argparse.ArgumentParser(description="Burgers simulation on scRNA graphs")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("config_graph.yaml"))
    parser.add_argument("--slice", type=str, help="Process single slice only")
    parser.add_argument("--level", type=int, help="只跑指定尺度层 (1-10)")
    parser.add_argument("--aggregate", action="store_true",
                        help="汇总 partial/ 为最终 trajectory/features")
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="仅跑指定耦合强度 ε（SLURM 单 job 单 ε）")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="覆盖 config 固定步数（消融实验）")
    parser.add_argument("--file-suffix", type=str, default="",
                        help="输出文件名后缀，如 ns2000 → stats_40-0_ns2000.npy")
    args = parser.parse_args()

    n_steps_override = args.n_steps
    if n_steps_override is None and os.environ.get("SCRNA_N_STEPS"):
        n_steps_override = int(os.environ["SCRNA_N_STEPS"])
    file_suffix = args.file_suffix or os.environ.get("SCRNA_FILE_SUFFIX", "")

    if args.level is not None and not args.slice:
        parser.error("--level 需要配合 --slice")

    cfg = load_yaml(args.config)
    base = Path(__file__).resolve().parent
    data_dir = resolve_path(base, cfg["paths"]["output_dir"])
    sim_dir = resolve_path(base, cfg["paths"].get("sim_dir", "burgers_sim"))
    cfg["paths"]["output_dir"] = str(data_dir)
    cfg["paths"]["sim_dir"] = str(sim_dir)

    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        return 1

    if args.aggregate:
        slices = [args.slice] if args.slice else cfg["slices"]
        ok = 0
        for s in slices:
            if aggregate_slice(s, sim_dir, cfg):
                ok += 1
        print(f"\n汇总完成: {ok}/{len(slices)} slices")
        return 0 if ok == len(slices) else 1

    slices = cfg["slices"]
    if args.slice:
        slices = [args.slice]

    sim_dir.mkdir(parents=True, exist_ok=True)
    smoke_tag = " [SMOKE]" if args.smoke else ""
    lvl_tag = f" L{args.level:02d}" if args.level else ""
    print(f"Burgers 模拟{smoke_tag}{lvl_tag}: {len(slices)} slices → {sim_dir}\n")

    graph_mode = resolve_graph_mode(cfg)
    for s in slices:
        slice_dir = data_dir / s
        if not slice_dir.exists():
            print(f"  ✗ {s}: 无数据目录")
            continue
        try:
            load_thresholds(data_dir, s, graph_mode)
        except FileNotFoundError:
            print(f"  ✗ {s}: 无 thresholds.json (graph_mode={graph_mode})")
            continue
        print(f"  {s}:", end=" ", flush=True) if not args.level else None
        process_slice(
            s, data_dir, sim_dir, cfg, args.smoke, level=args.level,
            epsilon=args.epsilon, n_steps_override=n_steps_override,
            file_suffix=file_suffix,
        )

    print(f"\n完成. 输出: {sim_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
