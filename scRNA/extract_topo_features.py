#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
从 scRNA 多层 Aij 提取拓扑谱特征（10 level × 6 feature，扰动实验）。

每个细胞: 60 维 = L01..L10 各 6 维拉普拉斯谱差分特征。
输出: scratch/topo_features/{slice}/{slice}_topo.npy + meta.json

用法:
    python -m scRNA.extract_topo_features
    python -m scRNA.extract_topo_features --slice GSE45719
    python -m scRNA.extract_topo_features --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.topo_features import (
    TOPO_FEATURE_NAMES,
    perturbation_from_aij,
    resolve_n_jobs,
    stack_level_features,
)
from scRNA.burgers_sim import graph_data_dir, load_aij_matrix, load_thresholds
from scRNA.classify_traj_stats import load_n_cells, load_yaml, resolve_path


def default_slices(cfg: dict) -> list[str]:
    """13 个主流程 slice（排除 evolution.exclude）。"""
    exclude = set(cfg.get("evolution", {}).get("exclude", []))
    return [s for s in cfg["slices"] if s not in exclude]


def resolve_aij_dir(cfg: dict, scrna_dir: Path, aij_dir: str | Path | None = None) -> Path:
    """Aij 根目录；PC3 等实验用独立 config/output_dir，勿与 PC30 混用。"""
    if aij_dir is not None:
        p = Path(aij_dir)
        return p if p.is_absolute() else scrna_dir / p
    raw = cfg["paths"]["output_dir"]
    p = Path(raw)
    return p if p.is_absolute() else scrna_dir / p


def topo_output_dir(
    cfg: dict,
    scrna_dir: Path,
    slice_name: str,
    output_tag: str | None = None,
) -> Path:
    paths = cfg.get("paths", {})
    root = paths.get("topo_features_dir")
    if root:
        base = Path(root)
    else:
        base = resolve_path(scrna_dir, paths["scratch_root"]) / "topo_features"
    tag = output_tag if output_tag is not None else paths.get("topo_features_tag")
    if tag:
        return base / tag / slice_name
    return base / slice_name


def column_names(num_levels: int) -> list[str]:
    return [
        f"{name}_L{lv:02d}"
        for lv in range(1, num_levels + 1)
        for name in TOPO_FEATURE_NAMES
    ]


def split_parallel_workers(n_cells: int, n_jobs: int, num_levels: int) -> tuple[int, int]:
    """(层并行数, 每层细胞并行数)；二者仅一侧 >1，避免嵌套进程池。"""
    if n_jobs <= 1:
        return 1, 1
    if n_cells <= 600:
        return min(num_levels, n_jobs), 1
    return 1, n_jobs


def _extract_level(
    lvl: int,
    slice_name: str,
    aij_dir: Path,
    graph_mode: str,
    n_cells: int,
    cell_workers: int,
) -> tuple[int, np.ndarray]:
    aij = load_aij_matrix(aij_dir, slice_name, lvl, graph_mode)
    if aij.shape[0] != n_cells:
        raise ValueError(
            f"{slice_name} L{lvl:02d}: Aij {aij.shape[0]} != n_cells {n_cells}"
        )
    feat = perturbation_from_aij(aij, n_jobs=cell_workers)
    return lvl, feat


def extract_level_only(
    slice_name: str,
    level: int,
    cfg: dict,
    scrna_dir: Path,
    graph_mode: str | None = None,
    n_jobs: int | None = None,
    output_tag: str | None = None,
    aij_dir: Path | None = None,
) -> dict:
    """提取单层拓扑特征 → {slice}_topo_L{level:02d}.npy。"""
    graph_mode = graph_mode or str(cfg.get("graph", {}).get("sim_mode", "pearson"))
    aij_dir = aij_dir or resolve_aij_dir(cfg, scrna_dir)
    out_dir = topo_output_dir(cfg, scrna_dir, slice_name, output_tag=output_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cells = load_n_cells(aij_dir, slice_name, graph_mode)
    cell_workers = resolve_n_jobs(n_jobs)
    t0 = time.time()
    _, feat = _extract_level(
        level, slice_name, aij_dir, graph_mode, n_cells, cell_workers,
    )
    out_path = out_dir / f"{slice_name}_topo_L{level:02d}.npy"
    np.save(out_path, feat)
    elapsed = round(time.time() - t0, 2)
    print(
        f"  L{level:02d}: shape={feat.shape} "
        f"range=[{feat.min():.4f}, {feat.max():.4f}] ({elapsed}s)",
        flush=True,
    )
    return {
        "slice": slice_name,
        "level": level,
        "n_cells": int(n_cells),
        "shape": list(feat.shape),
        "out_file": str(out_path),
        "elapsed_sec": elapsed,
        "cell_workers": cell_workers,
    }


def merge_slice_topo(
    slice_name: str,
    cfg: dict,
    scrna_dir: Path,
    graph_mode: str | None = None,
    num_levels: int | None = None,
    output_tag: str | None = None,
    aij_dir: Path | None = None,
) -> dict:
    """将已提取的 L01..L10 单层文件合并为 {slice}_topo.npy + meta.json。"""
    graph_mode = graph_mode or str(cfg.get("graph", {}).get("sim_mode", "pearson"))
    num_levels = num_levels or int(cfg["graph"]["num_levels"])
    aij_dir = aij_dir or resolve_aij_dir(cfg, scrna_dir)
    out_dir = topo_output_dir(cfg, scrna_dir, slice_name, output_tag=output_tag)
    effective_tag = output_tag if output_tag is not None else cfg.get("paths", {}).get("topo_features_tag")

    level_parts: list[np.ndarray] = []
    for lvl in range(1, num_levels + 1):
        p = out_dir / f"{slice_name}_topo_L{lvl:02d}.npy"
        if not p.exists():
            raise FileNotFoundError(f"缺少单层拓扑: {p}")
        level_parts.append(np.load(p))

    topo_all = stack_level_features(level_parts)
    topo_path = out_dir / f"{slice_name}_topo.npy"
    np.save(topo_path, topo_all)

    n_cells = load_n_cells(aij_dir, slice_name, graph_mode)
    _, thr_meta = load_thresholds(aij_dir, slice_name, graph_mode)
    meta = {
        "slice": slice_name,
        "n_cells": int(n_cells),
        "n_features": int(topo_all.shape[1]),
        "num_levels": num_levels,
        "feature_names": TOPO_FEATURE_NAMES,
        "column_names": column_names(num_levels),
        "graph_mode": graph_mode,
        "output_tag": effective_tag,
        "thresholds": thr_meta.get("thresholds"),
        "topo_file": str(topo_path),
        "level_files": [
            str(out_dir / f"{slice_name}_topo_L{lvl:02d}.npy")
            for lvl in range(1, num_levels + 1)
        ],
    }
    meta_path = out_dir / f"{slice_name}_topo_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  merge => {topo_path} shape={topo_all.shape}", flush=True)
    return meta


def extract_slice_topo(
    slice_name: str,
    cfg: dict,
    scrna_dir: Path,
    graph_mode: str | None = None,
    num_levels: int | None = None,
    n_jobs: int | None = None,
    output_tag: str | None = None,
    aij_dir: Path | None = None,
) -> dict:
    graph_mode = graph_mode or str(cfg.get("graph", {}).get("sim_mode", "pearson"))
    num_levels = num_levels or int(cfg["graph"]["num_levels"])
    aij_dir = aij_dir or resolve_aij_dir(cfg, scrna_dir)
    out_dir = topo_output_dir(cfg, scrna_dir, slice_name, output_tag=output_tag)
    effective_tag = output_tag if output_tag is not None else cfg.get("paths", {}).get("topo_features_tag")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cells = load_n_cells(aij_dir, slice_name, graph_mode)
    total_jobs = resolve_n_jobs(n_jobs)
    level_workers, cell_workers = split_parallel_workers(
        n_cells, total_jobs, num_levels,
    )
    print(
        f"  n_cells={n_cells}  parallel: levels={level_workers} "
        f"cells/level={cell_workers}  (total_cpus={total_jobs})",
        flush=True,
    )
    level_parts: list[np.ndarray] = [None] * num_levels  # type: ignore
    t0 = time.time()

    task_fn = partial(
        _extract_level,
        slice_name=slice_name,
        aij_dir=aij_dir,
        graph_mode=graph_mode,
        n_cells=n_cells,
        cell_workers=cell_workers,
    )
    levels = list(range(1, num_levels + 1))
    if level_workers <= 1:
        for lvl in levels:
            _, feat = task_fn(lvl)
            level_parts[lvl - 1] = feat
            np.save(out_dir / f"{slice_name}_topo_L{lvl:02d}.npy", feat)
            print(
                f"  L{lvl:02d}: {feat.shape} "
                f"range=[{feat.min():.4f}, {feat.max():.4f}]",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=level_workers) as pool:
            futures = {pool.submit(task_fn, lvl): lvl for lvl in levels}
            for fut in as_completed(futures):
                lvl, feat = fut.result()
                level_parts[lvl - 1] = feat
                np.save(out_dir / f"{slice_name}_topo_L{lvl:02d}.npy", feat)
                print(
                    f"  L{lvl:02d}: {feat.shape} "
                    f"range=[{feat.min():.4f}, {feat.max():.4f}]",
                    flush=True,
                )

    topo_all = stack_level_features(level_parts)
    topo_path = out_dir / f"{slice_name}_topo.npy"
    np.save(topo_path, topo_all)

    _, thr_meta = load_thresholds(aij_dir, slice_name, graph_mode)
    meta = {
        "slice": slice_name,
        "n_cells": int(n_cells),
        "n_features": int(topo_all.shape[1]),
        "num_levels": num_levels,
        "feature_names": TOPO_FEATURE_NAMES,
        "column_names": column_names(num_levels),
        "graph_mode": graph_mode,
        "output_tag": effective_tag,
        "thresholds": thr_meta.get("thresholds"),
        "topo_file": str(topo_path),
        "level_files": [
            str(out_dir / f"{slice_name}_topo_L{lvl:02d}.npy")
            for lvl in range(1, num_levels + 1)
        ],
        "elapsed_sec": round(time.time() - t0, 2),
        "n_jobs": total_jobs,
        "level_workers": level_workers,
        "cell_workers": cell_workers,
    }
    meta_path = out_dir / f"{slice_name}_topo_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"  => {topo_path}  shape={topo_all.shape}  ({meta['elapsed_sec']}s)",
        flush=True,
    )
    return meta


def task_index_to_slice_level(task_id: int, slices: list[str], num_levels: int) -> tuple[str, int]:
    """SLURM array 任务编号 → (slice, level)。"""
    if task_id < 0 or task_id >= len(slices) * num_levels:
        raise ValueError(
            f"task_id={task_id} 超出范围 [0, {len(slices) * num_levels - 1}]"
        )
    slice_idx = task_id // num_levels
    level = task_id % num_levels + 1
    return slices[slice_idx], level


def main() -> None:
    parser = argparse.ArgumentParser(description="scRNA 拓扑谱特征提取")
    parser.add_argument("--config", default=None, help="config_graph.yaml")
    parser.add_argument("--slice", default=None, help="单个 slice")
    parser.add_argument("--level", type=int, default=None, help="单层 level (1-10)")
    parser.add_argument("--all", action="store_true", help="处理全部 13 个 slice")
    parser.add_argument(
        "--merge", action="store_true",
        help="合并已有 L01..L10 为 {slice}_topo.npy（需 --slice 或 --all）",
    )
    parser.add_argument(
        "--task-id", type=int, default=None,
        help="SLURM array 任务编号 (0..N_slice*N_level-1)，与 --level 互斥",
    )
    parser.add_argument(
        "--graph-mode", default=None,
        help="Aij 子目录: pearson | euclidean | root",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=None,
        help="并行 CPU 数（默认 SLURM_CPUS_PER_TASK 或 -1→全部）",
    )
    parser.add_argument(
        "--output-tag", default=None,
        help="输出子目录 topo_features/{tag}/{slice}/（默认读 config paths.topo_features_tag）",
    )
    parser.add_argument(
        "--aij-dir", default=None,
        help="Aij 根目录（默认 config paths.output_dir；PC3 请用 config_graph_pc3.yaml）",
    )
    args = parser.parse_args()

    scrna_dir = Path(__file__).resolve().parent
    cfg_path = Path(args.config) if args.config else scrna_dir / "config_graph.yaml"
    cfg = load_yaml(cfg_path)
    aij_root = resolve_aij_dir(cfg, scrna_dir, args.aij_dir)
    num_levels = int(cfg["graph"]["num_levels"])
    n_jobs = args.n_jobs
    if n_jobs is None:
        par = cfg.get("parallel", {}).get("n_jobs")
        if par == -1:
            n_jobs = resolve_n_jobs(None)
        elif isinstance(par, int) and par > 0:
            n_jobs = par

    all_slices = default_slices(cfg)

    # SLURM array：单任务 = 1 slice × 1 level
    if args.task_id is not None:
        slice_name, level = task_index_to_slice_level(
            args.task_id, all_slices, num_levels,
        )
        print(
            f"=== task {args.task_id}: {slice_name} L{level:02d} ===",
            flush=True,
        )
        extract_level_only(
            slice_name, level, cfg, scrna_dir,
            graph_mode=args.graph_mode,
            n_jobs=n_jobs,
            output_tag=args.output_tag,
            aij_dir=aij_root,
        )
        return

    if args.all:
        slices = all_slices
    elif args.slice:
        slices = [args.slice]
    else:
        slices = all_slices

    if args.merge:
        print(f"合并拓扑: {len(slices)} slice(s)", flush=True)
        summaries = []
        for slice_name in slices:
            print(f"\n=== merge {slice_name} ===", flush=True)
            try:
                meta = merge_slice_topo(
                    slice_name, cfg, scrna_dir,
                    graph_mode=args.graph_mode,
                    output_tag=args.output_tag,
                    aij_dir=aij_root,
                )
                summaries.append(meta)
            except Exception as exc:
                print(f"  [FAIL] {slice_name}: {exc}", flush=True)
                summaries.append({"slice": slice_name, "error": str(exc)})
        summary_path = topo_output_dir(cfg, scrna_dir, "_summary")
        summary_path.mkdir(parents=True, exist_ok=True)
        tag = args.output_tag or cfg.get("paths", {}).get("topo_features_tag", "default")
        out = summary_path / f"merge_topo_{tag}.json"
        out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        ok = sum(1 for s in summaries if "error" not in s)
        print(f"\n合并完成 {ok}/{len(slices)} → {out}")
        return

    if args.level is not None:
        if len(slices) != 1:
            raise SystemExit("--level 需配合单个 --slice")
        print(f"=== {slices[0]} L{args.level:02d} ===", flush=True)
        extract_level_only(
            slices[0], args.level, cfg, scrna_dir,
            graph_mode=args.graph_mode,
            n_jobs=n_jobs,
            output_tag=args.output_tag,
            aij_dir=aij_root,
        )
        return

    print(
        f"拓扑特征提取: {len(slices)} slice(s), graph_mode="
        f"{args.graph_mode or cfg.get('graph', {}).get('sim_mode', 'pearson')}, "
        f"output_tag={args.output_tag or cfg.get('paths', {}).get('topo_features_tag')}, "
        f"n_jobs={resolve_n_jobs(n_jobs)}",
    )
    summaries = []
    for slice_name in slices:
        print(f"\n=== {slice_name} ===", flush=True)
        try:
            meta = extract_slice_topo(
                slice_name, cfg, scrna_dir,
                graph_mode=args.graph_mode,
                n_jobs=n_jobs,
                output_tag=args.output_tag,
            )
            summaries.append(meta)
        except Exception as exc:
            print(f"  [FAIL] {slice_name}: {exc}", flush=True)
            summaries.append({"slice": slice_name, "error": str(exc)})

    summary_path = topo_output_dir(cfg, scrna_dir, "_summary")
    summary_path.mkdir(parents=True, exist_ok=True)
    out = summary_path / "extract_topo_summary.json"
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    ok = sum(1 for s in summaries if "error" not in s)
    print(f"\n完成 {ok}/{len(slices)} → {out}")


if __name__ == "__main__":
    main()
