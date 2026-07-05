#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
按细胞数列出 config 中的 slice，供 burgers_sim.sh / classify_eps_scan.sh 排序提交。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def resolve_path(base: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def graph_data_dir(data_dir: Path, slice_name: str, graph_mode: str) -> Path:
    base = data_dir / slice_name
    if graph_mode in ("pearson", "euclidean"):
        sub = base / graph_mode
        if sub.exists():
            return sub
    return base


def load_n_cells(data_dir: Path, slice_name: str, graph_mode: str) -> int | None:
    aij_dir = graph_data_dir(data_dir, slice_name, graph_mode)
    th_path = aij_dir / "thresholds.json"
    if not th_path.exists():
        return None
    with open(th_path) as f:
        info = json.load(f)
    return int(info.get("n_cells", 0))


def evolution_exclude(cfg: dict) -> set[str]:
    """config evolution.exclude：默认不参与 Burgers 演化提交。"""
    raw = cfg.get("evolution", {}).get("exclude", []) or []
    return {str(s).strip() for s in raw if str(s).strip()}


def list_slices(cfg: dict, scrna_dir: Path, exclude: set[str]) -> list[tuple[int, str]]:
    data_dir = resolve_path(scrna_dir, cfg["paths"]["output_dir"])
    graph_mode = str(cfg.get("graph", {}).get("sim_mode", "pearson"))
    rows: list[tuple[int, str]] = []
    for slice_name in cfg.get("slices", []):
        if slice_name in exclude:
            continue
        n_cells = load_n_cells(data_dir, slice_name, graph_mode)
        if n_cells is None:
            continue
        rows.append((n_cells, slice_name))
    rows.sort(key=lambda x: (x[0], x[1]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="List slices sorted by cell count")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config_graph.yaml"),
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="逗号分隔，排除的 slice",
    )
    parser.add_argument(
        "--for-evolution",
        action="store_true",
        help="合并 config evolution.exclude（burgers_sim 提交用）",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="打印 细胞数 + slice 表格",
    )
    args = parser.parse_args()
    cfg_path = args.config.resolve()
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}
    if args.for_evolution:
        exclude |= evolution_exclude(cfg)
    rows = list_slices(cfg, cfg_path.parent, exclude)
    if args.table:
        print(f"{'n_cells':>8}  slice")
        print("-" * 32)
        for n_cells, slice_name in rows:
            print(f"{n_cells:>8}  {slice_name}")
    else:
        for _, slice_name in rows:
            print(slice_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
