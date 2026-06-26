#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
下载并组织 scRNA-seq 表达数据，与现有 labels 对齐。

用法:
    python download_scRNA.py                    # 下载所有数据集
    python download_scRNA.py --dataset GSE84133 # 下载单个
    python download_scRNA.py --dry-run          # 查看文件列表但不下载
"""

import argparse
import os
import shutil
import sys
import tarfile
import gzip
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlretrieve

# ── 8 个 GEO 数据集的元信息 ──────────────────────────────────────────────
# 手动整理自 GEO 页面，避免依赖 GEOparse 的复杂依赖
DATASETS = {
    "GSE45719": {
        "title": "Monoallelic expression in mouse embryos (Deng et al. 2014)",
        "organism": "mouse",
        "cells": 300,
        "files": [
            "GSE45719_RAW.tar",
        ],
        "note": "97.9 MB",
    },
    "GSE59114": {
        "title": "Aging hematopoietic stem/progenitor cells (Kowalczyk et al. 2015)",
        "organism": "mouse",
        "cells": 1428,
        "files": [
            "GSE59114_C57BL6_GEO_all.xlsx",
            "GSE59114_DBA_GEO_all.xlsx",
        ],
        "note": "98.1 + 44.4 MB (Excel)",
    },
    "GSE67835": {
        "title": "Human brain transcriptome (Darmanis et al. 2015)",
        "organism": "human",
        "cells": 466,
        "files": [
            "GSE67835_RAW.tar",
        ],
    },
    "GSE75748": {
        "title": "hESC endoderm progenitors (Chu et al. 2016)",
        "organism": "human",
        "cells": "1018 (snapshot) + 758 (timecourse) = 1776",
        "files": [
            "GSE75748_sc_cell_type_ec.csv.gz",
            "GSE75748_sc_time_course_ec.csv.gz",
        ],
    },
    "GSE82187": {
        "title": "Mouse striatum cell types (Stanley et al. 2016)",
        "organism": "mouse",
        "cells": 706,
        "files": [
            "GSE82187_cast_all_forGEO.csv.gz",
        ],
    },
    "GSE84133": {
        "title": "Human/mouse pancreas map (Baron et al. 2016)",
        "organism": "human+mouse",
        "cells": 12000,
        "files": [
            "GSE84133_RAW.tar",
        ],
    },
    "GSE89232": {
        "title": "Human pre-cDC heterogeneity (Oliveira et al. 2016)",
        "organism": "human",
        "cells": 957,
        "files": [
            "GSE89232_expMatrix.txt.gz",
        ],
    },
    "GSE94820": {
        "title": "Human blood dendritic cells/monocytes (See et al. 2017)",
        "organism": "human",
        "cells": 1140,
        "note": "Raw data in dbGaP (phs001294); processed expMatrix on GEO",
        "files": [
            "GSE94820_raw.expMatrix_DCnMono.discovery.set.submission.txt.gz",
            "GSE94820_raw.expMatrix_deeper.characterization.set.submission.txt.gz",
        ],
    },
}

# GEO supplementary 文件下载基 URL
GEO_DOWNLOAD_BASE = "https://www.ncbi.nlm.nih.gov/geo/download/"


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """下载单个文件，带进度提示。已存在则跳过。"""
    if dest.exists():
        print(f"  ✓ 已存在: {dest.name}")
        return True
    print(f"  ↓ 下载中: {desc or dest.name} ...", end=" ", flush=True)
    try:
        urlretrieve(url, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"OK ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"失败: {e}")
        return False


def extract_tar(tar_path: Path, dest_dir: Path) -> list[Path]:
    """解压 .tar 文件到目标目录，返回文件列表。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            if member.isdir():
                continue
            # 安全：防止路径穿越
            fname = Path(member.name).name
            out_path = dest_dir / fname
            if out_path.exists():
                extracted.append(out_path)
                continue
            with tar.extractfile(member) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(out_path)
            if fname.endswith(".gz"):
                # 顺便解压 .gz
                unzipped = out_path.with_suffix("")
                if not unzipped.exists():
                    with gzip.open(out_path, "rb") as gz_src, open(unzipped, "wb") as gz_dst:
                        shutil.copyfileobj(gz_src, gz_dst)
                    extracted.append(unzipped)
    return extracted


def gunzip_file(gz_path: Path) -> Path:
    """解压单个 .gz 文件。"""
    out_path = gz_path.with_suffix("")
    if out_path.exists():
        return out_path
    with gzip.open(gz_path, "rb") as src, open(out_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return out_path


def download_dataset(gse_id: str, info: dict, raw_dir: Path, processed_dir: Path,
                     dry_run: bool = False) -> bool:
    """下载单个 GSE 数据集的所有文件并解压到处理目录。"""
    print(f"\n{'='*60}")
    print(f"  {gse_id}: {info['title']}")
    print(f"  物种: {info['organism']}  |  细胞数: ~{info.get('cells', '?')}")
    print(f"{'='*60}")

    if dry_run:
        for fname in info["files"]:
            print(f"  [DRY-RUN] 将下载: {fname}")
        return True

    raw_ds = raw_dir / gse_id
    proc_ds = processed_dir / gse_id
    raw_ds.mkdir(parents=True, exist_ok=True)
    proc_ds.mkdir(parents=True, exist_ok=True)

    for fname in info["files"]:
        # _RAW.tar 文件使用 bulk download URL (format=file, 不带 &file=)
        if fname.endswith("_RAW.tar"):
            url = f"{GEO_DOWNLOAD_BASE}?acc={gse_id}&format=file"
        else:
            encoded = quote(fname, safe="")
            url = f"{GEO_DOWNLOAD_BASE}?acc={gse_id}&format=file&file={encoded}"

        raw_path = raw_ds / fname
        if not download_file(url, raw_path, fname):
            print(f"  ⚠ 跳过: {fname}")
            continue

        # 解压到 processed 目录
        if fname.endswith(".tar"):
            files = extract_tar(raw_path, proc_ds)
            print(f"  解压 {len(files)} 个文件 -> {proc_ds}")
        elif fname.endswith(".gz"):
            out = gunzip_file(raw_path)
            # 移动到 processed
            dest = proc_ds / out.name
            if not dest.exists():
                shutil.move(str(out), str(dest))
            print(f"  解压 -> {dest.name}")
        else:
            dest = proc_ds / raw_path.name
            if not dest.exists():
                shutil.copy2(raw_path, dest)

    return True


def link_labels(label_dir: Path, processed_dir: Path) -> None:
    """将已有的 label CSV 软链接到对应数据集的 processed 目录。"""
    for csv_path in sorted(label_dir.glob("*_full_labels.csv")):
        # 从文件名提取 GSE ID
        # e.g. GSE75748cell_full_labels.csv -> GSE75748
        name = csv_path.stem  # GSE75748cell_full_labels
        for gse_id in DATASETS:
            if name.startswith(gse_id):
                dest_dir = processed_dir / gse_id
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / csv_path.name
                if not dest.exists():
                    dest.symlink_to(csv_path.resolve())
                print(f"  labels: {csv_path.name} -> {dest_dir.name}/")
                break


def write_manifest(processed_dir: Path, dry_run: bool = False) -> None:
    """生成数据集清单。"""
    manifest = {"datasets": {}, "total_files": 0}
    for gse_id in sorted(DATASETS):
        ds_dir = processed_dir / gse_id
        if ds_dir.exists():
            files = [f.name for f in ds_dir.iterdir() if f.is_file()]
            manifest["datasets"][gse_id] = {
                "title": DATASETS[gse_id]["title"],
                "organism": DATASETS[gse_id]["organism"],
                "n_files": len(files),
                "files": files,
            }
            manifest["total_files"] += len(files)

    manifest["total_datasets"] = len(manifest["datasets"])
    if not dry_run:
        manifest_path = processed_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"\n清单: {manifest_path}")
        print(f"  {manifest['total_datasets']} 数据集, {manifest['total_files']} 个文件")


def main():
    parser = argparse.ArgumentParser(description="Download scRNA-seq data from GEO")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Download a single dataset (e.g. GSE84133)")
    parser.add_argument("--dry-run", action="store_true",
                       help="List files without downloading")
    parser.add_argument("--data-root", type=Path,
                       default=Path(__file__).resolve().parent / "scRNA_data",
                       help="Root directory for downloaded data")
    args = parser.parse_args()

    data_root = args.data_root
    raw_dir = data_root / "raw"           # 原始下载文件
    processed_dir = data_root / "processed"  # 解压后的表达矩阵

    data_root.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"数据根目录: {data_root}")
    print(f"  原始文件: {raw_dir}")
    print(f"  处理文件: {processed_dir}")

    if args.dataset:
        gse_id = args.dataset.upper()
        if gse_id not in DATASETS:
            print(f"未知数据集: {gse_id}")
            print(f"可选: {', '.join(DATASETS)}")
            return 1
        success = download_dataset(gse_id, DATASETS[gse_id], raw_dir, processed_dir,
                                   args.dry_run)
        if not success:
            return 1
    else:
        for gse_id, info in DATASETS.items():
            try:
                download_dataset(gse_id, info, raw_dir, processed_dir, args.dry_run)
            except Exception as e:
                print(f"  ✗ {gse_id} 失败: {e}")

    if not args.dry_run:
        # 链接已有的 label 文件
        label_dir = Path(__file__).resolve().parent / "Single Cell RNA Sequencing"
        if label_dir.exists():
            print(f"\n{'='*60}")
            print("关联 label 文件...")
            link_labels(label_dir, processed_dir)

        write_manifest(processed_dir)

    print(f"\n完成。数据结构:")
    print(f"  {processed_dir}/")  # 修复 f-string
    for gse_id in sorted(DATASETS):
        ds_dir = processed_dir / gse_id
        marker = " ✓" if ds_dir.exists() and any(ds_dir.iterdir()) else ""
        print(f"    {gse_id}/{marker}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
