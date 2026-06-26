#!/usr/bin/env python3
"""
预处理 13 个 scRNA-seq 数据片，统一为 (expression.csv.gz, labels.csv)。

输入: scRNA/scRNA_data/processed/ + Single Cell RNA Sequencing/
输出: scRNA/preprocessed/{slice_name}/expression.csv.gz, labels.csv

用法:
    python preprocess_scRNA.py                  # 处理全部 13 个
    python preprocess_scRNA.py --slice GSE84133_human1  # 处理单个
"""

import argparse
import csv
import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# ── 路径 ──
BASE = Path(__file__).resolve().parent
DATA = BASE / "scRNA_data" / "processed"
LABEL_DIR = BASE / "Single Cell RNA Sequencing"
OUTPUT = BASE / "preprocessed"


# ═══════════════════════════════════════════════════════════════════════════
# Type A: 每细胞一个文件，文件名含 GSM ID
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_gse45719():
    """300 cells, 每个 txt 有 RPKM 列, 文件名=GSM ID."""
    name = "GSE45719"
    data_dir = DATA / name
    label_path = LABEL_DIR / "GSE45719_full_labels.csv"
    out_dir = OUTPUT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读标签: GSM ID → cell_type
    gsm_to_type = {}
    with open(label_path) as f:
        for row in csv.DictReader(f):
            gsm_to_type[row["Sample Name"]] = row["Cell type"]

    # 读每个细胞文件, 收集基因集合
    cell_gsms = []
    cell_data = {}  # gsm → {gene: rpkm}
    all_genes = set()

    for fpath in sorted(data_dir.glob("GSM*_expression.txt")):
        gsm_id = fpath.name.split("_")[0]
        if gsm_id not in gsm_to_type:
            continue
        cell_gsms.append(gsm_id)
        gene_rpkm = {}
        with open(fpath) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    gene, _, rpkm_str = parts[0], parts[1], parts[2]
                    try:
                        rpkm = float(rpkm_str)
                    except ValueError:
                        rpkm = 0.0
                    gene_rpkm[gene] = rpkm
        cell_data[gsm_id] = gene_rpkm
        all_genes.update(gene_rpkm.keys())

    # 构建统一矩阵
    genes = sorted(all_genes)
    n_cells = len(cell_gsms)
    matrix = np.zeros((n_cells, len(genes)), dtype=np.float32)
    for i, gsm in enumerate(cell_gsms):
        for j, gene in enumerate(genes):
            matrix[i, j] = cell_data[gsm].get(gene, 0.0)

    # 保存
    _save_matrix(out_dir, matrix, cell_gsms, genes)
    _save_labels(out_dir, cell_gsms, gsm_to_type, name)
    print(f"  {name}: {n_cells} cells × {len(genes)} genes")


def preprocess_gse67835():
    """466 cells, 每个 csv 有 gene[TAB]count, 文件名=GSM ID."""
    name = "GSE67835"
    data_dir = DATA / name
    label_path = LABEL_DIR / "GSE67835_full_labels.csv"
    out_dir = OUTPUT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    gsm_to_type = {}
    with open(label_path) as f:
        for row in csv.DictReader(f):
            gsm_to_type[row["Sample Name"]] = row["Cell type"]

    cell_gsms = []
    cell_data = {}
    all_genes = set()

    for fpath in sorted(data_dir.glob("GSM*.csv")):
        gsm_id = fpath.name.split("_")[0]
        if gsm_id not in gsm_to_type:
            continue
        cell_gsms.append(gsm_id)
        gene_counts = {}
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    gene = parts[0].strip()
                    try:
                        cnt = float(parts[1])
                    except ValueError:
                        cnt = 0.0
                    gene_counts[gene] = cnt
        cell_data[gsm_id] = gene_counts
        all_genes.update(gene_counts.keys())

    genes = sorted(all_genes)
    n_cells = len(cell_gsms)
    matrix = np.zeros((n_cells, len(genes)), dtype=np.float32)
    for i, gsm in enumerate(cell_gsms):
        for j, gene in enumerate(genes):
            matrix[i, j] = cell_data[gsm].get(gene, 0.0)

    _save_matrix(out_dir, matrix, cell_gsms, genes)
    _save_labels(out_dir, cell_gsms, gsm_to_type, name)
    print(f"  {name}: {n_cells} cells × {len(genes)} genes")


# ═══════════════════════════════════════════════════════════════════════════
# Type B: 统一矩阵, ID 直接匹配
# ═══════════════════════════════════════════════════════════════════════════

def _preprocess_gse84133_slice(slice_name, exp_file, label_file):
    """GSE84133 子集: cells×genes CSV, cell ID 在 col 1."""
    exp_path = DATA / "GSE84133" / exp_file
    label_path = LABEL_DIR / label_file
    out_dir = OUTPUT / slice_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读标签
    cell_to_type = {}
    with open(label_path) as f:
        for row in csv.DictReader(f):
            cell_to_type[row["Sample Name"]] = row["Cell type"]

    # 读表达矩阵（逐行，不超过内存）
    # 格式: 第 1 列=cell_id(空header), 第 2 列=barcode, 第 3 列=assigned_cluster,
    #       后续列=基因表达值 (数字)
    cell_ids = []
    rows_list = []
    with open(exp_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        # 找到第一个纯数字列作为基因起点
        gene_start = None
        for i, col in enumerate(header):
            if col and col[0].isupper() and col != "barcode" and col != "assigned_cluster":
                gene_start = i
                break
        gene_names = header[gene_start:]
        for row in reader:
            if len(row) < gene_start + 1:
                continue
            cell_ids.append(row[0])  # 第 1 列是 cell_id (如 human1_lib1.final_cell_0001)
            values = []
            for v in row[gene_start:]:
                try:
                    values.append(float(v))
                except (ValueError, TypeError):
                    values.append(0.0)
            rows_list.append(np.array(values, dtype=np.float32))

    matrix = np.array(rows_list, dtype=np.float32)
    n_cells = len(cell_ids)

    _save_matrix(out_dir, matrix, cell_ids, gene_names)
    _save_labels(out_dir, cell_ids, cell_to_type, slice_name)
    print(f"  {slice_name}: {n_cells} cells × {len(gene_names)} genes")


def preprocess_gse84133():
    """5 个子集."""
    slices = [
        ("GSE84133_human1", "GSM2230757_human1_umifm_counts.csv", "GSE84133human1_full_labels.csv"),
        ("GSE84133_human2", "GSM2230758_human2_umifm_counts.csv", "GSE84133human2_full_labels.csv"),
        ("GSE84133_human4", "GSM2230760_human4_umifm_counts.csv", "GSE84133human4_full_labels.csv"),
        ("GSE84133_mouse1", "GSM2230761_mouse1_umifm_counts.csv", "GSE84133mouse1_full_labels.csv"),
        ("GSE84133_mouse2", "GSM2230762_mouse2_umifm_counts.csv", "GSE84133mouse2_full_labels.csv"),
    ]
    for slice_name, exp_file, label_file in slices:
        _preprocess_gse84133_slice(slice_name, exp_file, label_file)


def preprocess_gse82187():
    """706 cells, cells×genes CSV, 表达矩阵含 'type' 列直接作为细胞类型."""
    name = "GSE82187"
    exp_path = DATA / name / "GSE82187_cast_all_forGEO.csv"
    out_dir = OUTPUT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_ids = []
    cell_types = []
    rows_list = []
    with open(exp_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        gene_names = header[5:]  # 前5列: index, cell.name, type, experiment, protocol
        for row in reader:
            if len(row) < 6:
                continue
            cell_ids.append(row[1])  # cell.name
            cell_types.append(row[2])  # type
            values = []
            for v in row[5:]:
                try:
                    values.append(float(v))
                except (ValueError, TypeError):
                    values.append(0.0)
            rows_list.append(np.array(values, dtype=np.float32))

    matrix = np.array(rows_list, dtype=np.float32)
    type_counter = Counter(cell_types)
    type_to_label = {t: i for i, t in enumerate(sorted(type_counter))}

    _save_matrix_raw(out_dir, matrix, cell_ids)
    _save_labels_raw(out_dir, cell_ids, cell_types, type_to_label, name)
    print(f"  {name}: {len(cell_ids)} cells × {len(gene_names)} genes "
          f"({len(type_to_label)} cell types)")


# ═══════════════════════════════════════════════════════════════════════════
# Type C: 统一矩阵, 标签不匹配 → 从列名提取细胞类型
# ═══════════════════════════════════════════════════════════════════════════

def _extract_cell_type_gse75748(col_name):
    """H1_Exp1.001 → H1"""
    parts = col_name.split("_")
    if parts and not parts[0].startswith('"'):
        return parts[0]
    return col_name.split("_")[0].strip('"')


def _extract_cell_type_gse89232(col_name):
    """blood_bdca3_S1 → blood_bdca3 (CD141+ cDC); CB_pre_cDC_S1 → CB_pre_cDC"""
    parts = col_name.rsplit("_S", 1)
    return parts[0]


def _extract_cell_type_gse94820(col_name):
    """CD141_P10_S73 → CD141"""
    return col_name.split("_")[0]


def _preprocess_matrix_transposed(slice_name, exp_file, ds_dir,
                                   label_extractor, delim, has_gene_col,
                                   original_label_file=None):
    """处理基因×细胞矩阵（需转置为细胞×基因）。"""
    exp_path = DATA / ds_dir / exp_file
    out_dir = OUTPUT / slice_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读矩阵（基因×细胞 → 转置为细胞×基因）
    with open(exp_path) as f:
        reader = csv.reader(f, delimiter=delim)
        header = next(reader)
        # header: 可能包含空基因列名 (GSE75748: ["", "H1_Exp1.001", ...])
        # 或不含 (GSE89232: ["blood_bdca3_S1", ...])
        raw_ids = [c.strip('"') for c in header]
        if raw_ids and raw_ids[0] == "":
            cell_ids = raw_ids[1:]  # 跳过空的基因列名
        else:
            cell_ids = raw_ids

        rows = []
        for row in reader:
            if has_gene_col:
                vals = row[1:]  # 跳过第一列基因名
            else:
                vals = row
            try:
                rows.append([float(v) for v in vals])
            except (ValueError, TypeError):
                continue

    matrix = np.array(rows, dtype=np.float32).T  # 转置: 基因×细胞 → 细胞×基因
    n_cells = len(cell_ids)
    if n_cells != matrix.shape[0]:
        print(f"  ⚠ {slice_name}: cell_ids={n_cells} vs matrix rows={matrix.shape[0]}, fixing...")
        n_cells = min(n_cells, matrix.shape[0])
        cell_ids = cell_ids[:n_cells]
        matrix = matrix[:n_cells, :]

    # 从列名提取细胞类型
    cell_types = [label_extractor(c.strip('"')) for c in cell_ids]
    type_counter = Counter(cell_types)
    type_to_label = {t: i for i, t in enumerate(sorted(type_counter))}

    # 保存
    _save_matrix_raw(out_dir, matrix, cell_ids)
    _save_labels_raw(out_dir, cell_ids, cell_types, type_to_label, slice_name)
    print(f"  {slice_name}: {n_cells} cells × {matrix.shape[1]} genes "
          f"({len(type_to_label)} cell types: {dict(type_counter.most_common(8))})")


def preprocess_gse75748():
    """2 个子集: cell (snapshot) + time (timecourse)."""
    _preprocess_matrix_transposed(
        "GSE75748_cell", "GSE75748_sc_cell_type_ec.csv", "GSE75748",
        _extract_cell_type_gse75748, ",", has_gene_col=True)
    _preprocess_matrix_transposed(
        "GSE75748_time", "GSE75748_sc_time_course_ec.csv", "GSE75748",
        _extract_cell_type_gse75748, ",", has_gene_col=True)


def preprocess_gse89232():
    _preprocess_matrix_transposed(
        "GSE89232", "GSE89232_expMatrix.txt", "GSE89232",
        _extract_cell_type_gse89232, "\t", has_gene_col=True)


def preprocess_gse94820():
    _preprocess_matrix_transposed(
        "GSE94820_discovery", "GSE94820_raw.expMatrix_DCnMono.discovery.set.submission.txt",
        "GSE94820", _extract_cell_type_gse94820, "\t", has_gene_col=True)
    _preprocess_matrix_transposed(
        "GSE94820_deeper", "GSE94820_raw.expMatrix_deeper.characterization.set.submission.txt",
        "GSE94820", _extract_cell_type_gse94820, "\t", has_gene_col=True)


# ═══════════════════════════════════════════════════════════════════════════
# Type D: Excel 格式
# ═══════════════════════════════════════════════════════════════════════════

def _preprocess_gse59114_strain(slice_name, xlsx_file, strain):
    """处理单个品系的 Excel."""
    import re
    import pandas as pd

    xlsx_path = DATA / "GSE59114" / xlsx_file
    out_dir = OUTPUT / slice_name
    out_dir.mkdir(parents=True, exist_ok=True)

    def _cell_type_from_name(cell_name: str) -> str:
        """young_LT_HSC_2 → young_LT_HSC"""
        return re.sub(r'_\d+$', '', str(cell_name))

    df_raw = pd.read_excel(xlsx_path, header=None)
    col_names = df_raw.iloc[0].tolist()
    cell_names = df_raw.iloc[1].tolist()

    # 找数据列: 列名是数字的 (排除 Population Average)
    data_cols = []
    for i, cn in enumerate(col_names):
        if isinstance(cn, str) and 'Population' in cn:
            continue
        if isinstance(cn, (int, float)) and not pd.isna(cn):
            data_cols.append(i)

    genes = [str(g).strip("'") for g in df_raw.iloc[2:, 0].tolist()]
    n_genes = len(genes)

    cell_ids = []
    cell_types = []
    rows_list = []
    for col_idx in data_cols:
        cell_name = str(cell_names[col_idx])
        cell_type = _cell_type_from_name(cell_name)
        vals = [float(df_raw.iloc[r, col_idx]) if pd.notna(df_raw.iloc[r, col_idx]) else 0.0
                for r in range(2, len(df_raw))]
        cell_ids.append(cell_name)
        cell_types.append(cell_type)
        rows_list.append(np.array(vals, dtype=np.float32))

    matrix = np.array(rows_list, dtype=np.float32)
    type_counter = Counter(cell_types)
    type_to_label = {t: i for i, t in enumerate(sorted(type_counter))}

    _save_matrix_raw(out_dir, matrix, cell_ids)
    _save_labels_raw(out_dir, cell_ids, cell_types, type_to_label, slice_name)
    print(f"  {slice_name}: {len(cell_ids)} cells × {n_genes} genes "
          f"({len(type_to_label)} cell types)")


def preprocess_gse59114():
    _preprocess_gse59114_strain("GSE59114_C57BL6", "GSE59114_C57BL6_GEO_all.xlsx", "C57BL6")
    _preprocess_gse59114_strain("GSE59114_DBA", "GSE59114_DBA_GEO_all.xlsx", "DBA")


# ═══════════════════════════════════════════════════════════════════════════
# 保存函数
# ═══════════════════════════════════════════════════════════════════════════

def _save_matrix(out_dir, matrix, cell_ids, gene_names):
    """保存矩阵 (np.float32 gzip) 和元数据."""
    np.savez_compressed(out_dir / "expression.npz",
                        matrix=matrix, cell_ids=np.array(cell_ids),
                        gene_names=np.array(gene_names))
    with open(out_dir / "meta.json", "w") as f:
        json.dump({"n_cells": len(cell_ids), "n_genes": len(gene_names),
                   "dtype": "float32"}, f)
    # 也存一个 CSV.gz 备用
    with gzip.open(out_dir / "expression.csv.gz", "wt") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_id"] + gene_names)
        for i, cid in enumerate(cell_ids):
            writer.writerow([cid] + [f"{v:.6g}" for v in matrix[i]])


def _save_matrix_raw(out_dir, matrix, cell_ids):
    """保存矩阵，不保存基因名（不同数据集基因集合不同）。"""
    np.savez_compressed(out_dir / "expression.npz",
                        matrix=matrix, cell_ids=np.array(cell_ids))
    with open(out_dir / "meta.json", "w") as f:
        json.dump({"n_cells": len(cell_ids), "n_genes": int(matrix.shape[1]),
                   "dtype": "float32"}, f)


def _save_labels(out_dir, cell_ids, id_to_type, slice_name):
    """保存标签 CSV."""
    type_to_label = {}
    types_seen = set()
    for cid in cell_ids:
        ct = id_to_type.get(cid, "unknown")
        if ct not in types_seen:
            type_to_label[ct] = len(types_seen)
            types_seen.add(ct)

    with open(out_dir / "labels.csv", "w") as f:
        f.write("cell_id,cell_type,label\n")
        for cid in cell_ids:
            ct = id_to_type.get(cid, "unknown")
            f.write(f"{cid},{ct},{type_to_label[ct]}\n")
    print(f"    labels: {len(types_seen)} cell types")


def _save_labels_raw(out_dir, cell_ids, cell_types, type_to_label, slice_name):
    """保存从列名提取的标签."""
    with open(out_dir / "labels.csv", "w") as f:
        f.write("cell_id,cell_type,label\n")
        for cid, ct in zip(cell_ids, cell_types):
            f.write(f"{cid},{ct},{type_to_label[ct]}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

ALL_SLICES = [
    ("GSE45719", preprocess_gse45719),
    ("GSE67835", preprocess_gse67835),
    ("GSE82187", preprocess_gse82187),
    ("GSE89232", lambda: _preprocess_matrix_transposed(
        "GSE89232", "GSE89232_expMatrix.txt", "GSE89232",
        _extract_cell_type_gse89232, "\t", has_gene_col=True)),
    # GSE84133 5 个子集
    ("GSE84133_human1", lambda: _preprocess_gse84133_slice(
        "GSE84133_human1", "GSM2230757_human1_umifm_counts.csv", "GSE84133human1_full_labels.csv")),
    ("GSE84133_human2", lambda: _preprocess_gse84133_slice(
        "GSE84133_human2", "GSM2230758_human2_umifm_counts.csv", "GSE84133human2_full_labels.csv")),
    ("GSE84133_human4", lambda: _preprocess_gse84133_slice(
        "GSE84133_human4", "GSM2230760_human4_umifm_counts.csv", "GSE84133human4_full_labels.csv")),
    ("GSE84133_mouse1", lambda: _preprocess_gse84133_slice(
        "GSE84133_mouse1", "GSM2230761_mouse1_umifm_counts.csv", "GSE84133mouse1_full_labels.csv")),
    ("GSE84133_mouse2", lambda: _preprocess_gse84133_slice(
        "GSE84133_mouse2", "GSM2230762_mouse2_umifm_counts.csv", "GSE84133mouse2_full_labels.csv")),
    # GSE75748 2 个子集
    ("GSE75748_cell", lambda: _preprocess_matrix_transposed(
        "GSE75748_cell", "GSE75748_sc_cell_type_ec.csv", "GSE75748",
        _extract_cell_type_gse75748, ",", has_gene_col=True)),
    ("GSE75748_time", lambda: _preprocess_matrix_transposed(
        "GSE75748_time", "GSE75748_sc_time_course_ec.csv", "GSE75748",
        _extract_cell_type_gse75748, ",", has_gene_col=True)),
    # GSE94820 2 个子集
    ("GSE94820_discovery", lambda: _preprocess_matrix_transposed(
        "GSE94820_discovery", "GSE94820_raw.expMatrix_DCnMono.discovery.set.submission.txt",
        "GSE94820", _extract_cell_type_gse94820, "\t", has_gene_col=True)),
    ("GSE94820_deeper", lambda: _preprocess_matrix_transposed(
        "GSE94820_deeper", "GSE94820_raw.expMatrix_deeper.characterization.set.submission.txt",
        "GSE94820", _extract_cell_type_gse94820, "\t", has_gene_col=True)),
    ("GSE59114_C57BL6", lambda: _preprocess_gse59114_strain(
        "GSE59114_C57BL6", "GSE59114_C57BL6_GEO_all.xlsx", "C57BL6")),
    ("GSE59114_DBA", lambda: _preprocess_gse59114_strain(
        "GSE59114_DBA", "GSE59114_DBA_GEO_all.xlsx", "DBA")),
]


def main():
    parser = argparse.ArgumentParser(description="Preprocess scRNA-seq data")
    parser.add_argument("--slice", type=str, help="Process single slice only")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    if args.slice:
        targets = [(s, f) for s, f in ALL_SLICES if s == args.slice]
        if not targets:
            print(f"Unknown slice: {args.slice}")
            print(f"Available: {[s for s, _ in ALL_SLICES]}")
            return 1
    else:
        targets = ALL_SLICES

    print(f"Processing {len(targets)} slice(s) → {OUTPUT}\n")

    for slice_name, func in targets:
        try:
            func()
        except Exception as e:
            print(f"  ✗ {slice_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Output: {OUTPUT}")


if __name__ == "__main__":
    main()
