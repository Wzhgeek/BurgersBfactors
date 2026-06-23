# Author: Zihan Wang
# <wangzh011031@163.com>
"""读写 .xyzb 数据文件。"""

from pathlib import Path

import numpy as np


def load_xyzb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    加载 .xyzb 文件。

    每行格式: x y z b-factor（空格分隔），共 N 行对应 N 个原子。

    Returns:
        coords: (N, 3) 原子坐标
        labels: (N,) B-factor 标签
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                x, y, z, b = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue
            rows.append([x, y, z, b])

    data = np.asarray(rows, dtype=np.float64)
    coords = data[:, :3]
    labels = data[:, 3]
    return coords, labels


def pdb_id_from_filename(filename: str, suffix: str) -> str:
    """从文件名提取 PDB ID，例如 1AIE_ca.xyzb -> 1AIE。"""
    if filename.endswith(suffix):
        return filename[: -len(suffix)]
    return Path(filename).stem.split("_")[0]
