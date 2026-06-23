# Author: Zihan Wang
# <wangzh011031@163.com>
"""日志、IO 工具。"""
import logging
import sys
from pathlib import Path


def resolve_path(project_root: Path, path_value: str | Path) -> Path:
    """将配置中的路径解析为绝对路径（支持相对 project_root）。"""
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def setup_logging(log_path: Path, level=logging.INFO):
    """配置同时输出到文件和终端的日志。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def load_yaml(path: Path) -> dict:
    """加载 YAML 文件。"""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: Path):
    """保存 JSON。"""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=float)


def pcc_10digit(pcc: float) -> float:
    """保留10位有效数字。"""
    return float(f"{pcc:.10g}")
