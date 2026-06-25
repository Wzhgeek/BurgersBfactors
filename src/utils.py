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


def resolve_evaluation(n_samples: int, eval_cfg: dict) -> dict:
    """
    按样本数（原子数）解析实际评估策略。

    当 evaluation.use_cv=true 且 adaptive_cv.enabled=true 时：
      n < holdout_below       -> hold-out（不做 CV）
      holdout_below ≤ n < mid_below -> cv_folds_mid 折（默认 5）
      n ≥ mid_below           -> cv_folds 折（默认 10）

    当 evaluation.use_cv=false 时，全数据集统一 hold-out，忽略 adaptive_cv。
    """
    requested_use_cv = bool(eval_cfg.get("use_cv", True))
    test_size = float(eval_cfg.get("test_size", 0.2))
    random_state = int(eval_cfg.get("random_state", 42))
    max_cv_folds = int(eval_cfg.get("cv_folds", 10))

    adaptive = eval_cfg.get("adaptive_cv", {})
    if isinstance(adaptive, bool):
        adaptive_enabled = adaptive
        holdout_below = 10
        mid_below = 25
        cv_folds_mid = 5
    else:
        adaptive_enabled = bool(adaptive.get("enabled", True))
        holdout_below = int(adaptive.get("holdout_below", 10))
        mid_below = int(adaptive.get("mid_below", 25))
        cv_folds_mid = int(adaptive.get("cv_folds_mid", 5))

    base = {
        "test_size": test_size,
        "random_state": random_state,
        "requested_use_cv": requested_use_cv,
        "n_samples": int(n_samples),
    }

    if not requested_use_cv:
        return {
            **base,
            "use_cv": False,
            "cv_folds": 0,
            "adaptive_cv_applied": False,
            "eval_strategy": "holdout",
        }

    if adaptive_enabled and n_samples < holdout_below:
        return {
            **base,
            "use_cv": False,
            "cv_folds": 0,
            "adaptive_cv_applied": True,
            "eval_strategy": "holdout",
        }

    if adaptive_enabled and n_samples < mid_below:
        return {
            **base,
            "use_cv": True,
            "cv_folds": cv_folds_mid,
            "adaptive_cv_applied": True,
            "eval_strategy": f"cv_{cv_folds_mid}",
        }

    return {
        **base,
        "use_cv": True,
        "cv_folds": max_cv_folds,
        "adaptive_cv_applied": bool(adaptive_enabled),
        "eval_strategy": f"cv_{max_cv_folds}",
    }


def evaluation_for_json(eval_res: dict) -> dict:
    """result.json 中 evaluation 字段的标准结构。"""
    return {
        "use_cv": bool(eval_res["use_cv"]),
        "cv_folds": int(eval_res["cv_folds"]),
        "test_size": float(eval_res["test_size"]),
        "random_state": int(eval_res["random_state"]),
        "requested_use_cv": bool(eval_res.get("requested_use_cv", eval_res["use_cv"])),
        "adaptive_cv_applied": bool(eval_res.get("adaptive_cv_applied", False)),
        "eval_strategy": str(eval_res.get("eval_strategy", "")),
    }


def format_eval_log(eval_res: dict) -> str:
    """日志用评估策略描述。"""
    if eval_res["use_cv"]:
        suffix = " (adaptive)" if eval_res.get("adaptive_cv_applied") else ""
        return f"{eval_res['cv_folds']}-fold CV{suffix} [{eval_res.get('eval_strategy', '')}]"
    suffix = " (adaptive)" if eval_res.get("adaptive_cv_applied") else ""
    return f"hold-out test_size={eval_res['test_size']}{suffix} [holdout]"
