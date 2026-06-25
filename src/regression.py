# Author: Zihan Wang
# <wangzh011031@163.com>
"""
回归模块：5 种算法共享统一接口 evaluate_regressor(name, X, y, params)。

算法:
  - RF:   RandomForestRegressor (sklearn)
  - KNN:  KNeighborsRegressor (sklearn)
  - SVR:  SVR + StandardScaler (sklearn)
  - LR:   Ridge 线性回归 (L2 正则)
  - LSTM: PyTorch LSTM 模型 (hidden_size, num_layers 可配置)

评估:
  - cv_folds >= 2: K 折交叉验证（默认 10 折）
  - test_pcc: OOF 整体 PCC（无偏，适合正式比较）
  - best_fold_pcc / best_fold_idx: 各折验证 PCC 的最大值及折号（1-based）
  - mean_fold_pcc: 各折验证 PCC 的算术平均
  - cv_folds < 2:  单次 train/test 划分（hold-out）
"""

import time
import warnings

import numpy as np
from scipy.stats import pearsonr

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import torch
import torch.nn as nn
import torch.optim as optim


# ── 辅助 ──────────────────────────────────────────────────────────────

def _pcc_10digit(pcc_val: float) -> float:
    """将 PCC 保留 10 位有效数字。"""
    if np.isnan(pcc_val):
        return 0.0
    return float(f"{pcc_val:.10g}")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    计算 PCC / RMSE / R²。
    处理常量预测导致的 NaN（PCC 返回 0.0）。
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    std_true = float(np.std(y_true))
    std_pred = float(np.std(y_pred))
    if std_true < 1e-14 or std_pred < 1e-14:
        pcc_raw = 0.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pcc_stat = pearsonr(y_true, y_pred).statistic
        pcc_raw = float(pcc_stat) if not np.isnan(pcc_stat) else 0.0

    pcc_val = _pcc_10digit(pcc_raw)
    rmse_val = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2_val = float(r2_score(y_true, y_pred))
    return pcc_val, rmse_val, r2_val


# ── LSTM 模型定义 ──────────────────────────────────────────────────────

class _LSTMModel(nn.Module):
    """简单 LSTM → 全连接 回归模型。"""
    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _train_lstm_predict(model, X_train, y_train, X_pred,
                        epochs=200, lr=0.001, device="cpu"):
    """训练 LSTM 并预测 X_pred。"""
    model = model.to(device)
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_pred_t = torch.FloatTensor(X_pred).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train_t).squeeze(-1)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        return model(X_pred_t).squeeze(-1).cpu().numpy()


def _fit_and_predict_pair(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_a: np.ndarray,
    X_b: np.ndarray,
    params: dict,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """拟合一次，分别对 X_a、X_b 预测。"""
    name = name.upper()

    if name == "RF":
        model = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 5),
            random_state=params.get("random_state", random_state),
            n_jobs=params.get("n_jobs", 1),
        )
        model.fit(X_train, y_train)
        return model.predict(X_a), model.predict(X_b)

    if name == "KNN":
        model = KNeighborsRegressor(
            n_neighbors=_effective_knn_neighbors(params, len(y_train)),
            weights=params.get("weights", "uniform"),
            n_jobs=params.get("n_jobs", 1),
        )
        model.fit(X_train, y_train)
        return model.predict(X_a), model.predict(X_b)

    if name == "SVR":
        svr_scaler = StandardScaler()
        X_train_s = svr_scaler.fit_transform(X_train)
        model = SVR(
            kernel=params.get("kernel", "rbf"),
            C=params.get("C", 1.0),
            epsilon=params.get("epsilon", 0.1),
            gamma=params.get("gamma", "scale"),
        )
        model.fit(X_train_s, y_train)
        return model.predict(svr_scaler.transform(X_a)), model.predict(svr_scaler.transform(X_b))

    if name == "LR":
        model = Ridge(
            alpha=params.get("alpha", 1.0),
            random_state=params.get("random_state", random_state),
        )
        model.fit(X_train, y_train)
        return model.predict(X_a), model.predict(X_b)

    if name == "LSTM":
        n_feat = X_train.shape[1]
        lstm_model = _LSTMModel(
            input_size=1,
            hidden_size=params.get("hidden_size", 64),
            num_layers=params.get("num_layers", 2),
        )
        epochs = params.get("epochs", 200)
        lr = params.get("lr", 0.001)
        pred_a = _train_lstm_predict(
            lstm_model, X_train.reshape(-1, n_feat, 1), y_train,
            X_a.reshape(-1, n_feat, 1), epochs=epochs, lr=lr,
        )
        lstm_model2 = _LSTMModel(
            input_size=1,
            hidden_size=params.get("hidden_size", 64),
            num_layers=params.get("num_layers", 2),
        )
        pred_b = _train_lstm_predict(
            lstm_model2, X_train.reshape(-1, n_feat, 1), y_train,
            X_b.reshape(-1, n_feat, 1), epochs=epochs, lr=lr,
        )
        return pred_a, pred_b

    raise ValueError(
        f"Unknown regressor: '{name}'. Choose from RF, KNN, SVR, LR, LSTM."
    )


def _effective_knn_neighbors(params: dict, n_samples_fit: int) -> int:
    """KNN 邻居数不超过训练样本数（极小蛋白 + 多折 CV 时需要）。"""
    k = int(params.get("n_neighbors", 5))
    return max(1, min(k, n_samples_fit))


def _fit_and_predict(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    params: dict,
    random_state: int,
) -> np.ndarray:
    """在训练集上拟合，对 X_pred 样本预测。"""
    name = name.upper()

    if name == "RF":
        model = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 5),
            random_state=params.get("random_state", random_state),
            n_jobs=params.get("n_jobs", 1),
        )
        model.fit(X_train, y_train)
        return model.predict(X_pred)

    if name == "KNN":
        model = KNeighborsRegressor(
            n_neighbors=_effective_knn_neighbors(params, len(y_train)),
            weights=params.get("weights", "uniform"),
            n_jobs=params.get("n_jobs", 1),
        )
        model.fit(X_train, y_train)
        return model.predict(X_pred)

    if name == "SVR":
        svr_scaler = StandardScaler()
        X_train_s = svr_scaler.fit_transform(X_train)
        X_pred_s = svr_scaler.transform(X_pred)
        model = SVR(
            kernel=params.get("kernel", "rbf"),
            C=params.get("C", 1.0),
            epsilon=params.get("epsilon", 0.1),
            gamma=params.get("gamma", "scale"),
        )
        model.fit(X_train_s, y_train)
        return model.predict(X_pred_s)

    if name == "LR":
        model = Ridge(
            alpha=params.get("alpha", 1.0),
            random_state=params.get("random_state", random_state),
        )
        model.fit(X_train, y_train)
        return model.predict(X_pred)

    if name == "LSTM":
        n_feat = X_train.shape[1]
        X_train_l = X_train.reshape(X_train.shape[0], n_feat, 1)
        X_pred_l = X_pred.reshape(X_pred.shape[0], n_feat, 1)
        lstm_model = _LSTMModel(
            input_size=1,
            hidden_size=params.get("hidden_size", 64),
            num_layers=params.get("num_layers", 2),
        )
        return _train_lstm_predict(
            lstm_model, X_train_l, y_train, X_pred_l,
            epochs=params.get("epochs", 200),
            lr=params.get("lr", 0.001),
        )

    raise ValueError(
        f"Unknown regressor: '{name}'. Choose from RF, KNN, SVR, LR, LSTM."
    )


def _evaluate_holdout(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    test_size: float,
    random_state: int,
    *,
    as_single: bool = False,
) -> dict:
    """单次 train/test 划分评估。"""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=test_size, random_state=random_state)

    t_start = time.time()
    train_pred, test_pred = _fit_and_predict_pair(
        name, X_train, y_train, X_train, X_test, params, random_state)
    elapsed = time.time() - t_start

    train_pcc, train_rmse, train_r2 = _compute_metrics(y_train, train_pred)
    test_pcc, test_rmse, test_r2 = _compute_metrics(y_test, test_pred)

    if as_single:
        sp = _pcc_10digit(test_pcc)
        return {
            "cv_mode": "holdout",
            "cv_folds": 0,
            "train_pcc": train_pcc,
            "test_pcc": test_pcc,
            "single_pcc": sp,
            "oof_pcc": 0.0,
            "best_fold_pcc": 0.0,
            "best_fold_idx": 0,
            "mean_fold_pcc": 0.0,
            "test_pcc_std": 0.0,
            "fold_val_pccs": [0.0],
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "time_s": float(elapsed),
        }

    return {
        "cv_mode": "holdout",
        "cv_folds": 1,
        "train_pcc": train_pcc,
        "test_pcc": test_pcc,
        "single_pcc": _pcc_10digit(test_pcc),
        "oof_pcc": test_pcc,
        "best_fold_pcc": test_pcc,
        "best_fold_idx": 1,
        "mean_fold_pcc": test_pcc,
        "test_pcc_std": 0.0,
        "fold_val_pccs": [test_pcc],
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "time_s": float(elapsed),
    }


def _evaluate_kfold(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    cv_folds: int,
    random_state: int,
) -> dict:
    """K 折交叉验证：test_pcc 为 OOF 预测的整体 PCC。"""
    n_samples = len(y)
    n_splits = min(cv_folds, n_samples)
    if n_splits < 2:
        return _evaluate_holdout(name, X, y, params, test_size=0.2, random_state=random_state)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(n_samples, dtype=np.float64)
    fold_val_pccs = []
    fold_train_pccs = []

    t_start = time.time()
    for train_idx, val_idx in kf.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        y_train = y[train_idx]
        y_val = y[val_idx]

        train_pred, val_pred = _fit_and_predict_pair(
            name, X_train, y_train, X_train, X_val, params, random_state)

        oof_pred[val_idx] = val_pred
        vpcc, _, _ = _compute_metrics(y_val, val_pred)
        tpcc, _, _ = _compute_metrics(y_train, train_pred)
        fold_val_pccs.append(vpcc)
        fold_train_pccs.append(tpcc)

    elapsed = time.time() - t_start

    test_pcc, test_rmse, test_r2 = _compute_metrics(y, oof_pred)
    test_pcc_std = float(np.std(fold_val_pccs)) if fold_val_pccs else 0.0
    train_pcc = float(np.mean(fold_train_pccs)) if fold_train_pccs else 0.0
    mean_fold_pcc = float(np.mean(fold_val_pccs)) if fold_val_pccs else test_pcc
    best_fold_idx = int(np.argmax(fold_val_pccs)) + 1 if fold_val_pccs else 1
    best_fold_pcc = float(fold_val_pccs[best_fold_idx - 1]) if fold_val_pccs else test_pcc

    return {
        "cv_mode": "kfold",
        "cv_folds": n_splits,
        "train_pcc": _pcc_10digit(train_pcc),
        "test_pcc": test_pcc,
        "single_pcc": 0.0,
        "oof_pcc": test_pcc,
        "best_fold_pcc": _pcc_10digit(best_fold_pcc),
        "best_fold_idx": best_fold_idx,
        "mean_fold_pcc": _pcc_10digit(mean_fold_pcc),
        "test_pcc_std": _pcc_10digit(test_pcc_std),
        "fold_val_pccs": [_pcc_10digit(p) for p in fold_val_pccs],
        "train_rmse": 0.0,
        "test_rmse": test_rmse,
        "train_r2": 0.0,
        "test_r2": test_r2,
        "time_s": float(elapsed),
    }


# ── 统一评估入口 ───────────────────────────────────────────────────────

def evaluate_regressor(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    params: dict | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 10,
    use_cv: bool | None = None,
) -> dict:
    """
    训练并评估一个回归器。

    use_cv=True：K 折交叉验证，single_pcc=0，记录 oof/best_fold/mean_fold。
    use_cv=False：单次 train/test，single_pcc 有效，CV 相关指标填 0。

    Returns
    -------
    dict
        single_pcc, oof_pcc, best_fold_pcc, best_fold_idx, mean_fold_pcc, ...
    """
    if params is None:
        params = {}

    if use_cv is None:
        use_cv = cv_folds >= 2

    if use_cv:
        folds = cv_folds if cv_folds >= 2 else 10
        return _evaluate_kfold(name, X, y, params, folds, random_state)

    return _evaluate_holdout(
        name, X, y, params, test_size, random_state, as_single=True)
