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
"""

import time
import warnings

import numpy as np
from scipy.stats import pearsonr

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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

    # ---- PCC ----
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

    # ---- RMSE ----
    rmse_val = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # ---- R² ----
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
        # 取最后一个时间步的输出
        return self.fc(out[:, -1, :])


def _train_lstm(model, X_train, y_train, X_test, y_test,
                epochs=200, lr=0.001, device="cpu"):
    """训练 LSTM，返回 (train_pred, test_pred) numpy 数组。"""
    model = model.to(device)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

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
        train_pred = model(X_train_t).squeeze(-1).cpu().numpy()
        test_pred = model(X_test_t).squeeze(-1).cpu().numpy()

    return train_pred, test_pred


# ── 统一评估入口 ───────────────────────────────────────────────────────

def evaluate_regressor(name: str,
                       X: np.ndarray,
                       y: np.ndarray,
                       params: dict | None = None,
                       test_size: float = 0.2,
                       random_state: int = 42) -> dict:
    """
    训练并评估一个回归器。

    Parameters
    ----------
    name : str
        算法名: 'RF', 'KNN', 'SVR', 'LR', 'LSTM'。
    X : np.ndarray  shape (n_samples, n_features)
        特征矩阵。
    y : np.ndarray  shape (n_samples,)
        目标向量。
    params : dict | None
        模型超参数字典。不传则使用默认值。
    test_size : float
        测试集比例。
    random_state : int
        随机种子。

    Returns
    -------
    dict
        train_pcc, test_pcc, train_rmse, test_rmse,
        train_r2, test_r2, time_s
    """
    if params is None:
        params = {}

    name = name.upper()

    # 1. 标准化特征
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 2. 划分训练 / 测试
    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=test_size, random_state=random_state)

    t_start = time.time()

    # 3. 按算法名训练
    if name == "RF":
        model = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 5),
            random_state=params.get("random_state", random_state),
            n_jobs=params.get("n_jobs", 1),
        )
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

    elif name == "KNN":
        model = KNeighborsRegressor(
            n_neighbors=params.get("n_neighbors", 5),
            weights=params.get("weights", "uniform"),
            n_jobs=params.get("n_jobs", 1),
        )
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

    elif name == "SVR":
        # SVR 默认对特征尺度敏感，内部再做一次 StandardScaler
        svr_scaler = StandardScaler()
        X_train_s = svr_scaler.fit_transform(X_train)
        X_test_s = svr_scaler.transform(X_test)
        model = SVR(
            kernel=params.get("kernel", "rbf"),
            C=params.get("C", 1.0),
            epsilon=params.get("epsilon", 0.1),
            gamma=params.get("gamma", "scale"),
        )
        model.fit(X_train_s, y_train)
        train_pred = model.predict(X_train_s)
        test_pred = model.predict(X_test_s)

    elif name == "LR":
        model = Ridge(
            alpha=params.get("alpha", 1.0),
            random_state=params.get("random_state", random_state),
        )
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

    elif name == "LSTM":
        # 将 X 变形为 (batch, seq_len, 1)
        X_train_l = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_l = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        hidden_size = params.get("hidden_size", 64)
        num_layers = params.get("num_layers", 2)
        epochs = params.get("epochs", 200)
        lr = params.get("lr", 0.001)

        lstm_model = _LSTMModel(input_size=1,
                                hidden_size=hidden_size,
                                num_layers=num_layers)
        train_pred, test_pred = _train_lstm(
            lstm_model, X_train_l, y_train, X_test_l, y_test,
            epochs=epochs, lr=lr,
        )

    else:
        raise ValueError(
            f"Unknown regressor: '{name}'. "
            f"Choose from RF, KNN, SVR, LR, LSTM."
        )

    t_end = time.time()
    elapsed = float(t_end - t_start)

    # 4. 计算指标
    train_pcc, train_rmse, train_r2 = _compute_metrics(y_train, train_pred)
    test_pcc, test_rmse, test_r2 = _compute_metrics(y_test, test_pred)

    return {
        "train_pcc": train_pcc,
        "test_pcc": test_pcc,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "time_s": elapsed,
    }
