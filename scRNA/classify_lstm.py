#!/usr/bin/env python3
# Author: Zihan Wang
# <wangzh011031@163.com>
"""
LSTM 分类器：150 维汇总特征 / L01–L10 轨迹（100 点或完整保存轨迹）。

- stats6_topo6_l10_pca30: (10, 12) 层特征 + PCA30
- traj100_l10: (10, 100) Burgers u(t) 轨迹，每层 100 采样点
- trajfull_l10 / trajfull: (10, T) 或 (1, T) 完整保存轨迹（通常 T=3001）
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler


def resolve_lstm_device(device: str = "auto") -> str:
    """auto → cuda（可用时）否则 cpu。"""
    choice = (device or "auto").strip().lower()
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return choice


def split_l10_pca30(
    X: np.ndarray,
    num_levels: int = 10,
    level_dim: int = 12,
    pca_dim: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """(n, 150) → seq (n, 10, 12), pca (n, 30)。"""
    need = num_levels * level_dim + pca_dim
    if X.shape[1] != need:
        raise ValueError(f"期望 {need} 维, 实际 {X.shape[1]}")
    seq = X[:, : num_levels * level_dim].reshape(-1, num_levels, level_dim)
    pca = X[:, num_levels * level_dim :]
    return seq, pca


class LSTMSliceClassifier(nn.Module):
    """L01–L10 序列 LSTM + PCA30 拼接 → 多类 logits。"""

    def __init__(
        self,
        seq_dim: int = 12,
        hidden_size: int = 64,
        num_layers: int = 2,
        pca_dim: int = 30,
        n_classes: int = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            seq_dim, hidden_size, num_layers, batch_first=True,
        )
        self.head = nn.Linear(hidden_size + pca_dim, n_classes)

    def forward(self, seq: torch.Tensor, pca: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(seq)
        hidden = out[:, -1, :]
        return self.head(torch.cat([hidden, pca], dim=1))


def _scale_seq_pca(
    seq_train: np.ndarray,
    pca_train: np.ndarray,
    seq_eval: np.ndarray,
    pca_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """折内 StandardScaler：序列按 (n*levels, dim) 拟合，PCA 按 (n, 30) 拟合。"""
    n_levels = seq_train.shape[1]
    seq_dim = seq_train.shape[2]
    seq_scaler = StandardScaler()
    seq_tr = seq_scaler.fit_transform(
        seq_train.reshape(-1, seq_dim),
    ).reshape(seq_train.shape)
    seq_ev = seq_scaler.transform(
        seq_eval.reshape(-1, seq_dim),
    ).reshape(seq_eval.shape)

    pca_scaler = StandardScaler()
    pca_tr = pca_scaler.fit_transform(pca_train)
    pca_ev = pca_scaler.transform(pca_eval)
    return seq_tr, pca_tr, seq_ev, pca_ev


def train_lstm_classifier(
    seq_train: np.ndarray,
    pca_train: np.ndarray,
    y_train: np.ndarray,
    seq_eval: np.ndarray,
    pca_eval: np.ndarray,
    *,
    hidden_size: int = 64,
    num_layers: int = 2,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    训练 LSTM 分类器，返回 (y_pred, y_proba, classes_encoded)。

    y_train / y_eval 可为任意整型标签；内部 LabelEncoder 映射到 0..C-1。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    le = LabelEncoder()
    y_tr = le.fit_transform(y_train)
    classes = le.classes_
    n_classes = len(classes)

    model = LSTMSliceClassifier(
        seq_dim=seq_train.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        pca_dim=pca_train.shape[1],
        n_classes=n_classes,
    ).to(device)
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    seq_t = torch.as_tensor(seq_train, dtype=torch.float32, device=device)
    pca_t = torch.as_tensor(pca_train, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_tr, dtype=torch.long, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n = seq_train.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(seq_t[idx], pca_t[idx])
            loss = criterion(logits, y_t[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        seq_ev_t = torch.as_tensor(seq_eval, dtype=torch.float32, device=device)
        pca_ev_t = torch.as_tensor(pca_eval, dtype=torch.float32, device=device)
        logits = model(seq_ev_t, pca_ev_t)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        pred_enc = logits.argmax(dim=1).cpu().numpy()

    y_pred = le.inverse_transform(pred_enc)
    return y_pred, proba, classes


def compute_lstm(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    hidden_size: int = 64,
    num_layers: int = 2,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 1,
    num_levels: int = 10,
    level_dim: int = 12,
    pca_dim: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """单折 LSTM 分类，接口对齐 compute_rf。"""
    seq_tr, pca_tr = split_l10_pca30(
        X_train, num_levels=num_levels, level_dim=level_dim, pca_dim=pca_dim,
    )
    seq_te, pca_te = split_l10_pca30(
        X_test, num_levels=num_levels, level_dim=level_dim, pca_dim=pca_dim,
    )
    seq_tr, pca_tr, seq_te, pca_te = _scale_seq_pca(seq_tr, pca_tr, seq_te, pca_te)

    y_pred, y_proba, classes = train_lstm_classifier(
        seq_tr, pca_tr, y_train, seq_te, pca_te,
        hidden_size=hidden_size,
        num_layers=num_layers,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
    return y_pred, y_proba, classes


class TrajLevelLSTMClassifier(nn.Module):
    """每层完整/采样轨迹 LSTM（共享权重），各层隐状态拼接 → 分类。"""

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_levels: int = 10,
        n_classes: int = 2,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size * num_levels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_levels, traj_len)
        # 逐层过共享 LSTM，避免 (batch×levels) 一次性展开导致 OOM
        b, n_lv, _ = x.shape
        hiddens: list[torch.Tensor] = []
        for lvl in range(n_lv):
            seq = x[:, lvl, :].unsqueeze(-1)  # (b, traj_len, 1)
            out, _ = self.lstm(seq)
            hiddens.append(out[:, -1, :])
        return self.head(torch.cat(hiddens, dim=1))


def _scale_traj(
    tr_train: np.ndarray,
    tr_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """折内 StandardScaler：按 (n*levels, traj_len) 拟合。"""
    n, n_levels, traj_len = tr_train.shape
    scaler = StandardScaler()
    tr_tr = scaler.fit_transform(
        tr_train.reshape(-1, traj_len),
    ).reshape(n, n_levels, traj_len)
    tr_ev = scaler.transform(
        tr_eval.reshape(-1, traj_len),
    ).reshape(tr_eval.shape)
    return tr_tr, tr_ev


def train_traj_lstm_classifier(
    tr_train: np.ndarray,
    y_train: np.ndarray,
    tr_eval: np.ndarray,
    *,
    hidden_size: int = 64,
    num_layers: int = 2,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    le = LabelEncoder()
    y_tr = le.fit_transform(y_train)
    classes = le.classes_
    n_classes = len(classes)

    model = TrajLevelLSTMClassifier(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_levels=tr_train.shape[1],
        n_classes=n_classes,
    ).to(device)
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    x_t = torch.as_tensor(tr_train, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_tr, dtype=torch.long, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n = tr_train.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(x_t[idx])
            loss = criterion(logits, y_t[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        x_ev = torch.as_tensor(tr_eval, dtype=torch.float32, device=device)
        logits = model(x_ev)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        pred_enc = logits.argmax(dim=1).cpu().numpy()

    y_pred = le.inverse_transform(pred_enc)
    return y_pred, proba, classes


def compute_traj_lstm(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    hidden_size: int = 64,
    num_layers: int = 2,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """单折轨迹 LSTM：(n, levels, traj_len) → 分类。"""
    if X_train.ndim != 3:
        raise ValueError(f"轨迹 LSTM 期望 X shape (n, levels, traj_len), 得到 {X_train.shape}")
    tr_tr = _sanitize_traj(X_train)
    tr_te = _sanitize_traj(X_test)
    tr_tr, tr_te = _scale_traj(tr_tr, tr_te)
    y_pred, y_proba, classes = train_traj_lstm_classifier(
        tr_tr, y_train, tr_te,
        hidden_size=hidden_size,
        num_layers=num_layers,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
    return y_pred, y_proba, classes


def _sanitize_traj(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -1e6, 1e6)
