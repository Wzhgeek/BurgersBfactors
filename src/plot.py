# Author: Zihan Wang
# <wangzh011031@163.com>
"""
绘图模块：u-x 空间剖面图 & u-t 自轨迹图。

依赖 code/step3_dyn/burgers.rk4_step 进行仿真。
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── 固定项目根路径，确保能导入 code 包 ──
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def plot_ux(A: np.ndarray,
            nu: float,
            eps: float,
            pdb_id: str,
            level: int,
            pulse_idx: int,
            dt: float,
            dx: float,
            n_steps: int,
            save_path) -> None:
    """
    绘制 u-x 空间剖面图：在 pulse_idx 处施加脉冲，于 5 个快照时刻
    展示空间分布。

    颜色：Purple(t=0) → Cyan → Green → Yellow → Red(t=t_end)。
    包含 u=0.1 的水平虚线 (lightcoral)，y 轴刻度间隔 0.1。

    Parameters
    ----------
    A : np.ndarray  shape (N, N)
        邻接 / 耦合矩阵。
    nu : float
        扩散系数。
    eps : float
        图耦合强度。
    pdb_id : str
        PDB 标识。
    level : int
        结构层级编号。
    pulse_idx : int
        脉冲施加的原子索引。
    dt : float
        时间步长。
    dx : float
        空间步长。
    n_steps : int
        总步数。
    save_path : str or Path
        图片保存路径。
    """
    from code.step3_dyn.burgers import rk4_step

    N = A.shape[0]
    x_coord = np.linspace(0, 1, N, endpoint=False) + 0.5 / N

    # ---- 仿真 ----
    u = np.zeros(N, dtype=np.float64)
    u[pulse_idx] = 1.0
    snaps: dict[float, np.ndarray] = {0.0: u.copy()}
    snap_steps = {0, n_steps // 4, n_steps // 2,
                  3 * n_steps // 4, n_steps}

    for step in range(1, n_steps + 1):
        u = rk4_step(u, dt, A, nu, eps, dx, "graph_diffusion")
        if step in snap_steps:
            snaps[step * dt] = u.copy()

    # ---- 绘图 ----
    colors = ["purple", "cyan", "green", "yellow", "red"]
    ts = sorted(snaps)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)

    for k, t in enumerate(ts):
        ax.plot(x_coord, snaps[t],
                color=colors[k], lw=1.2,
                marker="o", ms=3, mew=0.3, mec="white",
                label=rf"$t={t:.2f}$")

    # u=0.1 参考线
    ax.axhline(y=0.1, color="lightcoral", linestyle="--",
               lw=1.2, alpha=0.7)

    # y 轴刻度 0.1 间隔
    y_max = max(float(np.max(snaps[t])) for t in ts)
    y_top = max(y_max, 1.0) + 0.05
    ax.set_yticks(np.arange(0, y_top + 0.1, 0.1))

    ax.set_title(rf"{pdb_id} L{level:02d}: "
                 rf"$\nu={nu}$ $\varepsilon={eps}$")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.2)
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_ut(trj: np.ndarray,
            nu: float,
            eps: float,
            pdb_id: str,
            level: int,
            dt: float,
            sample_every: int,
            save_path) -> None:
    """
    绘制 u-t 自轨迹图：每行一个原子，着色按原子索引 (viridis)。

    包含 u=0.1 的水平虚线 (lightcoral)，y 轴刻度间隔 0.1。

    Parameters
    ----------
    trj : np.ndarray  shape (N, n_pts)
        轨迹矩阵，trj[i, :] 为原子 i 的自身分量 u_i(t)。
    nu : float
        扩散系数。
    eps : float
        图耦合强度。
    pdb_id : str
        PDB 标识。
    level : int
        结构层级编号。
    dt : float
        时间步长。
    sample_every : int
        采样间隔（每隔多少步记录一次）。
    save_path : str or Path
        图片保存路径。
    """
    N, n_pts = trj.shape

    times = np.arange(n_pts) * sample_every * dt

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=130)

    denom = max(N - 1, 1)
    for i in range(N):
        ax.plot(times, trj[i],
                color=plt.cm.viridis(i / denom),
                lw=0.5, alpha=0.8)

    # u=0.1 参考线
    ax.axhline(y=0.1, color="lightcoral", linestyle="--",
               lw=1.2, alpha=0.7)

    # y 轴刻度 0.1 间隔
    y_max = float(np.nanmax(trj))
    y_top = max(y_max, 1.0) + 0.05
    ax.set_yticks(np.arange(0, y_top + 0.1, 0.1))

    ax.set_title(rf"{pdb_id} L{level:02d} u-t: "
                 rf"$\nu={nu}$ $\varepsilon={eps}$")
    ax.grid(True, alpha=0.2)
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
