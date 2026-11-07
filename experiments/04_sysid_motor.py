import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
"""实验 4：台架参数辨识 —— 从力矩-转速数据反推 J 与 b。

场景：车架在支架上空转（不接地、不受重力），给定力矩阶跃，测转速上升曲线。
模型：tau = J * dω/dt + b * ω
      两边都已知（tau 是我们给的，ω 是编码器测的），对 (α, ω) 做最小二乘
      就能把 J、b 解出来 —— 这就是"拿到一台新电机，先标定再用"的标准动作。

产出
  results/04_sysid.png     左：多组力矩阶跃的转速响应；右：拟合散点
  results/04_sysid.csv     原始数据
"""
import csv

import numpy as np

import plot
from env import BalanceCarEnv

DT = 0.002          # 500 Hz 采样，和编码器读取频率一致
T_END = 1.6
TORQUES = [0.15, 0.25, 0.40, 0.60, 0.90]
R_WHEEL = 0.05

curves = []
for tq in TORQUES:
    env = BalanceCarEnv(ctrl_hz=1 / DT, timestep=1e-3)
    env.bench_mode()                     # 关重力、断地面接触
    env.apply(0.0, 0.0)                  # 先让状态稳定
    W, Tq = [], []
    for k in range(int(T_END / DT)):
        env.spin(tq, tq, env.decim)
        w = float(np.mean(env.data.qvel[6:8]))
        W.append(w); Tq.append(tq)
    W = np.array(W)
    t = np.arange(len(W)) * DT
    alpha = np.gradient(W, DT)
    curves.append((t, W, alpha, tq))

# ---- 最小二乘：tau = [alpha, omega] · [J, b] ----
A_rows, y_rows = [], []
for t, W, alpha, tq in curves:
    sel = slice(5, -5)                   # 去掉头尾的数值微分毛刺
    A_rows.append(np.stack([alpha[sel], W[sel]], axis=1))
    y_rows.append(np.full(len(alpha[sel]), tq))
A = np.concatenate(A_rows); y = np.concatenate(y_rows)
(J, b), *_ = np.linalg.lstsq(A, y, rcond=None)
y_hat = A @ np.array([J, b])
r2 = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)

# 仿真里这块质量的真实等效惯量（用于验证辨识误差）
m_true = env.model.body_mass
j_true = float(env.model.body_inertia[
    __import__("mujoco").mj_name2id(env.model, __import__("mujoco").mjtObj.mjOBJ_BODY, "wheel_l"), 1])
j_true += float(env.model.dof_armature[6])

print(f"辨识结果: J = {J:.6f} kg·m^2   b = {b:.6f} N·m·s/rad   R2 = {r2:.4f}")
print(f"模型真值: J = {j_true:.6f} kg·m^2（轮体转动惯量 + armature）")
print(f"惯量辨识误差: {abs(J - j_true) / j_true * 100:.1f}%")

out = ROOT / "results"
out.mkdir(exist_ok=True)
with open(out / "04_sysid.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["t", "torque", "omega", "alpha"])
    for t, W, alpha, tq in curves:
        for i in range(len(t)):
            w.writerow([t[i], tq, W[i], alpha[i]])

plot.style()
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
for t, W, alpha, tq in curves:
    ax[0].plot(t, W, lw=1.3, label=f"{tq:.2f} N·m")
ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("ω [rad/s]")
ax[0].set_title("空载力矩阶跃响应（台架）", fontsize=10); ax[0].legend(ncol=2, fontsize=7)

# 把 b·ω 减掉之后，所有转速下的点应该塌到一条过原点的直线上，斜率就是 J
resid = y - b * A[:, 1]
ax[1].scatter(A[:, 0], resid, s=4, alpha=0.30, color=plot.BLUE, label="实测（已扣除 b·ω）")
lim = np.array([0.0, A[:, 0].max() * 1.05])
ax[1].plot(lim, J * lim, color=plot.ACCENT, lw=1.6, label=f"拟合 J = {J:.5f}")
ax[1].set_xlabel("α [rad/s^2]"); ax[1].set_ylabel("τ - b·ω [N·m]")
ax[1].set_title(f"扣除粘滞项后塌缩为直线   b={b:.5f}   R2={r2:.4f}", fontsize=9)
ax[1].legend(loc="upper left", fontsize=7)
print("saved", plot.save(fig, out / "04_sysid.png"))
