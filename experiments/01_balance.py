import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
"""实验 1：抗扰恢复 —— 初始倾斜 + 侧向撞击，看能不能站住、多快站稳。

产出
  results/01_balance.png   四联图：俯仰角 / 速度 / 左右轮力矩 / 位移
  results/01_balance.csv   原始数据（可直接拿去和真机比）
  stdout                   恢复时间、峰值力矩、残余倾角
"""
import csv

import numpy as np

import plot
from controller import BalanceController, Command, Gains
from env import BalanceCarEnv

DT = 0.005          # 200 Hz 控制周期，与 STM32 固件一致
T_END = 6.0
PUSH_T = 2.0
PUSH_F = 18.0       # N，推车的外力
PUSH_DUR = 0.10     # s，作用时长 -> 冲量 1.8 N·s，约相当于 1.0 m/s 的撞击

env = BalanceCarEnv(ctrl_hz=1 / DT, delay_ms=3.0, imu_noise=0.10, gyro_noise=0.4)
ctrl = BalanceController(Gains())
s = env.reset(pitch=0.10)          # 初始前倾 5.7°

log = []
for k in range(int(T_END / DT)):
    cmd = Command(v_ref=0.0)
    tau_l, tau_r = ctrl.step(s, cmd, DT)
    fx = PUSH_F if (PUSH_T <= s.t < PUSH_T + PUSH_DUR) else 0.0
    env.apply(tau_l, tau_r, fx=fx)
    s = env.state()
    log.append([s.t, s.pitch, s.pitch_rate, s.vx, s.x, tau_l, tau_r])

a = np.array(log)
t, pitch, prate, vx, x, tl, tr = a.T

# ---- 指标 ----
after = t >= PUSH_T
pa = pitch[after]; ta = t[after]
i_peak = int(np.argmax(np.abs(pa)))          # 先让它倾过去，再从峰值之后算恢复
band = np.abs(pa[i_peak:]) < np.deg2rad(2.0)
recover = float("nan")
if band.any():
    # 要求连续 0.2 s 都在带内，避免恰好穿过零点被误判恢复
    n = int(0.2 / DT)
    idx = np.nonzero(np.convolve(band, np.ones(n), "valid") == n)[0]
    if len(idx):
        recover = float(ta[i_peak + idx[0]] - PUSH_T)
print(f"撞击后峰值速度  {np.max(np.abs(vx[after])):.2f} m/s")
print(f"撞击后峰值倾角  {np.rad2deg(np.abs(pa).max()):.2f} deg")
print(f"残余倾角        {np.deg2rad(np.abs(pitch[-200:]).mean()):.3f} deg")
print(f"撞击后恢复(<2°) {recover*1000:.0f} ms")
print(f"峰值力矩        {np.max(np.abs(a[:,5:7])):.2f} N·m")
print(f"末速度          {vx[-1]:+.3f} m/s")
print(f"最大位移        {np.max(np.abs(x)):.3f} m")

out = ROOT / "results"
out.mkdir(exist_ok=True)
with open(out / "01_balance.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["t", "pitch", "pitch_rate", "vx", "x", "tau_l", "tau_r"])
    w.writerows(log)

plot.style()
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(9, 6), sharex=True)

ax[0, 0].plot(t, np.rad2deg(pitch), color=plot.BLUE, lw=1.4)
ax[0, 0].axvline(PUSH_T, ls="--", lw=1, color=plot.ACCENT)
ax[0, 0].set_ylabel("pitch [deg]")
ax[0, 0].set_title("机身俯仰角（虚线=撞击时刻）", fontsize=10)

ax[0, 1].plot(t, vx, color=plot.ACCENT, lw=1.4)
ax[0, 1].axhline(0, color=plot.GREY, lw=0.8)
ax[0, 1].set_ylabel("v_x [m/s]")
ax[0, 1].set_title("前进速度", fontsize=10)

ax[1, 0].plot(t, tl, lw=1.2, label="left")
ax[1, 0].plot(t, tr, lw=1.2, label="right")
ax[1, 0].axhline(2.0, ls=":", color=plot.GREY, lw=1)
ax[1, 0].axhline(-2.0, ls=":", color=plot.GREY, lw=1)
ax[1, 0].set_ylabel("torque [N·m]")
ax[1, 0].set_xlabel("time [s]")
ax[1, 0].set_title("轮端力矩（点线=饱和限幅）", fontsize=10)
ax[1, 0].legend(loc="upper right", ncol=2)

ax[1, 1].plot(t, x, color="#4b8f5a", lw=1.4)
ax[1, 1].set_ylabel("x [m]")
ax[1, 1].set_xlabel("time [s]")
ax[1, 1].set_title("位移", fontsize=10)

print("saved", plot.save(fig, out / "01_balance.png"))
