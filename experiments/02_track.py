import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
"""实验 2：速度指令跟踪 —— 方波指令，量化上升时间/超调/稳态误差。

产出
  results/02_track.png / .csv
  stdout: 每段阶跃的指标表
"""
import csv

import numpy as np

import plot
from controller import BalanceController, Command, Gains
from env import BalanceCarEnv
from metrics import step_metrics, tracking_metrics

DT = 0.005
# (起始时刻, 目标速度)
SEGMENTS = [(0.0, 0.0), (1.0, 0.5), (4.0, 0.0), (6.0, -0.3), (8.0, 0.0)]
T_END = 10.0

env = BalanceCarEnv(ctrl_hz=1 / DT, delay_ms=3.0, imu_noise=0.10, gyro_noise=0.4)
ctrl = BalanceController(Gains())
s = env.reset()

log, refs = [], []
idx = 0
for k in range(int(T_END / DT)):
    while idx + 1 < len(SEGMENTS) and s.t >= SEGMENTS[idx + 1][0]:
        idx += 1
    v_ref = SEGMENTS[idx][1]
    tau = ctrl.step(s, Command(v_ref=v_ref), DT)
    env.apply(*tau)
    s = env.state()
    log.append([s.t, v_ref, s.vx, s.pitch, tau[0], tau[1]])

a = np.array(log)
t, ref, vx, pitch = a[:, 0], a[:, 1], a[:, 2], a[:, 3]

print(f"{'段':<3}{'目标':>7}{'上升(ms)':>11}{'超调%':>8}{'调节(ms)':>11}{'稳态误差%':>11}")
for i in range(1, len(SEGMENTS)):
    t0 = SEGMENTS[i][0]
    t1 = SEGMENTS[i + 1][0] if i + 1 < len(SEGMENTS) else T_END
    # 分析窗口必须在本段结束前截断：否则下一段的反向指令会被算成"超调"
    m = (t >= t0) & (t < t1 - 0.05)
    y0 = vx[t < t0][-1] if (t < t0).any() else 0.0
    met = step_metrics(t[m] - t0, vx[m], y0, SEGMENTS[i][1])
    if met:
        print(f"{i:<3}{SEGMENTS[i][1]:>7.2f}{met['rise_time_s']*1000:>11.0f}"
              f"{met['overshoot_pct']:>8.1f}{met['settling_s']*1000:>11.0f}"
              f"{met['steady_error_pct']:>11.2f}")

tm = tracking_metrics(t, vx, ref)
print("\n全程跟踪: RMSE %.4f m/s  max|err| %.4f m/s" % (tm["rmse"], tm["max_abs_err"]))

out = ROOT / "results"
out.mkdir(exist_ok=True)
with open(out / "02_track.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["t", "v_ref", "vx", "pitch", "tau_l", "tau_r"])
    w.writerows(log)

plot.style()
import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 1, figsize=(9, 6.6), sharex=True)
ax[0].plot(t, ref, ls="--", lw=1.2, color=plot.GREY, label="v_ref")
ax[0].plot(t, vx, lw=1.5, color=plot.BLUE, label="v")
ax[0].set_ylabel("v [m/s]"); ax[0].legend(ncol=2); ax[0].set_title("速度跟踪", fontsize=10)
ax[1].plot(t, np.rad2deg(pitch), lw=1.4, color=plot.ACCENT)
ax[1].axhline(0, color=plot.GREY, lw=0.8); ax[1].set_ylabel("pitch [deg]"); ax[1].set_title("俯仰角（加速时前倾）", fontsize=10)
ax[2].plot(t, a[:, 4], lw=1.1); ax[2].plot(t, a[:, 5], lw=1.1)
ax[2].set_ylabel("torque [N·m]"); ax[2].set_xlabel("time [s]"); ax[2].set_title("轮端力矩", fontsize=10)
print("saved", plot.save(fig, out / "02_track.png"))
