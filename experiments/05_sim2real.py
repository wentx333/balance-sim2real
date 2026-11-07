import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
"""实验 5：真机数据回放 —— 把仿真和真机日志摆在同一张图上，并反推参数。

用法
  python experiments/05_sim2real.py            # 用 data/real_log.csv
  python experiments/05_sim2real.py --selftest # 没有真机日志时，用"已知增益的
                                               # 假真机"造一份，验证这套流程能
                                               # 把被改坏的参数找回来

真机日志格式（data/real_log.csv，200 Hz）
  t, v_ref, vx, pitch
  其中 pitch 用弧度，vx 用 m/s —— 和 env.State 的单位一致，别在导出时换成角度

产出
  results/05_sim2real.png
  stdout: 拟合前后的俯仰角 RMSE、搜到的最优参数
"""
import csv

import numpy as np

import plot
from controller import BalanceController, Command, Gains
from env import BalanceCarEnv

DT = 0.005
SELFTEST = "--selftest" in sys.argv
LOG = ROOT / "data" / "real_log.csv"

# 用来"演真机"的失真参数：IMU 装歪 1.2°、质量重 25%、地面更滑、
# 电机有死区、控制延迟 10 ms —— 每一项都是实车上真实会遇到的东西
TRUE_PERTURB = dict(mass_scale=1.25, friction_scale=0.70, delay_ms=10.0,
                    deadband=0.04, imu_bias_deg=1.2)


def v_profile(t: float) -> float:
    if t < 1.0:
        return 0.0
    if t < 3.0:
        return 0.40
    if t < 4.5:
        return -0.20
    return 0.0


def rollout(perturb: dict | None = None, t_end: float = 6.0):
    env = BalanceCarEnv(ctrl_hz=1 / DT, delay_ms=3.0,
                        imu_noise=0.10, gyro_noise=0.4, seed=7)
    if perturb:
        env.set_params(**perturb)
    ctrl = BalanceController(Gains())
    s = env.reset()
    T, V, P, R = [], [], [], []
    for _ in range(int(t_end / DT)):
        v_ref = v_profile(s.t)
        tau_l, tau_r = ctrl.step(s, Command(v_ref=v_ref), DT)
        env.apply(tau_l, tau_r)
        s = env.state()
        T.append(s.t); V.append(s.vx); P.append(s.pitch); R.append(v_ref)
    return (np.array(T), np.array(V), np.array(P), np.array(R))


# ---------- 1. 拿到"真机"日志 ----------
if SELFTEST:
    LOG.parent.mkdir(exist_ok=True)
    T, V, P, R = rollout(TRUE_PERTURB)
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "v_ref", "vx", "pitch"])
        for i in range(len(T)):
            w.writerow([f"{T[i]:.4f}", f"{R[i]:.4f}", f"{V[i]:.5f}", f"{P[i]:.6f}"])
    print(f"[selftest] 已生成一份带失真的\"真机\"日志: {LOG}")
    print(f"[selftest] 真实失真参数: {TRUE_PERTURB}")

rows = list(csv.DictReader(open(LOG, encoding="utf-8")))
rt = np.array([float(r["t"]) for r in rows])
rv = np.array([float(r["vx"]) for r in rows])
rp = np.array([float(r["pitch"]) for r in rows])
print(f"读入真机日志 {len(rows)} 点，时长 {rt[-1]:.1f} s")


def replay(perturb: dict | None = None, t_end: float = None):
    """按真机日志里的 v_ref 序列在仿真里重放，返回同一时间轴上的 pitch/vx。"""
    env = BalanceCarEnv(ctrl_hz=1 / DT, delay_ms=3.0,
                        imu_noise=0.10, gyro_noise=0.4, seed=7)
    if perturb:
        env.set_params(**perturb)
    ctrl = BalanceController(Gains())
    s = env.reset()
    n = len(rt) if t_end is None else int(t_end / DT)
    P, V = [], []
    i = 0
    while len(P) < min(n, len(rt)):
        v_ref = float(rows[i]["v_ref"])
        tau_l, tau_r = ctrl.step(s, Command(v_ref=v_ref), DT)
        env.apply(tau_l, tau_r)
        s = env.state()
        P.append(s.pitch); V.append(s.vx); i += 1
    return np.array(P), np.array(V)


def score(perturb: dict | None) -> float:
    P, _ = replay(perturb)
    return float(np.rad2deg(np.sqrt(np.mean((P - rp[:len(P)]) ** 2))))


base_err = score(None)
print(f"标称参数下的俯仰角 RMSE: {base_err:.2f} deg")

# ---------- 2. 分两步反推：先标 IMU 零偏，再标机械参数 ----------
# 先标零偏是因为它一阶主导误差，而且物理意义明确（装歪多少度）；
# 混在一起做网格会互相补偿，标出来的参数看着好、其实没有物理含义。
best_bias, best_bias_err = 0.0, base_err
for bias in np.arange(-2.0, 2.01, 0.1):
    e = score(dict(imu_bias_deg=float(bias)))
    if e < best_bias_err:
        best_bias, best_bias_err = float(bias), e
print(f"第一步 IMU 零偏: {best_bias:+.1f} deg -> RMSE {best_bias_err:.2f} deg")

best = (best_bias_err, dict(imu_bias_deg=best_bias))
for ms in [0.85, 0.95, 1.05, 1.15, 1.25, 1.35]:
    p = dict(mass_scale=ms, delay_ms=3.0, deadband=0.04, imu_bias_deg=best_bias)
    e = score(p)
    if e < best[0]:
        best = (e, p)
fit_err, fit_p = best
print(f"第二步 质量缩放: {fit_p['mass_scale']:.2f}x -> RMSE {fit_err:.2f} deg")

# 地面摩擦系数在这条轨迹上几乎不可辨识：轮子全程没打滑，摩擦只在
# 大扭矩时才有存在感。把这一点直接测出来，比硬塞一个数字诚实得多。
print("摩擦系数敏感性（固定其它参数）：")
for fs in [0.5, 0.7, 0.9, 1.1, 1.4]:
    e = score(dict(fit_p, friction_scale=fs))
    print(f"   mu x{fs:.1f} -> RMSE {e:.2f} deg")
print(f"参数搜索最优: RMSE {fit_err:.2f} deg  参数 {fit_p}")
print(f"失配降低 {base_err:.2f} -> {fit_err:.2f} deg（{(1 - fit_err / base_err) * 100:.0f}%）")

P0, V0 = replay(None)
P1, V1 = replay(fit_p)

out = ROOT / "results"
out.mkdir(exist_ok=True)
with open(out / "05_sim2real.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["t", "pitch_real", "pitch_sim_nominal", "pitch_sim_fitted",
                "vx_real", "vx_sim_nominal", "vx_sim_fitted"])
    n = min(len(rt), len(P0), len(P1))
    for i in range(n):
        w.writerow([rt[i], rp[i], P0[i], P1[i], rv[i], V0[i], V1[i]])

plot.style()
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
ax[0].plot(rt[:n], np.rad2deg(rp[:n]), color="black", lw=1.5, label="真机")
ax[0].plot(rt[:n], np.rad2deg(P0[:n]), color=plot.GREY, lw=1.2, ls="--",
           label=f"仿真·标称 (RMSE {base_err:.1f}°)")
ax[0].plot(rt[:n], np.rad2deg(P1[:n]), color=plot.ACCENT, lw=1.3,
           label=f"仿真·辨识后 (RMSE {fit_err:.1f}°)")
ax[0].set_ylabel("pitch [deg]"); ax[0].legend(ncol=3, fontsize=8)
ax[0].set_title("俯仰角：真机 vs 仿真", fontsize=10)

ax[1].plot(rt[:n], rv[:n], color="black", lw=1.5, label="真机")
ax[1].plot(rt[:n], V0[:n], color=plot.GREY, lw=1.2, ls="--", label="仿真·标称")
ax[1].plot(rt[:n], V1[:n], color=plot.ACCENT, lw=1.3, label="仿真·辨识后")
ax[1].set_ylabel("v_x [m/s]"); ax[1].set_xlabel("time [s]"); ax[1].legend(ncol=3, fontsize=8)
ax[1].set_title("速度：真机 vs 仿真", fontsize=10)
print("saved", plot.save(fig, out / "05_sim2real.png"))
