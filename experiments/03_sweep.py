import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
"""实验 3：鲁棒性扫描 —— 控制延迟 x 力矩上限，看性能在哪儿崩。

为什么扫这两个量
  延迟是 sim2real 最狠的杀手：仿真里 1 ms 的延迟，上机变成 CAN 传输 + 任务
  调度 + PWM 更新，轻松到 10 ms。力矩上限则决定了"救不救得回来"——车倒了
  一半，扭矩不够就是物理上救不回来，调参再多也没用。

产出
  results/03_sweep.png    左：恢复时间热图；右：峰值倾角热图；失稳格打叉
  results/03_sweep.csv    原始网格
"""
import csv

import numpy as np

import plot
from controller import BalanceController, Command, Gains
from env import BalanceCarEnv

DT = 0.005
T_END = 5.0
PUSH_T, PUSH_F, PUSH_DUR = 2.0, 18.0, 0.10
FALL = np.deg2rad(35.0)          # 超过这个角度就当倒了

# 200 Hz 控制周期是 5 ms，所以延迟按 5 ms 一档增加才有意义
DELAYS = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
TAUS = [2.0, 1.4, 1.0, 0.7, 0.5, 0.35]


def run(delay_ms: float, tau_max: float, seed: int = 0) -> dict:
    env = BalanceCarEnv(ctrl_hz=1 / DT, delay_ms=delay_ms,
                        imu_noise=0.10, gyro_noise=0.4, seed=seed)
    ctrl = BalanceController(Gains(tau_max=tau_max))
    s = env.reset(pitch=0.10)
    T, P, V = [], [], []
    for _ in range(int(T_END / DT)):
        tau_l, tau_r = ctrl.step(s, Command(), DT)
        fx = PUSH_F if (PUSH_T <= s.t < PUSH_T + PUSH_DUR) else 0.0
        env.apply(tau_l, tau_r, fx=fx)
        s = env.state()
        T.append(s.t); P.append(s.pitch); V.append(s.vx)
    T, P, V = np.array(T), np.array(P), np.array(V)

    fallen = bool(np.abs(P).max() > FALL)
    after = T >= PUSH_T
    pa, ta = P[after], T[after]
    i_pk = int(np.argmax(np.abs(pa)))
    band = np.abs(pa[i_pk:]) < np.deg2rad(2.0)
    rec = float("nan")
    if band.any():
        n = int(0.2 / DT)
        idx = np.nonzero(np.convolve(band, np.ones(n), "valid") == n)[0]
        if len(idx):
            rec = float(ta[i_pk + idx[0]] - PUSH_T)
    return dict(fallen=fallen, recover_s=rec,
                peak_tilt_deg=float(np.rad2deg(np.abs(pa).max())),
                rms_pitch_deg=float(np.rad2deg(np.sqrt(np.mean(pa ** 2)))),
                v_end=float(V[-1]))


rows = []
print(f"{'delay':>6}{'tau_max':>9}{'倒了':>6}{'恢复(s)':>10}{'峰值倾角':>10}{'末速度':>9}")
for d in DELAYS:
    for tq in TAUS:
        r = run(d, tq)
        rows.append([d, tq, int(r["fallen"]), r["recover_s"],
                     r["peak_tilt_deg"], r["rms_pitch_deg"], r["v_end"]])
        print(f"{d:>6.0f}{tq:>9.1f}{'是' if r['fallen'] else '否':>6}"
              f"{r['recover_s']:>10.2f}{r['peak_tilt_deg']:>10.1f}{r['v_end']:>9.3f}")

a = np.array(rows)
out = ROOT / "results"
out.mkdir(exist_ok=True)
with open(out / "03_sweep.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["delay_ms", "tau_max", "fallen", "recover_s",
                "peak_tilt_deg", "rms_pitch_deg", "v_end"])
    w.writerows(rows)

# ---- 热图 ----
rec = a[:, 3].reshape(len(DELAYS), len(TAUS))
tilt = a[:, 4].reshape(len(DELAYS), len(TAUS))
fall = a[:, 2].reshape(len(DELAYS), len(TAUS)).astype(bool)

plot.style()
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
for ax, data, title, cmap in [
    (axes[0], rec, "撞击后恢复时间 [s]", "viridis"),
    (axes[1], tilt, "撞击后峰值倾角 [deg]", "magma"),
]:
    masked = np.ma.masked_where(fall | np.isnan(data), data)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#e2e5ea")            # 倒地格单独用灰色，不和数值混淆
    vmin, vmax = float(masked.min()), float(masked.max())
    im = ax.imshow(masked, cmap=cm, aspect="auto", origin="lower",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(TAUS)), [f"{t:.1f}" for t in TAUS])
    ax.set_yticks(range(len(DELAYS)), [f"{d:.0f}" for d in DELAYS])
    ax.set_xlabel("力矩上限 [N·m]"); ax.set_ylabel("控制延迟 [ms]")
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    for i in range(len(DELAYS)):
        for j in range(len(TAUS)):
            if fall[i, j]:
                ax.text(j, i, "倒地", ha="center", va="center",
                        fontsize=7, color="#6b7280")
            elif np.isnan(data[i, j]):
                ax.text(j, i, "未恢复", ha="center", va="center",
                        fontsize=7, color="#6b7280")
            else:
                rel = (data[i, j] - vmin) / max(vmax - vmin, 1e-9)
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
                        fontsize=7, color="black" if rel > 0.6 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
fig.suptitle("鲁棒性扫描：灰色 = 倒地或 5 s 内未恢复", fontsize=10)
print("\n失稳格数:", int(fall.sum()), "/", fall.size)
print("saved", plot.save(fig, out / "03_sweep.png"))
