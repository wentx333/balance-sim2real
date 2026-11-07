import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
"""实验 6：把仿真渲染成 GIF —— 简历/邮件里放一张会动的图，比十张曲线管用。

产出
  results/06_demo.gif

注意：需要能创建 OpenGL 上下文（带显卡的普通 Windows 机器都行；
纯 SSH 无显示的环境要装 OSMesa 版 mujoco）。
"""
import numpy as np

from controller import BalanceController, Command, Gains
from env import BalanceCarEnv

DT = 0.005
FPS = 20
EVERY = int(round(1.0 / (FPS * DT)))     # 每 10 个控制周期存一帧
W, H = 480, 360                          # GIF 一大就几十 MB，480p 足够了
T_END = 6.0


def v_profile(t: float) -> float:
    if t < 1.0:
        return 0.0
    if t < 2.6:
        return 0.45
    if t < 4.2:
        return -0.25
    return 0.0


env = BalanceCarEnv(ctrl_hz=1 / DT, delay_ms=3.0, imu_noise=0.10, gyro_noise=0.4,
                    seed=3)
ctrl = BalanceController(Gains())
s = env.reset()

import mujoco
from PIL import Image

renderer = mujoco.Renderer(env.model, height=H, width=W)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = env._chassis
cam.distance = 1.15
cam.azimuth = 125.0
cam.elevation = -16.0

frames = []
for k in range(int(T_END / DT)):
    v_ref = v_profile(s.t)
    tau_l, tau_r = ctrl.step(s, Command(v_ref=v_ref), DT)
    env.apply(tau_l, tau_r)
    s = env.state()
    if k % EVERY == 0:
        renderer.update_scene(env.data, camera=cam)
        frames.append(Image.fromarray(renderer.render()))

out = ROOT / "results"
out.mkdir(exist_ok=True)
path = out / "06_demo.gif"

# 所有帧共用一张 96 色调色板：颜色不闪，GIF 也小一个数量级。
# 抖动(dither)会往画面里撒噪声，GIF 的 LZW 就压不动了 —— 不抖反而干净又小。
pal = frames[0].quantize(colors=64, method=Image.MEDIANCUT)
qframes = [fr.quantize(palette=pal, dither=Image.NONE) for fr in frames]
qframes[0].save(path, save_all=True, append_images=qframes[1:],
                duration=int(1000 / FPS), loop=0, optimize=True)
print(f"{len(frames)} 帧 -> {path}  ({path.stat().st_size / 1e6:.1f} MB)")

# 顺手存一帧静图，方便直接贴进 PDF 简历
frames[len(frames) // 2].save(out / "06_still.png")
print("still ->", out / "06_still.png")
