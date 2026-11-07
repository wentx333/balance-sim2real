"""MuJoCo 双轮平衡车仿真环境 —— 与真机接口逐项对齐。

设计原则：凡是真机上真实存在、会吃掉控制裕度的东西，都在这里显式建模，
否则 sim2real gap 无从分析，仿真就只是"动画"。

显式建模的非理想项：
  * 控制延迟 delay_ms        —— STM32 采样 + 计算 + PWM 输出 + CAN 传输
  * IMU 噪声 imu_noise       —— 陀螺零偏/白噪声
  * 编码器量化 enc_quant     —— N 线编码器的速度量化
  * 电机死区 deadband        —— 小力矩无法克服静摩擦
  * 力矩饱和                  —— ctrlrange 在 XML 里
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "balance_car.xml"


@dataclass
class State:
    """一帧可观测状态（真机上由 IMU + 编码器给出）。"""
    t: float
    x: float
    y: float
    vx: float
    pitch: float          # rad，+ 表示前倾
    pitch_rate: float     # rad/s
    roll: float
    yaw: float
    wz: float             # rad/s 偏航角速度
    wheel_w: np.ndarray   # rad/s [left, right]

    @property
    def wheel_speed(self) -> float:
        """两轮平均线速度（m/s）。"""
        return float(np.mean(self.wheel_w) * 0.05)


class BalanceCarEnv:
    def __init__(self, xml_path: str | Path | None = None, ctrl_hz: float = 200.0,
                 delay_ms: float = 3.0, imu_noise: float = 0.0,
                 gyro_noise: float = 0.0,
                 enc_quant: float | None = None, deadband: float = 0.0,
                 seed: int = 0, timestep: float = 1e-3):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path or MODEL_PATH))
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        self.dt_phys = timestep
        self.ctrl_hz = ctrl_hz
        self.decim = max(1, int(round(1.0 / (ctrl_hz * self.dt_phys))))
        self.delay_k = int(round(delay_ms / 1000.0 * ctrl_hz))
        # imu_noise：姿态角噪声（deg），对应互补滤波/卡尔曼输出的残余误差
        # gyro_noise：陀螺角速度噪声（deg/s），直接进 D 项，是力矩抖动的元凶
        self.imu_noise = float(imu_noise)
        self.gyro_noise = float(gyro_noise)
        # imu_bias：IMU 装歪 / 零偏未标定造成的恒定俯仰偏差（rad）。
        # 这是实车上最常见、也最容易被忽略的 sim2real gap：传感器认为车是正的，
        # 实际车身是斜的，控制器就会一直往一个方向"追"，表现为慢速漂移。
        self.imu_bias = 0.0
        self.enc_quant = float(enc_quant) if enc_quant else None
        self.deadband = float(deadband)

        self._delay: deque = deque()
        self._floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._chassis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self._mass0 = self.model.body_mass.copy()
        self._fric0 = self.model.geom_friction[:, 0].copy()
        self._quat = np.zeros(9)

    # ---------------- 初始化 ----------------
    def reset(self, x: float = 0.0, pitch: float = 0.0, vx: float = 0.0, z: float = 0.115):
        mujoco.mj_resetData(self.model, self.data)
        half = pitch / 2.0
        self.data.qpos[0:3] = (x, 0.0, z)
        self.data.qpos[3:7] = (np.cos(half), 0.0, np.sin(half), 0.0)  # 绕 y 轴转 pitch
        self.data.qpos[7:9] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qvel[0] = vx
        self._delay.clear()
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def set_params(self, mass_scale: float | None = None,
                   friction_scale: float | None = None,
                   delay_ms: float | None = None,
                   deadband: float | None = None,
                   imu_bias_deg: float | None = None):
        """在线改模型参数 —— 用来做"真机数据反推仿真参数"这一类的参数搜索。"""
        if mass_scale is not None:
            self.model.body_mass[1:] = self._mass0[1:] * mass_scale
        if friction_scale is not None:
            self.model.geom_friction[:, 0] = self._fric0 * friction_scale
        if delay_ms is not None:
            self.delay_k = int(round(delay_ms / 1000.0 * self.ctrl_hz))
        if deadband is not None:
            self.deadband = float(deadband)
        if imu_bias_deg is not None:
            self.imu_bias = float(np.deg2rad(imu_bias_deg))

    def bench_mode(self, z: float = 2.0):
        """台架实验：关重力、断地contact，等价于把车架在支架上空转。"""
        self.model.opt.gravity[:] = 0.0
        self.model.geom_contype[self._floor] = 0
        self.model.geom_conaffinity[self._floor] = 0
        self.reset(z=z)

    # ---------------- 观测 ----------------
    def state(self) -> State:
        d = self.data
        mujoco.mju_quat2Mat(self._quat, d.qpos[3:7])
        R = self._quat.reshape(3, 3)
        pitch = float(np.arctan2(R[0, 2], R[2, 2]))
        roll = float(np.arctan2(-R[1, 2], np.hypot(R[1, 0], R[1, 1])))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))

        pitch_rate = float(d.qvel[4])
        pitch += self.imu_bias
        if self.imu_noise > 0:
            pitch += self.rng.normal(0.0, np.deg2rad(self.imu_noise))
        if self.gyro_noise > 0:
            pitch_rate += self.rng.normal(0.0, np.deg2rad(self.gyro_noise))
        wz = float(d.qvel[5])

        wheel_w = np.array([d.qvel[6], d.qvel[7]], dtype=float)
        if self.enc_quant:
            wheel_w = np.round(wheel_w / self.enc_quant) * self.enc_quant

        return State(
            t=float(d.time), x=float(d.qpos[0]), y=float(d.qpos[1]),
            vx=float(d.qvel[0]), pitch=pitch, pitch_rate=pitch_rate,
            roll=roll, yaw=yaw, wz=wz, wheel_w=wheel_w,
        )

    # ---------------- 执行 ----------------
    def _apply_deadband(self, tau: float) -> float:
        if self.deadband <= 0:
            return tau
        if abs(tau) < self.deadband:
            return 0.0
        return tau - np.sign(tau) * self.deadband

    def apply(self, tau_l: float, tau_r: float, fx: float = 0.0):
        """施加一个控制周期的力矩，内部自动处理延迟 + 物理子步。

        fx：作用在车身上的水平外力（N），用来模拟"被人推一把"。用真实外力
        而不是直接改 qvel —— 后者相当于瞬移速度，图上是垂直的竖线，不物理。
        """
        lo = self.model.actuator_ctrlrange[0, 0]
        hi = self.model.actuator_ctrlrange[0, 1]
        cmd = (float(np.clip(self._apply_deadband(tau_l), lo, hi)),
               float(np.clip(self._apply_deadband(tau_r), lo, hi)))

        self._delay.append(cmd)
        if len(self._delay) > self.delay_k:
            out = self._delay.popleft()
        else:
            out = (0.0, 0.0)

        self.data.ctrl[0], self.data.ctrl[1] = out
        self.data.xfrc_applied[self._chassis, 0] = fx
        for _ in range(self.decim):
            mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[self._chassis, 0] = 0.0
        return out

    def spin(self, tau_l: float, tau_r: float, steps: int):
        """台架用：直接给恒定力矩跑若干个物理步。"""
        self.data.ctrl[0] = tau_l
        self.data.ctrl[1] = tau_r
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
