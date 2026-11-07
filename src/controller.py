"""串级 PID 平衡控制器 —— 与真机固件里跑的应该是同一套逻辑。

外环（速度/位置）：速度误差 -> 期望俯仰角 theta_ref
内环（姿态）    ：theta 与 theta_ref 的偏差 -> 轮端力矩
转向环          ：偏航角速度误差 -> 左右轮差动力矩

外环输出 theta_ref 就是"想往前倾多少"，这也是真机上人工推车时人做的事：
想加速就先倾倒，车轮追上来把身体接住。这个物理图像决定了所有符号。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from env import State


@dataclass
class Gains:
    kp_theta: float = 4.0      # N·m/rad
    kd_theta: float = 0.45     # N·m·s/rad
    kv_p: float = 0.60         # rad per (m/s)
    kv_i: float = 0.60
    theta_max: float = 0.12    # rad，外环限幅，防止"趴下加速"
    k_yaw: float = 0.30        # N·m per (rad/s)
    tau_max: float = 2.0
    slew: float = 120.0        # N·m/s，模拟驱动器电流上升率
    pos_gain: float = 0.0      # 位置保持（1/s），0 表示纯速度控制
    kd_wheel: float = 0.0      # N·m per (rad/s)，轮速阻尼
    acc_max: float = 1.2       # m/s^2，速度指令斜坡限幅，0 表示不限幅
    d_lpf_hz: float = 60.0     # Hz，D 项一阶低通；0 表示不滤波
    theta_slew: float = 0.0    # rad/s，俯仰目标变化率限幅；0 表示不限幅


@dataclass
class Command:
    v_ref: float = 0.0         # m/s
    wz_ref: float = 0.0        # rad/s
    x_ref: float = 0.0         # m（仅 pos_gain>0 时生效）


class BalanceController:
    def __init__(self, gains: Gains | None = None):
        self.g = gains or Gains()
        self.reset()

    def reset(self):
        self.v_int = 0.0
        self.v_cmd = 0.0
        self._pr_f = 0.0
        self._th_prev = 0.0
        self._tau_prev = 0.0, 0.0

    def step(self, s: State, cmd: Command, dt: float):
        g = self.g

        # ---- 指令斜坡：把阶跃变成受加速度约束的斜坡 ----
        # 真机上没有人会瞬移速度指令；不做这一步，0.5->0 的阶跃会让车先倒车
        # 冲过头（实测欠冲 -0.19 m/s），因为积分项要靠反向倾斜才能卸掉。
        if g.acc_max > 0:
            dv = cmd.v_ref - self.v_cmd
            lim = g.acc_max * dt
            self.v_cmd += float(np.clip(dv, -lim, lim))
        else:
            self.v_cmd = cmd.v_ref

        # ---- 外环：速度 -> 期望倾角 ----
        v_err = self.v_cmd - s.vx
        # 抗积分饱和（条件积分）：外环饱和时不再累加。
        # 不加这一条，被推一下之后 theta_ref 会在 ±theta_max 上顶 2 秒，
        # 积分项直接打满，松开后车会反向冲出去 —— 这是实车最容易踩的坑。
        theta_raw = g.kv_p * v_err + g.kv_i * self.v_int
        saturated = abs(theta_raw) > g.theta_max
        if not (saturated and np.sign(v_err) == np.sign(theta_raw)):
            self.v_int = float(np.clip(self.v_int + v_err * dt, -2.0, 2.0))
        theta_raw = g.kv_p * v_err + g.kv_i * self.v_int
        # 位置环（可选）：把车钉在 x_ref 上，抑制长距离漂移
        if g.pos_gain > 0:
            theta_raw += g.pos_gain * (cmd.x_ref - s.x)
        theta_ref = float(np.clip(theta_raw, -g.theta_max, g.theta_max))
        # 俯仰目标限速：不加这一步，速度指令阶跃会让车身猛地一栽，
        # 车轮因反作用先倒退一下再往前冲（真机上表现为"耸一下"）。
        if g.theta_slew > 0:
            lim = g.theta_slew * dt
            theta_ref = float(np.clip(theta_ref, self._th_prev - lim, self._th_prev + lim))
        self._th_prev = theta_ref

        # ---- 内环：姿态 -> 力矩 ----
        # D 项必须先低通：陀螺噪声被 kd 直接放大成力矩抖动，
        # 真机上车轮会嗡嗡响、电机发烫，这是最典型的"仿真干净、上机就抖"。
        pr = s.pitch_rate
        if g.d_lpf_hz > 0:
            a = dt / (dt + 1.0 / (2.0 * np.pi * g.d_lpf_hz))
            self._pr_f += a * (pr - self._pr_f)
            pr = self._pr_f
        else:
            self._pr_f = pr
        tau = g.kp_theta * (s.pitch - theta_ref) + g.kd_theta * pr
        # 轮速反馈微分项：抑制轮子与车体的高频耦合振荡
        if g.kd_wheel > 0:
            tau += g.kd_wheel * (self.v_cmd - s.wheel_speed)

        # ---- 转向环 ----
        d_yaw = g.k_yaw * (cmd.wz_ref - s.wz)

        tau_l = tau - d_yaw
        tau_r = tau + d_yaw

        # ---- 执行器非理想：饱和 + 变化率限幅 ----
        tau_l = float(np.clip(tau_l, -g.tau_max, g.tau_max))
        tau_r = float(np.clip(tau_r, -g.tau_max, g.tau_max))
        if g.slew > 0:
            lim = g.slew * dt
            tau_l = float(np.clip(tau_l, self._tau_prev[0] - lim, self._tau_prev[0] + lim))
            tau_r = float(np.clip(tau_r, self._tau_prev[1] - lim, self._tau_prev[1] + lim))
        self._tau_prev = tau_l, tau_r
        return tau_l, tau_r

    def as_dict(self):
        return asdict(self.g)
