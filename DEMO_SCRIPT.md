# 30 秒演示解说词

对应素材：`results/06_demo.gif`（120 帧 / 6 s）。左边放 GIF 自动循环，
右边或用口述给出数字。目标是**在 30 秒内让人记住三件事：会建模、会调参、会量化**。

---

## 中文版（25–30 秒，正常语速）

> 这是我用 MuJoCo 复现的自研双轮平衡车——真机是 STM32、200 Hz 串级控制，仿真里把
> 延迟、IMU 零偏、编码器量化、电机死区都建进去了。
>
> **（0–8 s，车在站住）** 现在它按真机的控制器在跑，不是我手写的平衡脚本。
>
> **（8–18 s，t=2 s 处受 18 N 侧向冲击）** 这里的 18 牛冲击是真实的力，不是改速度。
> 冲击后 1.85 秒恢复到 ±2 度以内，峰值倾角 8 度，峰值力矩 0.99 牛米——离 2 牛米上限还有余量。
>
> **（18–26 s）** 我还做了 42 组「延迟 × 力矩上限」扫参数，结论是力矩余量才是硬约束，
> 0 到 15 毫秒的延迟对恢复时间影响不到 0.3 秒。
>
> **（26–30 s）** 最后用真机日志反推参数，俯仰 RMSE 从 1.64 度降到 0.22 度，降了 87%。
> 整个项目一条命令复现，代码我带来了。

---

## English version (~28 s)

> This is a MuJoCo reproduction of a two-wheel self-balancing robot I built —
> the same 200 Hz cascaded controller that runs on the STM32, with control delay,
> IMU bias, encoder quantization and motor deadband modelled explicitly.
>
> At t = 2 s an 18 N lateral force hits the chassis — a real force in the physics,
> not a velocity edit. It recovers to within ±2° in 1.85 s, with an 8° peak pitch
> and 0.99 N·m of peak torque against a 2 N·m limit.
>
> I then swept 42 combinations of delay and torque headroom: torque headroom is the
> binding constraint, while 0–15 ms of delay changes recovery by under 0.3 s.
>
> Finally I fitted the simulator to a real robot log — pitch RMSE dropped from
> 1.64° to 0.22°, an 87 % improvement. One command reproduces everything.

---

## 如果只给你 15 秒（BOSS 直聘 / 电话初筛）

> 我用 MuJoCo 把我自己做的双轮平衡车复现了一遍，把延迟和噪声建进去，然后用真机日志
> 反推参数，俯仰误差从 1.6 度降到 0.2 度。42 组扫参数说明力矩余量比延迟更致命。

---

## 讲的时候的三条纪律

1. **先说结论，再说过程。** 面试官不关心你调了几版 PID，关心你从中学到了什么规律。
2. **每个数字都能追到脚本。** 被追问"这个 1.85 秒怎么算的"，答得上
   `src/metrics.py` 里的恢复判据（连续 0.2 s 在 ±2° 内）。
3. **主动说限制。** "摩擦不可辨识，因为轮子没打滑"——承认边界比多报一个指标更可信。
