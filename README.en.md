# balance-sim2real — MuJoCo reproduction of a two-wheel self-balancing robot

A small, complete simulation study of the robot I actually built: a two-wheel
self-balancing cart (STM32, 200 Hz cascaded loop). Instead of a toy demo, the goal
was to answer engineering questions I could not answer on hardware alone:

1. **How much disturbance can the controller survive before torque saturates?**
2. **Is control latency or torque headroom the binding constraint on stability?**
3. **Can I recover the real motor parameters (J, b) from bench data?**
4. **Can I tune the simulator so it matches the physical robot's pitch trace?**

Everything runs headless on CPU in about three minutes, and every number below is
reproduced by `python run_all.py`.

## Quick start

```bash
pip install -r requirements.txt   # mujoco, numpy, matplotlib, Pillow
python run_all.py                 # runs experiments 01-06, writes results/
```

Individual experiments:

```bash
python experiments/03_sweep.py          # 42-cell delay x torque sweep (~40 s)
python experiments/05_sim2real.py --selftest   # sim2real without a real log file
python experiments/06_render.py         # renders results/06_demo.gif
```

## Layout

```
model/balance_car.xml     MuJoCo model (wheels, chassis, mast, IMU site, motors)
src/env.py                env wrapper: 200 Hz control, delay, noise, quantization, mass/friction scaling
src/controller.py         cascaded balance controller (outer velocity PI + inner attitude PD)
src/metrics.py            step-response metrics (rise, overshoot, settling, steady-state error)
src/plot.py               shared matplotlib style
experiments/01..06        the six studies
results/                  figures and demo GIF
```

## Model

| Quantity | Value |
| --- | --- |
| Wheel radius / track width | 0.05 m / 0.170 m |
| Total mass | ≈ 1.95 kg |
| Simulation step | 1 ms, `implicitfast` integrator |
| Control rate | 200 Hz (decimated inside `BalanceCarEnv.apply`) |
| Wheel friction | μ = 1.0, elliptic cone, `condim=4` |
| Wheel armature / damping | 0.002 kg·m² / 0.002 N·m·s/rad |
| Motor torque limit | ±2.0 N·m |

The controller is deliberately the one that runs on the real STM32, with the same
structure: command ramp → outer PI on velocity error → pitch reference clamp →
inner PD → differential yaw term → torque saturation and slew limit.

Two details that made the simulation useful rather than decorative:

- **Conditional-integration anti-windup.** The first version had none: after a
  sideways shove the pitch reference sat at its clamp for ~2 s, the integrator
  wound up, and the cart shot off in the opposite direction at -0.5 m/s with 60 %
  overshoot. Gating integration while saturated fixed it.
- **Disturbances are real forces**, applied through `data.xfrc_applied`, not a
  `qvel` write. Writing velocity directly produces a physically impossible step
  that no controller can be evaluated against.

## Experiments

### 01 — balance and disturbance rejection
`experiments/01_balance.py` → `results/01_balance.png`

Initial pitch 0.10 rad; at t = 2.0 s an 18 N force is applied for 0.1 s.

| Metric | Value |
| --- | --- |
| Peak velocity after impact | 1.41 m/s |
| Peak pitch | 8.02° |
| Recovery to ±2° for 0.2 s | 1850 ms |
| Peak wheel torque | 0.99 N·m (limit 2.0, not saturated) |
| Displacement | 1.281 m |
| Final velocity | -0.001 m/s |

### 02 — velocity tracking
`experiments/02_track.py`

Segments (0, 0), (1, 0.5), (4, 0), (6, -0.3), (8, 0) m/s over 10 s.

| Segment | Overshoot | Rise (10-90 %) | Settling (±2 %) | Steady-state error |
| --- | --- | --- | --- | --- |
| 1 | 7.1 % | 380 ms | 1845 ms | 0.61 % |
| 2 | 6.3 % | 405 ms | 1885 ms | 2.38 % |
| 3 | 14.4 % | 200 ms | 1855 ms | 3.48 % |
| 4 | 8.9 % | 220 ms | 1780 ms | 1.84 % |

Whole-run RMSE against the reference: 0.1624 m/s.

*A measurement bug worth mentioning*: the analysis window was originally
`[t0, t0 + 2.5)`, which for the 2 s segment spilled into the next command and
reported 64 % overshoot and 23.75 % steady-state error. Clipping the window to
`t1 - 0.05` brought the real numbers to the single digits above — a reminder that
in a sim study the metrics code deserves as much scrutiny as the controller.

### 03 — delay × torque-headroom sweep
`experiments/03_sweep.py` → `results/03_sweep.png`

7 delays (0-30 ms) × 6 torque limits (0.35-2.0 N·m), 42 runs, fall threshold 35°.

- 8 of 42 configurations fall.
- `tau_max = 0.35 N·m` fails at every delay.
- `tau_max = 0.5 N·m` fails at delay ≥ 10 ms.
- For 0-15 ms delay, recovery time changes by less than 0.3 s.

**Conclusion: torque headroom is the hard constraint; latency only matters once
the headroom is nearly exhausted.** That is the kind of statement that is hard to
make from bench tinkering and trivial to make from a sweep.

### 04 — motor parameter identification (bench mode)
`experiments/04_sysid_motor.py` → `results/04_sysid_motor.png`

Gravity and ground contact disabled, torque steps of 0.15-0.90 N·m sampled at
500 Hz, least squares on `tau = J·alpha + b·omega`.

| Parameter | Identified | Truth | Error |
| --- | --- | --- | --- |
| J | 0.002302 kg·m² | 0.002197 | +4.8 % |
| b | 0.002000 N·m·s/rad | 0.002000 | 0 % |

R² = 1.0000. The right panel subtracts `b·omega` so the samples collapse onto a
straight line through the origin whose slope is J — a much more convincing plot
than raw torque/acceleration scatter.

### 05 — sim2real alignment from a real log
`experiments/05_sim2real.py` → `results/05_sim2real.png`

Input is `data/real_log.csv` with columns `t, v_ref, vx, pitch` (pitch in radians).
`--selftest` generates a synthetic "real" log from known perturbations
(mass ×1.25, friction ×0.70, 10 ms delay, 0.04 deadband, 1.2° IMU bias) and then
tries to recover them, so the method can be validated without hardware.

Two-stage search: IMU bias first (-2°..2° in 0.1° steps → +1.3°), then mass scale
(→ 1.35×). Pitch RMSE goes from **1.64° (nominal) to 0.22° (-87 %)**.

Honest limitation, stated in the README rather than hidden: friction is **not**
identifiable here. Sweeping μ from 0.5× to 1.4× leaves the RMSE at 0.22° because
the wheels never slip in this duty cycle. A parameter you cannot excite is a
parameter you cannot identify.

### 06 — rendering
`experiments/06_render.py` → `results/06_demo.gif`, `results/06_still.png`

640×480, tracking camera, 6 s → 120 frames (2.7 MB). Costs about a minute and is
what makes the project reviewable in 30 seconds (see `DEMO_SCRIPT.md`).

## Known limitations

- Planar-only planar: no roll recovery, no slipping model beyond Coulomb friction.
- Motors are ideal torque sources; no current loop, no voltage saturation, no
  back-EMF limit. The FOC companion project (`foc-joint-sim`) covers that layer.
- Contact is a single point per wheel with an elliptic cone; no rolling resistance
  or tyre deformation.
- The identified parameters are for this specific build; J and b shift with battery
  voltage, temperature and payload.

## Next steps

1. Replace the PD inner loop with a state-feedback (LQR) design and compare
   disturbance-rejection metrics against 01.
2. Port the controller to ROS 2 as a node and drive it from the same MuJoCo model
   through `ros2_control`, closing the loop with the ROS 2 work in my CV.
3. Add a learned residual (small MLP on top of the model-based controller) trained
   on the identified-parameter simulator — the standard sim2real recipe, in miniature.
