"""控制性能指标 —— 简历里那些数字就是用这里的函数算出来的。"""
from __future__ import annotations

import numpy as np


def step_metrics(t: np.ndarray, y: np.ndarray, y0: float, y_ref: float,
                 band: float = 0.02) -> dict:
    amp = y_ref - y0
    if abs(amp) < 1e-9:
        return {}
    yn = (y - y0) / amp
    out = {"overshoot_pct": float(max(0.0, (yn.max() - 1.0) * 100.0))}
    try:
        i90 = int(np.argmax(yn >= 0.9))
        i10 = int(np.argmax(yn >= 0.1))
        out["rise_time_s"] = float(t[i90] - t[i10])
    except Exception:
        out["rise_time_s"] = float("nan")
    inside = np.abs(yn - 1.0) > band
    if inside.any():
        last = int(np.nonzero(inside)[0][-1])
        out["settling_s"] = float(t[min(last + 1, len(t) - 1)])
    else:
        out["settling_s"] = 0.0
    tail = yn[int(0.8 * len(yn)):]
    out["steady_error_pct"] = float(abs(tail.mean() - 1.0) * 100.0)
    out["rms_err"] = float(np.sqrt(np.mean((yn - 1.0) ** 2)) * abs(amp))
    return out


def tracking_metrics(t: np.ndarray, y: np.ndarray, ref: np.ndarray) -> dict:
    err = y - ref
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs_err": float(np.max(np.abs(err))),
        "iae": float(np.trapezoid(np.abs(err), t)) if hasattr(np, "trapezoid") else float(np.trapz(np.abs(err), t)),
    }
