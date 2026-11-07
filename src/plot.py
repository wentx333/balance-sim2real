"""统一出图风格（深色网格、无边框，适合直接贴进作品集）。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACCENT = "#e8853a"
BLUE = "#3d7dd8"
GREY = "#7a8088"


def style():
    # 中文图注：Windows 自带雅黑，macOS/Linux 依次回落
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "PingFang SC",
        "Noto Sans CJK SC", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "figure.dpi": 140,
        "font.size": 9,
        "axes.edgecolor": "#c8ccd2",
        "axes.labelcolor": "#33383f",
        "axes.grid": True,
        "grid.color": "#e6e9ee",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
    })


def save(fig, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)
