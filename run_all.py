"""一键跑完所有实验。失败不会中断，最后统一汇报。"""
import os
import subprocess
import sys
from pathlib import Path

# Windows 下把 Python 的输出重定向到管道时，默认会退回 GBK，中文和 ° 直接报错。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
PY = sys.executable
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

SCRIPTS = [
    ("01_balance.py", []),
    ("02_track.py", []),
    ("03_sweep.py", []),
    ("04_sysid_motor.py", []),
    ("05_sim2real.py", ["--selftest"]),
    ("06_render.py", []),
]

fails = []
for name, args in SCRIPTS:
    print(f"\n=== {name} {' '.join(args)}")
    r = subprocess.run([PY, str(ROOT / "experiments" / name), *args],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=ENV)
    tail = (r.stdout or "").strip().splitlines()[-6:]
    print("\n".join(tail))
    if r.returncode != 0:
        fails.append(name)
        print("[FAILED]", (r.stderr or "").strip().splitlines()[-3:])

print("\n全部完成" if not fails else f"\n失败的实验: {fails}")
