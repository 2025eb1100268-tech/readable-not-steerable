"""Shared run plumbing: YAML configs, seeding, device choice, environment capture, results/LOG.md."""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(Path(path).as_posix())
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_threads(n: int | None) -> None:
    """Fix the CPU thread count: float results depend on it (reduction order)."""
    if n:
        torch.set_num_threads(int(n))


def pick_device(pref: str = "auto") -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config asks for CUDA but torch.cuda.is_available() is False")
    return pref


def git_state() -> str:
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        head = "no-commit"
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    return head + ("+dirty" if dirty else "")


def environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cpu_threads_torch": torch.get_num_threads(),
        "git": git_state(),
    }
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def append_log(title: str, cfg_path: str, command: str, outcome: str, details: dict | None = None,
               wall_s: float | None = None, device: str | None = None) -> None:
    """Append one run record to results/LOG.md (every run, including failed ones)."""
    RESULTS.mkdir(exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"\n## {stamp} — {title}", f"- config: `{cfg_path}`", f"- command: `{command}`",
             f"- git: `{git_state()}`"]
    if device is not None:
        lines.append(f"- device: {device}")
    if wall_s is not None:
        lines.append(f"- wall-clock: {wall_s:.1f} s")
    lines.append(f"- outcome: {outcome}")
    for k, v in (details or {}).items():
        lines.append(f"  - {k}: {v}")
    with open(RESULTS / "LOG.md", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def command_line() -> str:
    """The command as typed from the repo root (scripts are run as modules: python -m pkg.mod)."""
    rel = os.path.relpath(sys.argv[0], REPO).replace("\\", "/")
    module = rel[:-3].replace("/", ".") if rel.endswith(".py") else rel
    return "python -m " + " ".join([module] + sys.argv[1:])
