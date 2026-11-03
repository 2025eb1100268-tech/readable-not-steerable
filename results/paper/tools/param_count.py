"""Count the dial decoders' parameters by building the classes (paper pack, 2026-11-03). Loads no data, trains nothing.

Guard: refuses any file access whose path names a sealed or retired set. Run from the repo root:

    python results/paper/tools/param_count.py > results/paper/tools/param_count.txt
"""
import builtins
import os
import sys

import numpy as np

SEALED = ("confirm_v2", "confirm3", "matched_energy_confirm", "matched-energy-confirm", "test_id", "test_ood")


def _check(path):
    p = os.fspath(path).replace("\\", "/").lower()
    if any(s in p for s in SEALED):
        raise SystemExit(f"refused: sealed or retired data path {path}")


_open, _load = builtins.open, np.load
builtins.open = lambda f, *a, **k: (_check(f), _open(f, *a, **k))[1]
np.load = lambda f, *a, **k: (_check(f), _load(f, *a, **k))[1]

sys.path.insert(0, os.getcwd())
from dials.split import FiLMDecoder  # noqa: E402


def count(m, skip_film=False):
    return sum(p.numel() for n, p in m.named_parameters() if not (skip_film and n.startswith("film")))


for up in ("pixelshuffle", "resize"):
    dec = FiLMDecoder(120, 8, upsample=up)
    print(up, "total", count(dec), "without FiLM", count(dec, skip_film=True), "FiLM only", count(dec) - count(dec, True))
