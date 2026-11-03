"""List the 50 frozen checkpoints (Study 2: 39, Study 3: 11), check each SHA-256 against the frozen value, and
total their size (paper pack, 2026-11-03). Reads checkpoint files only; loads no data, trains nothing.

    python results/paper/tools/checkpoint_manifest.py > results/paper/tools/checkpoint_manifest.txt
"""
import hashlib
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
CKPT = REPO / "results/checkpoints"


def frozen():
    s2 = yaml.safe_load((REPO / "configs/explore/stage3_confirm.yaml").read_text(encoding="utf-8"))["checkpoint_sha256"]
    s3 = {k: v["sha256"] for k, v in json.loads((REPO / "configs/build/v3_frozen.json").read_text(encoding="utf-8"))["runs"].items()}
    return [("Study 2", k, v) for k, v in s2.items()] + [("Study 3", k, v) for k, v in s3.items()]


def main():
    rows, total, bad = frozen(), 0, 0
    print(f"{'study':<8} {'run':<40} {'bytes':>11}  sha256 check")
    for study, run, sha in rows:
        f = CKPT / f"{run}.pt"
        if not f.exists():
            print(f"{study:<8} {run:<40} {'missing':>11}")
            bad += 1
            continue
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        ok = h == sha
        bad += not ok
        total += f.stat().st_size
        print(f"{study:<8} {run:<40} {f.stat().st_size:>11}  {'ok' if ok else 'MISMATCH ' + h}")
    print(f"\n{len(rows)} checkpoints ({sum(r[0] == 'Study 2' for r in rows)} Study 2, "
          f"{sum(r[0] == 'Study 3' for r in rows)} Study 3); total {total} bytes ({total / 2**20:.1f} MiB); "
          f"{len(rows) - bad} of {len(rows)} match the frozen SHA-256")


if __name__ == "__main__":
    main()
