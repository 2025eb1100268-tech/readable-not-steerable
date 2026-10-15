"""[EXPLORE] Seal for the confirmation dataset (results/explore/BRIEF.md, rule 3).

A dataset whose manifest says ``"sealed": true`` cannot be loaded through ``data.dataset.Dataset``
until ``unseal()`` has been called in the same process. ``unseal()`` is meant to be called only by
the Stage 3 script, and refuses unless PREREGISTRATION_v2.md exists and is committed unmodified;
every call is logged in results/LOG.md.
"""
from __future__ import annotations

import subprocess

from common.run import REPO, append_log, command_line

_UNSEALED = False


class SealedDatasetError(PermissionError):
    pass


def check(root, manifest: dict) -> None:
    if manifest.get("sealed") and not _UNSEALED:
        raise SealedDatasetError(f"{root} is sealed (confirmation set); only the Stage 3 script may unseal it")


def unseal(reason: str) -> None:
    global _UNSEALED
    prereg = "PREREGISTRATION_v2.md"
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", prereg], capture_output=True).returncode == 0
    clean = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", prereg], capture_output=True, text=True).stdout.strip() == ""
    if not ((REPO / prereg).exists() and tracked and clean):
        raise SealedDatasetError(f"unseal refused: {prereg} must exist and be committed unmodified first")
    _UNSEALED = True
    append_log("[EXPLORE] UNSEAL confirmation dataset", "-", command_line(), "unsealed", {"reason": reason})
