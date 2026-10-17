"""Seals for confirmation datasets ([EXPLORE] brief rule 3; [BUILD] agreed change 9).

A dataset whose manifest says ``"sealed": true`` cannot be loaded through ``data.dataset.Dataset``
until ``unseal()`` has been called, in the same process, for the pre-registration that dataset names
(``"sealed_for"`` in its manifest; ``PREREGISTRATION_v2.md`` for the original confirm set, which
predates the field). ``unseal()`` refuses unless that file exists and is committed unmodified, and
every call is logged in results/LOG.md.
"""
from __future__ import annotations

import subprocess

from common.run import REPO, append_log, command_line

DEFAULT_PREREG = "PREREGISTRATION_v2.md"
_UNSEALED: set[str] = set()


class SealedDatasetError(PermissionError):
    pass


def check(root, manifest: dict) -> None:
    if manifest.get("sealed"):
        prereg = manifest.get("sealed_for", DEFAULT_PREREG)
        if prereg not in _UNSEALED:
            raise SealedDatasetError(f"{root} is sealed for {prereg}; only its confirmation script may unseal it")


def unseal(reason: str, prereg: str = DEFAULT_PREREG) -> None:
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", prereg], capture_output=True).returncode == 0
    clean = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", prereg], capture_output=True, text=True).stdout.strip() == ""
    if not ((REPO / prereg).exists() and tracked and clean):
        raise SealedDatasetError(f"unseal refused: {prereg} must exist and be committed unmodified first")
    _UNSEALED.add(prereg)
    append_log(f"UNSEAL datasets sealed for {prereg}", "-", command_line(), "unsealed", {"reason": reason})
