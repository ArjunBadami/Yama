"""Thin wrappers around `gcloud storage` for syncing runs/checkpoints.

Uses the gcloud CLI rather than the Python client so the training VM needs no
extra credentials plumbing: the VM's service account is already authorised.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def _gcloud() -> str | None:
    for name in ("gcloud", "gcloud.cmd"):
        path = shutil.which(name)
        if path:
            return path
    return None


def rsync(src: str | Path, dst: str, delete: bool = False) -> bool:
    """Recursive rsync local dir -> gs:// (or gs:// -> local). Returns success."""
    exe = _gcloud()
    if exe is None:
        log.warning("gcloud not found; skipping sync %s -> %s", src, dst)
        return False
    cmd = [exe, "storage", "rsync", "-r", str(src), str(dst)]
    if delete:
        cmd.append("--delete-unmatched-destination-objects")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        log.info("synced %s -> %s", src, dst)
        return True
    except subprocess.CalledProcessError as e:
        log.warning("gcs sync failed (%s): %s", " ".join(cmd), e.stderr.strip()[-500:])
        return False


def pull_if_missing(gcs_dir: str, local_dir: str | Path) -> bool:
    """If the local dir has no checkpoint but GCS does, pull it (for spot-VM resume)."""
    local = Path(local_dir)
    if (local / "last").exists():
        return False
    local.mkdir(parents=True, exist_ok=True)
    return rsync(gcs_dir, local)
