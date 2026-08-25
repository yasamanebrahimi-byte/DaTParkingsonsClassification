"""Safe archive and image discovery."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List


def discover_niftis(root: str | Path) -> Dict[str, List[Path]]:
    """Return UID to paths, recursively, for files named ``<uid>.nii.gz``."""
    paths = sorted(Path(root).rglob("*.nii.gz"))
    discovered: Dict[str, List[Path]] = {}
    for path in paths:
        uid = path.name[:-7]
        discovered.setdefault(uid, []).append(path)
    return discovered


def archive_members(archive_path: str | Path) -> List[str]:
    with zipfile.ZipFile(archive_path) as archive:
        return archive.namelist()


def safe_extract_zip(archive_path: str | Path, destination: str | Path) -> List[Path]:
    """Extract an archive while preventing path traversal."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
            archive.extract(member, destination)
            if not member.is_dir():
                extracted.append(target)
    return extracted


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

