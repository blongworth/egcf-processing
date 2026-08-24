"""Find and order lander SD-card log files (``gems_YYYY-MM-DD-HH-MM.txt``)."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

_GEMS_FILENAME_RE = re.compile(r"^gems_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})\.txt$")


def parse_rotation_ts(path: Path) -> datetime | None:
    """Parse the rotation timestamp embedded in a gems_*.txt filename, if present."""
    match = _GEMS_FILENAME_RE.match(path.name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M")


def find_gems_files(raw_dir: Path) -> list[Path]:
    """Find all gems_*.txt files under raw_dir, sorted by their rotation timestamp.

    Excludes 0-byte files (aborted runs/reboots) and anything that doesn't
    match the exact gems_YYYY-MM-DD-HH-MM.txt naming (e.g. gems_pump_*.csv,
    legacy bench-test files).
    """
    candidates = []
    for path in raw_dir.rglob("gems_*.txt"):
        rotation_ts = parse_rotation_ts(path)
        if rotation_ts is None:
            continue
        if path.stat().st_size == 0:
            continue
        candidates.append((rotation_ts, path))
    candidates.sort(key=lambda pair: pair[0])
    return [path for _, path in candidates]
