"""Find and order lander SD-card log files (``gems_YYYY-MM-DD-HH-MM.txt``) and surface
telemetry log files (``surface_YYYY-MM-DD-HH-MM_lander.log``), plus Odyssey PAR logger exports.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from egcf_processing.hobo import is_hobo_export
from egcf_processing.par import is_odyssey_export

_GEMS_FILENAME_RE = re.compile(r"^gems_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})\.txt$")
_SURFACE_FILENAME_RE = re.compile(r"^surface_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})_lander\.log$")
_SURFACE_EVENTS_FILENAME_RE = re.compile(r"^surface_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})_events\.log$")


def parse_rotation_ts(path: Path) -> datetime | None:
    """Parse the rotation timestamp embedded in a gems_*.txt filename, if present."""
    match = _GEMS_FILENAME_RE.match(path.name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M")


def parse_surface_rotation_ts(path: Path) -> datetime | None:
    """Parse the rotation timestamp embedded in a surface_*_lander.log filename, if present."""
    match = _SURFACE_FILENAME_RE.match(path.name)
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


def find_surface_files(raw_dir: Path) -> list[Path]:
    """Find all surface_*_lander.log files under raw_dir, sorted by rotation timestamp.

    Only the *_lander.log half of the surface log pair belongs in *this* list,
    which feeds reader.read_all -> parse_line. The paired *_events.log file
    (same rotation naming) records communication traffic
    (RX_CONSOLE/TX_LANDER/SYSTEM direction + payload) and never contains a
    parseable R:/V:/P:/!: payload, so it is excluded by not matching this glob,
    the same way legacy gems formats are excluded from find_gems_files. It does
    carry its own housekeeping telemetry, read via the separate
    find_surface_events_files -> events.read_all_events path.
    """
    candidates = []
    for path in raw_dir.rglob("surface_*_lander.log"):
        rotation_ts = parse_surface_rotation_ts(path)
        if rotation_ts is None:
            continue
        if path.stat().st_size == 0:
            continue
        candidates.append((rotation_ts, path))
    candidates.sort(key=lambda pair: pair[0])
    return [path for _, path in candidates]


def parse_surface_events_rotation_ts(path: Path) -> datetime | None:
    """Parse the rotation timestamp embedded in a surface_*_events.log filename, if present."""
    match = _SURFACE_EVENTS_FILENAME_RE.match(path.name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M")


def find_surface_events_files(raw_dir: Path) -> list[Path]:
    """Find all surface_*_events.log files under raw_dir, sorted by rotation timestamp.

    These are deliberately *not* part of find_all_files: they use a different
    grammar (see events.py) and are read by events.read_all_events instead of
    reader.read_all.
    """
    candidates = []
    for path in raw_dir.rglob("surface_*_events.log"):
        rotation_ts = parse_surface_events_rotation_ts(path)
        if rotation_ts is None:
            continue
        if path.stat().st_size == 0:
            continue
        candidates.append((rotation_ts, path))
    candidates.sort(key=lambda pair: pair[0])
    return [path for _, path in candidates]


def find_all_files(raw_dir: Path) -> list[Path]:
    """Find gems_*.txt and surface_*_lander.log files under raw_dir, merged in rotation-timestamp order."""
    gems = [(parse_rotation_ts(p), p) for p in find_gems_files(raw_dir)]
    surface = [(parse_surface_rotation_ts(p), p) for p in find_surface_files(raw_dir)]
    combined = sorted(gems + surface, key=lambda pair: pair[0])
    return [path for _, path in combined]


def find_par_files(raw_dir: Path) -> list[Path]:
    """Find Odyssey PAR logger exports (*.csv/*.CSV) under raw_dir, sorted by path.

    Identified by header content (see par.is_odyssey_export), not filename, so
    legacy CSVs like gems_pump_*.csv are excluded.
    """
    paths = {p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"}
    return sorted(p for p in paths if p.stat().st_size > 0 and is_odyssey_export(p))


def find_hobo_files(raw_dir: Path) -> list[Path]:
    """Find HOBO oxygen logger exports (*.csv/*.CSV) under raw_dir, sorted by path.

    Identified by header content (see hobo.is_hobo_export), not filename.
    """
    paths = {p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"}
    return sorted(p for p in paths if p.stat().st_size > 0 and is_hobo_export(p))
