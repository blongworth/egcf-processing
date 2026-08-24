"""Read gems_*.txt files into parsed record dicts, in chronological (filename) order."""

from __future__ import annotations

import logging
from pathlib import Path

from egcf_processing.lines import parse_line

logger = logging.getLogger(__name__)


def read_file(path: Path) -> list[dict]:
    records = []
    skipped = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = parse_line(line)
            if record is None:
                skipped += 1
                continue
            record["source_file"] = path.name
            records.append(record)
    if skipped:
        logger.debug("%s: skipped %d unrecognized/malformed lines", path.name, skipped)
    return records


def read_all(paths: list[Path]) -> list[dict]:
    """Read all files in the given order and concatenate their records.

    ``paths`` must already be sorted chronologically (see discovery.find_gems_files) --
    records within a file are chronological by construction, and embedded
    timestamps make cross-file ordering correct regardless of read order, but
    reading in rotation order keeps behavior predictable.
    """
    records: list[dict] = []
    for path in paths:
        records.extend(read_file(path))
    return records
