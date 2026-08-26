"""Read gems_*.txt and surface_*_lander.log files into parsed record dicts, in
chronological (filename) order.
"""

from __future__ import annotations

import logging
from pathlib import Path

from egcf_processing.lines import parse_line

logger = logging.getLogger(__name__)


def read_file(path: Path) -> list[dict]:
    """Parse one raw log file, dispatching on filename to the right line format.

    gems_*.txt lines are the payload directly. surface_*_lander.log lines wrap
    the same payload grammar in a "<surface_receipt_ts> <payload>" envelope (plus
    a "# ..." header) -- the surface receipt timestamp is discarded here, not
    used as ``ts``, since parse_line already extracts the embedded lander
    timestamp from the payload and that's what output timeseries are keyed on.
    """
    is_surface = path.name.startswith("surface_")
    records = []
    skipped = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if is_surface:
                if line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    skipped += 1
                    continue
                payload = parts[1]
            else:
                payload = line
            record = parse_line(payload)
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
