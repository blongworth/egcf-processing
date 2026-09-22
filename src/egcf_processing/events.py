"""Parse surface event-log lines (``surface_*_events.log``).

Kept separate from lines.py/reader.py because both halves differ: the envelope
is ``iso8601 direction payload`` (per the ``surface-event-log-v1`` header) and
the payload is prose, not the R:/V:/P:/!: grammar. Only the 10-second
``SYSTEM battery voltage=..V current=..A temp=..C`` housekeeping line carries
data; everything else (headers, RX_CONSOLE/TX_LANDER commands, status prose,
garbled serial) is skipped.

The envelope's surface receipt timestamp *is* used as ``ts`` here -- unlike the
lander logs, these lines have no lander-embedded timestamp to prefer.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from egcf_processing.lines import _parse_ts

logger = logging.getLogger(__name__)

_BATTERY_RE = re.compile(
    r"^(?P<ts>\S+)\s+SYSTEM\s+battery\s+voltage=(?P<voltage>[-\d.]+)V"
    r"\s+current=(?P<current>[-\d.]+)A\s+temp=(?P<temp>[-\d.]+)C$"
)


def parse_event_line(line: str) -> dict | None:
    """Parse one events-log line into an SH record, or None if it carries no data.

    ``temp`` rides the ``battery`` prefix but is the Teensy die temperature
    (~42-60 degC at ~0.03 A), not a battery-pack temperature -- see AGENTS.md.
    """
    match = _BATTERY_RE.match(line.strip())
    if not match:
        return None
    ts = _parse_ts(match.group("ts"))
    if ts is None:
        return None
    try:
        voltage = float(match.group("voltage"))
        current = float(match.group("current"))
        temp = float(match.group("temp"))
    except ValueError:
        return None
    return {"tag": "SH", "ts": ts, "voltage_v": voltage, "current_a": current, "teensy_temp_c": temp}


def read_events_file(path: Path) -> list[dict]:
    """Parse one surface_*_events.log file into SH records."""
    records = []
    skipped = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            record = parse_event_line(line)
            if record is None:
                skipped += 1
                continue
            record["source_file"] = path.name
            records.append(record)
    if skipped:
        # The overwhelming majority of events lines are legitimately non-data,
        # so this is DEBUG-level and never per-line.
        logger.debug("%s: skipped %d non-data/malformed lines", path.name, skipped)
    return records


def read_all_events(paths: list[Path]) -> list[dict]:
    """Read all events files in the given order and concatenate their records."""
    records: list[dict] = []
    for path in paths:
        records.extend(read_events_file(path))
    return records
