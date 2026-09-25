"""Parse HOBO dissolved-oxygen/temperature logger exports.

Like the Odyssey PAR logger (see par.py), each HOBO is a separately clocked,
separately recovered instrument, not a payload in the lander's line grammar,
so it gets its own reader feeding a Layer A ``hobo_oxygen`` table. Unlike PAR,
HOBO reports already-computed DO concentration (no slope/intercept
calibration needed), and its clock offset from UTC is stated explicitly in
the export header rather than drifting independently, so there's no
calibration file or time-offset CLI knob here.

Export format (one file per deployment location -- one HOBO physically lives
in each chamber, one in the mesocosm): a UTF-8 BOM, a ``"Plot Title: <location>"``
line (the location tag: "C1", "C2", or "mesocosm"), then a quoted header row
whose "Date Time" column embeds a fixed UTC offset (``GMT-04:00``) and whose
DO/Temp columns embed logger/sensor serial numbers -- located by substring,
since column count/position varies (some exports carry an extra "Sensor Data
Error" event column). Temperature is reported in Fahrenheit. Trailing rows
near logger retrieval carry the ``-888.88`` out-of-water/error sentinel or are
blank with only a stop/retrieval event flag set; both are dropped.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

HOBO_SCHEMA = {
    "ts": pl.Datetime,
    "location": pl.Utf8,
    "oxygen_mgl": pl.Float64,
    "temp_degc": pl.Float64,
    "serial_number": pl.Utf8,
}

_PLOT_TITLE_RE = re.compile(r'"?Plot Title:\s*([^"]+)"?')
_GMT_OFFSET_RE = re.compile(r"GMT([+-]\d{2}):(\d{2})")
_SERIAL_RE = re.compile(r"LGR S/N:\s*(\d+)")
_SENTINEL_DO = -888.88
_DATE_FORMAT = "%m/%d/%y %I:%M:%S %p"


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8-sig", errors="replace").splitlines()


def is_hobo_export(path: Path) -> bool:
    """True if the file's first line identifies it as a HOBO logger export."""
    try:
        with path.open(encoding="utf-8-sig", errors="replace") as f:
            first_line = f.readline()
    except OSError:
        return False
    return _PLOT_TITLE_RE.search(first_line) is not None


def _utc_offset(header: str) -> timedelta:
    """The fixed UTC offset stated in the header's "Date Time, GMT..." column, as a
    timedelta to *add* to a local timestamp to get UTC (e.g. GMT-04:00 -> +4h).
    """
    match = _GMT_OFFSET_RE.search(header)
    if match is None:
        return timedelta(0)
    hours, minutes = int(match.group(1)), int(match.group(2))
    return -timedelta(hours=hours, minutes=minutes)


def _column_index(header_fields: list[str], substring: str) -> int | None:
    for i, field in enumerate(header_fields):
        if substring in field:
            return i
    return None


def read_hobo_file(path: Path) -> pl.DataFrame:
    """Parse one HOBO export into ts/location/oxygen_mgl/temp_degc/serial_number rows."""
    lines = _read_lines(path)
    if len(lines) < 2:
        return pl.DataFrame(schema=HOBO_SCHEMA)

    title_match = _PLOT_TITLE_RE.search(lines[0])
    location = title_match.group(1).strip() if title_match else None

    header = lines[1]
    header_fields = next(csv.reader(io.StringIO(header)))
    do_idx = _column_index(header_fields, "DO conc")
    temp_idx = _column_index(header_fields, "Temp,")
    if do_idx is None or temp_idx is None:
        return pl.DataFrame(schema=HOBO_SCHEMA)

    serial_match = _SERIAL_RE.search(header)
    serial_number = serial_match.group(1) if serial_match else None
    offset = _utc_offset(header)

    rows = []
    for line in lines[2:]:
        if not line.strip():
            continue
        fields = [f.strip() for f in line.split(",")]
        try:
            do_str, temp_str = fields[do_idx], fields[temp_idx]
            if not do_str or not temp_str:
                continue
            do_mgl, temp_f = float(do_str), float(temp_str)
            if do_mgl == _SENTINEL_DO:
                continue
            ts = datetime.strptime(fields[1], _DATE_FORMAT) + offset
        except (IndexError, ValueError):
            continue
        rows.append(
            {
                "ts": ts,
                "location": location,
                "oxygen_mgl": do_mgl,
                "temp_degc": (temp_f - 32) * 5 / 9,
                "serial_number": serial_number,
            }
        )
    return pl.DataFrame(rows, schema=HOBO_SCHEMA) if rows else pl.DataFrame(schema=HOBO_SCHEMA)


def read_all_hobo(paths: list[Path]) -> pl.DataFrame:
    """Read and concatenate all HOBO exports."""
    frames = [read_hobo_file(path) for path in paths]
    return pl.concat(frames) if frames else pl.DataFrame(schema=HOBO_SCHEMA)
