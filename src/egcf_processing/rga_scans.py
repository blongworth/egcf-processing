"""Layer B window boundaries: RGA mass-scan-cycle detection.

A scan is one pass through the RGA's configured mass list. Boundaries are
detected from the data itself (a "masses seen in the current scan" set) so
they're robust to mass-list reordering or config changes, rather than
assuming any fixed mass order or count.
"""

from __future__ import annotations

import polars as pl

_EMPTY_WINDOWS_SCHEMA = {"window_start": pl.Datetime, "window_end": pl.Datetime}

_CONTEXT_SCHEMA = {
    "window_start": pl.Datetime,
    "window_end": pl.Datetime,
    "chamber": pl.Utf8,
    "experiment_number": pl.Int64,
    "elapsed_time": pl.Duration,
}


def rga_scan_windows(rga: pl.DataFrame) -> pl.DataFrame:
    """One row per RGA scan: [first reading's ts, next scan's first ts).

    Operates on ``rga`` in its given (arrival/file) order rather than
    re-sorting by ``ts``, consistent with cycles.chamber_cycle_windows.

    The trailing (last, open-ended) scan is dropped since its true end is
    unknown without a following scan to bound it.
    """
    if rga.is_empty():
        return pl.DataFrame(schema=_EMPTY_WINDOWS_SCHEMA)

    r = rga
    seen: set[int] = set()
    scan_ids = []
    scan_id = 0
    for mass in r["mass"]:
        if mass in seen:
            scan_id += 1
            seen = set()
        seen.add(mass)
        scan_ids.append(scan_id)

    r = r.with_columns(pl.Series("scan_id", scan_ids))
    scans = (
        r.group_by("scan_id")
        .agg(pl.col("ts").min().alias("window_start"))
        .sort("scan_id")
        .with_columns(window_end=pl.col("window_start").shift(-1))
        .filter(pl.col("window_end").is_not_null())
    )
    return scans.select(["window_start", "window_end"])


def attach_chamber_context(windows: pl.DataFrame, chamber_cycles: pl.DataFrame) -> pl.DataFrame:
    """Left-join chamber/experiment_number/elapsed_time from Layer C's cycle windows.

    A scan window gets chamber context only if its start falls inside a Layer
    C (Re-only, settle-adjusted) cycle window; otherwise those columns are
    null (e.g. during flush periods, the settle gap, or before any usable V:
    data exists).
    """
    if windows.is_empty() or chamber_cycles.is_empty():
        return windows.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("chamber"),
            pl.lit(None, dtype=pl.Int64).alias("experiment_number"),
            pl.lit(None, dtype=pl.Duration).alias("elapsed_time"),
        ).select(list(_CONTEXT_SCHEMA))

    cycles = chamber_cycles.sort("window_start").rename(
        {"window_start": "cycle_window_start", "window_end": "cycle_window_end"}
    )
    matched = windows.sort("window_start").join_asof(
        cycles, left_on="window_start", right_on="cycle_window_start", strategy="backward"
    )
    in_range = pl.col("cycle_window_end").is_not_null() & (pl.col("window_start") < pl.col("cycle_window_end"))
    matched = matched.with_columns(
        pl.when(in_range).then(pl.col("chamber")).otherwise(None).alias("chamber"),
        pl.when(in_range).then(pl.col("experiment_number")).otherwise(None).alias("experiment_number"),
        pl.when(in_range).then(pl.col("elapsed_time")).otherwise(None).alias("elapsed_time"),
    )
    return matched.select(list(_CONTEXT_SCHEMA))
