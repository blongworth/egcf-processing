"""Layer C window boundaries: chamber measurement cycles + experiment numbering.

Derived purely from the (deduped, new-format-only) V: sequence in
``valve.parquet``. A measurement cycle spans one V: transition into
(chamber, Re) to the next V: transition; Fl (flush) spans never produce
cycles. experiment_number increments, and the elapsed-time origin resets,
each time a (C1, Re) transition is observed that was preceded -- since the
last boundary -- by a (C2, Fl) transition (completing "flush C2" ends one
experiment and starts the next). The first Re transition in the dataset
starts experiment 1, regardless of chamber, to handle a dataset that begins
mid-cycle.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import polars as pl

logger = logging.getLogger(__name__)

_WINDOWS_SCHEMA = {
    "window_start": pl.Datetime,
    "window_end": pl.Datetime,
    "chamber": pl.Utf8,
    "experiment_number": pl.Int64,
    "elapsed_time": pl.Duration,
}


def dedupe_consecutive(valve: pl.DataFrame) -> pl.DataFrame:
    """Drop V: rows whose (chamber, flush_state) exactly repeats the prior row.

    Operates on ``valve`` in its given (arrival/file) order rather than
    re-sorting by ``ts`` -- preserving arrival order is what lets
    chamber_cycle_windows detect a non-monotonic embedded timestamp (a clock
    jump) as an ordering anomaly instead of silently normalizing it away.
    """
    if valve.is_empty():
        return valve
    v = valve.with_columns(
        prev_chamber=pl.col("chamber").shift(1),
        prev_flush=pl.col("flush_state").shift(1),
    )
    v = v.filter(
        pl.col("prev_chamber").is_null()
        | (pl.col("chamber") != pl.col("prev_chamber"))
        | (pl.col("flush_state") != pl.col("prev_flush"))
    )
    return v.drop(["prev_chamber", "prev_flush"])


def chamber_cycle_windows(valve: pl.DataFrame, settle_offset_s: float) -> tuple[pl.DataFrame, dict]:
    """Return (windows, stats). windows has one row per valid Re measurement cycle."""
    if valve.is_empty():
        return pl.DataFrame(schema=_WINDOWS_SCHEMA), {"total_cycles": 0, "dropped_too_short": 0}

    v = dedupe_consecutive(valve)

    experiment_number = None
    started = False
    saw_c2_flush = False
    exp_start_ts = None
    labeled = []
    for row in v.select(["ts", "chamber", "flush_state"]).iter_rows(named=True):
        ts, chamber, flush = row["ts"], row["chamber"], row["flush_state"]
        if not started and flush == "Re":
            started = True
            experiment_number = 1
            exp_start_ts = ts
        elif started and chamber == "C2" and flush == "Fl":
            saw_c2_flush = True
        elif started and chamber == "C1" and flush == "Re" and saw_c2_flush:
            experiment_number += 1
            exp_start_ts = ts
            saw_c2_flush = False
        labeled.append(
            {
                "ts": ts,
                "chamber": chamber,
                "flush_state": flush,
                "experiment_number": experiment_number,
                "exp_start_ts": exp_start_ts,
            }
        )

    labeled_df = pl.DataFrame(
        labeled,
        schema={
            "ts": pl.Datetime,
            "chamber": pl.Utf8,
            "flush_state": pl.Utf8,
            "experiment_number": pl.Int64,
            "exp_start_ts": pl.Datetime,
        },
    ).with_columns(next_ts=pl.col("ts").shift(-1))

    settle_offset = timedelta(seconds=settle_offset_s)
    candidates = labeled_df.filter(
        (pl.col("flush_state") == "Re")
        & pl.col("experiment_number").is_not_null()
        & pl.col("next_ts").is_not_null()
    ).with_columns(
        elapsed_time=pl.col("ts") - pl.col("exp_start_ts"),
        window_start=pl.col("ts") + pl.lit(settle_offset),
        window_end=pl.col("next_ts"),
    )

    long_enough = pl.col("window_end") > pl.col("window_start")
    valid = candidates.filter(long_enough)
    dropped = candidates.filter(~long_enough)

    stats = {"total_cycles": candidates.height, "dropped_too_short": dropped.height}
    if dropped.height:
        logger.warning(
            "dropped %d chamber cycle(s) shorter than settle_offset_s=%s: %s",
            dropped.height,
            settle_offset_s,
            dropped["ts"].to_list(),
        )

    windows = valid.select(["window_start", "window_end", "chamber", "experiment_number", "elapsed_time"])
    return windows, stats
