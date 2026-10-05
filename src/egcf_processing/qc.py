"""Data-quality checks over the Layer A tables: missingness and gross-range flags.

Kept free of Streamlit so it can be unit tested directly; the dashboard's
Status tab renders the results. Nothing here modifies or drops data -- it
only reports.

Missingness is scored against each stream's normal cadence (STREAM_CADENCE).
A gap is any interval between consecutive records longer than the stream's
gap threshold, *including* the interval from the start of the requested
range to the first record and from the last record to the end of the range,
so a stream that starts late or stops early counts as missing.

Gross-range flags follow QARTOD: ``fail`` means physically impossible or a
sensor error, ``suspect`` means plausible but outside what this site normally
sees. Null (or NaN) values are flagged ``missing``, never ``fail``.
"""

from __future__ import annotations

import math
from datetime import datetime

import polars as pl

STREAM_CADENCE: list[tuple[str, str | None, float, float]] = [
    ("status", None, 8.0, 60.0),
    ("system_health", None, 10.0, 60.0),
    ("rga", "mass", 8.0, 60.0),
    ("scalup", None, 17.0, 180.0),
    ("valve", None, 900.0, 3600.0),
    ("par", None, 300.0, 900.0),
    ("hobo_oxygen", "location", 60.0, 300.0),
]

# Smallest positive float: a fail_lo of _ABOVE_ZERO makes exactly 0 a failure,
# since limits are inclusive.
_ABOVE_ZERO = math.nextafter(0.0, 1.0)

RANGE_LIMITS: dict[tuple[str, str], tuple[float | None, float | None, float | None, float | None]] = {
    ("scalup", "ph"): (_ABOVE_ZERO, 14.0, 7.0, 8.8),
    ("scalup", "sal_psu"): (_ABOVE_ZERO, 42.0, 20.0, None),
    ("scalup", "temp_degc"): (-2.0, 40.0, None, None),
    ("scalup", "oxygen_mgl"): (0.0, 25.0, None, 15.0),
    ("scalup", "pressure_mbar"): (0.0, None, None, None),
    ("hobo_oxygen", "oxygen_mgl"): (0.0, 25.0, None, None),
    ("hobo_oxygen", "temp_degc"): (-2.0, 40.0, None, None),
    ("par", "par_umol_m2_s"): (0.0, 3000.0, None, None),
    ("system_health", "voltage_v"): (20.0, 30.0, 23.0, None),
    ("system_health", "teensy_temp_c"): (None, 85.0, None, 70.0),
    ("status", "turbo_error"): (0.0, 0.0, None, None),
    ("status", "raw_total_pressure_current"): (0.0, None, None, None),
    ("status", "turbo_speed_hz"): (None, None, 1200.0, None),
    ("rga", "current"): (_ABOVE_ZERO, None, None, None),
}

GAPS_SCHEMA = {"group": pl.Utf8, "gap_start": pl.Datetime, "gap_end": pl.Datetime, "gap_s": pl.Float64}

COMPLETENESS_SCHEMA = {
    "stream": pl.Utf8,
    "table": pl.Utf8,
    "group": pl.Utf8,
    "nominal_interval_s": pl.Float64,
    "gap_threshold_s": pl.Float64,
    "expected": pl.Float64,
    "observed": pl.Int64,
    "completeness": pl.Float64,
    "n_gaps": pl.Int64,
    "longest_gap_s": pl.Float64,
    "total_gap_s": pl.Float64,
    "last_record": pl.Datetime,
    "staleness_s": pl.Float64,
}

SEGMENTS_SCHEMA = {"stream": pl.Utf8, "seg_start": pl.Datetime, "seg_end": pl.Datetime}

RANGE_SUMMARY_SCHEMA = {
    "table": pl.Utf8,
    "column": pl.Utf8,
    "n": pl.Int64,
    "n_null": pl.Int64,
    "n_suspect": pl.Int64,
    "n_fail": pl.Int64,
    "pct_flagged": pl.Float64,
    "min": pl.Float64,
    "max": pl.Float64,
    "first_flag_ts": pl.Datetime,
    "last_flag_ts": pl.Datetime,
}

FLAGGED_ROWS_SCHEMA = {
    "table": pl.Utf8,
    "column": pl.Utf8,
    "ts": pl.Datetime,
    "value": pl.Float64,
    "flag": pl.Utf8,
}


def _stream_label(table: str, group: str | None) -> str:
    return table if group is None else f"{table} {group}"


def _group_ts(df: pl.DataFrame, ts_col: str, group_col: str | None) -> pl.DataFrame:
    group = pl.col(group_col).cast(pl.Utf8) if group_col else pl.lit(None, dtype=pl.Utf8)
    return df.select(group.alias("group"), pl.col(ts_col).cast(pl.Datetime).alias("ts")).drop_nulls("ts")


def find_gaps(
    df: pl.DataFrame,
    ts_col: str,
    group_col: str | None,
    threshold_s: float,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    """Intervals longer than threshold_s with no record, per group.

    With start/end given, records outside [start, end] are ignored and the
    leading (start -> first record) and trailing (last record -> end)
    intervals count as gaps too. A group with no records in range at all
    cannot be named, so a stream that is entirely empty yields one gap over
    the whole range with a null group.
    """
    ts = _group_ts(df, ts_col, group_col)
    if start is not None:
        ts = ts.filter(pl.col("ts") >= start)
    if end is not None:
        ts = ts.filter(pl.col("ts") <= end)
    if ts.is_empty():
        if start is None or end is None or (end - start).total_seconds() <= threshold_s:
            return pl.DataFrame(schema=GAPS_SCHEMA)
        return pl.DataFrame(
            {"group": [None], "gap_start": [start], "gap_end": [end], "gap_s": [(end - start).total_seconds()]},
            schema=GAPS_SCHEMA,
        )

    groups = ts.select("group").unique()
    sentinels = [
        groups.with_columns(pl.lit(bound, dtype=pl.Datetime).alias("ts")) for bound in (start, end) if bound is not None
    ]
    return (
        pl.concat([ts, *sentinels])
        .sort("group", "ts", nulls_last=True)
        .with_columns(pl.col("ts").shift(1).over("group").alias("gap_start"))
        .with_columns((pl.col("ts") - pl.col("gap_start")).dt.total_microseconds().truediv(1e6).alias("gap_s"))
        .filter(pl.col("gap_s") > threshold_s)
        .select("group", "gap_start", pl.col("ts").alias("gap_end"), "gap_s")
        .cast(GAPS_SCHEMA)
    )


def _nominal_interval_s(table: str, df: pl.DataFrame, default: float) -> float:
    if table == "par" and "interval_s" in df.columns:
        interval = df["interval_s"].drop_nulls().median()
        if interval is not None and interval > 0:
            return float(interval)
    return default


def _stream_frames(tables: dict[str, pl.DataFrame | None], config):
    """Yield (table, group_col, nominal, threshold, df, ts_col) for every loaded stream."""
    for table, group_col, nominal, threshold in config:
        df = tables.get(table)
        if df is None or "ts" not in df.columns:
            continue
        if group_col is not None and group_col not in df.columns:
            group_col = None
        yield table, group_col, _nominal_interval_s(table, df, nominal), threshold, df, "ts"


def _range_bounds(ts: pl.DataFrame, start: datetime | None, end: datetime | None):
    lo = start if start is not None else ts["ts"].min()
    hi = end if end is not None else ts["ts"].max()
    return lo, hi


def completeness(
    tables: dict[str, pl.DataFrame | None],
    config: list[tuple[str, str | None, float, float]] = STREAM_CADENCE,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    """One row per stream (and per group, for grouped streams) scoring missingness.

    ``observed`` counts distinct timestamps, so duplicated records don't
    inflate completeness. ``staleness_s`` is the time from the last record to
    the end of the range -- for live telemetry, how long the stream has been
    silent.
    """
    rows = []
    for table, group_col, nominal, threshold, df, ts_col in _stream_frames(tables, config):
        ts = _group_ts(df, ts_col, group_col)
        if start is not None:
            ts = ts.filter(pl.col("ts") >= start)
        if end is not None:
            ts = ts.filter(pl.col("ts") <= end)
        lo, hi = _range_bounds(ts, start, end)
        gaps = find_gaps(df, ts_col, group_col, threshold, start, end)
        groups = ts["group"].unique().sort(nulls_last=True).to_list() or [None]
        for group in groups:
            group_ts = ts.filter(pl.col("group").eq_missing(group))["ts"]
            group_gaps = gaps.filter(pl.col("group").eq_missing(group))
            span_s = (hi - lo).total_seconds() if lo is not None and hi is not None else 0.0
            expected = span_s / nominal
            observed = group_ts.n_unique()
            last_record = group_ts.max()
            rows.append(
                {
                    "stream": _stream_label(table, group),
                    "table": table,
                    "group": group,
                    "nominal_interval_s": nominal,
                    "gap_threshold_s": threshold,
                    "expected": expected,
                    "observed": observed,
                    "completeness": min(observed / expected, 1.0) if expected > 0 else None,
                    "n_gaps": group_gaps.height,
                    "longest_gap_s": group_gaps["gap_s"].max() if group_gaps.height else 0.0,
                    "total_gap_s": group_gaps["gap_s"].sum(),
                    "last_record": last_record,
                    "staleness_s": (hi - last_record).total_seconds()
                    if hi is not None and last_record is not None
                    else None,
                }
            )
    return pl.DataFrame(rows, schema=COMPLETENESS_SCHEMA)


def coverage_segments(
    tables: dict[str, pl.DataFrame | None],
    config: list[tuple[str, str | None, float, float]] = STREAM_CADENCE,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    """Contiguous runs of data per stream: the complement of find_gaps within the range."""
    segments = []
    for table, group_col, _nominal, threshold, df, ts_col in _stream_frames(tables, config):
        ts = _group_ts(df, ts_col, group_col)
        if start is not None:
            ts = ts.filter(pl.col("ts") >= start)
        if end is not None:
            ts = ts.filter(pl.col("ts") <= end)
        if ts.is_empty():
            continue
        lo, hi = _range_bounds(ts, start, end)
        gaps = find_gaps(df, ts_col, group_col, threshold, start, end)
        for group in ts["group"].unique().sort(nulls_last=True).to_list():
            group_gaps = gaps.filter(pl.col("group").eq_missing(group)).sort("gap_start")
            seg_starts = [lo, *group_gaps["gap_end"].to_list()]
            seg_ends = [*group_gaps["gap_start"].to_list(), hi]
            label = _stream_label(table, group)
            segments += [(label, s, e) for s, e in zip(seg_starts, seg_ends) if s < e]
    return pl.DataFrame(segments, schema=SEGMENTS_SCHEMA, orient="row")


def range_flags(column: str, limits: tuple[float | None, float | None, float | None, float | None]) -> pl.Expr:
    """``pass``/``suspect``/``fail``/``missing`` for each value of column.

    Limits are (fail_lo, fail_hi, suspect_lo, suspect_hi), each inclusive
    (a value equal to a limit is within it) and None for an open bound.
    """
    fail_lo, fail_hi, suspect_lo, suspect_hi = limits
    value = pl.col(column).cast(pl.Float64)

    def outside(lo: float | None, hi: float | None) -> pl.Expr:
        conds = []
        if lo is not None:
            conds.append(value < lo)
        if hi is not None:
            conds.append(value > hi)
        return pl.any_horizontal(conds) if conds else pl.lit(False)

    return (
        pl.when(value.is_null() | value.is_nan())
        .then(pl.lit("missing"))
        .when(outside(fail_lo, fail_hi))
        .then(pl.lit("fail"))
        .when(outside(suspect_lo, suspect_hi))
        .then(pl.lit("suspect"))
        .otherwise(pl.lit("pass"))
    )


def _flagged_frames(tables: dict[str, pl.DataFrame | None], config):
    for (table, column), limits in config.items():
        df = tables.get(table)
        if df is None or column not in df.columns or "ts" not in df.columns:
            continue
        yield table, column, df.select("ts", pl.col(column).cast(pl.Float64).alias("value"), range_flags(column, limits).alias("flag"))


def range_summary(
    tables: dict[str, pl.DataFrame | None],
    config: dict[tuple[str, str], tuple] = RANGE_LIMITS,
) -> pl.DataFrame:
    """Per (table, column) counts of missing/suspect/fail values, with the flagged time span."""
    rows = []
    for table, column, flagged in _flagged_frames(tables, config):
        is_flag = pl.col("flag").is_in(["suspect", "fail"])
        stats = flagged.select(
            pl.len().alias("n"),
            (pl.col("flag") == "missing").sum().alias("n_null"),
            (pl.col("flag") == "suspect").sum().alias("n_suspect"),
            (pl.col("flag") == "fail").sum().alias("n_fail"),
            pl.col("value").filter(pl.col("value").is_not_nan()).min().alias("min"),
            pl.col("value").filter(pl.col("value").is_not_nan()).max().alias("max"),
            pl.col("ts").filter(is_flag).min().alias("first_flag_ts"),
            pl.col("ts").filter(is_flag).max().alias("last_flag_ts"),
        ).row(0, named=True)
        n_flagged = stats["n_suspect"] + stats["n_fail"]
        rows.append(
            {
                "table": table,
                "column": column,
                **stats,
                "pct_flagged": 100.0 * n_flagged / stats["n"] if stats["n"] else None,
            }
        )
    return pl.DataFrame(rows, schema=RANGE_SUMMARY_SCHEMA)


def flagged_rows(
    tables: dict[str, pl.DataFrame | None],
    config: dict[tuple[str, str], tuple] = RANGE_LIMITS,
) -> pl.DataFrame:
    """Every suspect or failed value, long format, for drill-down."""
    frames = [
        flagged.filter(pl.col("flag").is_in(["suspect", "fail"])).select(
            pl.lit(table).alias("table"), pl.lit(column).alias("column"), "ts", "value", "flag"
        )
        for table, column, flagged in _flagged_frames(tables, config)
    ]
    if not frames:
        return pl.DataFrame(schema=FLAGGED_ROWS_SCHEMA)
    return pl.concat(frames).cast(FLAGGED_ROWS_SCHEMA).sort("table", "column", "ts")
