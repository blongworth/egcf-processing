"""Layer A: combine parsed records into one Parquet file per record type."""

from __future__ import annotations

from pathlib import Path

import polars as pl

RGA_SCHEMA = {"ts": pl.Datetime, "mass": pl.Int64, "current": pl.Float64}

VALVE_SCHEMA = {"ts": pl.Datetime, "chamber": pl.Utf8, "flush_state": pl.Utf8}

SCALUP_SCHEMA = {
    "ts": pl.Datetime,
    "ts_scalup": pl.Datetime,
    "temp_degc": pl.Float64,
    "sal_psu": pl.Float64,
    "pressure_mbar": pl.Float64,
    "oxygen_mgl": pl.Float64,
    "ph": pl.Float64,
}

STATUS_SCHEMA = {
    "ts": pl.Datetime,
    "turbo_error": pl.Float64,
    "turbo_speed_hz": pl.Float64,
    "turbo_power_w": pl.Float64,
    "turbo_voltage": pl.Float64,
    "turbo_etemp_c": pl.Float64,
    "turbo_btemp_c": pl.Float64,
    "turbo_mtemp_c": pl.Float64,
    "rga_filament": pl.Float64,
    "raw_total_pressure_current": pl.Float64,
    "pump_rpm": pl.Float64,
    "payload_raw": pl.Utf8,
}

_TAG_SCHEMAS = {
    "R": ("rga", RGA_SCHEMA),
    "V": ("valve", VALVE_SCHEMA),
    "P": ("scalup", SCALUP_SCHEMA),
    "!": ("status", STATUS_SCHEMA),
}


def _rows_for_schema(rows: list[dict], schema: dict) -> list[dict]:
    keys = list(schema.keys())
    return [{k: row.get(k) for k in keys} for row in rows]


def build_tables(records: list[dict]) -> dict[str, pl.DataFrame]:
    """Split parsed records by tag into one pl.DataFrame per record type."""
    buckets: dict[str, list[dict]] = {name: [] for _, (name, _) in _TAG_SCHEMAS.items()}
    for record in records:
        entry = _TAG_SCHEMAS.get(record["tag"])
        if entry is None:
            continue
        name, _ = entry
        buckets[name].append(record)

    tables = {}
    for _, (name, schema) in _TAG_SCHEMAS.items():
        rows = _rows_for_schema(buckets[name], schema)
        tables[name] = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
    return tables


def write_df(df: pl.DataFrame, out_dir: Path, name: str, output_format: str = "parquet") -> Path:
    """Write a table to out_dir/{name}.{ext} in the given format ("parquet" or "csv").

    CSV has no duration type, so any Duration column (elapsed_time) is
    converted to whole seconds (float) for the CSV output only -- the
    parquet output keeps the native Duration dtype.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_format == "csv":
        path = out_dir / f"{name}.csv"
        duration_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt.base_type() == pl.Duration]
        if duration_cols:
            df = df.with_columns((pl.col(c).dt.total_microseconds() / 1_000_000).alias(c) for c in duration_cols)
        df.write_csv(path)
    else:
        path = out_dir / f"{name}.parquet"
        df.write_parquet(path)
    return path


def write_tables(tables: dict[str, pl.DataFrame], out_dir: Path, output_format: str = "parquet") -> dict[str, Path]:
    return {name: write_df(df, out_dir, name, output_format) for name, df in tables.items()}
