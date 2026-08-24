"""Shared windowed aggregation used by both Layer B (RGA-scan-cycle) and
Layer C (chamber-cycle) outputs -- the two layers differ only in which
``windows`` table (window_start, window_end, chamber, experiment_number,
elapsed_time) they're aggregated onto.
"""

from __future__ import annotations

import polars as pl

_SCALUP_COLS = ["temp_degC", "sal_PSU", "pressure_mbar", "oxygen_mgL", "pH"]
_STATUS_COLS = ["turbo_speed_hz", "turbo_power_w", "water_pump_rpm", "total_pressure_amps"]


def _bucketize(readings: pl.DataFrame, windows: pl.DataFrame, ts_col: str = "ts") -> pl.DataFrame:
    """Match each reading to the window whose [window_start, window_end) it falls in.

    join_asof(backward) finds the latest window_start at or before the
    reading's timestamp; the explicit upper-bound filter then excludes
    readings that fall in a settle/flush gap or past the window's end.
    """
    if windows.is_empty():
        return readings.clear().with_columns(window_start=pl.lit(None, dtype=pl.Datetime))
    matched = readings.sort(ts_col).join_asof(
        windows.select(["window_start", "window_end"]).sort("window_start"),
        left_on=ts_col,
        right_on="window_start",
        strategy="backward",
    )
    return matched.filter(pl.col("window_start").is_not_null() & (pl.col(ts_col) < pl.col("window_end")))


def _empty_float_cols(windows: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return windows.select("window_start").with_columns(
        [pl.lit(None, dtype=pl.Float64).alias(c) for c in cols]
    )


def _aggregate_rga(rga: pl.DataFrame, windows: pl.DataFrame) -> pl.DataFrame:
    matched = _bucketize(rga, windows)
    if matched.is_empty():
        return windows.select("window_start")
    grouped = matched.group_by(["window_start", "mass"]).agg(pl.col("current").mean().alias("current"))
    pivoted = grouped.pivot(index="window_start", on="mass", values="current")
    rename_map = {c: f"mass_{c}_avg" for c in pivoted.columns if c != "window_start"}
    return pivoted.rename(rename_map)


def _aggregate_scalup(scalup: pl.DataFrame, windows: pl.DataFrame) -> pl.DataFrame:
    matched = _bucketize(scalup, windows)
    if matched.is_empty():
        return _empty_float_cols(windows, _SCALUP_COLS)
    return matched.group_by("window_start").agg(
        pl.col("temp_degc").mean().alias("temp_degC"),
        pl.col("sal_psu").mean().alias("sal_PSU"),
        pl.col("pressure_mbar").mean().alias("pressure_mbar"),
        pl.col("oxygen_mgl").mean().alias("oxygen_mgL"),
        pl.col("ph").mean().alias("pH"),
    )


def _aggregate_status(status: pl.DataFrame, windows: pl.DataFrame) -> pl.DataFrame:
    matched = _bucketize(status, windows)
    if matched.is_empty():
        return _empty_float_cols(windows, _STATUS_COLS)
    agg = matched.group_by("window_start").agg(
        pl.col("turbo_speed_hz").mean().alias("turbo_speed_hz"),
        pl.col("turbo_power_w").mean().alias("turbo_power_w"),
        pl.col("pump_rpm").mean().alias("water_pump_rpm"),
        pl.col("raw_total_pressure_current").mean().alias("_raw_tp_mean"),
    )
    return agg.with_columns((pl.col("_raw_tp_mean") * 1e-16).alias("total_pressure_amps")).drop("_raw_tp_mean")


def aggregate_onto_windows(
    windows: pl.DataFrame,
    rga: pl.DataFrame,
    scalup: pl.DataFrame,
    status: pl.DataFrame,
) -> pl.DataFrame:
    """Average rga/scalup/status readings onto each window and join with window context."""
    rga_agg = _aggregate_rga(rga, windows)
    scalup_agg = _aggregate_scalup(scalup, windows)
    status_agg = _aggregate_status(status, windows)

    result = (
        windows.join(rga_agg, on="window_start", how="left")
        .join(scalup_agg, on="window_start", how="left")
        .join(status_agg, on="window_start", how="left")
        .rename({"window_start": "timestamp"})
        .drop("window_end")
    )

    mass_cols = sorted(
        (c for c in rga_agg.columns if c.startswith("mass_")),
        key=lambda c: int(c.split("_")[1]),
    )
    final_cols = (
        ["timestamp", "experiment_number", "elapsed_time", "chamber"]
        + mass_cols
        + ["total_pressure_amps"]
        + _SCALUP_COLS
        + ["turbo_speed_hz", "turbo_power_w", "water_pump_rpm"]
    )
    return result.select(final_cols).sort("timestamp")
