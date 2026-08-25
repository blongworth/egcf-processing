"""Shared windowed aggregation used by both Layer B (RGA-scan-cycle) and
Layer C (chamber-cycle) outputs -- the two layers differ only in which
``windows`` table (window_start, window_end, chamber, experiment_number,
elapsed_time) they're aggregated onto.
"""

from __future__ import annotations

import polars as pl

_SCALUP_COLS = ["temp_degC", "sal_PSU", "pressure_mbar", "oxygen_mgL", "pH"]
_STATUS_COLS = [
    "turbo_speed_hz",
    "turbo_power_w",
    "water_pump_rpm",
    "total_pressure_amps",
    "total_pressure_torr",
]

# RGA R: readings and the status row's raw_total_pressure_current are both
# raw ion-current counts in units of 1e-16 A (RGAm.pdf RGA Command Set,
# Chapter 6: "Ion currents are represented as integers in units of 10-16
# Amps, and transmitted directly in Hex format").
RAW_CURRENT_AMPS_PER_COUNT = 1e-16

# Nominal Faraday-cup partial/total pressure sensitivity from the RGAm.pdf
# specifications table ("Sensitivity (A/Torr)*: 2e-4 (FC) ... Measured with
# N2 @ 28 amu ..."). This head's actual factory-calibrated SP/ST sensitivity
# isn't recoverable from the SD-card logs, so partial/total pressure in Torr
# is only as accurate as this nominal value -- pass a measured sensitivity
# in if one is available.
DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR = 2e-4
DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR = 2e-4


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


def _aggregate_rga(
    rga: pl.DataFrame,
    windows: pl.DataFrame,
    partial_pressure_sensitivity_a_per_torr: float,
) -> pl.DataFrame:
    matched = _bucketize(rga, windows)
    if matched.is_empty():
        return windows.select("window_start")
    grouped = matched.group_by(["window_start", "mass"]).agg(pl.col("current").mean().alias("current"))
    pivoted = grouped.pivot(index="window_start", on="mass", values="current")
    masses = [c for c in pivoted.columns if c != "window_start"]
    pivoted = pivoted.rename({m: f"mass_{m}_avg" for m in masses})
    derived = []
    for m in masses:
        amps = pl.col(f"mass_{m}_avg") * RAW_CURRENT_AMPS_PER_COUNT
        derived.append(amps.alias(f"mass_{m}_amps"))
        derived.append((amps / partial_pressure_sensitivity_a_per_torr).alias(f"mass_{m}_torr"))
    return pivoted.with_columns(derived)


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


def _aggregate_status(
    status: pl.DataFrame,
    windows: pl.DataFrame,
    total_pressure_sensitivity_a_per_torr: float,
) -> pl.DataFrame:
    matched = _bucketize(status, windows)
    if matched.is_empty():
        return _empty_float_cols(windows, _STATUS_COLS)
    agg = matched.group_by("window_start").agg(
        pl.col("turbo_speed_hz").mean().alias("turbo_speed_hz"),
        pl.col("turbo_power_w").mean().alias("turbo_power_w"),
        pl.col("pump_rpm").mean().alias("water_pump_rpm"),
        pl.col("raw_total_pressure_current").mean().alias("_raw_tp_mean"),
    )
    amps = pl.col("_raw_tp_mean") * RAW_CURRENT_AMPS_PER_COUNT
    return agg.with_columns(
        amps.alias("total_pressure_amps"),
        (amps / total_pressure_sensitivity_a_per_torr).alias("total_pressure_torr"),
    ).drop("_raw_tp_mean")


def aggregate_onto_windows(
    windows: pl.DataFrame,
    rga: pl.DataFrame,
    scalup: pl.DataFrame,
    status: pl.DataFrame,
    partial_pressure_sensitivity_a_per_torr: float = DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR,
    total_pressure_sensitivity_a_per_torr: float = DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR,
) -> pl.DataFrame:
    """Average rga/scalup/status readings onto each window and join with window context."""
    rga_agg = _aggregate_rga(rga, windows, partial_pressure_sensitivity_a_per_torr)
    scalup_agg = _aggregate_scalup(scalup, windows)
    status_agg = _aggregate_status(status, windows, total_pressure_sensitivity_a_per_torr)

    result = (
        windows.join(rga_agg, on="window_start", how="left")
        .join(scalup_agg, on="window_start", how="left")
        .join(status_agg, on="window_start", how="left")
        .rename({"window_start": "timestamp"})
        .drop("window_end")
    )

    masses = sorted(
        {int(c.split("_")[1]) for c in rga_agg.columns if c.startswith("mass_")}
    )
    mass_cols = [f"mass_{m}_{suffix}" for m in masses for suffix in ("avg", "amps", "torr")]
    final_cols = (
        ["timestamp", "experiment_number", "elapsed_time", "chamber"]
        + mass_cols
        + ["total_pressure_amps", "total_pressure_torr"]
        + _SCALUP_COLS
        + ["turbo_speed_hz", "turbo_power_w", "water_pump_rpm"]
    )
    return result.select(final_cols).sort("timestamp")
