from datetime import datetime, timedelta

import polars as pl

from egcf_processing.dashboard import (
    discover_masses,
    load_table,
    rga_current_to_unit,
    with_elapsed_time_s,
)


def test_load_table_prefers_parquet_over_csv(tmp_path):
    df = pl.DataFrame({"a": [1, 2]})
    df.write_parquet(tmp_path / "rga.parquet")
    df.write_csv(tmp_path / "rga.csv")
    loaded = load_table(tmp_path, "rga")
    assert loaded.height == 2


def test_load_table_falls_back_to_csv(tmp_path):
    df = pl.DataFrame({"a": [1, 2, 3]})
    df.write_csv(tmp_path / "rga.csv")
    loaded = load_table(tmp_path, "rga")
    assert loaded.height == 3


def test_load_table_missing_returns_none(tmp_path):
    assert load_table(tmp_path, "rga") is None


def test_with_elapsed_time_s_from_duration():
    df = pl.DataFrame(
        {"elapsed_time": [timedelta(seconds=90), timedelta(seconds=30)]},
        schema={"elapsed_time": pl.Duration},
    )
    result = with_elapsed_time_s(df)
    assert result["elapsed_time_s"].to_list() == [90.0, 30.0]


def test_with_elapsed_time_s_from_float():
    df = pl.DataFrame({"elapsed_time": [90.0, 30.0]})
    result = with_elapsed_time_s(df)
    assert result["elapsed_time_s"].to_list() == [90.0, 30.0]


def test_discover_masses():
    df = pl.DataFrame({"timestamp": [datetime(2026, 1, 1)], "mass_28_avg": [1.0], "mass_2_torr": [2.0]})
    assert discover_masses(df) == [2, 28]


def test_rga_current_to_unit():
    current = pl.Series("current", [1000.0])
    df = pl.DataFrame({"current": current})
    raw = df.select(rga_current_to_unit(pl.col("current"), "raw", 2e-4).alias("v"))["v"][0]
    amps = df.select(rga_current_to_unit(pl.col("current"), "amps", 2e-4).alias("v"))["v"][0]
    torr = df.select(rga_current_to_unit(pl.col("current"), "torr", 2e-4).alias("v"))["v"][0]
    assert raw == 1000.0
    assert amps == 1000.0 * 1e-16
    assert torr == (1000.0 * 1e-16) / 2e-4
