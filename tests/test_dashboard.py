from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from streamlit.testing.v1 import AppTest

from egcf_processing.dashboard import (
    discover_masses,
    load_table,
    mass_color_map,
    rga_current_to_unit,
    with_elapsed_time_s,
)

DASHBOARD_PATH = Path(__file__).resolve().parents[1] / "src" / "egcf_processing" / "dashboard.py"


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


def test_mass_color_map_is_stable_regardless_of_input_order():
    assert mass_color_map([28, 2, 15]) == mass_color_map([2, 15, 28])


def test_mass_color_map_gives_each_mass_its_own_color_within_a_palette_cycle():
    colors = mass_color_map([2, 15, 16])
    assert len(set(colors.values())) == 3


def test_experiment_tab_download_button_handles_parquet_duration(tmp_path):
    # Regression test: when egcf_chamber_cycles is loaded from parquet, elapsed_time
    # keeps its native Duration dtype (with_elapsed_time_s only adds a derived
    # elapsed_time_s float column alongside it) -- the download button's CSV export
    # must convert it too, or polars raises ComputeError on the Duration column.
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1)],
            "chamber": ["C1"],
            "experiment_number": [1],
            "elapsed_time": [timedelta(seconds=90)],
            "mass_2_avg": [10.0],
        },
        schema_overrides={"elapsed_time": pl.Duration},
    )
    df.write_parquet(tmp_path / "egcf_chamber_cycles.parquet")

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    at.tabs[2].radio(key="experiment_grain").set_value("Chamber cycle").run(timeout=60)
    assert not at.exception
