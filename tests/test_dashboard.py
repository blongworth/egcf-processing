from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from egcf_processing.dashboard import (
    attach_experiment_context,
    discover_masses,
    experiment_rates,
    experiment_start_times,
    linear_fit,
    load_table,
    mass_color_map,
    mass_to_argon_ratio_expr,
    rga_current_to_unit,
    rga_full_ratio_to_mass,
    rga_wide_ratio_to_mass,
    variable_value_expr,
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


def test_rga_full_ratio_to_mass_matches_nearest_in_time():
    rga = pl.DataFrame(
        {
            "ts": [
                datetime(2026, 1, 1, 0, 0, 0),
                datetime(2026, 1, 1, 0, 0, 1),
                datetime(2026, 1, 1, 0, 0, 10),
                datetime(2026, 1, 1, 0, 0, 11),
            ],
            "mass": [2, 40, 2, 40],
            "current": [10.0, 100.0, 20.0, 200.0],
        }
    )
    ratio = rga_full_ratio_to_mass(rga, [2, 40])
    assert ratio["mass"].to_list() == [2, 2]
    assert ratio["ratio"].to_list() == [0.1, 0.1]


def test_rga_full_ratio_to_mass_drops_zero_reference_and_excludes_reference_mass():
    rga = pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1), datetime(2026, 1, 1, 0, 0, 1)],
            "mass": [40, 2],
            "current": [0.0, 10.0],
        }
    )
    ratio = rga_full_ratio_to_mass(rga, [2, 40])
    assert ratio.is_empty()


def test_rga_full_ratio_to_mass_empty_without_reference_mass_data():
    rga = pl.DataFrame({"ts": [datetime(2026, 1, 1)], "mass": [2], "current": [10.0]})
    assert rga_full_ratio_to_mass(rga, [2]).is_empty()


def test_rga_wide_ratio_to_mass():
    table = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1), datetime(2026, 1, 1, 0, 1)],
            "mass_2_avg": [10.0, 30.0],
            "mass_40_avg": [100.0, 0.0],
        }
    )
    ratio = rga_wide_ratio_to_mass(table, [2, 40], ts_col="timestamp")
    assert ratio["mass"].to_list() == [2]
    assert ratio["ratio"].to_list() == [0.1]


def test_mass_to_argon_ratio_expr():
    table = pl.DataFrame({"mass_2_avg": [10.0, 30.0], "mass_40_avg": [100.0, 0.0]})
    result = table.select(mass_to_argon_ratio_expr(2).alias("ratio"))["ratio"].to_list()
    assert result == [0.1, None]


def test_variable_value_expr_mass():
    table = pl.DataFrame({"mass_2_avg": [10.0], "mass_40_avg": [100.0]})
    assert table.select(variable_value_expr("mass_2", True))["value"].to_list() == [0.1]


def test_variable_value_expr_non_mass():
    table = pl.DataFrame({"temp_degC": [12.5]})
    assert table.select(variable_value_expr("temp_degC", False))["value"].to_list() == [12.5]


def test_linear_fit_exact_line():
    # y = 2x + 1
    fit = linear_fit([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0])
    assert fit == (2.0, 1.0)


def test_linear_fit_none_with_fewer_than_two_points():
    assert linear_fit([1.0], [1.0]) is None
    assert linear_fit([], []) is None


def test_linear_fit_none_with_zero_x_variance():
    assert linear_fit([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) is None


def test_linear_fit_skips_null_pairs():
    fit = linear_fit([0.0, 1.0, None, 2.0], [1.0, 3.0, 99.0, 5.0])
    assert fit == (2.0, 1.0)


def test_experiment_rates_one_slope_per_experiment_and_chamber():
    source = pl.DataFrame(
        {
            "experiment_number": [1, 1, 1, 1, 2, 2],
            "chamber": ["C1", "C1", "C2", "C2", "C1", "C1"],
            "timestamp": [
                datetime(2026, 1, 1, 0, 0),
                datetime(2026, 1, 1, 0, 1),
                datetime(2026, 1, 1, 0, 0),
                datetime(2026, 1, 1, 0, 1),
                datetime(2026, 1, 2, 0, 0),
                datetime(2026, 1, 2, 0, 1),
            ],
            "elapsed_time_s": [0.0, 60.0, 0.0, 60.0, 0.0, 60.0],
            "mass_2_avg": [10.0, 20.0, 10.0, 40.0, 5.0, 5.0],
            "mass_40_avg": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
    rates = experiment_rates(source, "mass_2", is_mass_variable=True)
    assert rates.sort(["experiment_number", "chamber"])["rate"].to_list() == pytest.approx(
        [
            0.1,  # C1 exp 1: ratio 0.1 -> 0.2 over 1 min
            0.3,  # C2 exp 1: ratio 0.1 -> 0.4 over 1 min
            0.0,  # C1 exp 2: ratio 0.05 -> 0.05, flat
        ]
    )


def test_experiment_rates_empty_source_returns_empty_schema():
    source = pl.DataFrame(
        {
            "experiment_number": pl.Series([], dtype=pl.Int64),
            "chamber": pl.Series([], dtype=pl.Utf8),
            "timestamp": pl.Series([], dtype=pl.Datetime),
            "elapsed_time_s": pl.Series([], dtype=pl.Float64),
            "temp_degC": pl.Series([], dtype=pl.Float64),
        }
    )
    result = experiment_rates(source, "temp_degC", is_mass_variable=False)
    assert result.is_empty()
    assert result.columns == ["experiment_number", "chamber", "experiment_start", "rate"]


def test_experiment_start_times_min_of_ts_minus_elapsed_per_experiment():
    source = pl.DataFrame(
        {
            "experiment_number": [1, 1, 2],
            "window_start": [
                datetime(2026, 1, 1, 0, 1, 0),
                datetime(2026, 1, 1, 0, 11, 0),
                datetime(2026, 1, 2, 0, 5, 0),
            ],
            "elapsed_time": [timedelta(seconds=60), timedelta(seconds=660), timedelta(seconds=300)],
        }
    )
    result = experiment_start_times(source, "window_start")
    assert result == {1: datetime(2026, 1, 1, 0, 0, 0), 2: datetime(2026, 1, 2, 0, 0, 0)}


def test_attach_experiment_context_uses_each_readings_own_timestamp():
    # Two readings inside the same cycle, at different times: their elapsed_time
    # must differ (each relative to its own ts), not both equal the window's
    # single, constant per-cycle elapsed_time.
    windows = pl.DataFrame(
        {
            "window_start": [datetime(2026, 1, 1, 0, 0, 10)],
            "window_end": [datetime(2026, 1, 1, 0, 0, 30)],
            "chamber": ["C1"],
            "experiment_number": [1],
            "elapsed_time": [timedelta(seconds=10)],  # exp_start = window_start - elapsed_time = 00:00:00
        }
    )
    readings = pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1, 0, 0, 12), datetime(2026, 1, 1, 0, 0, 22)],
            "mass": [2, 2],
            "current": [1.0, 2.0],
        }
    )
    result = attach_experiment_context(readings, windows, ts_col="ts")
    assert result["chamber"].to_list() == ["C1", "C1"]
    assert result["experiment_number"].to_list() == [1, 1]
    assert [d.total_seconds() for d in result["elapsed_time"].to_list()] == [12.0, 22.0]


def test_attach_experiment_context_marks_settled_out_without_dropping_rows():
    # Readings within settle_offset_s of the window's own start (the valve switch)
    # are flagged settled_out, without removing any rows -- Full data shows every
    # reading, greying out the settled-out ones.
    windows = pl.DataFrame(
        {
            "window_start": [datetime(2026, 1, 1, 0, 0, 0)],
            "window_end": [datetime(2026, 1, 1, 0, 1, 0)],
            "chamber": ["C1"],
            "experiment_number": [1],
            "elapsed_time": [timedelta(seconds=0)],
        }
    )
    readings = pl.DataFrame(
        {
            "ts": [
                datetime(2026, 1, 1, 0, 0, 1),
                datetime(2026, 1, 1, 0, 0, 2),
                datetime(2026, 1, 1, 0, 0, 3),
            ],
            "mass": [2, 2, 2],
            "current": [1.0, 2.0, 3.0],
        }
    )
    result = attach_experiment_context(readings, windows, ts_col="ts", settle_offset_s=2.0)
    assert result.height == 3
    assert result.sort("ts")["settled_out"].to_list() == [True, False, False]


def test_attach_experiment_context_drops_readings_outside_any_window():
    windows = pl.DataFrame(
        {
            "window_start": [datetime(2026, 1, 1, 0, 0, 10)],
            "window_end": [datetime(2026, 1, 1, 0, 0, 20)],
            "chamber": ["C1"],
            "experiment_number": [1],
            "elapsed_time": [timedelta(seconds=0)],
        }
    )
    readings = pl.DataFrame({"ts": [datetime(2026, 1, 1)], "mass": [2], "current": [1.0]})
    assert attach_experiment_context(readings, windows, ts_col="ts").is_empty()


def _write_one_cycle_valve_and_rga(tmp_path):
    # Cycle spans 5 minutes so it survives the default 60s settle offset, with
    # readings well past that mark so they aren't settled out either.
    valve_df = pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1, 0, 0, 0), datetime(2026, 1, 1, 0, 5, 0)],
            "chamber": ["C1", "C2"],
            "flush_state": ["Re", "Fl"],
        }
    )
    valve_df.write_parquet(tmp_path / "valve.parquet")
    rga_df = pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1, 0, 1, 40), datetime(2026, 1, 1, 0, 1, 50)],
            "mass": [2, 40],
            "current": [10.0, 100.0],
        }
    )
    rga_df.write_parquet(tmp_path / "rga.parquet")


def test_experiment_tab_cycle_averages_shows_fit_and_rates_plot(tmp_path):
    # Two C1 cycles, both in experiment 1 (no C2 flush ever occurs, so the
    # experiment boundary never advances), 5 min apart and each 5 min long so
    # both survive the default 60s settle offset -- linear_fit needs 2 points.
    valve_df = pl.DataFrame(
        {
            "ts": [
                datetime(2026, 1, 1, 0, 0, 0),
                datetime(2026, 1, 1, 0, 5, 0),
                datetime(2026, 1, 1, 0, 10, 0),
                datetime(2026, 1, 1, 0, 15, 0),
            ],
            "chamber": ["C1", "C1", "C1", "C1"],
            "flush_state": ["Re", "Fl", "Re", "Fl"],
        }
    )
    valve_df.write_parquet(tmp_path / "valve.parquet")
    rga_df = pl.DataFrame(
        {
            "ts": [
                datetime(2026, 1, 1, 0, 2, 30),
                datetime(2026, 1, 1, 0, 2, 40),
                datetime(2026, 1, 1, 0, 12, 30),
                datetime(2026, 1, 1, 0, 12, 40),
            ],
            "mass": [2, 40, 2, 40],
            "current": [10.0, 100.0, 20.0, 100.0],
        }
    )
    rga_df.write_parquet(tmp_path / "rga.parquet")

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    at.tabs[2].radio(key="experiment_grain").set_value("Cycle averages").run(timeout=60)
    assert not at.exception

    tab = at.tabs[2]
    charts = tab.get("plotly_chart")
    assert len(charts) == 2
    main_spec = charts[0].proto.spec
    assert "fit (" in main_spec
    assert "Started 2026-01-01 00:00:00" in main_spec
    rates_spec = charts[1].proto.spec
    assert "rate per experiment" in rates_spec


def test_experiment_tab_full_data_grain_shows_experiment_start_subtitle(tmp_path):
    valve_df = pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1, 0, 0, 0), datetime(2026, 1, 1, 0, 5, 0)],
            "chamber": ["C1", "C2"],
            "flush_state": ["Re", "Fl"],
        }
    )
    valve_df.write_parquet(tmp_path / "valve.parquet")
    rga_df = pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1, 0, 1, 0), datetime(2026, 1, 1, 0, 1, 10)],
            "mass": [2, 40],
            "current": [10.0, 100.0],
        }
    )
    rga_df.write_parquet(tmp_path / "rga.parquet")

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    assert not at.exception

    tab = at.tabs[2]
    fig_spec = tab.get("plotly_chart")[0].proto.spec
    assert "Started 2026-01-01 00:00:00" in fig_spec

    exp_select = [s for s in tab.get("selectbox") if s.label == "Experiment"][0]
    assert exp_select.options == ["1 (2026-01-01 00:00:00)"]


def test_experiment_tab_cycle_averages_download_button_handles_duration(tmp_path):
    # Regression test: the live-computed cycle-average table's elapsed_time is
    # always a Duration (from cycles.chamber_cycle_windows) -- the download
    # button's CSV export must convert it, or polars raises ComputeError.
    _write_one_cycle_valve_and_rga(tmp_path)

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    at.tabs[2].radio(key="experiment_grain").set_value("Cycle averages").run(timeout=60)
    assert not at.exception


def test_experiment_tab_cycle_averages_variable_options_and_settle_slider(tmp_path):
    _write_one_cycle_valve_and_rga(tmp_path)

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    tab = at.tabs[2].radio(key="experiment_grain").set_value("Cycle averages").run(timeout=60)
    assert not at.exception

    tab = at.tabs[2]
    variable_select = tab.selectbox(key="experiment_variable")
    # egcf_chamber_cycles always carries the full output schema even when a source
    # table is absent (aggregate_onto_windows fills it with all-null columns) --
    # scalup/status options are present here even though no scalup/status.parquet
    # was written, matching how a real pipeline-generated table always looks.
    assert variable_select.options == [
        "mass_2",
        "total_pressure_amps",
        "total_pressure_torr",
        "temp_degC",
        "sal_PSU",
        "pressure_mbar",
        "oxygen_mgL",
        "pH",
        "turbo_speed_hz",
        "turbo_power_w",
        "water_pump_rpm",
    ]
    assert [r.label for r in tab.get("radio")] == ["Grain"]
    assert [s.label for s in tab.get("slider")] == ["Settling time after valve switch (s)"]
    exp_select = [s for s in tab.get("selectbox") if s.label == "Experiment"][0]
    assert exp_select.options == ["1 (2026-01-01 00:00:00)"]


def test_experiment_tab_full_data_grain_uses_per_reading_elapsed_time(tmp_path):
    # Two chamber cycles in experiment 1: readings within the same cycle must get
    # distinct elapsed_time values (their own ts), not the cycle's single constant.
    valve_df = pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1, 0, 0, 0), datetime(2026, 1, 1, 0, 0, 20), datetime(2026, 1, 1, 0, 0, 40)],
            "chamber": ["C1", "C2", "C1"],
            "flush_state": ["Re", "Fl", "Re"],
        }
    )
    valve_df.write_parquet(tmp_path / "valve.parquet")
    rga_df = pl.DataFrame(
        {
            "ts": [
                datetime(2026, 1, 1, 0, 0, 2),
                datetime(2026, 1, 1, 0, 0, 4),
                datetime(2026, 1, 1, 0, 0, 8),
                datetime(2026, 1, 1, 0, 0, 10),
            ],
            "mass": [2, 40, 2, 40],
            "current": [10.0, 100.0, 20.0, 200.0],
        }
    )
    rga_df.write_parquet(tmp_path / "rga.parquet")

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    assert not at.exception

    tab = at.tabs[2]
    assert [r.options for r in tab.get("radio") if r.label == "Grain"] == [["Full data", "Cycle averages"]]
    var_select = [s for s in tab.get("selectbox") if s.label == "Variable"][0]
    assert var_select.options == ["mass_2"]

    tab = at.tabs[2]
    variable_select = tab.selectbox(key="experiment_variable")
    assert variable_select.options == ["mass_2"]


def test_experiment_tab_full_data_grain_settling_slider_greys_out_dropped_points(tmp_path):
    valve_df = pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1, 0, 0, 0), datetime(2026, 1, 1, 0, 1, 0)],
            "chamber": ["C1", "C2"],
            "flush_state": ["Re", "Fl"],
        }
    )
    valve_df.write_parquet(tmp_path / "valve.parquet")
    rga_df = pl.DataFrame(
        {
            "ts": [
                datetime(2026, 1, 1, 0, 0, 1),
                datetime(2026, 1, 1, 0, 0, 2),
                datetime(2026, 1, 1, 0, 0, 3),
                datetime(2026, 1, 1, 0, 0, 1, 500000),
                datetime(2026, 1, 1, 0, 0, 2, 500000),
            ],
            "mass": [2, 2, 2, 40, 40],
            "current": [10.0, 20.0, 30.0, 100.0, 100.0],
        }
    )
    rga_df.write_parquet(tmp_path / "rga.parquet")

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    assert not at.exception

    tab = at.tabs[2]
    slider = [s for s in tab.get("slider") if "settl" in s.label.lower()][0]
    slider.set_value(2).run(timeout=60)
    assert not at.exception

    tab = at.tabs[2]
    fig_spec = tab.get("plotly_chart")[0].proto.spec
    assert '"dropped (settling)"' in fig_spec
    assert '"#B0B0B0"' in fig_spec


def test_experiment_tab_full_data_grain_includes_non_rga_variables(tmp_path):
    valve_df = pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1, 0, 0, 0), datetime(2026, 1, 1, 0, 0, 20), datetime(2026, 1, 1, 0, 0, 40)],
            "chamber": ["C1", "C2", "C1"],
            "flush_state": ["Re", "Fl", "Re"],
        }
    )
    valve_df.write_parquet(tmp_path / "valve.parquet")
    scalup_df = pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1, 0, 0, 5)],
            "ts_scalup": [datetime(2026, 1, 1, 0, 0, 5)],
            "temp_degc": [12.5],
            "sal_psu": [30.0],
            "pressure_mbar": [1010.0],
            "oxygen_mgl": [8.0],
            "ph": [7.8],
        }
    )
    scalup_df.write_parquet(tmp_path / "scalup.parquet")
    status_df = pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1, 0, 0, 5)],
            "turbo_error": [0.0],
            "turbo_speed_hz": [1200.0],
            "turbo_power_w": [50.0],
            "turbo_voltage": [24.0],
            "turbo_etemp_c": [30.0],
            "turbo_btemp_c": [28.0],
            "turbo_mtemp_c": [29.0],
            "rga_filament": [1.0],
            "raw_total_pressure_current": [2000.0],
            "pump_rpm": [8760.0],
            "payload_raw": [None],
        }
    )
    status_df.write_parquet(tmp_path / "status.parquet")

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    assert not at.exception

    tab = at.tabs[2]
    var_select = [s for s in tab.get("selectbox") if s.label == "Variable"][0]
    assert var_select.options == [
        "temp_degC",
        "sal_PSU",
        "pressure_mbar",
        "oxygen_mgL",
        "pH",
        "turbo_speed_hz",
        "turbo_power_w",
        "water_pump_rpm",
        "total_pressure_amps",
        "total_pressure_torr",
    ]

    for variable in ["temp_degC", "turbo_power_w", "total_pressure_amps", "total_pressure_torr"]:
        var_select = [s for s in at.tabs[2].get("selectbox") if s.label == "Variable"][0]
        var_select.set_value(variable).run(timeout=60)
        assert not at.exception, f"selecting {variable} raised: {at.exception}"
    assert [r.label for r in tab.get("radio")] == ["Grain"]


def test_measurements_tab_rga_data_source_control(tmp_path):
    rga_df = pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1, 0, 0, i) for i in range(4)],
            "mass": [2, 40, 2, 40],
            "current": [10.0, 100.0, 20.0, 200.0],
        }
    )
    rga_df.write_parquet(tmp_path / "rga.parquet")
    cycles_df = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1), datetime(2026, 1, 1, 0, 1)],
            "chamber": ["C1", "C2"],
            "experiment_number": [1, 1],
            "elapsed_time": [0.0, 60.0],
            "mass_2_avg": [10.0, 30.0],
            "mass_40_avg": [100.0, 100.0],
        }
    )
    cycles_df.write_parquet(tmp_path / "egcf_chamber_cycles.parquet")

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    assert not at.exception

    tab = at.tabs[1]
    data_source_radio = [r for r in tab.get("radio") if r.label == "RGA data source"][0]
    assert data_source_radio.options == ["Full RGA data", "Chamber cycle averages"]

    data_source_radio.set_value("Chamber cycle averages").run(timeout=60)
    assert not at.exception
