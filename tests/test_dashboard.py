import json
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from egcf_processing.combine import STATUS_SCHEMA, VALVE_SCHEMA
from egcf_processing.par import PAR_SCHEMA
from egcf_processing.dashboard import (
    active_chamber_spans,
    align_slider_bounds,
    date_range_bounds,
    filter_tables_to_range,
    preset_time_range,
    table_ts_col,
    tables_time_bounds,
    attach_experiment_context,
    chamber_color_map,
    discover_masses,
    experiment_rates,
    experiment_start_times,
    flux_variable_units,
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


def test_chamber_color_map_is_stable_regardless_of_input_order():
    assert chamber_color_map(["C2", "C1"]) == chamber_color_map(["C1", "C2"])
    assert len(set(chamber_color_map(["C1", "C2"]).values())) == 2


def test_flux_variable_units():
    fluxes = pl.DataFrame(
        {
            "variable": ["oxygen", "oxygen", "temp_degC"],
            "output_unit": ["umol m-2 h-1", "umol m-2 h-1", "degC h-1"],
        }
    )
    assert flux_variable_units(fluxes) == {"oxygen": "umol m-2 h-1", "temp_degC": "degC h-1"}


def _write_two_experiments_one_thin(tmp_path):
    """Experiment 1 gets a single cycle (unfittable); experiment 2 gets two.

    experiment_number advances when a (C1, Re) follows a (C2, Fl), so the
    C1 Fl -> C2 Fl pair in the middle closes experiment 1 after just one
    measurement cycle.
    """
    valve_df = pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1, 0, m) for m in (0, 5, 10, 15, 20, 25, 30)],
            "chamber": ["C1", "C1", "C2", "C1", "C1", "C1", "C1"],
            "flush_state": ["Re", "Fl", "Fl", "Re", "Fl", "Re", "Fl"],
        }
    )
    valve_df.write_parquet(tmp_path / "valve.parquet")
    stamps = [datetime(2026, 1, 1, 0, 2, 30), datetime(2026, 1, 1, 0, 17, 30), datetime(2026, 1, 1, 0, 27, 30)]
    pl.DataFrame(
        {
            "ts": stamps,
            "ts_scalup": stamps,
            "temp_degc": [12.0, 12.0, 12.5],
            "sal_psu": [32.0, 32.0, 32.0],
            "pressure_mbar": [1013.0, 1013.0, 1013.0],
            "oxygen_mgl": [8.0, 8.0, 7.0],
            "ph": [8.1, 8.1, 8.0],
        }
    ).write_parquet(tmp_path / "scalup.parquet")


def _flux_over_time_spec(tab):
    for chart in tab.get("plotly_chart"):
        spec = json.loads(chart.proto.spec)
        title = spec.get("layout", {}).get("title") or {}
        if "Flux over time" in (title.get("text") or ""):
            return spec
    return None


def test_flux_over_time_renders_even_when_selected_experiment_has_no_fit(tmp_path):
    # Landing on a thin experiment used to look like "no flux data at all"; the
    # deployment-wide plot must still show the experiments that do have fits.
    _write_two_experiments_one_thin(tmp_path)

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    at.tabs[2].radio(key="experiment_grain").set_value("Cycle averages").run(timeout=60)
    at.sidebar.number_input[2].set_value(4.0).run(timeout=60)
    at.sidebar.number_input[3].set_value(0.06).run(timeout=60)
    assert not at.exception

    tab = at.tabs[2]
    # The selectbox value is the bare experiment number; only its label is formatted.
    assert tab.selectbox(key="experiment_number").value == "1"
    assert any("see the deployment-wide plot below" in i.value for i in tab.info)
    assert not tab.dataframe  # no per-experiment table for the thin experiment
    spec = _flux_over_time_spec(tab)
    assert spec is not None and spec["data"]


def test_flux_variable_selector_filters_the_flux_over_time_subplots(tmp_path):
    _write_two_experiments_one_thin(tmp_path)

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    at.tabs[2].radio(key="experiment_grain").set_value("Cycle averages").run(timeout=60)
    at.sidebar.number_input[2].set_value(4.0).run(timeout=60)
    at.sidebar.number_input[3].set_value(0.06).run(timeout=60)

    selector = at.tabs[2].multiselect(key="flux_variables")
    assert selector.value == selector.options  # defaults to every variable
    all_subplots = len(_flux_over_time_spec(at.tabs[2])["layout"]["annotations"])

    at.tabs[2].multiselect(key="flux_variables").set_value(["oxygen"]).run(timeout=60)
    assert not at.exception
    spec = _flux_over_time_spec(at.tabs[2])
    titles = [a["text"] for a in spec["layout"]["annotations"]]
    assert len(titles) == 1 < all_subplots
    assert titles[0].startswith("oxygen (umol m-2 h-1)")

    at.tabs[2].multiselect(key="flux_variables").set_value([]).run(timeout=60)
    assert not at.exception
    assert _flux_over_time_spec(at.tabs[2]) is None
    assert any("Select at least one variable" in i.value for i in at.tabs[2].info)


def test_experiment_flux_bar_chart_has_one_subplot_per_variable(tmp_path):
    _write_two_experiments_one_thin(tmp_path)

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    at.tabs[2].radio(key="experiment_grain").set_value("Cycle averages").run(timeout=60)
    at.sidebar.number_input[2].set_value(4.0).run(timeout=60)
    at.sidebar.number_input[3].set_value(0.06).run(timeout=60)
    exp2 = [o for o in at.tabs[2].selectbox(key="experiment_number").options if o.startswith("2 ")][0]
    at.tabs[2].selectbox(key="experiment_number").set_value(exp2).run(timeout=60)
    assert not at.exception

    charts = [json.loads(c.proto.spec) for c in at.tabs[2].get("plotly_chart")]
    bars = [s for s in charts if "flux by variable" in ((s.get("layout", {}).get("title") or {}).get("text") or "")]
    assert len(bars) == 1
    spec = bars[0]
    assert {d["type"] for d in spec["data"]} == {"bar"}
    # Each variable gets its own x-axis: units and magnitudes don't share one.
    assert len({d["xaxis"] for d in spec["data"]}) == len(spec["layout"]["annotations"])
    # The chamber legend entry appears exactly once despite repeating per subplot.
    assert sum(1 for d in spec["data"] if d.get("showlegend") and d.get("name") == "C1") == 1
    assert at.tabs[2].dataframe  # exact numbers still available underneath


def test_experiment_tab_flux_table_needs_chamber_geometry(tmp_path):
    # Same two-C1-cycle setup as the fit/rates test, plus scalup oxygen so
    # there's a flux to compute once geometry is supplied.
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
    scalup_df = pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1, 0, 2, 30), datetime(2026, 1, 1, 0, 12, 30)],
            "ts_scalup": [datetime(2026, 1, 1, 0, 2, 30), datetime(2026, 1, 1, 0, 12, 30)],
            "temp_degc": [12.0, 12.0],
            "sal_psu": [32.0, 32.0],
            "pressure_mbar": [1013.0, 1013.0],
            "oxygen_mgl": [8.0, 7.0],
            "ph": [8.1, 8.0],
        }
    )
    scalup_df.write_parquet(tmp_path / "scalup.parquet")

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    at.tabs[2].radio(key="experiment_grain").set_value("Cycle averages").run(timeout=60)
    assert not at.exception
    # Geometry defaults to 0, so the flux table is replaced by a prompt.
    assert not at.tabs[2].dataframe
    assert any("chamber volume" in info.value for info in at.tabs[2].info)

    at.sidebar.number_input[2].set_value(4.0).run(timeout=60)
    at.sidebar.number_input[3].set_value(0.06).run(timeout=60)
    assert not at.exception

    tables = at.tabs[2].dataframe
    assert len(tables) == 1
    flux_table = pl.from_pandas(tables[0].value)
    assert set(flux_table["variable"].to_list()) == {"oxygen", "h_ion", "temp_degC"}
    oxygen = flux_table.filter(pl.col("variable") == "oxygen")
    # 8.0 -> 7.0 mg/L over the 10 min between the two cycles' window starts.
    assert oxygen["slope_native_per_min"][0] == pytest.approx(-0.1)
    assert oxygen["output_value"][0] == pytest.approx(-0.1 * (1000 / 32) * 4.0 / 0.06 * 60)


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


def _write_status(tmp_path):
    pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1, 0, 0, 5), datetime(2026, 1, 1, 0, 0, 15)],
            "turbo_error": [0.0, 0.0],
            "turbo_speed_hz": [1200.0, 1210.0],
            "turbo_power_w": [50.0, 51.0],
            "turbo_voltage": [24.0, 24.0],
            "turbo_etemp_c": [30.0, 30.5],
            "turbo_btemp_c": [28.0, 28.5],
            "turbo_mtemp_c": [29.0, 29.5],
            "rga_filament": [1.0, 1.0],
            "raw_total_pressure_current": [2000.0, 2100.0],
            "pump_rpm": [8760.0, 8760.0],
            "payload_raw": [None, None],
        }
    ).write_parquet(tmp_path / "status.parquet")


def _write_system_health(tmp_path):
    pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1, 0, 0, 3), datetime(2026, 1, 1, 0, 0, 13)],
            "voltage_v": [27.02, 22.81],
            "current_a": [0.033, 0.040],
            "teensy_temp_c": [42.5, 59.8],
        }
    ).write_parquet(tmp_path / "system_health.parquet")


def _status_tab_spec(tmp_path):
    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    assert not at.exception
    return at, at.tabs[0]


def test_status_tab_renders_system_health_when_status_is_empty(tmp_path):
    pl.DataFrame(schema=STATUS_SCHEMA).write_parquet(tmp_path / "status.parquet")
    _write_system_health(tmp_path)

    _at, tab = _status_tab_spec(tmp_path)
    charts = tab.get("plotly_chart")
    assert len(charts) == 1
    assert not tab.get("info")
    spec = json.loads(charts[0].proto.spec)
    titles = [a["text"] for a in spec["layout"]["annotations"]]
    assert titles == ["Supply voltage (V)", "Supply current (A)", "Teensy temperature (degC)"]


def test_status_tab_renders_all_eight_panels_when_both_tables_populated(tmp_path):
    _write_status(tmp_path)
    _write_system_health(tmp_path)

    _at, tab = _status_tab_spec(tmp_path)
    charts = tab.get("plotly_chart")
    assert len(charts) == 1
    spec = json.loads(charts[0].proto.spec)
    titles = [a["text"] for a in spec["layout"]["annotations"]]
    assert titles == [
        "Turbo speed (Hz)",
        "Turbo power (W)",
        "Turbo temperatures (degC)",
        "Water pump (RPM)",
        "Total pressure (Torr)",
        "Supply voltage (V)",
        "Supply current (A)",
        "Teensy temperature (degC)",
    ]


def test_status_tab_empty_state_when_both_tables_missing(tmp_path):
    _at, tab = _status_tab_spec(tmp_path)
    assert not tab.get("plotly_chart")
    assert "No status data" in tab.get("info")[0].value


def _write_valve_two_chambers(tmp_path):
    pl.DataFrame(
        {
            "ts": [
                datetime(2026, 1, 1, 0, 0, 0),
                datetime(2026, 1, 1, 0, 0, 10),
                datetime(2026, 1, 1, 0, 0, 20),
                datetime(2026, 1, 1, 0, 0, 30),
            ],
            "chamber": ["C1", "C2", "C2", "C1"],
            "flush_state": ["Re", "Re", "Fl", "Re"],
        }
    ).write_parquet(tmp_path / "valve.parquet")


def test_active_chamber_spans_from_valve_transitions():
    valve = pl.DataFrame(
        {
            "ts": [
                datetime(2026, 1, 1, 0, 0, 0),
                datetime(2026, 1, 1, 0, 0, 10),
                datetime(2026, 1, 1, 0, 0, 20),
                datetime(2026, 1, 1, 0, 0, 30),
            ],
            "chamber": ["C1", "C2", "C2", "C1"],
            "flush_state": ["Re", "Re", "Fl", "Re"],
        }
    )
    spans = active_chamber_spans(valve)
    # The trailing (C1, Re) has no following transition, so it is not a span;
    # the (C2, Fl) flush span is skipped.
    assert spans.select("start", "end", "chamber").rows() == [
        (datetime(2026, 1, 1, 0, 0, 0), datetime(2026, 1, 1, 0, 0, 10), "C1"),
        (datetime(2026, 1, 1, 0, 0, 10), datetime(2026, 1, 1, 0, 0, 20), "C2"),
    ]


def test_active_chamber_spans_empty_without_valve_data():
    assert active_chamber_spans(None).is_empty()
    assert active_chamber_spans(pl.DataFrame(schema=VALVE_SCHEMA)).is_empty()
    assert active_chamber_spans(None).columns == ["start", "end", "chamber"]


def test_status_tab_chamber_shading_toggle_draws_bands(tmp_path):
    _write_status(tmp_path)
    _write_valve_two_chambers(tmp_path)

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    assert not at.exception

    shade = [c for c in at.tabs[0].get("checkbox") if c.label == "Shade by active chamber"][0]
    assert shade.value is False
    assert not json.loads(at.tabs[0].get("plotly_chart")[0].proto.spec)["layout"].get("shapes")

    shade.set_value(True).run(timeout=60)
    assert not at.exception
    spec = json.loads(at.tabs[0].get("plotly_chart")[0].proto.spec)
    shapes = spec["layout"]["shapes"]
    assert len(shapes) == 2
    assert {s["fillcolor"] for s in shapes} == set(chamber_color_map(["C1", "C2"]).values())
    # One band spanning the whole stacked figure, behind the traces.
    assert all(s["yref"] == "paper" and s["layer"] == "below" for s in shapes)
    assert [d["name"] for d in spec["data"] if d["name"].endswith("active")] == ["C1 active", "C2 active"]


def test_measurements_tab_chamber_shading_toggle_draws_bands(tmp_path):
    pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1, 0, 0, i) for i in range(4)],
            "mass": [2, 40, 2, 40],
            "current": [10.0, 100.0, 20.0, 200.0],
        }
    ).write_parquet(tmp_path / "rga.parquet")
    _write_valve_two_chambers(tmp_path)

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    assert not at.exception

    shade = [c for c in at.tabs[1].get("checkbox") if c.label == "Shade by active chamber"][0]
    shade.set_value(True).run(timeout=60)
    assert not at.exception
    spec = json.loads(at.tabs[1].get("plotly_chart")[0].proto.spec)
    assert len(spec["layout"]["shapes"]) == 2


def test_chamber_shading_toggle_absent_without_valve_data(tmp_path):
    _write_status(tmp_path)

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    assert not at.exception
    assert not [c for c in at.tabs[0].get("checkbox") if c.label == "Shade by active chamber"]
    assert not [c for c in at.tabs[1].get("checkbox") if c.label == "Shade by active chamber"]


def test_table_ts_col_prefers_timestamp_over_ts():
    assert table_ts_col(pl.DataFrame({"ts": [1]})) == "ts"
    assert table_ts_col(pl.DataFrame({"timestamp": [1], "ts": [1]})) == "timestamp"
    assert table_ts_col(pl.DataFrame({"mass": [1]})) is None


def test_tables_time_bounds_spans_every_timestamped_table():
    tables = {
        "rga": pl.DataFrame({"ts": [datetime(2026, 1, 1, 0, 5), datetime(2026, 1, 1, 0, 9)]}),
        "egcf_chamber_cycles": pl.DataFrame({"timestamp": [datetime(2026, 1, 1, 0, 1)]}),
        "status": None,
        "valve": pl.DataFrame(schema={"ts": pl.Datetime}),
    }
    assert tables_time_bounds(tables) == (datetime(2026, 1, 1, 0, 1), datetime(2026, 1, 1, 0, 9))


def test_tables_time_bounds_none_without_any_timestamps():
    assert tables_time_bounds({"status": None, "rga": pl.DataFrame(schema={"ts": pl.Datetime})}) is None


def test_filter_tables_to_range_is_inclusive_and_leaves_untimestamped_tables_alone():
    tables = {
        "rga": pl.DataFrame({"ts": [datetime(2026, 1, 1, 0, m) for m in range(5)], "mass": list(range(5))}),
        "egcf_chamber_cycles": pl.DataFrame({"timestamp": [datetime(2026, 1, 1, 0, m) for m in range(5)]}),
        "status": None,
        "other": pl.DataFrame({"mass": [1, 2]}),
    }
    filtered = filter_tables_to_range(tables, datetime(2026, 1, 1, 0, 1), datetime(2026, 1, 1, 0, 3))
    assert filtered["rga"]["mass"].to_list() == [1, 2, 3]
    assert filtered["egcf_chamber_cycles"].height == 3
    assert filtered["status"] is None
    assert filtered["other"].height == 2


def test_preset_time_range_anchors_at_the_end_of_the_data():
    lo, hi = datetime(2026, 8, 13, 18, 23, 12), datetime(2026, 9, 21, 17, 55, 36)
    assert preset_time_range("All data", lo, hi) == (lo, hi)
    assert preset_time_range("Last 24 hours", lo, hi) == (datetime(2026, 9, 20, 17, 55, 36), hi)
    # A window longer than the dataset clamps to the dataset, it does not run off the front.
    assert preset_time_range("Last 7 days", lo, datetime(2026, 8, 14)) == (lo, datetime(2026, 8, 14))


def test_date_range_bounds_widens_to_whole_days_and_clamps():
    lo, hi = datetime(2026, 9, 20, 6, 30), datetime(2026, 9, 21, 17, 55, 36)
    assert date_range_bounds((date(2026, 9, 20), date(2026, 9, 21)), lo, hi) == (lo, hi)
    assert date_range_bounds((date(2026, 9, 21),), lo, hi) == (datetime(2026, 9, 21), hi)
    assert date_range_bounds(date(2026, 9, 21), lo, hi) == (datetime(2026, 9, 21), hi)


def test_align_slider_bounds_makes_the_upper_bound_reachable():
    step = timedelta(minutes=1)
    lo, hi = datetime(2026, 9, 21), datetime(2026, 9, 21, 17, 55, 36)
    aligned_lo, aligned_hi = align_slider_bounds(lo, hi, step)
    assert aligned_lo == lo
    assert aligned_hi == datetime(2026, 9, 21, 17, 56)
    # Reachable means an exact whole number of steps from the lower bound, and
    # never short of the real maximum.
    assert (aligned_hi - aligned_lo) % step == timedelta(0)
    assert aligned_hi >= hi


def test_align_slider_bounds_leaves_an_exact_multiple_alone():
    step = timedelta(minutes=1)
    lo, hi = datetime(2026, 9, 21), datetime(2026, 9, 21, 0, 30)
    assert align_slider_bounds(lo, hi, step) == (lo, hi)


def test_align_slider_bounds_keeps_at_least_one_step():
    step = timedelta(minutes=1)
    lo = datetime(2026, 9, 21)
    assert align_slider_bounds(lo, lo, step) == (lo, lo + step)


def _write_system_health_over(tmp_path, stamps):
    pl.DataFrame(
        {
            "ts": stamps,
            "voltage_v": [24.0] * len(stamps),
            "current_a": [0.03] * len(stamps),
            "teensy_temp_c": [50.0] * len(stamps),
        }
    ).write_parquet(tmp_path / "system_health.parquet")


def _status_points(at):
    spec = json.loads(at.tabs[0].get("plotly_chart")[0].proto.spec)
    return spec["data"][0]["x"]


def test_sidebar_time_range_preset_narrows_the_plotted_data(tmp_path):
    _write_system_health_over(tmp_path, [datetime(2026, 1, 1, h) for h in range(6)])

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    assert not at.exception

    preset = at.sidebar.selectbox[0]
    assert preset.label == "Preset"
    assert preset.value == "All data"
    assert len(_status_points(at)) == 6

    preset.set_value("Last hour").run(timeout=60)
    assert not at.exception
    assert len(_status_points(at)) == 2

    # The overview still describes the whole dataset, not the visible slice.
    assert any("6 rows" in m.value for m in at.get("markdown"))


def test_sidebar_custom_time_range_can_reach_the_final_partial_day(tmp_path):
    # Two days of data ending at an awkward 17:55:36 -- with a whole-day slider
    # step this tail was unselectable.
    stamps = [datetime(2026, 9, 20, 12), datetime(2026, 9, 21, 0, 30), datetime(2026, 9, 21, 17, 55, 36)]
    _write_system_health_over(tmp_path, stamps)

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    at.sidebar.selectbox[0].set_value("Custom").run(timeout=60)
    assert not at.exception

    at.sidebar.date_input[0].set_value((date(2026, 9, 21), date(2026, 9, 21))).run(timeout=60)
    assert not at.exception

    fine = at.sidebar.slider[0]
    assert fine.value == (datetime(2026, 9, 21), datetime(2026, 9, 21, 17, 56))
    points = _status_points(at)
    assert len(points) == 2
    assert points[-1].startswith("2026-09-21T17:55:36")

    fine.set_range(datetime(2026, 9, 21), datetime(2026, 9, 21, 1)).run(timeout=60)
    assert not at.exception
    assert len(_status_points(at)) == 1


def test_sidebar_time_range_filter_absent_for_a_single_instant_dataset(tmp_path):
    _write_status(tmp_path)
    pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1, 0, 0, 5)],
            "voltage_v": [24.0],
            "current_a": [0.03],
            "teensy_temp_c": [50.0],
        }
    ).write_parquet(tmp_path / "system_health.parquet")
    pl.read_parquet(tmp_path / "status.parquet").head(1).write_parquet(tmp_path / "status.parquet")

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    assert not at.exception
    assert not at.sidebar.selectbox
    assert not at.sidebar.slider


def _write_par(tmp_path, calibrated=True):
    pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1, 0, 0, 1), datetime(2026, 1, 1, 0, 0, 11)],
            "scan_no": [1, 2],
            "par_raw": [100.0, 300.0],
            "par_umol_m2_s": [52.9, 145.9] if calibrated else [None, None],
            "serial_number": ["50472", "50472"],
            "sensor_number": [1, 1],
            "cal_date": [None, None],
            "interval_s": [300.0, 300.0],
        },
        schema=PAR_SCHEMA,
    ).write_parquet(tmp_path / "par.parquet")


def _measurements_subplot_titles(tmp_path):
    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=60)
    assert not at.exception
    spec = json.loads(at.tabs[1].get("plotly_chart")[0].proto.spec)
    return at, spec, [a["text"] for a in spec["layout"]["annotations"]]


def test_measurements_tab_renders_par_panel_on_shared_axis(tmp_path):
    pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1, 0, 0, i) for i in range(4)],
            "mass": [2, 40, 2, 40],
            "current": [10.0, 100.0, 20.0, 200.0],
        }
    ).write_parquet(tmp_path / "rga.parquet")
    _write_par(tmp_path)

    _at, spec, titles = _measurements_subplot_titles(tmp_path)
    assert titles[-1] == "PAR (µmol photons m⁻² s⁻¹)"
    par_trace = [d for d in spec["data"] if d["name"] == "par_umol_m2_s"][0]
    assert spec["layout"][par_trace["xaxis"].replace("x", "xaxis")]["matches"] == "x"


def test_measurements_tab_falls_back_to_raw_par_when_uncalibrated(tmp_path):
    _write_par(tmp_path, calibrated=False)

    at, _spec, titles = _measurements_subplot_titles(tmp_path)
    assert titles == ["PAR (raw counts, uncalibrated)"]
    assert any("No PAR calibration matched" in i.value for i in at.tabs[1].info)
