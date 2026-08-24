from datetime import datetime, timedelta

import polars as pl

from egcf_processing.aggregate import aggregate_onto_windows
from egcf_processing.combine import RGA_SCHEMA, SCALUP_SCHEMA, STATUS_SCHEMA

WINDOWS_SCHEMA = {
    "window_start": pl.Datetime,
    "window_end": pl.Datetime,
    "chamber": pl.Utf8,
    "experiment_number": pl.Int64,
    "elapsed_time": pl.Duration,
}


def _ts(seconds):
    return datetime(2026, 1, 1) + timedelta(seconds=seconds)


def _windows(rows):
    return pl.DataFrame(
        [
            {
                "window_start": _ts(start),
                "window_end": _ts(end),
                "chamber": chamber,
                "experiment_number": exp,
                "elapsed_time": timedelta(seconds=elapsed),
            }
            for start, end, chamber, exp, elapsed in rows
        ],
        schema=WINDOWS_SCHEMA,
    )


def _empty(schema):
    return pl.DataFrame(schema=schema)


def test_half_open_window_boundary():
    windows = _windows([(10, 20, "C1", 1, 0)])
    rga = pl.DataFrame(
        [
            {"ts": _ts(10), "mass": 2, "current": 1.0},  # at window_start: included
            {"ts": _ts(19), "mass": 2, "current": 3.0},  # inside: included
            {"ts": _ts(20), "mass": 2, "current": 100.0},  # at window_end: excluded
            {"ts": _ts(5), "mass": 2, "current": -100.0},  # before window: excluded
        ],
        schema=RGA_SCHEMA,
    )
    result = aggregate_onto_windows(windows, rga, _empty(SCALUP_SCHEMA), _empty(STATUS_SCHEMA))
    assert result["mass_2_avg"][0] == 2.0  # mean of 1.0 and 3.0 only


def test_mass_pivot_missing_mass_is_null_not_dropped():
    windows = _windows([(0, 10, "C1", 1, 0), (10, 20, "C2", 1, 10)])
    rga = pl.DataFrame(
        [
            {"ts": _ts(1), "mass": 2, "current": 1.0},
            {"ts": _ts(2), "mass": 15, "current": 2.0},
            {"ts": _ts(11), "mass": 2, "current": 5.0},
            # mass 15 has no reading in the second window
        ],
        schema=RGA_SCHEMA,
    )
    result = aggregate_onto_windows(windows, rga, _empty(SCALUP_SCHEMA), _empty(STATUS_SCHEMA))
    assert result.height == 2
    assert result["mass_2_avg"].to_list() == [1.0, 5.0]
    assert result["mass_15_avg"].to_list() == [2.0, None]


def test_status_fields_and_total_pressure_conversion():
    windows = _windows([(0, 100, "C1", 1, 0)])
    status = pl.DataFrame(
        [
            {
                "ts": _ts(10),
                "turbo_error": 0.0,
                "turbo_speed_hz": 1200.0,
                "turbo_power_w": 50.0,
                "turbo_voltage": 24.0,
                "turbo_etemp_c": 30.0,
                "turbo_btemp_c": 28.0,
                "turbo_mtemp_c": 29.0,
                "rga_filament": 1.0,
                "raw_total_pressure_current": 2000.0,
                "pump_rpm": 8760.0,
                "payload_raw": None,
            }
        ],
        schema=STATUS_SCHEMA,
    )
    result = aggregate_onto_windows(windows, _empty(RGA_SCHEMA), _empty(SCALUP_SCHEMA), status)
    assert result["turbo_speed_hz"][0] == 1200.0
    assert result["turbo_power_w"][0] == 50.0
    assert result["water_pump_rpm"][0] == 8760.0
    assert result["total_pressure_amps"][0] == 2000.0 * 1e-16


def test_all_empty_sources_produce_null_schema_present_columns():
    windows = _windows([(0, 100, "C1", 1, 0)])
    result = aggregate_onto_windows(windows, _empty(RGA_SCHEMA), _empty(SCALUP_SCHEMA), _empty(STATUS_SCHEMA))
    assert result.height == 1
    for col in ["total_pressure_amps", "temp_degC", "turbo_speed_hz", "turbo_power_w", "water_pump_rpm"]:
        assert col in result.columns
        assert result[col][0] is None


def test_same_raw_data_different_window_grains():
    scalup = pl.DataFrame(
        [
            {
                "ts": _ts(t),
                "ts_scalup": _ts(t),
                "temp_degc": temp,
                "sal_psu": 30.0,
                "pressure_mbar": 1010.0,
                "oxygen_mgl": 8.0,
                "ph": 7.5,
            }
            for t, temp in [(5, 10.0), (15, 20.0), (25, 30.0)]
        ],
        schema=SCALUP_SCHEMA,
    )
    fine_windows = _windows([(0, 10, "C1", 1, 0), (10, 20, "C1", 1, 10), (20, 30, "C1", 1, 20)])
    coarse_windows = _windows([(0, 30, "C1", 1, 0)])

    fine = aggregate_onto_windows(fine_windows, _empty(RGA_SCHEMA), scalup, _empty(STATUS_SCHEMA))
    coarse = aggregate_onto_windows(coarse_windows, _empty(RGA_SCHEMA), scalup, _empty(STATUS_SCHEMA))

    assert fine["temp_degC"].to_list() == [10.0, 20.0, 30.0]
    assert coarse["temp_degC"].to_list() == [20.0]  # mean of all three
