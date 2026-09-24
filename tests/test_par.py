import logging
from datetime import date, datetime

import polars as pl
import pytest

from egcf_processing.discovery import find_par_files
from egcf_processing.par import (
    CALIBRATION_SCHEMA,
    calibrate,
    daily_max_trend,
    daily_par,
    is_odyssey_export,
    read_all_par,
    read_odyssey_file,
    select_calibration,
)

ODYSSEY_HEADER = [
    "﻿Site Name ,ESL-EGCF",
    "Site Number ,11",
    "Logger ,Integrating Light Sensor",
    "Logger Serial Number ,50472",
    "",
    "",
    "Scan No ,Date and Time,       Integrating Light,        ,",
    "        ,        ,RAW VALUE ,CALIBRATED VALUE,",
    "",
]

ODYSSEY_ROWS = [
    "1,18/09/2026 , 16:11:14,2223,2223",
    "2,18/09/2026 , 16:16:14,0,0",
    "garbled",
    "3,18/09/2026 , 16:21:14,3293,3293",
]


def _write_odyssey(path, rows=ODYSSEY_ROWS, serial="50472"):
    header = [line.replace("50472", serial) for line in ODYSSEY_HEADER]
    path.write_bytes("\r\n".join(header + rows).encode("utf-8"))
    return path


def _cals(rows):
    return pl.DataFrame(rows, schema=CALIBRATION_SCHEMA)


def _cal(serial="50472", cal_date=None, interval_s=None, slope=0.4647, intercept=6.4541, sensor=1):
    return {
        "sensor_number": sensor,
        "serial_number": serial,
        "cal_date": cal_date,
        "interval_s": interval_s,
        "slope": slope,
        "intercept": intercept,
        "notes": None,
    }


def test_read_odyssey_file_parses_header_serial_and_dd_mm_dates(tmp_path):
    df = read_odyssey_file(_write_odyssey(tmp_path / "par.CSV"))
    assert df["serial_number"].unique().to_list() == ["50472"]
    assert df["scan_no"].to_list() == [1, 2, 3]
    assert df["par_raw"].to_list() == [2223.0, 0.0, 3293.0]
    assert df["ts"].to_list() == [
        datetime(2026, 9, 18, 16, 11, 14),
        datetime(2026, 9, 18, 16, 16, 14),
        datetime(2026, 9, 18, 16, 21, 14),
    ]


def test_read_odyssey_file_applies_time_offset(tmp_path):
    df = read_odyssey_file(_write_odyssey(tmp_path / "par.CSV"), time_offset_h=4.0)
    assert df["ts"][0] == datetime(2026, 9, 18, 20, 11, 14)


def test_calibrate_formula_and_zero_clamp():
    df = calibrate(pl.DataFrame({"par_raw": [100.0, 0.0]}), slope=0.2266, intercept=-2.6368)
    assert df["par_umol_m2_s"].to_list() == pytest.approx([0.2266 * 100 - 2.6368, 0.0])


def test_select_calibration_latest_dated_on_or_before_first_ts_beats_blank():
    cals = _cals(
        [
            _cal(cal_date=None, slope=1.0),
            _cal(cal_date=date(2026, 6, 1), slope=2.0),
            _cal(cal_date=date(2026, 12, 1), slope=3.0),
            _cal(serial="99999", cal_date=date(2026, 7, 1), slope=4.0),
        ]
    )
    assert select_calibration(cals, "50472", datetime(2026, 9, 18))["slope"] == 2.0
    assert select_calibration(cals, "50472", datetime(2026, 1, 1))["slope"] == 1.0
    assert select_calibration(cals, "12345", datetime(2026, 9, 18)) is None


def test_read_all_par_calibrates_and_warns_when_interval_unchecked(tmp_path, caplog):
    path = _write_odyssey(tmp_path / "par.CSV")
    with caplog.at_level(logging.WARNING):
        df = read_all_par([path], _cals([_cal()]))
    assert df["par_umol_m2_s"].to_list() == pytest.approx([0.4647 * 2223 + 6.4541, 6.4541, 0.4647 * 3293 + 6.4541])
    assert df["sensor_number"].unique().to_list() == [1]
    assert df["interval_s"].unique().to_list() == [300.0]
    assert "no interval_s" in caplog.text


def test_read_all_par_interval_mismatch_leaves_calibrated_null(tmp_path, caplog):
    path = _write_odyssey(tmp_path / "par.CSV")
    with caplog.at_level(logging.WARNING):
        df = read_all_par([path], _cals([_cal(interval_s=900.0)]))
    assert df["par_umol_m2_s"].is_null().all()
    assert df["par_raw"].null_count() == 0
    assert "not rescaled" in caplog.text


def test_read_all_par_matching_interval_calibrates_without_warning(tmp_path, caplog):
    path = _write_odyssey(tmp_path / "par.CSV")
    with caplog.at_level(logging.WARNING):
        df = read_all_par([path], _cals([_cal(interval_s=300.0)]))
    assert df["par_umol_m2_s"].null_count() == 0
    assert not caplog.records


def test_read_all_par_unknown_serial_leaves_calibrated_null(tmp_path, caplog):
    path = _write_odyssey(tmp_path / "par.CSV", serial="11111")
    with caplog.at_level(logging.WARNING):
        df = read_all_par([path], _cals([_cal()]))
    assert df["par_umol_m2_s"].is_null().all()
    assert df["par_raw"].to_list() == [2223.0, 0.0, 3293.0]
    assert "no PAR calibration for serial 11111" in caplog.text


def test_read_all_par_start_end_trim_is_half_open(tmp_path):
    path = _write_odyssey(tmp_path / "par.CSV")
    df = read_all_par(
        [path],
        _cals([_cal()]),
        start=datetime(2026, 9, 18, 16, 16, 14),
        end=datetime(2026, 9, 18, 16, 21, 14),
    )
    assert df["scan_no"].to_list() == [2]


def test_read_all_par_no_files_gives_empty_typed_table():
    df = read_all_par([], _cals([_cal()]))
    assert df.is_empty()
    assert df.schema["par_umol_m2_s"] == pl.Float64
    assert df.schema["cal_date"] == pl.Date


def test_is_odyssey_export_rejects_other_csvs(tmp_path):
    other = tmp_path / "gems_pump_2025-07-12.csv"
    other.write_text("ts,rpm\n2025-07-12T00:00:00Z,8760\n")
    assert is_odyssey_export(_write_odyssey(tmp_path / "par.CSV"))
    assert not is_odyssey_export(other)


def test_find_par_files_by_content_skipping_zero_byte(tmp_path):
    (tmp_path / "PAR").mkdir()
    good = _write_odyssey(tmp_path / "PAR" / "ESL-EGCF_011_001.CSV")
    (tmp_path / "PAR" / "empty.csv").write_text("")
    (tmp_path / "gems_pump_2025-07-12.csv").write_text("ts,rpm\n")
    assert find_par_files(tmp_path) == [good]


def _par_days(days, per_day_values, interval_s=3600.0):
    """Hourly readings: one list of 24 values per day starting 2026-09-19."""
    rows = []
    for d, values in zip(range(days), per_day_values):
        for h, v in enumerate(values):
            rows.append({"ts": datetime(2026, 9, 19 + d, h), "par_umol_m2_s": v, "interval_s": interval_s})
    return pl.DataFrame(rows, schema={"ts": pl.Datetime, "par_umol_m2_s": pl.Float64, "interval_s": pl.Float64})


def test_daily_par_dli_max_and_coverage():
    full = [0.0] * 10 + [500.0, 1000.0, 500.0] + [0.0] * 11
    daily = daily_par(_par_days(2, [full, [200.0] * 6]))
    assert daily["date"].to_list() == [date(2026, 9, 19), date(2026, 9, 20)]
    assert daily["n_readings"].to_list() == [24, 6]
    assert daily["coverage"].to_list() == pytest.approx([1.0, 0.25])
    assert daily["dli_mol_m2_d"].to_list() == pytest.approx([2000 * 3600 / 1e6, 1200 * 3600 / 1e6])
    assert daily["max_par_umol_m2_s"].to_list() == [1000.0, 200.0]


def test_daily_par_uncalibrated_day_is_null_not_zero():
    daily = daily_par(_par_days(1, [[None] * 24]))
    assert daily["dli_mol_m2_d"][0] is None
    assert daily["max_par_umol_m2_s"][0] is None


def test_daily_max_trend_uses_only_full_days():
    days = [[0.0] * 11 + [m] + [0.0] * 12 for m in (1000.0, 900.0, 800.0)] + [[5000.0] * 3]
    trend = daily_max_trend(daily_par(_par_days(4, days)))
    assert trend["n_days"] == 3
    assert trend["slope_umol_m2_s_per_day"] == pytest.approx(-100.0)
    assert trend["pct_per_day"] == pytest.approx(-100.0 / 900.0 * 100)


def test_daily_max_trend_none_with_fewer_than_two_full_days():
    assert daily_max_trend(daily_par(_par_days(2, [[100.0] * 24, [100.0] * 3]))) is None
