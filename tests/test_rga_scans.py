from datetime import datetime, timedelta

import polars as pl

from egcf_processing.rga_scans import attach_chamber_context, rga_scan_windows

RGA_SCHEMA = {"ts": pl.Datetime, "mass": pl.Int64, "current": pl.Float64}


def _ts(seconds):
    return datetime(2026, 1, 1) + timedelta(seconds=seconds)


def test_rga_scan_windows_basic_two_scans():
    # Two clean 3-mass scans; the second (trailing) scan is dropped since it's open-ended.
    rows = [
        {"ts": _ts(0), "mass": 2, "current": 1.0},
        {"ts": _ts(1), "mass": 15, "current": 2.0},
        {"ts": _ts(2), "mass": 16, "current": 3.0},
        {"ts": _ts(10), "mass": 2, "current": 1.5},
        {"ts": _ts(11), "mass": 15, "current": 2.5},
        {"ts": _ts(12), "mass": 16, "current": 3.5},
    ]
    rga = pl.DataFrame(rows, schema=RGA_SCHEMA)
    windows = rga_scan_windows(rga)
    assert windows.height == 1
    assert windows["window_start"][0] == _ts(0)
    assert windows["window_end"][0] == _ts(10)


def test_rga_scan_windows_repeated_mass_mid_scan_closes_scan():
    # Mass 2 repeats before the scan naturally would have -- treated as a new scan start.
    rows = [
        {"ts": _ts(0), "mass": 2, "current": 1.0},
        {"ts": _ts(1), "mass": 15, "current": 2.0},
        {"ts": _ts(2), "mass": 2, "current": 1.1},  # repeat -> closes scan 0, starts scan 1
        {"ts": _ts(3), "mass": 15, "current": 2.1},
        {"ts": _ts(4), "mass": 2, "current": 1.2},  # closes scan 1, starts scan 2 (dropped, open-ended)
    ]
    rga = pl.DataFrame(rows, schema=RGA_SCHEMA)
    windows = rga_scan_windows(rga)
    assert windows.height == 2
    assert windows["window_start"].to_list() == [_ts(0), _ts(2)]
    assert windows["window_end"].to_list() == [_ts(2), _ts(4)]


def test_rga_scan_windows_empty_input():
    rga = pl.DataFrame(schema=RGA_SCHEMA)
    windows = rga_scan_windows(rga)
    assert windows.height == 0
    assert set(windows.columns) == {"window_start", "window_end"}


def test_attach_chamber_context_inside_and_outside_cycle():
    windows = pl.DataFrame(
        {"window_start": [_ts(5), _ts(50)], "window_end": [_ts(15), _ts(60)]},
        schema={"window_start": pl.Datetime, "window_end": pl.Datetime},
    )
    chamber_cycles = pl.DataFrame(
        {
            "window_start": [_ts(0)],
            "window_end": [_ts(20)],
            "chamber": ["C1"],
            "experiment_number": [1],
            "elapsed_time": [timedelta(seconds=0)],
        },
        schema={
            "window_start": pl.Datetime,
            "window_end": pl.Datetime,
            "chamber": pl.Utf8,
            "experiment_number": pl.Int64,
            "elapsed_time": pl.Duration,
        },
    )
    result = attach_chamber_context(windows, chamber_cycles)
    assert result["chamber"].to_list() == ["C1", None]
    assert result["experiment_number"].to_list() == [1, None]


def test_attach_chamber_context_no_cycles():
    windows = pl.DataFrame(
        {"window_start": [_ts(5)], "window_end": [_ts(15)]},
        schema={"window_start": pl.Datetime, "window_end": pl.Datetime},
    )
    empty_cycles = pl.DataFrame(
        schema={
            "window_start": pl.Datetime,
            "window_end": pl.Datetime,
            "chamber": pl.Utf8,
            "experiment_number": pl.Int64,
            "elapsed_time": pl.Duration,
        }
    )
    result = attach_chamber_context(windows, empty_cycles)
    assert result["chamber"].to_list() == [None]
