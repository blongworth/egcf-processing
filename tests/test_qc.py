from datetime import datetime, timedelta

import polars as pl
import pytest

from egcf_processing.qc import (
    RANGE_LIMITS,
    completeness,
    coverage_segments,
    find_gaps,
    flagged_rows,
    range_flags,
    range_summary,
)

T0 = datetime(2026, 1, 1)


def _regular(n: int, step_s: float, skip: range = range(0), offset_s: float = 0.0) -> list[datetime]:
    return [T0 + timedelta(seconds=offset_s + i * step_s) for i in range(n) if i not in skip]


def _status(ts: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame({"ts": ts, "turbo_error": [0.0] * len(ts)})


def test_find_gaps_detects_injected_gap():
    # 10 s cadence, records 20..29 removed -> one 110 s gap between t=190 and t=300.
    df = _status(_regular(50, 10, skip=range(20, 30)))
    gaps = find_gaps(df, "ts", None, 60.0)
    assert gaps.rows() == [(None, T0 + timedelta(seconds=190), T0 + timedelta(seconds=300), 110.0)]


def test_find_gaps_counts_leading_and_trailing_gaps():
    df = _status(_regular(10, 10, offset_s=300))
    start, end = T0, T0 + timedelta(seconds=1000)
    gaps = find_gaps(df, "ts", None, 60.0, start, end)
    assert gaps.select("gap_start", "gap_end", "gap_s").rows() == [
        (start, T0 + timedelta(seconds=300), 300.0),
        (T0 + timedelta(seconds=390), end, 610.0),
    ]


def test_find_gaps_ignores_records_outside_range():
    df = _status([T0 - timedelta(days=1), *_regular(10, 10)])
    assert find_gaps(df, "ts", None, 60.0, T0, T0 + timedelta(seconds=90)).is_empty()


def test_find_gaps_whole_range_when_no_records():
    gaps = find_gaps(_status([]), "ts", None, 60.0, T0, T0 + timedelta(hours=1))
    assert gaps.rows() == [(None, T0, T0 + timedelta(hours=1), 3600.0)]
    assert find_gaps(_status([]), "ts", None, 60.0).is_empty()


def test_completeness_scores_regular_series_with_gap():
    ts = _regular(100, 8, skip=range(40, 60))
    end = T0 + timedelta(seconds=800)
    scores = completeness({"status": _status(ts)}, start=T0, end=end)
    row = scores.row(0, named=True)
    assert row["stream"] == "status"
    assert row["expected"] == 100.0
    assert row["observed"] == 80
    assert row["completeness"] == pytest.approx(0.8)
    assert row["n_gaps"] == 1
    assert row["longest_gap_s"] == 168.0
    assert row["last_record"] == T0 + timedelta(seconds=792)
    assert row["staleness_s"] == 8.0


def test_completeness_counts_duplicate_timestamps_once_and_caps_at_one():
    ts = _regular(10, 8)
    scores = completeness({"status": _status(ts + ts)}, start=T0, end=T0 + timedelta(seconds=72))
    assert scores["observed"].item() == 10
    assert scores["completeness"].item() == 1.0


def test_completeness_scores_grouped_stream_per_group():
    ts = _regular(100, 8)
    rga = pl.concat(
        [
            pl.DataFrame({"ts": ts, "mass": [2] * 100, "current": [1.0] * 100}),
            pl.DataFrame({"ts": ts[:50], "mass": [40] * 50, "current": [1.0] * 50}),
        ]
    )
    end = T0 + timedelta(seconds=800)
    scores = completeness({"rga": rga}, start=T0, end=end)
    by_stream = {r["stream"]: r for r in scores.iter_rows(named=True)}
    assert set(by_stream) == {"rga 2", "rga 40"}
    assert by_stream["rga 2"]["n_gaps"] == 0
    assert by_stream["rga 40"]["n_gaps"] == 1
    assert by_stream["rga 40"]["completeness"] == pytest.approx(0.5)
    assert by_stream["rga 40"]["staleness_s"] == 800.0 - 392.0


def test_completeness_uses_par_logging_interval():
    par = pl.DataFrame({"ts": _regular(4, 600), "par_umol_m2_s": [1.0] * 4, "interval_s": [600.0] * 4})
    scores = completeness({"par": par}, start=T0, end=T0 + timedelta(seconds=2400))
    assert scores["nominal_interval_s"].item() == 600.0
    assert scores["completeness"].item() == 1.0


def test_completeness_handles_none_and_empty_tables():
    tables = {"status": None, "scalup": pl.DataFrame(schema={"ts": pl.Datetime, "ph": pl.Float64})}
    scores = completeness(tables, start=T0, end=T0 + timedelta(hours=1))
    assert scores["stream"].to_list() == ["scalup"]
    row = scores.row(0, named=True)
    assert row["observed"] == 0
    assert row["completeness"] == 0.0
    assert row["n_gaps"] == 1
    assert row["last_record"] is None
    assert completeness({}).is_empty()


def test_coverage_segments_are_complement_of_gaps():
    df = _status(_regular(50, 10, skip=range(20, 30), offset_s=100))
    end = T0 + timedelta(seconds=1000)
    segments = coverage_segments({"status": df}, start=T0, end=end)
    assert segments.rows() == [
        ("status", T0 + timedelta(seconds=100), T0 + timedelta(seconds=290)),
        ("status", T0 + timedelta(seconds=400), T0 + timedelta(seconds=590)),
    ]


def test_coverage_segments_skip_empty_streams():
    assert coverage_segments({"status": _status([])}, start=T0, end=T0 + timedelta(hours=1)).is_empty()


def _flags(values, limits):
    return pl.DataFrame({"v": values}, schema={"v": pl.Float64}).select(range_flags("v", limits).alias("flag"))["flag"].to_list()


def test_range_flags_boundaries_are_inclusive():
    limits = (0.0, 10.0, 2.0, 8.0)
    assert _flags([-0.1, 0.0, 1.9, 2.0, 5.0, 8.0, 8.1, 10.0, 10.1], limits) == [
        "fail", "suspect", "suspect", "pass", "pass", "pass", "suspect", "suspect", "fail",
    ]


def test_range_flags_null_and_nan_are_missing():
    assert _flags([None, float("nan"), 5.0], (0.0, 10.0, None, None)) == ["missing", "missing", "pass"]


def test_range_flags_open_bounds():
    assert _flags([-1e9, 1e9], (None, None, None, None)) == ["pass", "pass"]


def test_ph_zero_fails_but_small_positive_is_suspect():
    assert _flags([0.0, 0.01, 7.9, 14.1], RANGE_LIMITS[("scalup", "ph")]) == ["fail", "suspect", "pass", "fail"]


def test_turbo_error_nonzero_fails():
    assert _flags([0.0, 1.0, -1.0], RANGE_LIMITS[("status", "turbo_error")]) == ["pass", "fail", "fail"]


def test_range_summary_and_flagged_rows():
    ts = _regular(4, 10)
    scalup = pl.DataFrame({"ts": ts, "ph": [7.9, 0.0, 6.5, None]})
    status = pl.DataFrame({"ts": ts, "turbo_error": [0.0, 0.0, 3.0, 0.0]})
    tables = {"scalup": scalup, "status": status, "rga": None}
    config = {
        ("scalup", "ph"): RANGE_LIMITS[("scalup", "ph")],
        ("status", "turbo_error"): (0.0, 0.0, None, None),
        ("rga", "current"): RANGE_LIMITS[("rga", "current")],
        ("scalup", "absent"): (0.0, 1.0, None, None),
    }

    summary = range_summary(tables, config)
    assert summary.select("table", "column").rows() == [("scalup", "ph"), ("status", "turbo_error")]
    ph = summary.row(0, named=True)
    assert (ph["n"], ph["n_null"], ph["n_suspect"], ph["n_fail"]) == (4, 1, 1, 1)
    assert ph["pct_flagged"] == 50.0
    assert (ph["min"], ph["max"]) == (0.0, 7.9)
    assert (ph["first_flag_ts"], ph["last_flag_ts"]) == (ts[1], ts[2])

    rows = flagged_rows(tables, config)
    assert rows.rows() == [
        ("scalup", "ph", ts[1], 0.0, "fail"),
        ("scalup", "ph", ts[2], 6.5, "suspect"),
        ("status", "turbo_error", ts[2], 3.0, "fail"),
    ]


def test_range_checks_handle_no_tables():
    assert range_summary({}).is_empty()
    assert flagged_rows({}).is_empty()
    assert flagged_rows({}).columns == ["table", "column", "ts", "value", "flag"]
