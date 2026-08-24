from datetime import datetime, timedelta

import polars as pl

from egcf_processing.cycles import chamber_cycle_windows, dedupe_consecutive

VALVE_SCHEMA = {"ts": pl.Datetime, "chamber": pl.Utf8, "flush_state": pl.Utf8}


def _ts(seconds):
    return datetime(2026, 1, 1) + timedelta(seconds=seconds)


def _valve(rows):
    return pl.DataFrame(
        [{"ts": _ts(t), "chamber": c, "flush_state": f} for t, c, f in rows], schema=VALVE_SCHEMA
    )


def test_dedupe_consecutive_removes_exact_repeat():
    valve = pl.DataFrame(
        [
            {"ts": _ts(0), "chamber": "C1", "flush_state": "Fl"},
            {"ts": _ts(0), "chamber": "C1", "flush_state": "Fl"},
            {"ts": _ts(30), "chamber": "C2", "flush_state": "Fl"},
        ],
        schema=VALVE_SCHEMA,
    )
    result = dedupe_consecutive(valve)
    assert result.height == 2


def test_full_documented_experiment_sequence():
    valve = _valve(
        [
            (0, "C1", "Re"),
            (30, "C2", "Re"),
            (60, "C1", "Re"),
            (90, "C2", "Re"),
            (120, "C1", "Fl"),
            (150, "C2", "Fl"),
            (180, "C1", "Re"),
            (210, "C2", "Fl"),
        ]
    )
    windows, stats = chamber_cycle_windows(valve, settle_offset_s=0)
    assert windows.height == 5
    assert windows["experiment_number"].to_list() == [1, 1, 1, 1, 2]
    assert windows["elapsed_time"].to_list() == [
        timedelta(seconds=s) for s in (0, 30, 60, 90, 0)
    ]
    assert windows["chamber"].to_list() == ["C1", "C2", "C1", "C2", "C1"]
    assert stats["dropped_too_short"] == 0


def test_mid_cycle_start_first_re_is_c2():
    valve = _valve([(0, "C2", "Re"), (30, "C1", "Re")])
    windows, _ = chamber_cycle_windows(valve, settle_offset_s=0)
    assert windows.height == 1
    assert windows["experiment_number"].to_list() == [1]
    assert windows["chamber"].to_list() == ["C2"]


def test_too_short_cycle_dropped():
    valve = _valve([(0, "C1", "Re"), (30, "C2", "Re")])
    windows, stats = chamber_cycle_windows(valve, settle_offset_s=60)
    assert windows.height == 0
    assert stats["total_cycles"] == 1
    assert stats["dropped_too_short"] == 1


def test_clock_jump_produces_no_valid_cycle():
    valve = _valve([(100, "C1", "Re"), (50, "C2", "Re")])
    windows, stats = chamber_cycle_windows(valve, settle_offset_s=0)
    assert windows.height == 0
    assert stats["dropped_too_short"] == 1


def test_trailing_open_cycle_dropped():
    valve = _valve([(0, "C1", "Fl"), (30, "C2", "Fl"), (60, "C1", "Re")])
    windows, _ = chamber_cycle_windows(valve, settle_offset_s=0)
    # C1,Re at t=60 has no following transition -> open-ended, dropped.
    assert windows.height == 0


def test_empty_valve_returns_empty_windows():
    windows, stats = chamber_cycle_windows(pl.DataFrame(schema=VALVE_SCHEMA), settle_offset_s=60)
    assert windows.height == 0
    assert stats == {"total_cycles": 0, "dropped_too_short": 0}
