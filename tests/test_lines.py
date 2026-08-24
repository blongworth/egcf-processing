from datetime import datetime

from egcf_processing.lines import parse_line


def test_parse_r():
    rec = parse_line("R:2026-06-02T14:30:00Z,28,12345")
    assert rec == {"tag": "R", "ts": datetime(2026, 6, 2, 14, 30, 0), "mass": 28, "current": 12345.0}


def test_parse_r_malformed():
    assert parse_line("R:2026-06-02T14:30:00Z,28") is None
    assert parse_line("R:not-a-timestamp,28,12345") is None


def test_parse_v_new_format():
    rec = parse_line("V:2026-08-10T19:52:34Z,C1,Fl")
    assert rec == {
        "tag": "V",
        "ts": datetime(2026, 8, 10, 19, 52, 34),
        "chamber": "C1",
        "flush_state": "Fl",
    }


def test_parse_v_unknown_flush_state_is_valid():
    rec = parse_line("V:2026-08-10T19:52:03Z,C1,Unknown")
    assert rec["flush_state"] == "Unknown"


def test_parse_v_old_format_rejected():
    old = "V:2026-07-06T18:36:32Z,FLUSH_RECIRCULATE,CHAMBER=Unknown,FLUSH=Moving to B"
    assert parse_line(old) is None


def test_parse_v_invalid_chamber_rejected():
    assert parse_line("V:2026-08-10T19:52:34Z,C3,Fl") is None


def test_parse_p_7_field():
    rec = parse_line("P:2026-08-11T21:18:03Z,2026-08-11T21:18:26,20.310,21.430,1020.630,6.720,7.970")
    assert rec["pressure_mbar"] == 1020.630
    assert rec["temp_degc"] == 20.310
    assert rec["ph"] == 7.970


def test_parse_p_6_field_missing_pressure():
    rec = parse_line("P:2019-01-01T00:00:18Z,2019-01-01T00:00:18,23.010,0.000,9.280,9.050")
    assert rec["pressure_mbar"] is None
    assert rec["temp_degc"] == 23.010
    assert rec["oxygen_mgl"] == 9.280
    assert rec["ph"] == 9.050


def test_parse_p_nan_field():
    rec = parse_line("P:2019-01-01T00:00:17Z,2026-07-14T15:25:35,23.830,0.000,9.080,nan")
    assert rec["oxygen_mgl"] == 9.080
    assert rec["ph"] is None


def test_parse_status_detailed():
    line = "!:2026-06-02T14:30:00Z,0,1200,50,24,30,28,29,1,1591,8760"
    rec = parse_line(line)
    assert rec["turbo_error"] == 0.0
    assert rec["turbo_speed_hz"] == 1200.0
    assert rec["turbo_power_w"] == 50.0
    assert rec["raw_total_pressure_current"] == 1591.0
    assert rec["pump_rpm"] == 8760.0
    assert rec["payload_raw"] is None


def test_parse_status_na_total_pressure():
    line = "!:2026-06-02T14:30:00Z,0,1200,50,24,30,28,29,1,NA,8760"
    rec = parse_line(line)
    assert rec["raw_total_pressure_current"] is None


def test_parse_status_generic_fallback():
    rec = parse_line("!:2026-06-02T14:30:00Z,5")
    assert rec["tag"] == "!"
    assert rec["payload_raw"] == "5"
    assert rec["turbo_speed_hz"] is None


def test_parse_unrecognized_tags_return_none():
    assert parse_line("PM:2026-07-21T15:53:35Z,0.0,0.0") is None
    assert parse_line("TS,ERR=0,SPD=1200,PWR=5,V=2357") is None
    assert parse_line("CFG,RGA_MASSES=2,15,16") is None
    assert parse_line("") is None
    assert parse_line("garbage line") is None
