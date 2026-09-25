from datetime import datetime

from egcf_processing.hobo import is_hobo_export, read_all_hobo, read_hobo_file

C1_HEADER = (
    '﻿"Plot Title: C1"\r\n'
    '"#","Date Time, GMT-04:00","DO conc, mg/L (LGR S/N: 20113801, SEN S/N: 20113801, LBL: DO)",'
    '"Temp, °F (LGR S/N: 20113801, SEN S/N: 20113801, LBL: temp)",'
    '"Sensor Data Error (LGR S/N: 20113801)","Coupler Attached (LGR S/N: 20113801)",'
    '"Host Connected (LGR S/N: 20113801)","Stopped (LGR S/N: 20113801)","End Of File (LGR S/N: 20113801)"\r\n'
)
C1_ROWS = [
    "1,09/23/26 11:00:00 AM,8.73,73.58,,,,,",
    "2,09/23/26 11:01:00 AM,8.72,73.58,,,,,",
    "3,09/23/26 11:02:00 AM,-888.88,-888.88,Logged,,,,",
    "4,09/23/26 11:02:40 AM,,,,Logged,,,",
]

MESOCOSM_HEADER = (
    '﻿"Plot Title: mesocosm"\r\n'
    '"#","Date Time, GMT-04:00","DO conc, mg/L (LGR S/N: 20601917, SEN S/N: 20601917)",'
    '"Temp, °F (LGR S/N: 20601917, SEN S/N: 20601917)",'
    '"Coupler Attached (LGR S/N: 20601917)","Host Connected (LGR S/N: 20601917)",'
    '"Stopped (LGR S/N: 20601917)","End Of File (LGR S/N: 20601917)"\r\n'
)
MESOCOSM_ROWS = [
    "1,09/23/26 11:00:00 AM,8.83,73.00,,,,",
    "2,09/23/26 11:01:00 AM,8.83,73.00,,,,",
]


def _write(path, header, rows):
    path.write_bytes((header + "\r\n".join(rows)).encode("utf-8"))
    return path


def test_is_hobo_export_true_for_plot_title_header(tmp_path):
    path = _write(tmp_path / "C1.csv", C1_HEADER, C1_ROWS)
    assert is_hobo_export(path)


def test_is_hobo_export_false_for_unrelated_csv(tmp_path):
    path = tmp_path / "other.csv"
    path.write_text("a,b,c\n1,2,3\n")
    assert not is_hobo_export(path)


def test_read_hobo_file_parses_location_serial_and_oxygen(tmp_path):
    df = read_hobo_file(_write(tmp_path / "C1.csv", C1_HEADER, C1_ROWS))
    assert df["location"].unique().to_list() == ["C1"]
    assert df["serial_number"].unique().to_list() == ["20113801"]
    assert df["oxygen_mgl"].to_list() == [8.73, 8.72]


def test_read_hobo_file_temp_f_to_c_conversion(tmp_path):
    df = read_hobo_file(_write(tmp_path / "C1.csv", C1_HEADER, C1_ROWS))
    assert abs(df["temp_degc"][0] - (73.58 - 32) * 5 / 9) < 1e-9


def test_read_hobo_file_drops_sentinel_and_event_only_rows(tmp_path):
    df = read_hobo_file(_write(tmp_path / "C1.csv", C1_HEADER, C1_ROWS))
    assert df.height == 2


def test_read_hobo_file_converts_local_time_to_utc(tmp_path):
    df = read_hobo_file(_write(tmp_path / "C1.csv", C1_HEADER, C1_ROWS))
    assert df["ts"].to_list() == [
        datetime(2026, 9, 23, 15, 0, 0),
        datetime(2026, 9, 23, 15, 1, 0),
    ]


def test_read_hobo_file_handles_fewer_event_columns(tmp_path):
    df = read_hobo_file(_write(tmp_path / "mesocosm.csv", MESOCOSM_HEADER, MESOCOSM_ROWS))
    assert df["location"].unique().to_list() == ["mesocosm"]
    assert df.height == 2


def test_read_all_hobo_concatenates_locations(tmp_path):
    c1 = _write(tmp_path / "C1.csv", C1_HEADER, C1_ROWS)
    mesocosm = _write(tmp_path / "mesocosm.csv", MESOCOSM_HEADER, MESOCOSM_ROWS)
    df = read_all_hobo([c1, mesocosm])
    assert sorted(df["location"].unique().to_list()) == ["C1", "mesocosm"]
    assert df.height == 4
