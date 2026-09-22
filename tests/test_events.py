from datetime import datetime

from egcf_processing.events import parse_event_line, read_events_file

EVENTS_LOG = "\n".join(
    [
        "# surface-event-log-v1",
        "# fields: iso8601 direction payload",
        "2026-09-16T15:11:03Z SYSTEM startup complete",
        "2026-09-16T15:11:03Z SYSTEM battery voltage=27.02V current=0.033A temp=42.5C",
        "2026-09-16T15:11:04Z RX_CONSOLE VSTAT",
        "2026-09-16T15:11:05Z TX_LANDER FOFF",
        "2026-09-16T15:11:13Z SYSTEM battery voltage=22.81V current=0.040A temp=59.8C",
        "2026-09-16T15:11:14Z SYSTEM battery voltage below threshold; sending OFF to lander",
    ]
)


def test_parse_event_line_battery():
    record = parse_event_line("2026-09-16T15:11:03Z SYSTEM battery voltage=27.02V current=0.033A temp=42.5C")
    assert record == {
        "tag": "SH",
        "ts": datetime(2026, 9, 16, 15, 11, 3),
        "voltage_v": 27.02,
        "current_a": 0.033,
        "teensy_temp_c": 42.5,
    }


def test_parse_event_line_returns_none_for_non_data_lines():
    non_data = [
        "# surface-event-log-v1",
        "# fields: iso8601 direction payload",
        "2026-09-16T15:11:04Z RX_CONSOLE VSTAT",
        "2026-09-16T15:11:05Z TX_LANDER FOFF",
        "2026-09-16T15:11:03Z SYSTEM startup complete",
        "2026-09-16T15:11:14Z SYSTEM battery voltage below threshold; sending OFF to lander",
        "2026-09-16T15:11:15Z SYSTEM low voltage shutdown confirmed by lander",
        "",
        "\x00garbled serial",
    ]
    for line in non_data:
        assert parse_event_line(line) is None, line


def test_read_events_file_returns_only_data_rows_with_source_file(tmp_path):
    path = tmp_path / "surface_2026-09-16-15-11_events.log"
    path.write_text(EVENTS_LOG)

    records = read_events_file(path)
    assert len(records) == 2
    assert [r["voltage_v"] for r in records] == [27.02, 22.81]
    assert [r["current_a"] for r in records] == [0.033, 0.040]
    assert [r["teensy_temp_c"] for r in records] == [42.5, 59.8]
    assert {r["source_file"] for r in records} == {path.name}
