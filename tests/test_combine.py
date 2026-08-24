import polars as pl

from egcf_processing.combine import STATUS_SCHEMA, build_tables, write_tables
from egcf_processing.lines import parse_line

LINES = [
    "R:2026-06-02T14:00:00Z,2,100",
    "R:2026-06-02T14:00:01Z,15,200",
    "V:2026-06-02T14:00:02Z,C1,Re",
    "P:2026-06-02T14:00:03Z,2026-06-02T14:00:03,20.0,30.0,1010.0,8.0,7.5",
]


def _records():
    return [parse_line(line) for line in LINES]


def test_build_tables_row_counts_and_schema():
    tables = build_tables(_records())
    assert tables["rga"].height == 2
    assert tables["valve"].height == 1
    assert tables["scalup"].height == 1
    assert tables["status"].height == 0
    assert list(tables["status"].schema.keys()) == list(STATUS_SCHEMA.keys())


def test_build_tables_ignores_unrecognized_records():
    # parse_line already filters these out, but build_tables should not choke
    # if handed an unknown tag defensively.
    tables = build_tables(_records() + [{"tag": "PM", "ts": None}])
    assert tables["rga"].height == 2


def test_write_tables_creates_parquet_files(tmp_path):
    tables = build_tables(_records())
    written = write_tables(tables, tmp_path)
    assert set(written) == {"rga", "valve", "scalup", "status"}
    for name, path in written.items():
        assert path.exists()
        assert pl.read_parquet(path).height == tables[name].height
