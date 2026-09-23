import polars as pl
import pytest

from egcf_processing.cli import main

# Placeholder chamber geometry -- required flags, exercising flux arithmetic only.
GEOMETRY = ["--chamber-volume-l", "4.0", "--chamber-area-m2", "0.06"]

# Two 10s-long cycles: short enough that the default 60s settle offset drops
# both as too-short, but long enough to survive with --settle-offset-s 0.
RAW = "\n".join(
    [
        "V:2026-01-01T00:00:00Z,C1,Re",
        "R:2026-01-01T00:00:01Z,2,10",
        "V:2026-01-01T00:00:10Z,C2,Re",
        "R:2026-01-01T00:00:11Z,2,20",
        "V:2026-01-01T00:00:20Z,C1,Fl",
    ]
)


def _write_raw(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "gems_2026-01-01-00-00.txt").write_text(RAW)
    return raw_dir


def test_main_uses_default_settle_offset_and_parquet_format(tmp_path):
    raw_dir = _write_raw(tmp_path)
    out_dir = tmp_path / "processed"

    main([str(raw_dir), "--out-dir", str(out_dir), *GEOMETRY])

    # default settle_offset_s (60s) drops both 10s-long cycles as too-short.
    assert (out_dir / "egcf_chamber_cycles.parquet").exists()
    assert pl.read_parquet(out_dir / "egcf_chamber_cycles.parquet").is_empty()


def test_main_settle_offset_flag_reaches_pipeline(tmp_path):
    raw_dir = _write_raw(tmp_path)
    out_dir = tmp_path / "processed"

    main([str(raw_dir), "--out-dir", str(out_dir), *GEOMETRY, "--settle-offset-s", "0"])

    chamber_cycles = pl.read_parquet(out_dir / "egcf_chamber_cycles.parquet")
    assert chamber_cycles.height == 2
    assert chamber_cycles["chamber"].to_list() == ["C1", "C2"]


def test_main_format_flag_writes_csv_not_parquet(tmp_path):
    raw_dir = _write_raw(tmp_path)
    out_dir = tmp_path / "processed"

    main([str(raw_dir), "--out-dir", str(out_dir), *GEOMETRY, "--settle-offset-s", "0", "--format", "csv"])

    assert (out_dir / "egcf_chamber_cycles.csv").exists()
    assert not (out_dir / "egcf_chamber_cycles.parquet").exists()


def test_main_partial_pressure_sensitivity_flag_reaches_pipeline(tmp_path):
    raw_dir = _write_raw(tmp_path)
    out_dir = tmp_path / "processed"

    main(
        [
            str(raw_dir),
            "--out-dir",
            str(out_dir),
            *GEOMETRY,
            "--settle-offset-s",
            "0",
            "--partial-pressure-sensitivity",
            "1e-3",
        ]
    )

    chamber_cycles = pl.read_parquet(out_dir / "egcf_chamber_cycles.parquet")
    # mass_2_avg=10 raw counts -> amps = 10 * 1e-16; torr = amps / sensitivity.
    assert chamber_cycles["mass_2_torr"][0] == pytest.approx((10 * 1e-16) / 1e-3)


def test_main_total_pressure_sensitivity_flag_reaches_pipeline(tmp_path):
    raw = "\n".join(
        [
            "V:2026-01-01T00:00:00Z,C1,Re",
            "!:2026-01-01T00:00:01Z,0,1200,50,24,30,28,29,1,2000,8760",
            "V:2026-01-01T00:00:20Z,C2,Fl",
        ]
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "gems_2026-01-01-00-00.txt").write_text(raw)
    out_dir = tmp_path / "processed"

    main(
        [
            str(raw_dir),
            "--out-dir",
            str(out_dir),
            *GEOMETRY,
            "--settle-offset-s",
            "0",
            "--total-pressure-sensitivity",
            "5e-4",
        ]
    )

    chamber_cycles = pl.read_parquet(out_dir / "egcf_chamber_cycles.parquet")
    # raw_total_pressure_current=2000 -> amps = 2000 * 1e-16; torr = amps / sensitivity.
    assert chamber_cycles["total_pressure_torr"][0] == pytest.approx((2000 * 1e-16) / 5e-4)


ODYSSEY_EXPORT = "\r\n".join(
    [
        "\ufeffSite Name ,ESL-EGCF",
        "Logger Serial Number ,50472",
        "",
        "Scan No ,Date and Time,       Integrating Light,        ,",
        "        ,        ,RAW VALUE ,CALIBRATED VALUE,",
        "",
        "1,01/01/2026 , 00:00:02,100,100",
        "2,01/01/2026 , 00:00:04,300,300",
        "3,01/01/2026 , 00:00:12,500,500",
    ]
)


def test_main_par_dir_flag_searches_outside_raw_dir(tmp_path):
    raw_dir = _write_raw(tmp_path)
    par_dir = tmp_path / "PAR"
    par_dir.mkdir()
    (par_dir / "ESL-EGCF_011_001.CSV").write_bytes(ODYSSEY_EXPORT.encode("utf-8"))
    out_dir = tmp_path / "processed"

    main([str(raw_dir), "--out-dir", str(out_dir), *GEOMETRY])
    assert pl.read_parquet(out_dir / "par.parquet").is_empty()

    main([str(raw_dir), "--out-dir", str(out_dir), *GEOMETRY, "--par-dir", str(par_dir)])
    assert pl.read_parquet(out_dir / "par.parquet").height == 3
