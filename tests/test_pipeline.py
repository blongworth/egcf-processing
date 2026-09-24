from datetime import datetime

import polars as pl
import pytest

from egcf_processing.combine import SYSTEM_HEALTH_SCHEMA
from egcf_processing.par import PAR_SCHEMA
from egcf_processing.pipeline import run

# Placeholder chamber geometry -- exercises the flux arithmetic only, not real
# EGFC dimensions.
TEST_VOLUME_L = 4.0
TEST_AREA_M2 = 0.06

FILE_1 = "\n".join(
    [
        "V:2026-01-01T00:00:00Z,C1,Re",
        "R:2026-01-01T00:00:01Z,2,10",
        "R:2026-01-01T00:00:02Z,15,20",
        "V:2026-01-01T00:00:10Z,C2,Re",
        "R:2026-01-01T00:00:11Z,2,11",
        "R:2026-01-01T00:00:12Z,15,21",
    ]
)

# Rotated file (simulates a rotation mid-experiment): flush both chambers, then
# start experiment 2.
FILE_2 = "\n".join(
    [
        "V:2026-01-01T04:00:00Z,C1,Fl",
        "V:2026-01-01T04:00:05Z,C2,Fl",
        "V:2026-01-01T04:00:10Z,C1,Re",
        "R:2026-01-01T04:00:11Z,2,12",
        "R:2026-01-01T04:00:12Z,15,22",
        "V:2026-01-01T04:00:20Z,C2,Fl",
    ]
)


def test_end_to_end_pipeline_merges_gems_and_surface_sources(tmp_path):
    raw_dir = tmp_path / "raw"
    (raw_dir / "lander").mkdir(parents=True)
    (raw_dir / "surface").mkdir(parents=True)
    (raw_dir / "lander" / "gems_2026-01-01-00-00.txt").write_text(FILE_1)
    (raw_dir / "surface" / "surface_2026-01-01-04-00_lander.log").write_text(
        "\n".join(
            [
                "# surface-lander-log-v1",
                "# fields: iso8601 payload",
                "2026-01-01T04:00:01Z V:2026-01-01T04:00:00Z,C1,Fl",
                "2026-01-01T04:00:06Z V:2026-01-01T04:00:05Z,C2,Fl",
                "2026-01-01T04:00:11Z V:2026-01-01T04:00:10Z,C1,Re",
                "2026-01-01T04:00:12Z R:2026-01-01T04:00:11Z,2,12",
                "2026-01-01T04:00:13Z R:2026-01-01T04:00:12Z,15,22",
                "2026-01-01T04:00:21Z V:2026-01-01T04:00:20Z,C2,Fl",
            ]
        )
    )

    out_dir = tmp_path / "processed"
    stats = run(raw_dir, out_dir, TEST_VOLUME_L, TEST_AREA_M2, settle_offset_s=0)

    assert stats["n_files"] == 2
    assert stats["cycle_stats"]["total_cycles"] == 3

    chamber_cycles = pl.read_parquet(out_dir / "egcf_chamber_cycles.parquet")
    assert chamber_cycles["experiment_number"].to_list() == [1, 1, 2]
    assert chamber_cycles["chamber"].to_list() == ["C1", "C2", "C1"]

    rga = pl.read_parquet(out_dir / "rga.parquet")
    assert rga.height == 6
    assert rga["ts"].dt.hour().to_list() == [0, 0, 0, 0, 4, 4]


def test_end_to_end_pipeline(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "gems_2026-01-01-00-00.txt").write_text(FILE_1)
    (raw_dir / "gems_2026-01-01-04-00.txt").write_text(FILE_2)
    (raw_dir / "gems_2026-01-01-08-00.txt").write_text("")  # 0-byte file, must be skipped

    out_dir = tmp_path / "processed"
    stats = run(raw_dir, out_dir, TEST_VOLUME_L, TEST_AREA_M2, settle_offset_s=0)

    assert stats["n_files"] == 2
    assert stats["cycle_stats"]["total_cycles"] == 3
    assert stats["cycle_stats"]["dropped_too_short"] == 0

    for name in ["status", "rga", "scalup", "valve"]:
        assert (out_dir / f"{name}.parquet").exists()

    chamber_cycles = pl.read_parquet(out_dir / "egcf_chamber_cycles.parquet")
    assert chamber_cycles.height == 3
    assert chamber_cycles["experiment_number"].to_list() == [1, 1, 2]
    assert chamber_cycles["chamber"].to_list() == ["C1", "C2", "C1"]

    rga_scans = pl.read_parquet(out_dir / "egcf_rga_scans.parquet")
    assert rga_scans.height == 2
    assert rga_scans["mass_2_avg"].to_list() == [10.0, 11.0]

    # No scalup data in this raw text, so every flux source column is null --
    # the file is still written, with the full schema, just empty.
    fluxes = pl.read_parquet(out_dir / "egcf_fluxes.parquet")
    assert fluxes.is_empty()
    assert "output_value" in fluxes.columns


def test_end_to_end_pipeline_computes_oxygen_flux(tmp_path):
    # Two C1 measurement cycles 2 min apart (a flush span between them),
    # oxygen 8.0 -> 7.0 mg/L: -0.5 mg/L/min uptake.
    raw = "\n".join(
        [
            "V:2026-01-01T00:00:00Z,C1,Re",
            "P:2026-01-01T00:00:01Z,2026-01-01T00:00:01Z,12.0,32.0,8.0,8.1",
            "V:2026-01-01T00:01:00Z,C1,Fl",
            "V:2026-01-01T00:02:00Z,C1,Re",
            "P:2026-01-01T00:02:01Z,2026-01-01T00:02:01Z,12.0,32.0,7.0,8.1",
            "V:2026-01-01T00:03:00Z,C2,Fl",
        ]
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "gems_2026-01-01-00-00.txt").write_text(raw)

    out_dir = tmp_path / "processed"
    run(raw_dir, out_dir, TEST_VOLUME_L, TEST_AREA_M2, settle_offset_s=0)

    fluxes = pl.read_parquet(out_dir / "egcf_fluxes.parquet")
    oxygen = fluxes.filter(pl.col("variable") == "oxygen")
    assert oxygen.height == 1
    assert oxygen["chamber"][0] == "C1"
    # -0.5 mg/L/min * (1000/32) umol/mg * 4.0 L / 0.06 m^2 * 60 min/h.
    assert oxygen["slope_native_per_min"][0] == pytest.approx(-0.5)
    assert oxygen["output_value"][0] == pytest.approx(-0.5 * (1000 / 32) * 4.0 / 0.06 * 60)
    assert oxygen["output_unit"][0] == "umol m-2 h-1"


def test_end_to_end_pipeline_csv_format(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "gems_2026-01-01-00-00.txt").write_text(FILE_1)

    out_dir = tmp_path / "processed"
    run(raw_dir, out_dir, TEST_VOLUME_L, TEST_AREA_M2, settle_offset_s=0, output_format="csv")

    for name in ["status", "rga", "scalup", "valve", "egcf_rga_scans", "egcf_chamber_cycles", "egcf_fluxes"]:
        assert (out_dir / f"{name}.csv").exists()
        assert not (out_dir / f"{name}.parquet").exists()

    chamber_cycles = pl.read_csv(out_dir / "egcf_chamber_cycles.csv")
    assert chamber_cycles["elapsed_time"].to_list() == [0.0]


EVENTS_LOG = "\n".join(
    [
        "# surface-event-log-v1",
        "# fields: iso8601 direction payload",
        "2026-01-01T04:00:02Z SYSTEM startup complete",
        "2026-01-01T04:00:03Z SYSTEM battery voltage=27.02V current=0.033A temp=42.5C",
        "2026-01-01T04:00:13Z SYSTEM battery voltage=22.81V current=0.040A temp=59.8C",
        "2026-01-01T04:00:14Z RX_CONSOLE VSTAT",
    ]
)


def test_end_to_end_pipeline_reads_paired_events_log(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "surface_2026-01-01-04-00_lander.log").write_text(
        "\n".join(
            [
                "# surface-lander-log-v1",
                "2026-01-01T04:00:11Z V:2026-01-01T04:00:10Z,C1,Re",
                "2026-01-01T04:00:12Z R:2026-01-01T04:00:11Z,2,12",
            ]
        )
    )
    (raw_dir / "surface_2026-01-01-04-00_events.log").write_text(EVENTS_LOG)

    out_dir = tmp_path / "processed"
    stats = run(raw_dir, out_dir, TEST_VOLUME_L, TEST_AREA_M2, settle_offset_s=0)

    assert stats["n_system_health_rows"] == 2
    system_health = pl.read_parquet(out_dir / "system_health.parquet")
    assert system_health["voltage_v"].to_list() == [27.02, 22.81]
    assert system_health["current_a"].to_list() == [0.033, 0.040]
    assert system_health["teensy_temp_c"].to_list() == [42.5, 59.8]
    assert system_health["ts"][0] == datetime(2026, 1, 1, 4, 0, 3)


def test_end_to_end_pipeline_gems_only_writes_empty_typed_system_health(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "gems_2026-01-01-00-00.txt").write_text(FILE_1)

    out_dir = tmp_path / "processed"
    stats = run(raw_dir, out_dir, TEST_VOLUME_L, TEST_AREA_M2, settle_offset_s=0)

    assert stats["n_system_health_rows"] == 0
    system_health = pl.read_parquet(out_dir / "system_health.parquet")
    assert system_health.is_empty()
    assert dict(system_health.schema) == SYSTEM_HEALTH_SCHEMA


ODYSSEY_EXPORT = "\r\n".join(
    [
        "﻿Site Name ,ESL-EGCF",
        "Site Number ,11",
        "Logger ,Integrating Light Sensor",
        "Logger Serial Number ,50472",
        "",
        "",
        "Scan No ,Date and Time,       Integrating Light,        ,",
        "        ,        ,RAW VALUE ,CALIBRATED VALUE,",
        "",
        "1,01/01/2026 , 00:00:02,100,100",
        "2,01/01/2026 , 00:00:04,300,300",
        "3,01/01/2026 , 00:00:12,500,500",
    ]
)


def test_end_to_end_pipeline_writes_par_and_averages_it_onto_cycles(tmp_path):
    raw_dir = tmp_path / "raw"
    (raw_dir / "PAR").mkdir(parents=True)
    (raw_dir / "gems_2026-01-01-00-00.txt").write_text(FILE_1)
    (raw_dir / "PAR" / "ESL-EGCF_011_001.CSV").write_bytes(ODYSSEY_EXPORT.encode("utf-8"))

    out_dir = tmp_path / "processed"
    stats = run(raw_dir, out_dir, TEST_VOLUME_L, TEST_AREA_M2, settle_offset_s=0)

    assert stats["n_par_rows"] == 3
    par = pl.read_parquet(out_dir / "par.parquet")
    assert par["serial_number"].unique().to_list() == ["50472"]
    assert par["sensor_number"].unique().to_list() == [1]
    assert par["par_umol_m2_s"].to_list() == pytest.approx([0.4647 * r + 6.4541 for r in (100, 300, 500)])

    # FILE_1 closes only the C1 cycle [00:00:00, 00:00:10); the 00:00:12 scan
    # falls in the still-open C2 cycle and is not averaged.
    chamber_cycles = pl.read_parquet(out_dir / "egcf_chamber_cycles.parquet")
    assert chamber_cycles["par_raw"].to_list() == [200.0]

    fluxes = pl.read_parquet(out_dir / "egcf_fluxes.parquet")
    assert {"par_mean_umol_m2_s", "par_integrated_mol_m2", "par_coverage"} <= set(fluxes.columns)

    # No scalup in FILE_1, so no O2 flux: metabolism and the P-I fit are written empty, with schema.
    assert pl.read_parquet(out_dir / "egcf_metabolism.parquet").columns[-2:] == ["ncp_umol_m2_h", "gpp_umol_m2_h"]
    assert "pmax_umol_m2_h" in pl.read_parquet(out_dir / "egcf_pi_fit.parquet").columns

    par_daily = pl.read_parquet(out_dir / "par_daily.parquet")
    assert stats["n_par_days"] == 1
    assert par_daily["max_par_umol_m2_s"][0] == pytest.approx(0.4647 * 500 + 6.4541)


def test_end_to_end_pipeline_without_par_writes_empty_typed_par(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "gems_2026-01-01-00-00.txt").write_text(FILE_1)

    out_dir = tmp_path / "processed"
    stats = run(raw_dir, out_dir, TEST_VOLUME_L, TEST_AREA_M2, settle_offset_s=0)

    assert stats["n_par_rows"] == 0
    par = pl.read_parquet(out_dir / "par.parquet")
    assert par.is_empty()
    assert par.columns == list(PAR_SCHEMA)
    assert par.schema["par_umol_m2_s"] == pl.Float64

    chamber_cycles = pl.read_parquet(out_dir / "egcf_chamber_cycles.parquet")
    assert chamber_cycles["par_umol_m2_s"].is_null().all()
