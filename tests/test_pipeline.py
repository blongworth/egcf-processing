import polars as pl

from egcf_processing.pipeline import run

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


def test_end_to_end_pipeline(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "gems_2026-01-01-00-00.txt").write_text(FILE_1)
    (raw_dir / "gems_2026-01-01-04-00.txt").write_text(FILE_2)
    (raw_dir / "gems_2026-01-01-08-00.txt").write_text("")  # 0-byte file, must be skipped

    out_dir = tmp_path / "processed"
    stats = run(raw_dir, out_dir, settle_offset_s=0)

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
