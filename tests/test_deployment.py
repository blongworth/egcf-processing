from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from egcf_processing import cli
from egcf_processing.cli import main
from egcf_processing.flux import chamber_geometry
from egcf_processing.dashboard import clip_tables, deployment_label, list_processed_dirs, parse_window_time
from egcf_processing.deployment import DEPLOYMENT_SCHEMA, clip, deployment_table, load_deployment, resolve_processing
from egcf_processing.pipeline import run

DASHBOARD_PATH = Path(__file__).resolve().parents[1] / "src" / "egcf_processing" / "dashboard.py"

VALID_TOML = """
site = "Hadley Harbor"
lat = 41.5145
lon = -70.6990
start = 2026-01-01T00:01:00Z   # excludes the first cycle
end   = 2026-01-01T05:00:00-00:00
test  = false
c1 = "grass"
c2 = "sediment"
comments = \"\"\"
Calm, two divers.
\"\"\"
"""

# One experiment (only C1 flushes), whose first cycle falls before the window.
RAW = "\n".join(
    [
        "V:2026-01-01T00:00:00Z,C1,Re",
        "P:2026-01-01T00:00:01Z,2026-01-01T00:00:01Z,12.0,32.0,8.0,8.1",
        "R:2026-01-01T00:00:02Z,2,10",
        "V:2026-01-01T00:00:10Z,C1,Fl",
        "V:2026-01-01T04:00:00Z,C1,Re",
        "P:2026-01-01T04:00:01Z,2026-01-01T04:00:01Z,12.0,32.0,8.0,8.1",
        "R:2026-01-01T04:00:02Z,2,12",
        "V:2026-01-01T04:01:00Z,C1,Fl",
        "V:2026-01-01T04:02:00Z,C1,Re",
        "P:2026-01-01T04:02:01Z,2026-01-01T04:02:01Z,12.0,32.0,7.0,8.1",
        "R:2026-01-01T04:02:02Z,2,14",
        "V:2026-01-01T04:03:00Z,C2,Fl",
    ]
)


def _deployment_dir(tmp_path, toml=VALID_TOML, name="2026-01-01_hadley"):
    raw_dir = tmp_path / name
    raw_dir.mkdir()
    (raw_dir / "deployment.toml").write_text(toml)
    return raw_dir


def test_load_deployment_valid(tmp_path):
    meta = load_deployment(_deployment_dir(tmp_path))
    assert meta["deployment_id"] == "2026-01-01_hadley"
    assert meta["site"] == "Hadley Harbor"
    assert meta["start"] == datetime(2026, 1, 1, 0, 1)
    assert meta["start"].tzinfo is None
    assert meta["end"] == datetime(2026, 1, 1, 5)
    assert (meta["c1"], meta["c2"], meta["test"]) == ("grass", "sediment", False)
    assert meta["comments"].strip() == "Calm, two divers."


def test_load_deployment_converts_offset_to_utc(tmp_path):
    toml = VALID_TOML.replace("2026-01-01T00:01:00Z", "2026-01-01T00:01:00-04:00")
    assert load_deployment(_deployment_dir(tmp_path, toml))["start"] == datetime(2026, 1, 1, 4, 1)


def test_load_deployment_missing_file_returns_none(tmp_path):
    assert load_deployment(tmp_path) is None


@pytest.mark.parametrize(
    "old, new, match",
    [
        ("2026-01-01T00:01:00Z", "2026-01-01T00:01:00", "timezone"),
        ('c1 = "grass"', 'c1 = "seagrass"', "c1"),
        ("test  = false", "test  = false\nsiet = 'x'", "unknown key"),
        ('site = "Hadley Harbor"', "", "missing"),
        ("2026-01-01T05:00:00-00:00", "2025-12-31T00:00:00Z", "after start"),
    ],
)
def test_load_deployment_rejects_bad_toml(tmp_path, old, new, match):
    with pytest.raises(ValueError, match=match):
        load_deployment(_deployment_dir(tmp_path, VALID_TOML.replace(old, new)))


def test_deployment_table_is_one_row_with_schema(tmp_path):
    table = deployment_table(load_deployment(_deployment_dir(tmp_path)))
    assert table.height == 1
    assert table.columns == list(DEPLOYMENT_SCHEMA)
    assert table["start"].to_list() == [datetime(2026, 1, 1, 0, 1)]


def test_clip_is_half_open_and_noop_without_meta():
    df = pl.DataFrame({"ts": [datetime(2026, 1, 1, h) for h in range(4)]})
    meta = {"start": datetime(2026, 1, 1, 1), "end": datetime(2026, 1, 1, 3)}
    assert clip(df, meta)["ts"].dt.hour().to_list() == [1, 2]
    assert clip(df, {**meta, "end": None})["ts"].dt.hour().to_list() == [1, 2, 3]
    assert clip(df, None).equals(df)


def test_pipeline_trims_layers_b_to_e_but_not_layer_a(tmp_path):
    raw_dir = _deployment_dir(tmp_path)
    (raw_dir / "gems_2026-01-01-00-00.txt").write_text(RAW)
    out_dir = tmp_path / "processed"

    stats = run(raw_dir, out_dir, 4.0, 0.06, settle_offset_s=0)
    assert stats["deployment_id"] == "2026-01-01_hadley"

    assert pl.read_parquet(out_dir / "valve.parquet").height == 6
    assert pl.read_parquet(out_dir / "rga.parquet").height == 3

    cycles = pl.read_parquet(out_dir / "egcf_chamber_cycles.parquet")
    assert cycles["timestamp"].min() >= datetime(2026, 1, 1, 4)
    assert cycles["experiment_number"].unique().to_list() == [1]

    fluxes = pl.read_parquet(out_dir / "egcf_fluxes.parquet")
    o2 = fluxes.filter(pl.col("variable") == "oxygen")
    assert o2.height == 1
    assert o2["condition"].to_list() == ["grass"]
    assert "condition" in pl.read_parquet(out_dir / "egcf_metabolism.parquet").columns
    assert "condition" in pl.read_parquet(out_dir / "egcf_pi_fit.parquet").columns

    for path in out_dir.glob("*.parquet"):
        df = pl.read_parquet(path)
        assert df.columns[0] == "deployment_id", path.name
        assert df["deployment_id"].drop_nulls().unique().to_list() in ([], ["2026-01-01_hadley"])

    assert pl.read_parquet(out_dir / "deployment.parquet")["site"].to_list() == ["Hadley Harbor"]
    assert (out_dir / "deployment.toml").read_text() == VALID_TOML


def test_cli_defaults_out_dir_to_processed_deployment_id(tmp_path, monkeypatch):
    raw_dir = _deployment_dir(tmp_path)
    (raw_dir / "gems_2026-01-01-00-00.txt").write_text(RAW)
    monkeypatch.chdir(tmp_path)
    main([str(raw_dir)])
    assert (tmp_path / "data" / "processed" / "2026-01-01_hadley" / "deployment.parquet").exists()


def _processed(tmp_path, name, test):
    raw_dir = _deployment_dir(tmp_path / "raw", VALID_TOML.replace("test  = false", f"test  = {str(test).lower()}"), name)
    (raw_dir / "gems_2026-01-01-00-00.txt").write_text(RAW)
    out_dir = tmp_path / "data" / "processed" / name
    run(raw_dir, out_dir, 4.0, 0.06, settle_offset_s=0)
    return out_dir


def test_list_processed_dirs_hides_tests_and_skips_non_output(tmp_path):
    (tmp_path / "raw").mkdir()
    _processed(tmp_path, "2026-01-01_hadley", test=False)
    _processed(tmp_path, "2026-01-02_bench", test=True)
    root = tmp_path / "data" / "processed"
    (root / "scratch").mkdir()
    (root / "scratch" / "notes.html").write_text("")

    assert [p.name for p, _ in list_processed_dirs(root)] == ["2026-01-01_hadley"]
    shown = list_processed_dirs(root, show_test=True)
    assert [p.name for p, _ in shown] == ["2026-01-02_bench", "2026-01-01_hadley"]
    assert deployment_label(*shown[0]) == "2026-01-02_bench — Hadley Harbor (TEST)"


def test_parse_window_time_and_clip_tables():
    assert parse_window_time("2026-01-01T04:00:00-04:00") == datetime(2026, 1, 1, 8)
    assert parse_window_time("2026-01-01T04:00:00") == datetime(2026, 1, 1, 4)
    assert parse_window_time(" ") is None
    tables = {
        "rga": pl.DataFrame({"ts": [datetime(2026, 1, 1, 0), datetime(2026, 1, 1, 5)]}),
        "egcf_fluxes": pl.DataFrame({"experiment_start": [datetime(2026, 1, 1, 0), datetime(2026, 1, 1, 5)]}),
        "egcf_pi_fit": pl.DataFrame({"chamber": ["C1"]}),
        "par": None,
    }
    clipped = clip_tables(tables, datetime(2026, 1, 1, 1), None)
    assert clipped["rga"].height == 1
    assert clipped["egcf_fluxes"].height == 1
    assert clipped["egcf_pi_fit"].height == 1
    assert clipped["par"] is None


def test_dashboard_deployment_selectbox_metadata_and_window(tmp_path, monkeypatch):
    (tmp_path / "raw").mkdir()
    _processed(tmp_path, "2026-01-01_hadley", test=False)
    _processed(tmp_path, "2026-01-02_bench", test=True)
    monkeypatch.chdir(tmp_path)

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    assert not at.exception
    assert at.sidebar.selectbox(key="deployment").options == ["2026-01-01_hadley — Hadley Harbor"]
    assert "C1: **grass**" in " ".join(m.value for m in at.markdown)
    assert not at.warning

    at.sidebar.checkbox(key="show_test_deployments").check().run(timeout=60)
    assert len(at.sidebar.selectbox(key="deployment").options) == 2
    at.sidebar.selectbox(key="deployment").set_value("2026-01-02_bench — Hadley Harbor (TEST)").run(timeout=60)
    assert not at.exception
    assert "TEST deployment" in at.warning[0].value

    # The toml window starts the experiment at the 04:00 cycle; widening it brings back the 00:00 one.
    def experiments():
        return [s for s in at.tabs[2].get("selectbox") if s.label == "Experiment"][0].options

    assert experiments() == ["1 (2026-01-01 04:00:00)"]
    at.sidebar.text_input(key="window_start_2026-01-02_bench").set_value("2026-01-01T00:00:00Z").run(timeout=60)
    assert not at.exception
    assert experiments() == ["1 (2026-01-01 00:00:00)"]


def test_dashboard_lists_deployments_under_the_selected_directory(tmp_path):
    (tmp_path / "raw").mkdir()
    _processed(tmp_path, "2026-01-01_hadley", test=False)
    root = tmp_path / "data" / "processed"

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(root)).run(timeout=60)
    assert not at.exception
    assert at.sidebar.selectbox(key="deployment").options == ["2026-01-01_hadley — Hadley Harbor"]

    # Pointing straight at one deployment's output loads it with no picker.
    at.sidebar.text_input[0].set_value(str(root / "2026-01-01_hadley")).run(timeout=60)
    assert not at.exception
    assert "deployment" not in [s.key for s in at.sidebar.selectbox]
    assert "C1: **grass**" in " ".join(m.value for m in at.markdown)

    # Raw deployment folders carry a toml too, but aren't processed output: no picker, empty dataset as before.
    at.sidebar.text_input[0].set_value(str(tmp_path / "raw")).run(timeout=60)
    assert not at.exception
    assert "deployment" not in [s.key for s in at.sidebar.selectbox]


PROCESSING_TOML = VALID_TOML + """
[processing]
chamber_volume_l = 25
chamber_area_m2 = 0.25
"""


def test_load_deployment_processing_table(tmp_path):
    assert load_deployment(_deployment_dir(tmp_path))["processing"] == {}
    meta = load_deployment(_deployment_dir(tmp_path, PROCESSING_TOML, "with_processing"))
    assert meta["processing"] == {"chamber_volume_l": 25.0, "chamber_area_m2": 0.25}


@pytest.mark.parametrize(
    "old, new, match",
    [
        ("chamber_area_m2 = 0.25", "chamber_area_m2 = 0", "positive"),
        ("chamber_area_m2 = 0.25", "chamber_area = 0.25", r"unknown \[processing\] key"),
        ("chamber_volume_l = 25", 'chamber_volume_l = "25"', "positive"),
    ],
)
def test_load_deployment_rejects_bad_processing(tmp_path, old, new, match):
    with pytest.raises(ValueError, match=match):
        load_deployment(_deployment_dir(tmp_path, PROCESSING_TOML.replace(old, new)))


def test_resolve_processing_precedence(tmp_path):
    meta = load_deployment(_deployment_dir(tmp_path, VALID_TOML + "\n[processing]\nchamber_volume_l = 25\n"))
    defaults = {"chamber_volume_l": 30.0, "chamber_area_m2": 0.30}
    assert resolve_processing(meta, {}, defaults) == {
        "chamber_volume_l": (25.0, "deployment.toml"),
        "chamber_area_m2": (0.30, "default"),
    }
    assert resolve_processing(meta, {"chamber_volume_l": 10, "chamber_area_m2": None}, defaults)["chamber_volume_l"] == (
        10.0,
        "argument",
    )
    assert resolve_processing(None, {}, defaults)["chamber_volume_l"] == (30.0, "default")


def test_cli_geometry_from_toml_unless_flag_given(tmp_path, monkeypatch):
    raw_dir = _deployment_dir(tmp_path, PROCESSING_TOML)
    seen = {}
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: seen.update(kwargs))
    main([str(raw_dir), "--out-dir", str(tmp_path / "out")])
    assert (seen["chamber_volume_l"], seen["chamber_area_m2"]) == (25.0, 0.25)
    main([str(raw_dir), "--out-dir", str(tmp_path / "out"), "--chamber-area-m2", "0.5"])
    assert (seen["chamber_volume_l"], seen["chamber_area_m2"]) == (25.0, 0.5)


def test_pipeline_records_processing_parameters_used(tmp_path):
    raw_dir = _deployment_dir(tmp_path, PROCESSING_TOML)
    (raw_dir / "gems_2026-01-01-00-00.txt").write_text(RAW)
    run(raw_dir, tmp_path / "out", 4.0, 0.06, settle_offset_s=0)
    processing = pl.read_parquet(tmp_path / "out" / "processing.parquet")
    assert processing.to_dicts() == [
        {
            "deployment_id": "2026-01-01_hadley",
            "chamber_volume_l": 4.0,
            "chamber_area_m2": 0.06,
            "chamber_height_m": pytest.approx(4.0 / 0.06 / 1000),
            "chamber_geometry_from": "volume",
        }
    ]


def test_dashboard_geometry_defaults_to_the_runs_recorded_values(tmp_path):
    root = tmp_path / "processed"
    for name, volume in [("a_site", 4.0), ("b_site", 8.0)]:
        raw_dir = _deployment_dir(tmp_path, name=name)
        (raw_dir / "gems_2026-01-01-00-00.txt").write_text(RAW)
        run(raw_dir, root / name, volume, 0.06, settle_offset_s=0)

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(root)).run(timeout=60)
    assert not at.exception
    assert at.sidebar.number_input(key="chamber_volume_l").value == 8.0  # b_site sorts first
    assert at.sidebar.number_input(key="chamber_area_m2").value == 0.06

    at.sidebar.number_input(key="chamber_volume_l").set_value(5.0).run(timeout=60)
    assert at.sidebar.number_input(key="chamber_volume_l").value == 5.0

    at.sidebar.selectbox(key="deployment").set_value("a_site — Hadley Harbor").run(timeout=60)
    assert not at.exception
    assert at.sidebar.number_input(key="chamber_volume_l").value == 4.0


def test_chamber_geometry_from_volume_or_height():
    from_volume = chamber_geometry(0.30, chamber_volume_l=30.0)
    assert from_volume["chamber_height_m"] == pytest.approx(0.1)
    from_height = chamber_geometry(0.30, chamber_height_m=0.1)
    assert from_height["chamber_volume_l"] == pytest.approx(30.0)
    for kwargs in ({}, {"chamber_volume_l": 30.0, "chamber_height_m": 0.1}):
        with pytest.raises(ValueError, match="exactly one"):
            chamber_geometry(0.30, **kwargs)
    with pytest.raises(ValueError, match="chamber_height_m must be positive"):
        chamber_geometry(0.30, chamber_height_m=0)


def test_flux_from_height_matches_equivalent_volume(tmp_path):
    # Flux depends only on V/A: 4 L over 0.06 m2 is the same chamber as 1/15 m of water over any area.
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
    run(raw_dir, tmp_path / "by_volume", 4.0, 0.06, settle_offset_s=0)
    run(raw_dir, tmp_path / "by_height", None, 0.5, settle_offset_s=0, chamber_height_m=4.0 / 0.06 / 1000)

    def o2(out):
        fluxes = pl.read_parquet(tmp_path / out / "egcf_fluxes.parquet")
        return fluxes.filter(pl.col("variable") == "oxygen")["output_value"].to_list()

    # -0.5 mg/L/min / 32 mg/mmol * 60 min/h = -0.9375 mmol/L/h, times V/A = 4/0.06 L/m2 = -62.5.
    assert o2("by_volume") == pytest.approx([-62.5])
    assert o2("by_height") == pytest.approx(o2("by_volume"))


def test_processing_height_resolution_and_record(tmp_path, monkeypatch):
    toml = VALID_TOML + "\n[processing]\nchamber_height_m = 0.1\nchamber_area_m2 = 0.2\n"
    raw_dir = _deployment_dir(tmp_path, toml)
    (raw_dir / "gems_2026-01-01-00-00.txt").write_text(RAW)
    main([str(raw_dir), "--out-dir", str(tmp_path / "out"), "--settle-offset-s", "0"])
    processing = pl.read_parquet(tmp_path / "out" / "processing.parquet").row(0, named=True)
    assert processing["chamber_geometry_from"] == "height"
    assert processing["chamber_volume_l"] == pytest.approx(20.0)

    # A volume flag beats the toml's height rather than leaving both set.
    seen = {}
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: seen.update(kwargs))
    main([str(raw_dir), "--out-dir", str(tmp_path / "out"), "--chamber-volume-l", "12"])
    assert (seen["chamber_volume_l"], seen["chamber_height_m"], seen["chamber_area_m2"]) == (12.0, None, 0.2)

    with pytest.raises(SystemExit):
        main([str(raw_dir), "--chamber-volume-l", "12", "--chamber-height-m", "0.1"])
    with pytest.raises(ValueError, match="only one of"):
        load_deployment(_deployment_dir(tmp_path, toml + "chamber_volume_l = 20\n", "both"))


def test_dashboard_geometry_height_mode(tmp_path):
    raw_dir = _deployment_dir(tmp_path, VALID_TOML + "\n[processing]\nchamber_height_m = 0.1\nchamber_area_m2 = 0.2\n")
    (raw_dir / "gems_2026-01-01-00-00.txt").write_text(RAW)
    run(raw_dir, tmp_path / "out", None, 0.2, settle_offset_s=0, chamber_height_m=0.1)

    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.run(timeout=60)
    at.sidebar.text_input[0].set_value(str(tmp_path / "out")).run(timeout=60)
    assert not at.exception
    assert at.sidebar.radio(key="chamber_geometry_from").value == "Height (m)"
    assert at.sidebar.number_input(key="chamber_height_m").value == 0.1
    assert "Volume: 20.000 L" in " ".join(c.value for c in at.sidebar.caption)

    at.sidebar.number_input(key="chamber_height_m").set_value(0.2).run(timeout=60)
    at.sidebar.radio(key="chamber_geometry_from").set_value("Volume (L)").run(timeout=60)
    assert not at.exception
    assert at.sidebar.number_input(key="chamber_volume_l").value == pytest.approx(40.0)
