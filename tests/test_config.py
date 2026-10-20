import tomllib
from pathlib import Path

import pytest

from egcf_processing import cli
from egcf_processing.cli import main
from egcf_processing.config import CONFIG_ENV_VAR, SECTIONS, load_config
from egcf_processing.dashboard import DEFAULT_EXPERIMENT_VARIABLE, DEFAULT_PROCESSED_ROOT, DEFAULT_TIME_RANGE_PRESET
from egcf_processing.deployment import PROCESSING_KEYS
from egcf_processing.pipeline import DEFAULT_OUTPUT_FORMAT, DEFAULT_PAR_CALIBRATIONS_PATH, PROCESSING_DEFAULTS

REPO_CONFIG = Path(__file__).resolve().parents[1] / "egcf.toml"

DEPLOYMENT_TOML = """
site = "Hadley Harbor"
start = 2026-01-01T00:00:00Z
test = false
c1 = "grass"
c2 = "sediment"

[processing]
settle_offset_s = 30
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_every_processing_key_has_a_section():
    sectioned = [k for keys in SECTIONS.values() for k in keys]
    assert set(PROCESSING_KEYS) <= set(sectioned)
    assert len(sectioned) == len(set(sectioned))


def test_repo_config_matches_code_defaults():
    settings = load_config(REPO_CONFIG)
    root = REPO_CONFIG.parent
    assert settings.pop("raw_root") == (root / cli.DEFAULT_RAW_ROOT).resolve()
    assert settings.pop("processed_root") == (root / DEFAULT_PROCESSED_ROOT).resolve()
    assert settings.pop("par_calibrations") == DEFAULT_PAR_CALIBRATIONS_PATH.resolve()
    assert settings == {
        **PROCESSING_DEFAULTS,
        "output_format": DEFAULT_OUTPUT_FORMAT,
        "time_range_preset": DEFAULT_TIME_RANGE_PRESET,
        "experiment_variable": DEFAULT_EXPERIMENT_VARIABLE,
        "show_test_deployments": False,
    }


def test_missing_default_config_is_empty(tmp_path, monkeypatch):
    monkeypatch.delenv(CONFIG_ENV_VAR)
    monkeypatch.chdir(tmp_path)
    assert load_config() == {}


def test_explicit_config_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")


def test_relative_paths_resolve_against_config_folder(tmp_path):
    settings = load_config(_write(tmp_path / "egcf.toml", '[paths]\nraw_root = "raw"\n'))
    assert settings["raw_root"] == (tmp_path / "raw").resolve()


def test_local_config_overrides_key_by_key(tmp_path):
    _write(tmp_path / "egcf.toml", "[chamber]\nchamber_area_m2 = 0.3\nchamber_volume_l = 30\n")
    _write(tmp_path / "egcf.local.toml", "[chamber]\nchamber_height_m = 0.2\n")
    settings = load_config(tmp_path / "egcf.toml")
    assert settings == {"chamber_area_m2": 0.3, "chamber_height_m": 0.2}


@pytest.mark.parametrize(
    "text, match",
    [
        ("[chamber]\nchamber_area = 0.3\n", r"unknown \[chamber\] key"),
        ("[rga]\nsettle_offset_s = 60\n", r"unknown \[rga\] key"),
        ("[misc]\nx = 1\n", "unknown section"),
        ("[chamber]\nchamber_area_m2 = -1\n", "positive"),
        ("[chamber]\nchamber_volume_l = 30\nchamber_height_m = 0.1\n", "only one of"),
        ('[output]\noutput_format = "xlsx"\n', "output_format"),
        ('[cycles]\nsettle_offset_s = "60"\n', "must be a number"),
        ('[dashboard]\nshow_test_deployments = "no"\n', "true or false"),
    ],
)
def test_config_rejects_bad_values(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        load_config(_write(tmp_path / "egcf.toml", text))


def test_deployment_processing_accepts_every_setting(tmp_path):
    raw_dir = tmp_path / "dep"
    raw_dir.mkdir()
    _write(raw_dir / "deployment.toml", DEPLOYMENT_TOML.replace("settle_offset_s = 30", "par_time_offset_h = -4"))
    assert cli.load_deployment(raw_dir)["processing"] == {"par_time_offset_h": -4.0}


def test_cli_precedence_flag_over_deployment_over_config_over_default(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    raw_dir = raw_root / "2026-01-01_hadley"
    raw_dir.mkdir(parents=True)
    _write(raw_dir / "deployment.toml", DEPLOYMENT_TOML)
    config = _write(
        tmp_path / "egcf.toml",
        '[paths]\nraw_root = "raw"\nprocessed_root = "out"\n'
        "[cycles]\nsettle_offset_s = 45\n"
        "[metabolism]\ndark_par_threshold_umol_m2_s = 10\n",
    )
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config))
    seen = {}
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: seen.update(kwargs, args=args))

    main(["2026-01-01_hadley"])
    assert seen["args"] == (raw_dir.resolve(), (tmp_path / "out" / "2026-01-01_hadley").resolve())
    assert seen["settle_offset_s"] == 30.0  # deployment.toml beats egcf.toml
    assert seen["dark_par_threshold_umol_m2_s"] == 10.0  # egcf.toml beats default
    assert seen["min_par_coverage"] == PROCESSING_DEFAULTS["min_par_coverage"]

    main(["2026-01-01_hadley", "--settle-offset-s", "5"])
    assert seen["settle_offset_s"] == 5.0
