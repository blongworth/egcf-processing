"""Repo-wide settings from a human-edited ``egcf.toml``.

The file is found at $EGCF_CONFIG, else ./egcf.toml; a missing default file
means "use the code defaults". An ``egcf.local.toml`` beside it (gitignored,
for machine-specific paths) overrides it key by key.

Sections only group keys for reading: key names are flat and match the CLI
options and deployment.toml [processing], so a line can be copied between
them. Precedence is CLI flag > deployment.toml [processing] > egcf.toml >
code default. Relative paths resolve against the config file's folder.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from egcf_processing.deployment import PROCESSING_GROUPS, check_processing_value

CONFIG_FILENAME = "egcf.toml"
LOCAL_CONFIG_FILENAME = "egcf.local.toml"
CONFIG_ENV_VAR = "EGCF_CONFIG"

OUTPUT_FORMATS = ("parquet", "csv", "both")

SECTIONS = {
    "paths": ("raw_root", "processed_root", "par_calibrations"),
    "output": ("output_format",),
    "chamber": ("chamber_area_m2", "chamber_volume_l", "chamber_height_m"),
    "cycles": ("settle_offset_s",),
    "rga": (
        "partial_pressure_sensitivity_a_per_torr",
        "total_pressure_sensitivity_a_per_torr",
        "n2_ar_sensitivity_ratio",
    ),
    "par": ("par_time_offset_h",),
    "metabolism": (
        "dark_par_threshold_umol_m2_s",
        "min_par_coverage",
        "metabolism_min_r2",
        "chamber_par_transmittance",
    ),
    "dashboard": ("time_range_preset", "experiment_variable", "show_test_deployments"),
}


def config_path() -> Path:
    return Path(os.environ.get(CONFIG_ENV_VAR) or CONFIG_FILENAME)


def _check_value(key: str, value, path: Path):
    if key in SECTIONS["paths"]:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{path}: {key} must be a non-empty path string")
        return (path.parent / value).resolve()
    if key == "output_format":
        if value not in OUTPUT_FORMATS:
            raise ValueError(f"{path}: output_format = {value!r} must be one of {list(OUTPUT_FORMATS)}")
        return value
    if key in ("time_range_preset", "experiment_variable"):
        if not isinstance(value, str):
            raise ValueError(f"{path}: {key} must be a string")
        return value
    if key == "show_test_deployments":
        if not isinstance(value, bool):
            raise ValueError(f"{path}: {key} must be true or false")
        return value
    return check_processing_value(key, value, f"{path}:")


def parse_config(raw: dict, path: Path) -> dict:
    """Validate a parsed toml dict and flatten it to {key: value}."""
    unknown = sorted(set(raw) - set(SECTIONS))
    if unknown:
        raise ValueError(f"{path}: unknown section(s) {unknown}; allowed: {list(SECTIONS)}")
    settings = {}
    for section, table in raw.items():
        if not isinstance(table, dict):
            raise ValueError(f"{path}: {section} must be a [{section}] table")
        unknown = sorted(set(table) - set(SECTIONS[section]))
        if unknown:
            raise ValueError(f"{path}: unknown [{section}] key(s) {unknown}; allowed: {list(SECTIONS[section])}")
        for key, value in table.items():
            settings[key] = _check_value(key, value, path)
    for group in PROCESSING_GROUPS:
        given = [k for k in group if k in settings]
        if len(given) > 1:
            raise ValueError(f"{path}: give only one of {given}")
    return settings


def _read(path: Path) -> dict:
    with path.open("rb") as f:
        return parse_config(tomllib.load(f), path)


def load_config(path: Path | None = None) -> dict:
    """Merged settings from egcf.toml and egcf.local.toml; {} if neither exists.

    An explicit path (argument or $EGCF_CONFIG) must exist.
    """
    explicit = path is not None or bool(os.environ.get(CONFIG_ENV_VAR))
    path = Path(path) if path is not None else config_path()
    if explicit and not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    settings = _read(path) if path.is_file() else {}
    local = path.parent / LOCAL_CONFIG_FILENAME
    if local.is_file():
        overrides = _read(local)
        for group in PROCESSING_GROUPS:
            if any(k in overrides for k in group):
                for k in group:
                    settings.pop(k, None)
        settings.update(overrides)
    return settings
