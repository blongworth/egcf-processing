"""Per-deployment metadata from a human-edited ``deployment.toml``.

One raw folder per deployment; the folder name is the ``deployment_id``. The
toml records the site, chamber treatments, whether the run was a test, and the
in-water window ``[start, end)`` that Layers B-E (and the dashboard) are
trimmed to. Layer A stays untrimmed, so changing the window never needs a
re-parse.

Times must carry an explicit offset and are normalized to naive UTC, matching
every other timestamp in the pipeline.
"""

from __future__ import annotations

import re
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

DEPLOYMENT_FILENAME = "deployment.toml"

_TOML_POSITION = re.compile(r"at line (\d+), column (\d+)")
_UNPADDED_DATE = re.compile(r"\b\d{4}-(\d-|\d{2}-\d\b)")

CONDITIONS = ("grass", "sediment", "none")

DEPLOYMENT_SCHEMA = {
    "deployment_id": pl.Utf8,
    "site": pl.Utf8,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "start": pl.Datetime,
    "end": pl.Datetime,
    "test": pl.Boolean,
    "c1": pl.Utf8,
    "c2": pl.Utf8,
    "comments": pl.Utf8,
}

PROCESSING_SCHEMA = {
    "deployment_id": pl.Utf8,
    "chamber_volume_l": pl.Float64,
    "chamber_area_m2": pl.Float64,
    "chamber_height_m": pl.Float64,
    "chamber_geometry_from": pl.Utf8,
    "settle_offset_s": pl.Float64,
    "partial_pressure_sensitivity_a_per_torr": pl.Float64,
    "total_pressure_sensitivity_a_per_torr": pl.Float64,
    "n2_ar_sensitivity_ratio": pl.Float64,
    "par_time_offset_h": pl.Float64,
    "dark_par_threshold_umol_m2_s": pl.Float64,
    "min_par_coverage": pl.Float64,
    "metabolism_min_r2": pl.Float64,
    "chamber_par_transmittance": pl.Float64,
}

# Each group is one quantity: a source (argument, toml, default) supplies at most one key per group.
PROCESSING_GROUPS = (
    ("chamber_area_m2",),
    ("chamber_volume_l", "chamber_height_m"),
    ("settle_offset_s",),
    ("partial_pressure_sensitivity_a_per_torr",),
    ("total_pressure_sensitivity_a_per_torr",),
    ("n2_ar_sensitivity_ratio",),
    ("par_time_offset_h",),
    ("dark_par_threshold_umol_m2_s",),
    ("min_par_coverage",),
    ("metabolism_min_r2",),
    ("chamber_par_transmittance",),
)
PROCESSING_KEYS = [k for group in PROCESSING_GROUPS for k in group]
_POSITIVE_KEYS = (
    "chamber_area_m2",
    "chamber_volume_l",
    "chamber_height_m",
    "partial_pressure_sensitivity_a_per_torr",
    "total_pressure_sensitivity_a_per_torr",
    "n2_ar_sensitivity_ratio",
    "chamber_par_transmittance",
)

_REQUIRED = ("site", "start", "c1", "c2", "test")
_ALLOWED = ("site", "lat", "lon", "start", "end", "test", "c1", "c2", "comments", "processing")


def read_toml(path: Path) -> dict:
    """tomllib.loads, but a syntax error names the file and points at the offending line."""
    text = path.read_text(encoding="utf-8")
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        message = f"{path}: invalid TOML: {e}"
        position = _TOML_POSITION.search(str(e))
        lines = text.splitlines()
        if position and int(position[1]) <= len(lines):
            line, column = int(position[1]), int(position[2])
            source = lines[line - 1]
            message += f"\n  {line} | {source}\n  {' ' * len(str(line))} | {' ' * (column - 1)}^"
            if _UNPADDED_DATE.search(source):
                message += "\n  hint: dates must be zero-padded, e.g. 2026-08-29T21:30:00Z"
        raise ValueError(message) from None


def _to_naive_utc(value, key: str, path: Path) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{path}: {key} must be a datetime with a timezone offset, got {value!r}")
    if value.tzinfo is None:
        raise ValueError(f"{path}: {key} = {value.isoformat()} has no timezone offset -- add Z or +hh:mm")
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def parse_deployment(raw: dict, deployment_id: str, path: Path) -> dict:
    """Validate a parsed toml dict and normalize it into a deployment meta dict."""
    unknown = sorted(set(raw) - set(_ALLOWED))
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {unknown}; allowed: {list(_ALLOWED)}")
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"{path}: missing required key(s) {missing}")

    if not isinstance(raw["site"], str):
        raise ValueError(f"{path}: site must be a string")
    if not isinstance(raw["test"], bool):
        raise ValueError(f"{path}: test must be true or false")
    for key in ("c1", "c2"):
        if raw[key] not in CONDITIONS:
            raise ValueError(f"{path}: {key} = {raw[key]!r} must be one of {list(CONDITIONS)}")
    for key in ("lat", "lon"):
        if key in raw and (isinstance(raw[key], bool) or not isinstance(raw[key], (int, float))):
            raise ValueError(f"{path}: {key} must be a number in decimal degrees")
    if "comments" in raw and not isinstance(raw["comments"], str):
        raise ValueError(f"{path}: comments must be a string")

    processing = _parse_processing(raw.get("processing", {}), path)

    start = _to_naive_utc(raw["start"], "start", path)
    end = _to_naive_utc(raw["end"], "end", path) if "end" in raw else None
    if end is not None and end <= start:
        raise ValueError(f"{path}: end ({end}) must be after start ({start})")

    return {
        "deployment_id": deployment_id,
        "site": raw["site"],
        "lat": float(raw["lat"]) if "lat" in raw else None,
        "lon": float(raw["lon"]) if "lon" in raw else None,
        "start": start,
        "end": end,
        "test": raw["test"],
        "c1": raw["c1"],
        "c2": raw["c2"],
        "comments": raw.get("comments"),
        "processing": processing,
    }


def check_processing_value(key: str, value, where: str) -> float:
    if key in _POSITIVE_KEYS:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{where} {key} must be a positive number, got {value!r}")
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} {key} must be a number, got {value!r}")
    return float(value)


def _parse_processing(raw, path: Path) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: processing must be a [processing] table")
    unknown = sorted(set(raw) - set(PROCESSING_KEYS))
    if unknown:
        raise ValueError(f"{path}: unknown [processing] key(s) {unknown}; allowed: {PROCESSING_KEYS}")
    for group in PROCESSING_GROUPS:
        given = [k for k in group if k in raw]
        if len(given) > 1:
            raise ValueError(f"{path}: [processing] give only one of {given}")
    return {k: check_processing_value(k, v, f"{path}: [processing]") for k, v in raw.items()}


def resolve_processing(
    meta: dict | None, overrides: dict, defaults: dict, config: dict | None = None
) -> dict[str, tuple[float, str]]:
    """Each processing parameter's value and where it came from: argument > deployment.toml > egcf.toml > default.

    Resolved per group, so e.g. --chamber-height-m beats a toml chamber_volume_l
    rather than leaving both set; the result has one key per group. An override
    of None means "not given"; defaults must supply one key per group. config
    is the egcf.toml settings; its non-processing keys are ignored.
    """
    from_toml = meta["processing"] if meta is not None else {}
    given = {k: v for k, v in overrides.items() if v is not None}
    sources = (
        (given, "argument"),
        (from_toml, DEPLOYMENT_FILENAME),
        (config or {}, "egcf.toml"),
        (defaults, "default"),
    )
    resolved = {}
    for group in PROCESSING_GROUPS:
        for values, source in sources:
            hits = [k for k in group if k in values]
            if len(hits) > 1:
                raise ValueError(f"give only one of {hits} ({source})")
            if hits:
                resolved[hits[0]] = (float(values[hits[0]]), source)
                break
    return resolved


def load_deployment(raw_dir: Path) -> dict | None:
    """Read and validate raw_dir/deployment.toml; None if the folder has none."""
    path = Path(raw_dir) / DEPLOYMENT_FILENAME
    if not path.exists():
        return None
    return parse_deployment(read_toml(path), Path(raw_dir).resolve().name, path)


def find_deployments(root: Path) -> list[Path]:
    """Every folder at or below root holding a deployment.toml, sorted by path.

    Folders without a toml (legacy layouts, loose PAR/HOBO folders) are not
    deployments and are skipped. A deployment nested inside another, or two
    folders with the same name (= deployment_id), is an error: discovery is
    recursive, so the outer one would ingest the inner one's files, and equal
    ids would write to the same output folder.
    """
    found = sorted(path.parent for path in Path(root).rglob(DEPLOYMENT_FILENAME))
    for outer in found:
        nested = [p for p in found if p != outer and outer in p.parents]
        if nested:
            raise ValueError(f"deployment {outer} contains other deployment(s) {nested}; move them out")
    names = [p.name for p in found]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise ValueError(f"deployment id(s) {duplicated} used by more than one folder under {root}")
    return found


def deployment_table(meta: dict) -> pl.DataFrame:
    return pl.DataFrame([{k: meta[k] for k in DEPLOYMENT_SCHEMA}], schema=DEPLOYMENT_SCHEMA)


def processing_table(meta: dict, values: dict) -> pl.DataFrame:
    """One row recording the processing parameters a run actually used (given and derived)."""
    return pl.DataFrame([{"deployment_id": meta["deployment_id"], **values}], schema=PROCESSING_SCHEMA)


def clip(df: pl.DataFrame, meta: dict | None, ts_col: str = "ts") -> pl.DataFrame:
    """Keep rows with ts_col in the deployment window [start, end); df unchanged if meta is None."""
    if meta is None:
        return df
    keep = pl.col(ts_col) >= meta["start"]
    if meta["end"] is not None:
        keep = keep & (pl.col(ts_col) < meta["end"])
    return df.filter(keep)


def chamber_condition_expr(meta: dict) -> pl.Expr:
    """Map chamber C1/C2 to the deployment's c1/c2 treatment, as a `condition` column."""
    return (
        pl.when(pl.col("chamber") == "C1")
        .then(pl.lit(meta["c1"]))
        .when(pl.col("chamber") == "C2")
        .then(pl.lit(meta["c2"]))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("condition")
    )
