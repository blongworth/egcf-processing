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

import tomllib
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

DEPLOYMENT_FILENAME = "deployment.toml"

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

_REQUIRED = ("site", "start", "c1", "c2", "test")
_ALLOWED = ("site", "lat", "lon", "start", "end", "test", "c1", "c2", "comments")


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
    }


def load_deployment(raw_dir: Path) -> dict | None:
    """Read and validate raw_dir/deployment.toml; None if the folder has none."""
    path = Path(raw_dir) / DEPLOYMENT_FILENAME
    if not path.exists():
        return None
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return parse_deployment(raw, Path(raw_dir).resolve().name, path)


def deployment_table(meta: dict) -> pl.DataFrame:
    return pl.DataFrame([{k: meta[k] for k in DEPLOYMENT_SCHEMA}], schema=DEPLOYMENT_SCHEMA)


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
