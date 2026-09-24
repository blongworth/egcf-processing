"""Parse and calibrate Odyssey integrating light logger exports (PAR).

The Odyssey is a separately clocked, separately recovered logger, not a
payload in the lander's line grammar, so it gets its own reader (like
events.py) feeding a Layer A ``par`` table.

Export format (``ESL-EGCF_011_001.CSV``): a UTF-8 BOM, ``key ,value`` header
lines (``Site Name``, ``Logger Serial Number``, ...), two column-header rows
(``Scan No ,Date and Time,...`` / ``...,RAW VALUE ,CALIBRATED VALUE,``), then
``n,dd/mm/yyyy , HH:MM:SS,raw,cal`` rows with CRLF line endings.

Calibration is a port of CRISPEE's loadPAR.m: ``max(0, slope*raw + intercept)``,
with per-serial coefficients kept in par_calibrations.csv rather than in code.
The Odyssey integrates over its logging interval, so raw counts depend on it;
a calibration fit at a different interval is refused, never rescaled.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from egcf_processing.flux import ols_fit

logger = logging.getLogger(__name__)

DEFAULT_CALIBRATIONS_PATH = Path(__file__).parent / "par_calibrations.csv"

PAR_SCHEMA = {
    "ts": pl.Datetime,
    "scan_no": pl.Int64,
    "par_raw": pl.Float64,
    "par_umol_m2_s": pl.Float64,
    "serial_number": pl.Utf8,
    "sensor_number": pl.Int64,
    "cal_date": pl.Date,
    "interval_s": pl.Float64,
}

PAR_DAILY_SCHEMA = {
    "date": pl.Date,
    "n_readings": pl.Int64,
    "coverage": pl.Float64,
    "dli_mol_m2_d": pl.Float64,
    "max_par_umol_m2_s": pl.Float64,
}

SECONDS_PER_DAY = 86_400
DEFAULT_MIN_DAY_COVERAGE = 0.9

CALIBRATION_SCHEMA = {
    "sensor_number": pl.Int64,
    "serial_number": pl.Utf8,
    "cal_date": pl.Date,
    "interval_s": pl.Float64,
    "slope": pl.Float64,
    "intercept": pl.Float64,
    "notes": pl.Utf8,
}

_DATE_FORMAT = "%d/%m/%Y %H:%M:%S"


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8-sig", errors="replace").splitlines()


def is_odyssey_export(path: Path) -> bool:
    """True if the file's header identifies it as an Odyssey logger export."""
    try:
        with path.open(encoding="utf-8-sig", errors="replace") as f:
            head = [f.readline() for _ in range(10)]
    except OSError:
        return False
    if not head[0].startswith("Site Name"):
        return False
    return any(line.startswith("Logger Serial Number") for line in head)


def read_odyssey_file(path: Path, time_offset_h: float = 0.0) -> pl.DataFrame:
    """Parse one Odyssey export into ts/scan_no/par_raw/serial_number rows.

    ``time_offset_h`` is added to every logger timestamp (loadPAR.m's
    ``timeShift``) -- the logger clock is independent of the lander's.
    """
    serial = None
    rows = []
    skipped = 0
    in_data = False
    offset = timedelta(hours=time_offset_h)
    for line in _read_lines(path):
        if not in_data:
            if line.startswith("Logger Serial Number"):
                serial = line.split(",", 1)[1].strip() if "," in line else None
            elif "RAW VALUE" in line:
                in_data = True
            continue
        if not line.strip():
            continue
        fields = [f.strip() for f in line.split(",")]
        try:
            ts = datetime.strptime(f"{fields[1]} {fields[2]}", _DATE_FORMAT) + offset
            rows.append({"ts": ts, "scan_no": int(fields[0]), "par_raw": float(fields[3])})
        except (IndexError, ValueError):
            skipped += 1
    if skipped:
        logger.debug("%s: skipped %d malformed lines", path.name, skipped)
    schema = {k: PAR_SCHEMA[k] for k in ("ts", "scan_no", "par_raw")}
    df = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
    return df.with_columns(pl.lit(serial, dtype=pl.Utf8).alias("serial_number"))


def load_calibrations(path: Path = DEFAULT_CALIBRATIONS_PATH) -> pl.DataFrame:
    return pl.read_csv(path, schema=CALIBRATION_SCHEMA)


def select_calibration(cals: pl.DataFrame, serial: str | None, first_ts: datetime) -> dict | None:
    """Latest calibration for ``serial`` dated on or before ``first_ts``.

    A blank cal_date sorts as the oldest, so any dated calibration added
    later takes precedence over it.
    """
    if serial is None:
        return None
    candidates = cals.filter(
        (pl.col("serial_number") == serial)
        & (pl.col("cal_date").is_null() | (pl.col("cal_date") <= first_ts.date()))
    ).sort("cal_date", nulls_last=False)
    if candidates.is_empty():
        return None
    return candidates.row(-1, named=True)


def logging_interval_s(df: pl.DataFrame) -> float | None:
    """The file's logging interval: median spacing between consecutive timestamps."""
    if df.height < 2:
        return None
    return df["ts"].diff().drop_nulls().dt.total_microseconds().median() / 1_000_000


def calibrate(df: pl.DataFrame, slope: float, intercept: float) -> pl.DataFrame:
    """PAR in umol photons m^-2 s^-1, clamped at 0 (as loadPAR.m)."""
    par = (pl.col("par_raw") * slope + intercept).clip(lower_bound=0.0)
    return df.with_columns(par.alias("par_umol_m2_s"))


def _calibrate_file(path: Path, df: pl.DataFrame, cals: pl.DataFrame) -> pl.DataFrame:
    serial = df["serial_number"][0] if not df.is_empty() else None
    interval = logging_interval_s(df)
    cal = select_calibration(cals, serial, df["ts"].min()) if not df.is_empty() else None
    df = df.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("par_umol_m2_s"),
        pl.lit(cal["sensor_number"] if cal else None, dtype=pl.Int64).alias("sensor_number"),
        pl.lit(cal["cal_date"] if cal else None, dtype=pl.Date).alias("cal_date"),
        pl.lit(interval, dtype=pl.Float64).alias("interval_s"),
    )
    if cal is None:
        logger.warning("%s: no PAR calibration for serial %s; par_umol_m2_s left null", path.name, serial)
        return df
    if cal["interval_s"] is None:
        logger.warning(
            "%s: calibration for serial %s has no interval_s; applying it without checking against the "
            "file's %.0f s logging interval",
            path.name,
            serial,
            interval if interval is not None else float("nan"),
        )
    elif interval is None or cal["interval_s"] != interval:
        logger.warning(
            "%s: calibration for serial %s was fit at %.0f s but the file logs at %s s; "
            "par_umol_m2_s left null (counts are not rescaled)",
            path.name,
            serial,
            cal["interval_s"],
            interval,
        )
        return df
    logger.info(
        "%s: serial %s -> sensor %s, slope=%g intercept=%g, cal_date=%s",
        path.name,
        serial,
        cal["sensor_number"],
        cal["slope"],
        cal["intercept"],
        cal["cal_date"],
    )
    return calibrate(df, cal["slope"], cal["intercept"])


def read_all_par(
    paths: list[Path],
    calibrations: pl.DataFrame,
    time_offset_h: float = 0.0,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    """Read, calibrate, and concatenate Odyssey exports, optionally trimmed to [start, end)."""
    frames = []
    for path in paths:
        df = read_odyssey_file(path, time_offset_h)
        frames.append(_calibrate_file(path, df, calibrations).select(list(PAR_SCHEMA)))
    par = pl.concat(frames) if frames else pl.DataFrame(schema=PAR_SCHEMA)
    if start is not None:
        par = par.filter(pl.col("ts") >= start)
    if end is not None:
        par = par.filter(pl.col("ts") < end)
    return par


def daily_par(par: pl.DataFrame) -> pl.DataFrame:
    """Per-UTC-day light QC: daily light integral and daily maximum PAR.

    UTC day boundaries fall at ~20:00 EDT, after Woods Hole sunset, so each
    UTC day holds one whole photoperiod. ``dli_mol_m2_d`` sums calibrated
    readings times their interval and is not extrapolated over gaps;
    ``coverage`` (logged seconds / 86400, capped at 1) marks the partial first
    and last days. Uncalibrated days get null DLI and max, never zeros.
    """
    if par.is_empty():
        return pl.DataFrame(schema=PAR_DAILY_SCHEMA)
    calibrated = pl.col("par_umol_m2_s").is_not_null()
    return (
        par.group_by(pl.col("ts").dt.date().alias("date"))
        .agg(
            pl.len().cast(pl.Int64).alias("n_readings"),
            (pl.col("interval_s").sum() / SECONDS_PER_DAY).clip(upper_bound=1.0).alias("coverage"),
            pl.when(calibrated.any())
            .then((pl.col("par_umol_m2_s") * pl.col("interval_s")).sum() / 1e6)
            .alias("dli_mol_m2_d"),
            pl.col("par_umol_m2_s").max().alias("max_par_umol_m2_s"),
        )
        .cast(PAR_DAILY_SCHEMA)
        .sort("date")
    )


def daily_max_trend(daily: pl.DataFrame, min_coverage: float = DEFAULT_MIN_DAY_COVERAGE) -> dict | None:
    """OLS trend of daily max PAR over full days, a screen for diffuser biofouling.

    Fouling reads progressively low, so a sustained decline in the daily
    maximum is the signature -- but cloudy days lower it too, so this flags
    days to inspect rather than proving fouling. Returns slope in
    umol m^-2 s^-1 per day and as a percent of the mean daily max, or None
    with fewer than two full days.
    """
    full = daily.filter((pl.col("coverage") >= min_coverage) & pl.col("max_par_umol_m2_s").is_not_null())
    if full.height < 2:
        return None
    day0 = full["date"].min()
    x = [(d - day0).days for d in full["date"].to_list()]
    fit = ols_fit(x, full["max_par_umol_m2_s"].to_list())
    if fit is None:
        return None
    mean_max = full["max_par_umol_m2_s"].mean()
    return {
        "slope_umol_m2_s_per_day": fit["slope"],
        "intercept_umol_m2_s": fit["intercept"],
        "pct_per_day": 100 * fit["slope"] / mean_max if mean_max else None,
        "r2": fit["r2"],
        "n_days": fit["n"],
        "first_date": day0,
    }
