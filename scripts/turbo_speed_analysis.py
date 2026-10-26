"""Turbo-speed RGA performance test: tag RGA/TP readings with the turbo setpoint in
effect, drop a settle period after acquisition restarts (OK,AON) following each
change, and report per-mass mean/RSD per speed plus speed vs total pressure.

    uv run scripts/turbo_speed_analysis.py <raw_dir> --out-dir <dir> [--settle-min 10]
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import plotly.graph_objects as go
import numpy as np
import polars as pl
from plotly.subplots import make_subplots
from scipy import optimize, stats

from egcf_processing.aggregate import (
    DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR,
    DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR,
    RAW_CURRENT_AMPS_PER_COUNT,
)
from egcf_processing.combine import build_tables, write_df
from egcf_processing.discovery import find_all_files, find_surface_events_files
from egcf_processing.lines import _parse_ts
from egcf_processing.reader import read_all

_CMD_RE = re.compile(r"\bSPD(\d{3,4})\b")
_ACK_RE = re.compile(r"\bOK,SPD(\d{3,4})?\b")
_AON_RE = re.compile(r"\bOK,AON\b")
_LEADING_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T[\d:.]+Z?)")
_TS_KEYS = {"SPD": "speed_hz", "PWR": "power_w", "ETEMP": "etemp_c", "BTEMP": "btemp_c", "MTEMP": "mtemp_c", "TP": "tp_raw"}

REF_MASS = 40

_LEG_COLORS = {"down": "#2a78d6", "up": "#eb6834"}
_TEXT_SECONDARY = "#52514e"
_GRID = "#e4e3df"

SPEED_CHANGES_SCHEMA = {"ts": pl.Datetime, "setpoint": pl.Int64, "source": pl.Utf8, "source_file": pl.Utf8}
TURBO_SCHEMA = {
    "ts": pl.Datetime,
    "source": pl.Utf8,
    "speed_hz": pl.Float64,
    "power_w": pl.Float64,
    "etemp_c": pl.Float64,
    "btemp_c": pl.Float64,
    "mtemp_c": pl.Float64,
    "tp_raw": pl.Float64,
}


def _iter_timestamped_lines(raw_dir: Path, commands_log: Path | None):
    """Yield (ts, payload, source_file) from lander/gems logs, events logs, and the stepper log.

    Surface and events lines carry a leading receipt timestamp; TS, and OK, payloads
    carry no embedded one, so the receipt time is the only time available. gems lines
    are used only if the payload itself starts with a timestamp.
    """
    for path in find_all_files(raw_dir):
        is_surface = path.name.startswith("surface_")
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if is_surface:
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2:
                        continue
                    ts = _parse_ts(parts[0])
                    payload = parts[1]
                else:
                    match = _LEADING_TS_RE.match(line)
                    ts = _parse_ts(match.group(1)) if match else None
                    payload = line
                if ts is not None:
                    yield ts, payload, path.name
    extra = [commands_log] if commands_log is not None else []
    for path in [*find_surface_events_files(raw_dir), *extra]:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split(maxsplit=2)
                if len(parts) != 3 or parts[0].startswith("#"):
                    continue
                ts = _parse_ts(parts[0])
                if ts is not None:
                    yield ts, parts[2], path.name


def scan_raw(raw_dir: Path, commands_log: Path | None) -> tuple[list[dict], list[dict], list[datetime], list[dict]]:
    """Return (commands, acks, aon_acks, ts_rows) found in the raw lines."""
    commands, acks, aon_acks, ts_rows = [], [], [], []
    for ts, payload, source_file in _iter_timestamped_lines(raw_dir, commands_log):
        ack = _ACK_RE.search(payload)
        if ack:
            setpoint = int(ack.group(1)) if ack.group(1) else None
            acks.append({"ts": ts, "setpoint": setpoint, "source_file": source_file})
            continue
        if _AON_RE.search(payload):
            aon_acks.append(ts)
            continue
        cmd = _CMD_RE.search(payload)
        if cmd:
            commands.append({"ts": ts, "setpoint": int(cmd.group(1)), "source_file": source_file})
            continue
        if payload.startswith("TS,"):
            row = {"ts": ts, "source": "TS"}
            for field in payload[3:].split(","):
                key, _, val = field.partition("=")
                if key in _TS_KEYS:
                    try:
                        row[_TS_KEYS[key]] = float(val)
                    except ValueError:
                        row[_TS_KEYS[key]] = None
            ts_rows.append(row)
    return commands, acks, aon_acks, ts_rows


def build_speed_changes(commands: list[dict], acks: list[dict], ack_window_s: float) -> pl.DataFrame:
    """Turn raw command/ack sightings into one row per setpoint change.

    The same command is seen several times (RX_CONSOLE, TX_LANDER, the stepper log), so
    repeats of a setpoint within ack_window_s are collapsed. A digitless ack takes the
    setpoint of the most recent command. A command with no ack within ack_window_s falls
    back to the command time, with a warning.
    """
    window = timedelta(seconds=ack_window_s)
    commands = sorted(commands, key=lambda r: r["ts"])
    acks = sorted(acks, key=lambda r: r["ts"])

    requests: list[dict] = []
    for cmd in commands:
        prev = requests[-1] if requests else None
        if prev and prev["setpoint"] == cmd["setpoint"] and cmd["ts"] - prev["ts"] <= window:
            continue
        requests.append(cmd)

    accepted: list[dict] = []
    for ack in acks:
        setpoint = ack["setpoint"]
        if setpoint is None:
            preceding = [c for c in requests if c["ts"] <= ack["ts"]]
            if not preceding:
                print(f"WARNING: ack at {ack['ts']} ({ack['source_file']}) has no setpoint and no preceding command; ignored", file=sys.stderr)
                continue
            setpoint = preceding[-1]["setpoint"]
        prev = accepted[-1] if accepted else None
        if prev and prev["setpoint"] == setpoint and ack["ts"] - prev["ts"] <= window:
            continue
        accepted.append({"ts": ack["ts"], "setpoint": setpoint, "source": "ack", "source_file": ack["source_file"]})

    changes = list(accepted)
    for req in requests:
        matched = any(
            a["setpoint"] == req["setpoint"] and req["ts"] - window <= a["ts"] <= req["ts"] + window for a in accepted
        )
        if not matched:
            print(
                f"WARNING: SPD{req['setpoint']} at {req['ts']} ({req['source_file']}) has no OK,SPD ack within "
                f"{ack_window_s:g} s; using command time",
                file=sys.stderr,
            )
            changes.append({**req, "source": "command"})

    changes.sort(key=lambda r: r["ts"])
    deduped: list[dict] = []
    for change in changes:
        if deduped and deduped[-1]["setpoint"] == change["setpoint"]:
            continue
        deduped.append(change)
    df = pl.DataFrame(deduped, schema=SPEED_CHANGES_SCHEMA) if deduped else pl.DataFrame(schema=SPEED_CHANGES_SCHEMA)
    return add_segments(df)


def add_segments(changes: pl.DataFrame) -> pl.DataFrame:
    """Number each hold and label it as part of the down or up leg of the ramp.

    A hold's leg is the direction of the change into it; the first hold takes the leg
    of the change out of it.
    """
    diff = pl.col("setpoint").diff()
    leg = pl.when(diff < 0).then(pl.lit("down")).when(diff > 0).then(pl.lit("up"))
    changes = changes.with_columns(pl.int_range(pl.len()).alias("segment"), leg.alias("leg"))
    if changes.height > 1 and changes["leg"][0] is None:
        changes = changes.with_columns(pl.col("leg").backward_fill())
    return changes.with_columns(pl.col("leg").fill_null("down"))


def build_turbo(ts_rows: list[dict], status: pl.DataFrame) -> pl.DataFrame:
    ts_df = pl.DataFrame(ts_rows, schema=TURBO_SCHEMA) if ts_rows else pl.DataFrame(schema=TURBO_SCHEMA)
    status_df = status.select(
        "ts",
        pl.lit("status").alias("source"),
        pl.col("turbo_speed_hz").alias("speed_hz"),
        pl.col("turbo_power_w").alias("power_w"),
        pl.col("turbo_etemp_c").alias("etemp_c"),
        pl.col("turbo_btemp_c").alias("btemp_c"),
        pl.col("turbo_mtemp_c").alias("mtemp_c"),
        pl.col("raw_total_pressure_current").alias("tp_raw"),
    ).cast(TURBO_SCHEMA)
    return pl.concat([ts_df, status_df]).filter(pl.col("tp_raw").is_not_null() | pl.col("speed_hz").is_not_null())


def add_acq_on(changes: pl.DataFrame, aon_acks: list[datetime]) -> pl.DataFrame:
    """Attach the first OK,AON after each change and before the next one as acq_on_ts.

    The speed is changed with acquisition off (AOFF, SPD, wait for TURBO=ready, AON), so
    the settle period is timed from acquisition restarting. A change with no AON ack
    falls back to the change time, with a warning.
    """
    aon = pl.DataFrame({"acq_on_ts": sorted(set(aon_acks))}, schema={"acq_on_ts": pl.Datetime})
    out = (
        changes.sort("ts")
        .join_asof(aon, left_on="ts", right_on="acq_on_ts", strategy="forward")
        .with_columns(
            pl.when(pl.col("ts").shift(-1).is_null() | (pl.col("acq_on_ts") < pl.col("ts").shift(-1)))
            .then(pl.col("acq_on_ts"))
            .alias("acq_on_ts")
        )
    )
    for row in out.filter(pl.col("acq_on_ts").is_null()).iter_rows(named=True):
        print(
            f"WARNING: SPD{row['setpoint']} change at {row['ts']} has no OK,AON before the next change; "
            "settle timed from the change instead",
            file=sys.stderr,
        )
    return out


def tag(readings: pl.DataFrame, changes: pl.DataFrame, settle_s: float) -> pl.DataFrame:
    """Attach the setpoint in effect to each reading; drop readings before the first change."""
    right = changes.select(
        pl.col("ts").alias("change_ts"), "acq_on_ts", "setpoint", "segment", "leg", pl.col("source").alias("change_source")
    )
    tagged = readings.sort("ts").join_asof(right, left_on="ts", right_on="change_ts", strategy="backward")
    t_since = (pl.col("ts") - pl.col("change_ts")).dt.total_microseconds() / 1e6
    return (
        tagged.filter(pl.col("setpoint").is_not_null())
        .with_columns(
            t_since.alias("t_since_change_s"),
            ((pl.col("ts") - pl.coalesce("acq_on_ts", "change_ts")).dt.total_microseconds() / 1e6).alias("t_since_acq_on_s"),
        )
        .with_columns((pl.col("t_since_acq_on_s") < settle_s).alias("in_settle"))
    )


def _stats(col: str, prefix: str) -> list[pl.Expr]:
    return [
        pl.col(col).mean().alias(f"{prefix}_mean"),
        pl.col(col).std().alias(f"{prefix}_std"),
    ]


def mass_stats(rga_tagged: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Per-mass mean/std, plus RSD two ways.

    rsd_pct uses the std over the hold, so slow drift counts as noise. rsd_diff_pct uses
    the std of successive differences within a segment / sqrt(2), which estimates the
    reading-to-reading (white) noise and is insensitive to drift.
    """
    kept = rga_tagged.filter(~pl.col("in_settle")).with_columns(
        pl.col("current").diff().over(["segment", "mass"]).alias("current_diff")
    )
    return (
        kept.group_by([*keys, "mass"])
        .agg(
            pl.len().alias("n"),
            *_stats("current", "current_raw"),
            (pl.col("current_diff").std() / math.sqrt(2)).alias("current_raw_noise"),
            *_stats("amps", "amps"),
            *_stats("torr", "torr"),
        )
        .with_columns(
            (100 * pl.col("current_raw_std") / pl.col("current_raw_mean")).alias("rsd_pct"),
            (100 * pl.col("current_raw_noise") / pl.col("current_raw_mean")).alias("rsd_diff_pct"),
        )
        .sort([*keys, "mass"])
    )


def add_ratio(rga_tagged: pl.DataFrame, ref_mass: int) -> pl.DataFrame:
    """Add scan_id and ratio = current / the ref_mass current from the same scan.

    Scans are split where a mass repeats, as in rga_scans.rga_scan_windows. ratio is
    null where the scan has no ref_mass reading or it is zero.
    """
    seen: set[int] = set()
    scan_ids, scan_id = [], 0
    for mass in rga_tagged["mass"]:
        if mass in seen:
            scan_id += 1
            seen = set()
        seen.add(mass)
        scan_ids.append(scan_id)
    r = rga_tagged.with_columns(pl.Series("scan_id", scan_ids, dtype=pl.Int64))
    ref = r.filter(pl.col("mass") == ref_mass).select("scan_id", pl.col("current").alias("_ref"))
    return (
        r.join(ref, on="scan_id", how="left", maintain_order="left")
        .with_columns(pl.when(pl.col("_ref") != 0).then(pl.col("current") / pl.col("_ref")).alias("ratio"))
        .drop("_ref")
    )


def ratio_stats(rga_tagged: pl.DataFrame, keys: list[str], ref_mass: int) -> pl.DataFrame:
    """Per-mass mean/std of the ratio to ref_mass, with rsd_pct and rsd_diff_pct as in mass_stats."""
    kept = rga_tagged.filter(
        ~pl.col("in_settle") & (pl.col("mass") != ref_mass) & pl.col("ratio").is_not_null()
    ).with_columns(pl.col("ratio").diff().over(["segment", "mass"]).alias("ratio_diff"))
    return (
        kept.group_by([*keys, "mass"])
        .agg(
            pl.len().alias("n"),
            *_stats("ratio", "ratio"),
            (pl.col("ratio_diff").std() / math.sqrt(2)).alias("ratio_noise"),
        )
        .with_columns(
            (100 * pl.col("ratio_std") / pl.col("ratio_mean")).alias("rsd_pct"),
            (100 * pl.col("ratio_noise") / pl.col("ratio_mean")).alias("rsd_diff_pct"),
        )
        .sort([*keys, "mass"])
    )


RSD_TESTS_SCHEMA = {
    "mass": pl.Int64,
    "test": pl.Utf8,
    "n_setpoints": pl.Int64,
    "n": pl.Int64,
    "statistic": pl.Float64,
    "p_value": pl.Float64,
    "p_adj": pl.Float64,
}


def rsd_speed_tests(rga_tagged: pl.DataFrame, col: str = "current", relative: bool = True) -> pl.DataFrame:
    """Test whether turbo setpoint affects drift-robust RSD of col ("current" or "ratio"),
    or with relative=False, the absolute noise.

    Per mass: Brown-Forsythe test for equal spread of the successive differences across
    setpoints, divided by the hold mean when relative. Only non-overlapping pairs (x1-x0, x3-x2, ...)
    are used, since overlapping differences are correlated. p_adj is Benjamini-Hochberg
    across masses.
    """
    kept = rga_tagged.filter(~pl.col("in_settle") & pl.col(col).is_not_null()).with_columns(
        (pl.col(col).diff() / (pl.col(col).mean() if relative else 1)).over(["segment", "mass"]).alias("rel_diff"),
        pl.int_range(pl.len()).over(["segment", "mass"]).alias("i"),
    )
    pairs = kept.filter((pl.col("i") % 2 == 1) & pl.col("rel_diff").is_finite())
    rows = []
    for (mass,), d in pairs.group_by("mass"):
        groups = [g["rel_diff"].to_numpy() for _, g in d.group_by("setpoint") if g.height >= 3]
        if len(groups) < 2:
            continue
        res = stats.levene(*groups, center="median")
        rows.append({"mass": mass, "test": "Brown-Forsythe", "n_setpoints": len(groups), "n": sum(map(len, groups)),
                     "statistic": float(res.statistic), "p_value": float(res.pvalue)})
    rows.sort(key=lambda r: r["mass"])
    if rows:
        for r, adj in zip(rows, stats.false_discovery_control([r["p_value"] for r in rows])):
            r["p_adj"] = float(adj)
    return pl.DataFrame(rows, schema=RSD_TESTS_SCHEMA) if rows else pl.DataFrame(schema=RSD_TESTS_SCHEMA)


def noise_floor_summary(
    mass_by_speed: pl.DataFrame, rsd_tests: pl.DataFrame, noise_tests: pl.DataFrame
) -> pl.DataFrame:
    """Per mass, signal, drift-robust noise and RSD at the lowest setpoint relative to the
    highest, with the speed-effect p-values for RSD and absolute noise.

    A noise-floor-limited mass shows signal_lo_hi > 1, noise_lo_hi ~ 1 and
    rsd_lo_hi ~ 1 / signal_lo_hi; a mass limited by noise proportional to signal shows
    noise_lo_hi ~ signal_lo_hi and rsd_lo_hi ~ 1.
    """
    d = mass_by_speed.filter(pl.col("current_raw_noise") > 0)
    if d.is_empty():
        return pl.DataFrame(
            schema={"mass": pl.Int64, "setpoint_lo": pl.Int64, "setpoint_hi": pl.Int64, "signal_lo_hi": pl.Float64,
                    "noise_lo_hi": pl.Float64, "rsd_lo_hi": pl.Float64, "p_adj_rsd": pl.Float64, "p_adj_noise": pl.Float64}
        )
    lo, hi = d["setpoint"].min(), d["setpoint"].max()
    at = lambda sp: d.filter(pl.col("setpoint") == sp).select("mass", "current_raw_mean", "current_raw_noise", "rsd_diff_pct")
    return (
        at(lo).join(at(hi), on="mass", suffix="_hi")
        .select(
            "mass",
            pl.lit(lo, dtype=pl.Int64).alias("setpoint_lo"),
            pl.lit(hi, dtype=pl.Int64).alias("setpoint_hi"),
            (pl.col("current_raw_mean") / pl.col("current_raw_mean_hi")).alias("signal_lo_hi"),
            (pl.col("current_raw_noise") / pl.col("current_raw_noise_hi")).alias("noise_lo_hi"),
            (pl.col("rsd_diff_pct") / pl.col("rsd_diff_pct_hi")).alias("rsd_lo_hi"),
        )
        .join(rsd_tests.select("mass", pl.col("p_adj").alias("p_adj_rsd")), on="mass", how="left")
        .join(noise_tests.select("mass", pl.col("p_adj").alias("p_adj_noise")), on="mass", how="left")
        .sort("mass")
    )


def fit_noise_model(seg_stats: pl.DataFrame) -> dict[str, float] | None:
    """Fit noise^2 = floor^2 + shot * S + (prop * S)^2 (raw counts) to every mass and hold,
    in log space so small and large signals weigh equally. None if under 3 holds."""
    d = seg_stats.filter((pl.col("current_raw_noise") > 0) & (pl.col("current_raw_mean") > 0))
    if d.height < 3:
        return None
    sig = d["current_raw_mean"].to_numpy()
    noise = d["current_raw_noise"].to_numpy()

    def resid(p):
        return np.log(noise) - 0.5 * np.log(p[0] ** 2 + p[1] * sig + (p[2] * sig) ** 2)

    x0 = [noise.min(), noise.min() ** 2 / np.median(sig), np.median(noise / sig)]
    fit = optimize.least_squares(resid, x0, bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), x_scale="jac")
    floor, shot, prop = fit.x
    return {"floor": float(floor), "shot": float(shot), "prop": float(prop)}


def tp_stats(turbo_tagged: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    kept = turbo_tagged.filter(~pl.col("in_settle"))
    return (
        kept.group_by(keys)
        .agg(
            pl.col("tp_raw").is_not_null().sum().alias("n"),
            *_stats("speed_hz", "speed_hz"),
            *_stats("power_w", "power_w"),
            *_stats("tp_raw", "tp_raw"),
            *_stats("tp_amps", "tp_amps"),
            *_stats("tp_torr", "tp_torr"),
        )
        .with_columns((100 * pl.col("tp_raw_std") / pl.col("tp_raw_mean")).alias("tp_rsd_pct"))
        .sort(keys)
    )


def analyze(
    tables: dict[str, pl.DataFrame],
    changes: pl.DataFrame,
    ts_rows: list[dict],
    start: datetime | None,
    end: datetime | None,
    settle_s: float,
    pp_sens: float,
    tp_sens: float,
    ref_mass: int = REF_MASS,
) -> dict[str, pl.DataFrame]:
    """Window the changes and readings to [start, end], then tag the readings with the
    setpoint in effect and compute the per-speed and per-segment stats.

    Changes are windowed too, so readings between start and the first change inside
    the window have no setpoint and are dropped.
    """

    def window(df: pl.DataFrame) -> pl.DataFrame:
        if start is not None:
            df = df.filter(pl.col("ts") >= start)
        if end is not None:
            df = df.filter(pl.col("ts") <= end)
        return df

    changes = add_segments(window(changes).drop("segment", "leg")).select(changes.columns)
    rga = window(tables["rga"]).with_columns((pl.col("current") * RAW_CURRENT_AMPS_PER_COUNT).alias("amps"))
    rga = rga.with_columns((pl.col("amps") / pp_sens).alias("torr"))
    turbo = window(build_turbo(ts_rows, tables["status"])).with_columns(
        (pl.col("tp_raw") * RAW_CURRENT_AMPS_PER_COUNT).alias("tp_amps")
    )
    turbo = turbo.with_columns((pl.col("tp_amps") / tp_sens).alias("tp_torr"))

    rga_tagged = add_ratio(tag(rga, changes, settle_s), ref_mass)
    turbo_tagged = tag(turbo, changes, settle_s)
    seg_keys = ["segment", "leg", "setpoint", "change_ts"]
    mass_by_speed = mass_stats(rga_tagged, ["setpoint"])
    rsd_tests = rsd_speed_tests(rga_tagged)
    noise_tests = rsd_speed_tests(rga_tagged, relative=False)
    return {
        "speed_changes": changes,
        "rga_tagged": rga_tagged,
        "turbo_tagged": turbo_tagged,
        "mass_stats_by_speed": mass_by_speed,
        "mass_stats_by_segment": mass_stats(rga_tagged, seg_keys),
        "tp_by_speed": tp_stats(turbo_tagged, ["setpoint"]),
        "tp_by_segment": tp_stats(turbo_tagged, seg_keys),
        "rsd_speed_tests": rsd_tests,
        "noise_speed_tests": noise_tests,
        "noise_floor_summary": noise_floor_summary(mass_by_speed, rsd_tests, noise_tests),
        "ratio_stats_by_speed": ratio_stats(rga_tagged, ["setpoint"], ref_mass),
        "ratio_stats_by_segment": ratio_stats(rga_tagged, seg_keys, ref_mass),
        "ratio_speed_tests": rsd_speed_tests(
            rga_tagged.filter(pl.col("mass") != ref_mass), col="ratio"
        ),
    }


def _style(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        font={"color": _TEXT_SECONDARY},
        hovermode="closest",
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0, "xanchor": "left"},
    )
    fig.update_xaxes(gridcolor=_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=_GRID, zeroline=False)
    return fig


def fig_timeseries(turbo_tagged: pl.DataFrame, changes: pl.DataFrame, settle_s: float) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04)
    tp = turbo_tagged.filter(pl.col("tp_torr").is_not_null())
    fig.add_trace(
        go.Scattergl(x=tp["ts"], y=tp["tp_torr"], mode="markers", marker={"size": 3, "color": "#2a78d6"}, name="TP"),
        row=1, col=1,
    )
    spd = turbo_tagged.filter(pl.col("speed_hz").is_not_null())
    fig.add_trace(
        go.Scattergl(x=spd["ts"], y=spd["speed_hz"], mode="markers", marker={"size": 3, "color": "#2a78d6"}, name="measured speed"),
        row=2, col=1,
    )
    end = turbo_tagged["ts"].max() if turbo_tagged.height else None
    step_x = [*changes["ts"].to_list(), *([end] if end is not None and changes.height else [])]
    step_y = [*changes["setpoint"].to_list(), *(changes["setpoint"].to_list()[-1:] if end is not None else [])]
    fig.add_trace(
        go.Scatter(x=step_x, y=step_y, mode="lines", line={"shape": "hv", "width": 2, "color": "#eb6834"}, name="setpoint"),
        row=3, col=1,
    )
    for ts, acq_on in changes.select("ts", "acq_on_ts").iter_rows():
        settle_end = (acq_on or ts) + timedelta(seconds=settle_s)
        fig.add_vrect(x0=ts, x1=settle_end, fillcolor="#8a8984", opacity=0.15, line_width=0)
    fig.update_yaxes(title_text="TP (Torr)", type="log", exponentformat="E", showexponent="all", row=1, col=1)
    fig.update_yaxes(title_text="measured speed (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="setpoint", row=3, col=1)
    return _style(fig, "Total pressure, measured speed and setpoint (shaded: settle period, excluded)", 750)


def fig_mass_facets(
    seg_stats: pl.DataFrame, value: str, error: str | None, y_title: str, log_y: bool, hover_fmt: str = ".3e"
) -> go.Figure:
    masses = sorted(seg_stats["mass"].unique().to_list())
    ncols = 4
    nrows = max(1, math.ceil(len(masses) / ncols))
    fig = make_subplots(
        rows=nrows, cols=ncols, subplot_titles=[f"mass {m}" for m in masses],
        horizontal_spacing=0.06, vertical_spacing=0.08,
    )
    for i, mass in enumerate(masses):
        row, col = i // ncols + 1, i % ncols + 1
        for leg, color in _LEG_COLORS.items():
            d = seg_stats.filter((pl.col("mass") == mass) & (pl.col("leg") == leg)).sort("setpoint")
            if d.is_empty():
                continue
            fig.add_trace(
                go.Scatter(
                    x=d["setpoint"], y=d[value],
                    error_y={"type": "data", "array": d[error].to_list(), "thickness": 1} if error else None,
                    mode="lines+markers", line={"width": 2, "color": color, "dash": "solid" if leg == "down" else "dash"},
                    marker={"size": 8, "color": color, "symbol": "circle" if leg == "down" else "diamond"},
                    name=f"{leg} leg", legendgroup=leg, showlegend=i == 0,
                    customdata=d.select("segment", "n").to_numpy(),
                    hovertemplate=f"mass {mass}<br>setpoint %{{x}}<br>{y_title} %{{y:{hover_fmt}}}<br>segment %{{customdata[0]}}, n=%{{customdata[1]}}<extra>{leg}</extra>",
                ),
                row=row, col=col,
            )
        if log_y:
            fig.update_yaxes(type="log", exponentformat="E", showexponent="all", row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text=y_title, row=row, col=col)
        if row == nrows:
            fig.update_xaxes(title_text="setpoint", row=row, col=col)
    return _style(fig, f"{y_title} vs turbo setpoint, per mass (settle excluded)", 280 * nrows + 120)


def _fig_vs_speed(seg_tp: pl.DataFrame, col: str, y_title: str, hover: str, log_y: bool, title: str) -> go.Figure:
    fig = go.Figure()
    for leg, color in _LEG_COLORS.items():
        d = seg_tp.filter((pl.col("leg") == leg) & pl.col(f"{col}_mean").is_not_null()).sort("setpoint")
        if d.is_empty():
            continue
        fig.add_trace(
            go.Scatter(
                x=d["speed_hz_mean"], y=d[f"{col}_mean"],
                error_x={"type": "data", "array": d["speed_hz_std"].to_list(), "thickness": 1},
                error_y={"type": "data", "array": d[f"{col}_std"].to_list(), "thickness": 1},
                mode="lines+markers", line={"width": 2, "color": color, "dash": "solid" if leg == "down" else "dash"},
                marker={"size": 8, "color": color, "symbol": "circle" if leg == "down" else "diamond"},
                name=f"{leg} leg", customdata=d.select("segment", "n", "setpoint").to_numpy(),
                hovertemplate=f"speed %{{x:.0f}} Hz (setpoint %{{customdata[2]}})<br>{hover}"
                "<br>segment %{customdata[0]}, n=%{customdata[1]}<extra></extra>",
            )
        )
    if not fig.data:
        fig.add_annotation(text=f"no {y_title} readings", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
    if log_y:
        fig.update_yaxes(type="log", exponentformat="E", showexponent="all")
    fig.update_yaxes(title_text=y_title)
    fig.update_xaxes(title_text="measured speed (Hz)")
    return _style(fig, title, 480)


def fig_tp(seg_tp: pl.DataFrame) -> go.Figure:
    return _fig_vs_speed(
        seg_tp, "tp_torr", "TP (Torr)", "TP %{y:.3e} Torr", True,
        "Total pressure vs measured turbo speed (mean ± std, settle excluded)",
    )


def fig_power(seg_tp: pl.DataFrame) -> go.Figure:
    return _fig_vs_speed(
        seg_tp, "power_w", "pump power (W)", "power %{y:.2f} W", False,
        "Turbo pump power vs measured speed (mean ± std, settle excluded)",
    )


def _fmt_p(p: float | None) -> str:
    if p is None or math.isnan(p):
        return "n/a"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def fig_rsd(seg_stats: pl.DataFrame, tests: pl.DataFrame, label: str = "Drift-robust RSD") -> go.Figure:
    """Drift-robust RSD per mass and hold on a log colour scale, with the per-mass
    speed-effect p-values from rsd_speed_tests."""
    rsd = (
        seg_stats.filter(pl.col("rsd_diff_pct") > 0)
        .group_by("mass", "setpoint", "leg")
        .agg(pl.col("rsd_diff_pct").mean(), pl.col("rsd_pct").mean())
    )
    cols = rsd.select("setpoint", "leg").unique().sort(["setpoint", pl.col("leg") == "up"])
    col_labels = [f"{sp} {'↓' if leg == 'down' else '↑'}" for sp, leg in cols.iter_rows()]
    masses = sorted(rsd["mass"].unique().to_list())
    grid = cols.join(pl.DataFrame({"mass": masses}, schema={"mass": pl.Int64}), how="cross").join(rsd, on=["mass", "setpoint", "leg"], how="left")
    rsd_diff = [grid.filter(pl.col("mass") == m)["rsd_diff_pct"].to_list() for m in masses]
    rsd_plain = [grid.filter(pl.col("mass") == m)["rsd_pct"].to_list() for m in masses]

    p_by_mass = dict(tests.select("mass", "p_adj").iter_rows())

    all_vals = [v for r in rsd_diff for v in r if v]
    lo, hi = (math.floor(math.log10(min(all_vals))), math.ceil(math.log10(max(all_vals)))) if all_vals else (-1, 1)
    hi = max(hi, lo + 1)
    ticks = [t / 2 for t in range(2 * lo, 2 * hi + 1)]
    x = list(range(len(col_labels)))
    y = list(range(len(masses)))
    fig = go.Figure(
        go.Heatmap(
            x=x, y=y, zmin=lo, zmax=hi, colorscale="Blues", reversescale=True,
            z=[[math.log10(v) if v else None for v in r] for r in rsd_diff],
            text=[[f"{v:.2f}" if v else "" for v in r] for r in rsd_diff], texttemplate="%{text}",
            customdata=[
                [[f"m{m}", c, a, b] for c, a, b in zip(col_labels, r1, r2)] for m, r1, r2 in zip(masses, rsd_diff, rsd_plain)
            ],
            hovertemplate="%{customdata[0]} %{customdata[1]}<br>RSD (drift-robust) %{customdata[2]:.3g}%"
            "<br>RSD (incl. drift) %{customdata[3]:.3g}%<extra></extra>",
            colorbar={"title": "RSD (%)", "tickvals": ticks, "ticktext": [f"{10 ** t:.2g}" for t in ticks]},
        )
    )
    fig.update_xaxes(tickvals=x, ticktext=col_labels, title_text="setpoint (↓ down leg, ↑ up leg)")
    fig.update_yaxes(tickvals=y, ticktext=[f"m{m} ({_fmt_p(p_by_mass.get(m))})" for m in masses], autorange="reversed")
    fig = _style(
        fig,
        f"{label} (%) by mass and speed<br><sup>p: Brown-Forsythe across setpoints, BH-adjusted</sup>",
        28 * len(masses) + 220,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def fig_noise_vs_signal(seg_stats: pl.DataFrame) -> go.Figure:
    """Drift-robust noise and RSD against mean signal, one point per mass and hold, coloured
    by setpoint, with the fitted noise model. If the speed effect works only through
    signal level, every point lies on the one curve whatever its setpoint."""
    d = seg_stats.filter((pl.col("current_raw_noise") > 0) & (pl.col("current_raw_mean") > 0)).with_columns(
        (pl.col("current_raw_noise") * RAW_CURRENT_AMPS_PER_COUNT).alias("noise_amps")
    )
    fig = make_subplots(rows=1, cols=2, subplot_titles=["noise vs signal", "RSD vs signal"], horizontal_spacing=0.1)
    panels = [(1, "noise_amps"), (2, "rsd_diff_pct")]
    for (mass,), m in d.sort("mass").group_by("mass", maintain_order=True):
        m = m.sort("current_raw_mean")
        labels = [""] * (m.height - 1) + [f"m{mass}"]
        for col, y in panels:
            fig.add_trace(
                go.Scatter(
                    x=m["amps_mean"], y=m[y], mode="lines+text", text=labels, textposition="middle right",
                    line={"color": "#b4b2ac", "width": 1}, showlegend=False, hoverinfo="skip",
                ),
                row=1, col=col,
            )
    custom = d.select("mass", "setpoint", "leg", "segment", "n", "noise_amps", "rsd_diff_pct").to_numpy()
    for col, y in panels:
        fig.add_trace(
            go.Scatter(
                x=d["amps_mean"], y=d[y], mode="markers", customdata=custom, showlegend=False,
                marker={
                    "size": 9, "color": d["setpoint"], "colorscale": "Viridis", "line": {"width": 0.5, "color": "white"},
                    "symbol": ["circle" if leg == "down" else "diamond" for leg in d["leg"]],
                    "colorbar": {"title": "setpoint"} if col == 1 else None, "showscale": col == 1,
                },
                hovertemplate="m%{customdata[0]} setpoint %{customdata[1]} (%{customdata[2]} leg, segment %{customdata[3]}, n=%{customdata[4]})"
                "<br>signal %{x:.3e} A<br>noise %{customdata[5]:.3e} A<br>RSD %{customdata[6]:.3g}%<extra></extra>",
            ),
            row=1, col=col,
        )
    model = fit_noise_model(seg_stats)
    subtitle = "fit needs 3 or more holds"
    if model is not None:
        s = np.geomspace(d["current_raw_mean"].min(), d["current_raw_mean"].max(), 200)
        sigma = np.sqrt(model["floor"] ** 2 + model["shot"] * s + (model["prop"] * s) ** 2)
        for col, y in [(1, sigma * RAW_CURRENT_AMPS_PER_COUNT), (2, 100 * sigma / s)]:
            fig.add_trace(
                go.Scatter(
                    x=s * RAW_CURRENT_AMPS_PER_COUNT, y=y, mode="lines", line={"color": "#eb6834", "width": 2, "dash": "dash"},
                    name="noise model (all masses, all speeds)", legendgroup="model", showlegend=col == 1, hoverinfo="skip",
                ),
                row=1, col=col,
            )
        subtitle = (
            f"model: noise² = floor² + shot·S + (prop·S)², floor {model['floor'] * RAW_CURRENT_AMPS_PER_COUNT:.2e} A, "
            f"prop {100 * model['prop']:.2f}%"
        )
    fig.update_xaxes(type="log", exponentformat="E", showexponent="all", title_text="mean signal (A)")
    fig.update_yaxes(type="log", exponentformat="E", showexponent="all", title_text="drift-robust noise (A)", row=1, col=1)
    fig.update_yaxes(type="log", title_text="drift-robust RSD (%)", row=1, col=2)
    return _style(
        fig,
        f"Noise vs signal, per mass and hold (colour: setpoint; circle ↓ leg, diamond ↑ leg)<br><sup>{subtitle}</sup>",
        560,
    )


def build_figs(outputs: dict[str, pl.DataFrame], settle_s: float) -> list[go.Figure]:
    seg_stats = outputs["mass_stats_by_segment"]
    return [
        fig_timeseries(outputs["turbo_tagged"], outputs["speed_changes"], settle_s),
        fig_mass_facets(seg_stats, "torr_mean", "torr_std", "partial pressure (Torr)", log_y=True),
        fig_mass_facets(seg_stats, "rsd_pct", None, "RSD (%)", log_y=False, hover_fmt=".3g"),
        fig_rsd(seg_stats, outputs["rsd_speed_tests"]),
        fig_tp(outputs["tp_by_segment"]),
        fig_power(outputs["tp_by_segment"]),
    ]


def build_ratio_figs(outputs: dict[str, pl.DataFrame], ref_mass: int = REF_MASS) -> list[go.Figure]:
    seg = outputs["ratio_stats_by_segment"]
    if seg.is_empty():
        return []
    return [
        fig_mass_facets(seg, "ratio_mean", "ratio_std", f"signal / m{ref_mass}", log_y=True, hover_fmt=".4g"),
        fig_mass_facets(seg, "rsd_pct", None, f"RSD of signal / m{ref_mass} (%)", log_y=False, hover_fmt=".3g"),
        fig_rsd(seg, outputs["ratio_speed_tests"], label=f"Drift-robust RSD of signal / m{ref_mass}"),
    ]


def write_report(figs: list[go.Figure], path: Path) -> None:
    parts = [fig.to_html(full_html=False, include_plotlyjs=(i == 0)) for i, fig in enumerate(figs)]
    path.write_text(
        "<html><head><meta charset='utf-8'><title>Turbo speed test</title></head>"
        "<body style='font-family:sans-serif;background:#fcfcfb'>" + "\n".join(parts) + "</body></html>",
        encoding="utf-8",
    )


def _parse_iso(s: str) -> datetime:
    ts = _parse_ts(s)
    if ts is None:
        raise argparse.ArgumentTypeError(f"not an ISO-8601 timestamp: {s}")
    return ts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--settle-min", type=float, default=10.0, help="minutes dropped after acquisition restarts (OK,AON) following each speed change (default 10)")
    parser.add_argument("--start", type=_parse_iso, help="ignore readings and changes before this ISO time (UTC)")
    parser.add_argument("--end", type=_parse_iso, help="ignore readings and changes after this ISO time (UTC)")
    parser.add_argument("--commands-log", type=Path, help="turbo_speed_commands.log from the stepper, as an extra event source")
    parser.add_argument("--ack-window-s", type=float, default=120.0, help="max command-to-ack gap, and dedup window (default 120)")
    parser.add_argument("--ref-mass", type=int, default=REF_MASS, help=f"mass each signal is normalised to (default {REF_MASS})")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--partial-pressure-sensitivity", type=float, default=DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR, help="A/Torr")
    parser.add_argument("--total-pressure-sensitivity", type=float, default=DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR, help="A/Torr")
    args = parser.parse_args(argv)
    settle_s = args.settle_min * 60

    tables = build_tables(read_all(find_all_files(args.raw_dir)))
    commands, acks, aon_acks, ts_rows = scan_raw(args.raw_dir, args.commands_log)
    changes = add_acq_on(build_speed_changes(commands, acks, args.ack_window_s), aon_acks)
    outputs = analyze(
        tables, changes, ts_rows, args.start, args.end, settle_s,
        args.partial_pressure_sensitivity, args.total_pressure_sensitivity, args.ref_mass,
    )
    changes = outputs["speed_changes"]
    print(f"found {len(commands)} SPD command sightings, {len(acks)} OK,SPD acks -> {changes.height} speed changes")
    if changes.is_empty():
        print("No speed changes found; nothing to analyse.", file=sys.stderr)
        return 1

    for name, df in outputs.items():
        write_df(df, args.out_dir, name, args.format)

    report = args.out_dir / "turbo_speed_report.html"
    write_report(
        [*build_figs(outputs, settle_s), fig_noise_vs_signal(outputs["mass_stats_by_segment"]), *build_ratio_figs(outputs, args.ref_mass)],
        report,
    )

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, fmt_float="mixed"):
        print(changes)
        print(outputs["tp_by_speed"].select("setpoint", "n", "speed_hz_mean", "tp_torr_mean", "tp_torr_std", "tp_rsd_pct"))
        print(
            outputs["mass_stats_by_speed"]
            .pivot(index="setpoint", on="mass", values="rsd_pct")
            .rename(lambda c: c if c == "setpoint" else f"rsd_m{c}")
        )
        print(outputs["rsd_speed_tests"])
        print(outputs["noise_floor_summary"])
        print(f"signal / m{args.ref_mass}:")
        print(outputs["ratio_speed_tests"])
    print(f"wrote outputs and {report} to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
