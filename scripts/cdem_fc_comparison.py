"""Compare RGA CDEM vs Faraday-cup (FC) detector runs from one surface lander log.

Detector state comes from the OK,EMON / OK,EMOFF control lines (CDEM between
them, FC otherwise); runs are OK,AON -> OK,AOFF spans. Control lines only carry
the surface receipt timestamp, so readings are segmented on their receipt
timestamp too (the embedded lander ts is a few seconds earlier).

Usage:
    uv run scripts/cdem_fc_comparison.py <surface_*_lander.log> --out-dir <dir> [--stable-skip-s 60] \
        [--fc-window <start> <end>] [--cdem-window <start> <end>]

--fc-window/--cdem-window (UTC ISO datetimes, surface receipt time) restrict the
"stable" statistics to that span, overriding --stable-skip-s for that detector.

Partial pressure: FC Torr = I / SP. CDEM Torr = I / (SP * G), with a single
CDEM gain G for all masses (as the SRS RGA applies it), defaulting to the
CDEM/FC gain measured here at --cdem-gain-mass (m/z 28, N2). Override with
--cdem-gain. SP is the nominal FC sensitivity unless
--partial-pressure-sensitivity-ma-per-torr is given.

Total pressure from the !: status lines (I / ST) is overlaid on the FC panel of
the partial-pressure plot, restricted to FC acquisition (AON->AOFF, CDEM off);
it reads 0 while the CDEM is on.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from egcf_processing.aggregate import (
    DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR,
    DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR,
    RAW_CURRENT_AMPS_PER_COUNT,
)
from egcf_processing.lines import parse_line

_CONTROL_RE = re.compile(r"^OK,(EMON|EMOFF|AON|AOFF)$")
_DETECTOR_COLORS = {"FC": "#1f77b4", "CDEM": "#d62728"}
_NORM_MASS = 40
_DEFAULT_SP_MA_PER_TORR = DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR * 1e3
_DEFAULT_ST_MA_PER_TORR = DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR * 1e3

_READINGS_SCHEMA = {
    "receipt_ts": pl.Datetime("us"),
    "ts": pl.Datetime("us"),
    "mass": pl.Int64,
    "counts": pl.Float64,
}
_EVENTS_SCHEMA = {"receipt_ts": pl.Datetime("us"), "event": pl.Utf8}
_STATUS_SCHEMA = {"receipt_ts": pl.Datetime("us"), "ts": pl.Datetime("us"), "raw_total_pressure_current": pl.Float64}


def _parse_receipt_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def read_log(path: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    readings, events, status = [], [], []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            receipt_ts = _parse_receipt_ts(parts[0])
            m = _CONTROL_RE.match(parts[1])
            if m:
                events.append({"receipt_ts": receipt_ts, "event": m.group(1)})
                continue
            rec = parse_line(parts[1])
            if rec and rec["tag"] == "!" and rec["raw_total_pressure_current"] is not None:
                status.append(
                    {"receipt_ts": receipt_ts, "ts": rec["ts"], "raw_total_pressure_current": rec["raw_total_pressure_current"]}
                )
            elif rec and rec["tag"] == "R":
                readings.append(
                    {"receipt_ts": receipt_ts, "ts": rec["ts"], "mass": rec["mass"], "counts": rec["current"]}
                )
    return (
        pl.DataFrame(readings, schema=_READINGS_SCHEMA),
        pl.DataFrame(events, schema=_EVENTS_SCHEMA),
        pl.DataFrame(status, schema=_STATUS_SCHEMA),
    )


def total_pressure(status: pl.DataFrame, events: pl.DataFrame, st_a_per_torr: float) -> pl.DataFrame:
    cdem_spans = _spans(events, "EMON", "EMOFF")
    run_spans = _spans(events, "AON", "AOFF")
    keep = [
        any(a <= ts <= b for a, b in run_spans) and not any(a <= ts < b for a, b in cdem_spans)
        for ts in status["receipt_ts"]
    ]
    return status.filter(pl.Series(keep, dtype=pl.Boolean) & (pl.col("raw_total_pressure_current") > 0)).with_columns(
        total_pressure_torr=pl.col("raw_total_pressure_current") * RAW_CURRENT_AMPS_PER_COUNT / st_a_per_torr
    )


def _spans(events: pl.DataFrame, on: str, off: str) -> list[tuple[datetime, datetime]]:
    spans, start = [], None
    for ts, ev in events.iter_rows():
        if ev == on and start is None:
            start = ts
        elif ev == off and start is not None:
            spans.append((start, ts))
            start = None
    if start is not None:
        spans.append((start, datetime.max))
    return spans


def label_readings(
    readings: pl.DataFrame,
    events: pl.DataFrame,
    stable_skip_s: float,
    windows: dict[str, tuple[datetime, datetime]],
) -> pl.DataFrame:
    cdem_spans = _spans(events, "EMON", "EMOFF")
    run_spans = _spans(events, "AON", "AOFF")

    detectors, runs, run_starts = [], [], []
    for ts in readings["receipt_ts"]:
        detectors.append("CDEM" if any(a <= ts < b for a, b in cdem_spans) else "FC")
        run = next((i for i, (a, b) in enumerate(run_spans) if a <= ts <= b), None)
        runs.append(run)
        run_starts.append(run_spans[run][0] if run is not None else None)

    df = readings.with_columns(
        pl.Series("detector", detectors, dtype=pl.Utf8),
        pl.Series("run", runs, dtype=pl.Int64),
        pl.Series("run_start", run_starts, dtype=pl.Datetime("us")),
    ).filter(pl.col("run").is_not_null())

    scan_ids = []
    for _, group in df.group_by("run", maintain_order=True):
        seen: set[int] = set()
        scan_id = 0
        for mass in group["mass"]:
            if mass in seen:
                scan_id += 1
                seen = set()
            seen.add(mass)
            scan_ids.append(scan_id)

    elapsed_s = (pl.col("receipt_ts") - pl.col("run_start")).dt.total_milliseconds() / 1000
    stable = elapsed_s >= stable_skip_s
    for det, (start, end) in windows.items():
        stable = (
            pl.when(pl.col("detector") == det)
            .then(pl.col("receipt_ts").is_between(start, end))
            .otherwise(stable)
        )
    return df.with_columns(
        pl.Series("scan", scan_ids, dtype=pl.Int64),
        amps=pl.col("counts") * RAW_CURRENT_AMPS_PER_COUNT,
        elapsed_s=elapsed_s,
        stable=stable,
    )


def _slope_per_min(t_s: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 3:
        return float("nan")
    return float(np.polyfit(t_s / 60, y, 1)[0])


def detector_stats(df: pl.DataFrame) -> pl.DataFrame:
    norm = (
        df.filter(pl.col("mass") == _NORM_MASS)
        .select("run", "scan", norm_amps=pl.col("amps"))
    )
    df = df.join(norm, on=["run", "scan"], how="left").with_columns(
        ratio_to_norm=pl.col("amps") / pl.col("norm_amps")
    )

    rows = []
    for (detector, mass), g in df.sort("receipt_ts").group_by(["detector", "mass"], maintain_order=True):
        amps = g["amps"].to_numpy()
        s = g.filter(pl.col("stable"))
        s_amps = s["amps"].to_numpy()
        s_t = s["elapsed_s"].to_numpy()
        s_ratio = s["ratio_to_norm"].drop_nulls().to_numpy()
        mean_s = float(s_amps.mean()) if len(s_amps) else float("nan")
        sd_s = float(s_amps.std(ddof=1)) if len(s_amps) > 1 else float("nan")
        d = np.diff(s_amps)
        noise = float(d.std(ddof=1) / np.sqrt(2)) if len(s_amps) > 2 else float("nan")
        # MAD-based: robust to sample transients (e.g. the air pulse mid-CDEM run)
        noise_mad = float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) if len(s_amps) > 2 else float("nan")
        rows.append(
            {
                "detector": detector,
                "mass": mass,
                "n": len(amps),
                "n_stable": len(s_amps),
                "mean_amps": float(amps.mean()),
                "sd_amps": float(amps.std(ddof=1)),
                "rsd_pct": float(amps.std(ddof=1) / amps.mean() * 100),
                "mean_stable_amps": mean_s,
                "sd_stable_amps": sd_s,
                "se_stable_amps": sd_s / np.sqrt(len(s_amps)) if len(s_amps) else float("nan"),
                "rsd_stable_pct": sd_s / mean_s * 100,
                "noise_rsd_pct": noise / mean_s * 100,
                "noise_mad_rsd_pct": noise_mad / mean_s * 100,
                "drift_pct_per_min": _slope_per_min(s_t, s_amps) / mean_s * 100,
                f"ratio_to_m{_NORM_MASS}_rsd_pct": (
                    float(s_ratio.std(ddof=1) / s_ratio.mean() * 100) if len(s_ratio) > 1 else float("nan")
                ),
            }
        )
    return pl.DataFrame(rows).sort("mass", "detector")


def gain_table(stats: pl.DataFrame) -> pl.DataFrame:
    cols = [
        "mass", "mean_stable_amps", "se_stable_amps", "noise_rsd_pct", "noise_mad_rsd_pct",
        f"ratio_to_m{_NORM_MASS}_rsd_pct",
    ]
    fc = stats.filter(pl.col("detector") == "FC").select(cols).rename(lambda c: c if c == "mass" else f"fc_{c}")
    cdem = stats.filter(pl.col("detector") == "CDEM").select(cols).rename(lambda c: c if c == "mass" else f"cdem_{c}")
    g = pl.col("cdem_mean_stable_amps") / pl.col("fc_mean_stable_amps")
    return (
        fc.join(cdem, on="mass", how="full", coalesce=True)
        .with_columns(cdem_fc_gain=g)
        .with_columns(
            cdem_fc_gain_se=pl.col("cdem_fc_gain")
            * (
                (pl.col("cdem_se_stable_amps") / pl.col("cdem_mean_stable_amps")) ** 2
                + (pl.col("fc_se_stable_amps") / pl.col("fc_mean_stable_amps")) ** 2
            ).sqrt(),
            cdem_sensitivity_a_per_torr_from_nominal_fc=pl.col("cdem_fc_gain")
            * DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR,
            noise_improvement_fc_over_cdem=pl.col("fc_noise_mad_rsd_pct") / pl.col("cdem_noise_mad_rsd_pct"),
        )
        .sort("mass")
    )


def add_partial_pressure(df: pl.DataFrame, sp_a_per_torr: float, cdem_gain: float) -> pl.DataFrame:
    gain = pl.when(pl.col("detector") == "CDEM").then(pl.lit(cdem_gain)).otherwise(pl.lit(1.0))
    return df.with_columns(torr=pl.col("amps") / (sp_a_per_torr * gain))


def _add_event_lines(fig: go.Figure, events: pl.DataFrame) -> None:
    for ts, ev in events.iter_rows():
        fig.add_vline(x=ts, line_dash="dot", line_color="grey", line_width=1)
        fig.add_annotation(
            x=ts, y=1.0, yref="paper", text=ev, showarrow=False, textangle=-90,
            xanchor="right", yanchor="top", font={"size": 9, "color": "grey"},
        )


def _stable_spans(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("stable"))
        .group_by("detector")
        .agg(start=pl.col("receipt_ts").min(), end=pl.col("receipt_ts").max())
    )


def _add_stable_shading(fig: go.Figure, df: pl.DataFrame) -> None:
    for det, start, end in _stable_spans(df).iter_rows():
        fig.add_vrect(x0=start, x1=end, fillcolor=_DETECTOR_COLORS[det], opacity=0.08, line_width=0)


def plot_timeseries(
    df: pl.DataFrame, events: pl.DataFrame, out: Path,
    value_col: str = "amps", y_title: str = "ion current (A)", title: str = "RGA ion current by mass",
) -> None:
    masses = sorted(df["mass"].unique().to_list())
    ncols = 3
    nrows = -(-len(masses) // ncols)
    fig = make_subplots(
        rows=nrows, cols=ncols, shared_xaxes=True,
        subplot_titles=[f"m/z {m}" for m in masses], vertical_spacing=0.06,
    )
    for i, m in enumerate(masses):
        r, c = i // ncols + 1, i % ncols + 1
        for det, color in _DETECTOR_COLORS.items():
            g = df.filter((pl.col("mass") == m) & (pl.col("detector") == det))
            fig.add_trace(
                go.Scatter(
                    x=g["receipt_ts"], y=g[value_col], mode="lines+markers", marker={"size": 3},
                    line={"color": color, "width": 1}, name=det, legendgroup=det, showlegend=i == 0,
                ),
                row=r, col=c,
            )
        fig.update_yaxes(type="log", exponentformat="power", row=r, col=c)
    for ts, ev in events.iter_rows():
        fig.add_vline(x=ts, line_dash="dot", line_color="grey", line_width=1)
    _add_stable_shading(fig, df)
    fig.update_yaxes(title_text=y_title, col=1)
    fig.update_layout(
        title=f"{title}: FC vs CDEM (dotted: EMON/EMOFF/AON/AOFF; shaded: analysis windows)",
        height=300 * nrows, template="plotly_white",
    )
    fig.write_html(out)


def plot_by_detector(
    df: pl.DataFrame, events: pl.DataFrame, out: Path,
    value_col: str = "amps", y_title: str = "ion current (A)", title: str = "RGA ion current",
    total_pressure: pl.DataFrame | None = None,
) -> None:
    fig = make_subplots(rows=1, cols=2, subplot_titles=["FC", "CDEM"], horizontal_spacing=0.08)
    if total_pressure is not None:
        fig.add_trace(
            go.Scatter(
                x=total_pressure["receipt_ts"], y=total_pressure["total_pressure_torr"], mode="lines+markers",
                marker={"size": 3}, line={"color": "black", "width": 2}, name="total pressure (!: status)",
            ),
            row=1, col=1,
        )
    colors = px.colors.qualitative.D3
    for i, m in enumerate(sorted(df["mass"].unique().to_list())):
        for c, det in enumerate(["FC", "CDEM"], start=1):
            g = df.filter((pl.col("mass") == m) & (pl.col("detector") == det))
            fig.add_trace(
                go.Scatter(
                    x=g["receipt_ts"], y=g[value_col], mode="lines", line={"color": colors[i % len(colors)]},
                    name=f"m/z {m}", legendgroup=str(m), showlegend=c == 1,
                ),
                row=1, col=c,
            )
    _add_stable_shading(fig, df)
    fig.update_yaxes(type="log", exponentformat="power", title_text=y_title)
    fig.update_layout(
        title=f"{title}, all masses, per detector run (shaded: analysis windows)", height=550, template="plotly_white"
    )
    fig.write_html(out)


def plot_gain(gains: pl.DataFrame, out: Path) -> None:
    fig = go.Figure(
        go.Scatter(
            x=gains["mass"], y=gains["cdem_fc_gain"], mode="markers+lines",
            error_y={"type": "data", "array": gains["cdem_fc_gain_se"].to_list()},
            marker={"size": 9, "color": _DETECTOR_COLORS["CDEM"]},
        )
    )
    fig.update_layout(
        title="CDEM / FC gain by mass (stable-window means; sequential runs, assumes unchanged sample)",
        xaxis_title="m/z", yaxis_title="CDEM / FC current ratio", template="plotly_white",
    )
    fig.update_xaxes(tickmode="array", tickvals=gains["mass"].to_list())
    fig.write_html(out)


def plot_relative_error(stats: pl.DataFrame, out: Path) -> None:
    metrics = {
        "noise_mad_rsd_pct": "scan-to-scan noise RSD, MAD-based (%)",
        "noise_rsd_pct": "scan-to-scan noise RSD, SD-based (%)",
        "rsd_stable_pct": "stable-window RSD (%)",
        f"ratio_to_m{_NORM_MASS}_rsd_pct": f"RSD of ratio to m/z {_NORM_MASS} (%)",
    }
    fig = make_subplots(rows=len(metrics), cols=1, shared_xaxes=True, subplot_titles=list(metrics.values()))
    for r, col in enumerate(metrics, start=1):
        for det, color in _DETECTOR_COLORS.items():
            g = stats.filter(pl.col("detector") == det)
            fig.add_trace(
                go.Bar(
                    x=g["mass"].cast(pl.Utf8), y=g[col], name=det, marker_color=color,
                    legendgroup=det, showlegend=r == 1,
                ),
                row=r, col=1,
            )
    fig.update_xaxes(title_text="m/z", type="category", row=len(metrics), col=1)
    fig.update_yaxes(type="log")
    fig.update_layout(
        title="Relative error by mass: FC vs CDEM", barmode="group", height=1100, template="plotly_white"
    )
    fig.write_html(out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("lander_log", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--stable-skip-s", type=float, default=60.0, help="seconds after AON excluded from stable stats")
    p.add_argument("--fc-window", nargs=2, type=datetime.fromisoformat, metavar=("START", "END"))
    p.add_argument("--cdem-window", nargs=2, type=datetime.fromisoformat, metavar=("START", "END"))
    p.add_argument("--partial-pressure-sensitivity-ma-per-torr", type=float, default=_DEFAULT_SP_MA_PER_TORR)
    p.add_argument("--total-pressure-sensitivity-ma-per-torr", type=float, default=_DEFAULT_ST_MA_PER_TORR)
    p.add_argument("--cdem-gain", type=float, help="CDEM gain for all masses (default: measured at --cdem-gain-mass)")
    p.add_argument("--cdem-gain-mass", type=int, default=28)
    args = p.parse_args(argv)

    readings, events, status = read_log(args.lander_log)
    windows = {d: tuple(w) for d, w in (("FC", args.fc_window), ("CDEM", args.cdem_window)) if w}
    df = label_readings(readings, events, args.stable_skip_s, windows)
    stats = detector_stats(df)
    gains = gain_table(stats)
    sp_a_per_torr = args.partial_pressure_sensitivity_ma_per_torr * 1e-3
    cdem_gain = args.cdem_gain or gains.filter(pl.col("mass") == args.cdem_gain_mass)["cdem_fc_gain"].item()
    df = add_partial_pressure(df, sp_a_per_torr, cdem_gain)
    torr_means = (
        df.filter(pl.col("stable"))
        .group_by("mass", "detector")
        .agg(pl.col("torr").mean())
        .pivot(index="mass", on="detector", values="torr")
        .select("mass", fc_mean_stable_torr="FC", cdem_mean_stable_torr="CDEM")
        .with_columns(cdem_over_fc_torr=pl.col("cdem_mean_stable_torr") / pl.col("fc_mean_stable_torr"))
    )
    gains = gains.join(torr_means, on="mass", how="left").with_columns(
        cdem_gain_applied=pl.lit(cdem_gain), sp_ma_per_torr=pl.lit(args.partial_pressure_sensitivity_ma_per_torr)
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.drop("run_start").write_parquet(args.out_dir / "cdem_fc_rga_long.parquet")
    stats.write_parquet(args.out_dir / "cdem_fc_detector_stats.parquet")
    stats.write_csv(args.out_dir / "cdem_fc_detector_stats.csv")
    gains.write_parquet(args.out_dir / "cdem_fc_summary.parquet")
    gains.write_csv(args.out_dir / "cdem_fc_summary.csv")
    plot_timeseries(df, events, args.out_dir / "rga_timeseries.html")
    plot_by_detector(df, events, args.out_dir / "rga_timeseries_by_detector.html")
    pp_title = (
        f"RGA partial pressure (SP {args.partial_pressure_sensitivity_ma_per_torr:.4g} mA/Torr; "
        f"CDEM gain {cdem_gain:.0f} applied to all masses)"
    )
    plot_timeseries(df, events, args.out_dir / "rga_partial_pressure.html", "torr", "partial pressure (Torr)", pp_title)
    tp = total_pressure(status, events, args.total_pressure_sensitivity_ma_per_torr * 1e-3)
    tp.write_parquet(args.out_dir / "total_pressure.parquet")
    plot_by_detector(
        df, events, args.out_dir / "rga_partial_pressure_by_detector.html", "torr", "pressure (Torr)",
        f"{pp_title}; total pressure ST {args.total_pressure_sensitivity_ma_per_torr:.4g} mA/Torr", tp,
    )
    plot_gain(gains, args.out_dir / "gain_by_mass.html")
    plot_relative_error(stats, args.out_dir / "relative_error_by_mass.html")

    pl.Config.set_tbl_rows(50)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    print("control events:")
    print(events)
    print("analysis windows (actual first/last reading used):")
    print(_stable_spans(df).sort("start"))
    print(
        df.group_by("run", "detector")
        .agg(start=pl.col("receipt_ts").min(), end=pl.col("receipt_ts").max(), scans=pl.col("scan").n_unique())
        .sort("run")
    )
    print(
        stats.select(
            "detector", "mass", "n", "n_stable", "mean_stable_amps", "rsd_pct", "rsd_stable_pct",
            "noise_rsd_pct", "noise_mad_rsd_pct", "drift_pct_per_min", f"ratio_to_m{_NORM_MASS}_rsd_pct",
        )
    )
    print(
        gains.select(
            "mass", "cdem_fc_gain", "cdem_fc_gain_se", "cdem_sensitivity_a_per_torr_from_nominal_fc",
            "fc_noise_mad_rsd_pct", "cdem_noise_mad_rsd_pct", "noise_improvement_fc_over_cdem",
        )
    )
    print(f"partial pressure: SP={args.partial_pressure_sensitivity_ma_per_torr:.4g} mA/Torr, CDEM gain={cdem_gain:.1f}")
    print(gains.select("mass", "fc_mean_stable_torr", "cdem_mean_stable_torr", "cdem_over_fc_torr"))
    print(f"total pressure: ST={args.total_pressure_sensitivity_ma_per_torr:.4g} mA/Torr, {tp.height} FC-acquisition status readings")
    print(tp.select(pl.col("receipt_ts").min().alias("first"), pl.col("receipt_ts").max().alias("last"),
                    pl.col("total_pressure_torr").min().alias("min_torr"), pl.col("total_pressure_torr").max().alias("max_torr")))
    fc_win = _stable_spans(df).filter(pl.col("detector") == "FC")
    if fc_win.height:
        tp_win = tp.filter(pl.col("receipt_ts").is_between(fc_win["start"].item(), fc_win["end"].item()))
        pp_sum = df.filter(pl.col("stable") & (pl.col("detector") == "FC")).group_by("scan").agg(pl.col("torr").sum())
        print(f"FC window: mean total pressure {tp_win['total_pressure_torr'].mean():.3e} Torr, "
              f"mean sum of partial pressures {pp_sum['torr'].mean():.3e} Torr")
    print(f"wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
