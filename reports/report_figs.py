"""Figure and table builders for egcf_report.qmd, with no Streamlit dependency.

Deployment figures reuse the dashboard's helpers (colors, chamber shading,
linked subplots, metabolism scatter, P-I curves) so the report matches what the
dashboard shows. Tables are read from the saved pipeline output, never
recomputed, so every number matches processing.parquet. The turbo and CDEM
tests reuse the figure code in scripts/.

Every builder returns (figure, caption) pairs; captions state any averaging.
"""

from __future__ import annotations

import base64
import importlib
import math
import re
import sys
from datetime import timedelta
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from IPython.display import Markdown, display
from plotly.subplots import make_subplots

from egcf_processing import dashboard as db
from egcf_processing import qc
from egcf_processing.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
cfc = importlib.import_module("cdem_fc_comparison")
tsa = importlib.import_module("turbo_speed_analysis")

AVERAGE_EVERY = "1m"
GAP_BREAK = timedelta(minutes=5)
PANEL_HEIGHT = 210
CDEM_TEST = "cdem_test_2026-10-13"
CDEM_LOG = Path("tests/egcf_cdem_test_2026-10-13/surface_2026-10-13-16-00_lander.log")
TURBO_TEST = "turbo_test"

FLUX_SHORT = {"oxygen": "O₂", "n2_denitrification": "N₂", "h_ion": "H⁺", "temp_degC": "ΔT"}
FLUX_VARIABLES = {
    "oxygen": "O₂ flux",
    "n2_denitrification": "N₂ flux",
    "h_ion": "H⁺ flux",
    "temp_degC": "Temperature change",
}
_UNIT_LABELS = {"mmol m-2 h-1": "mmol m⁻² h⁻¹", "degC h-1": "°C h⁻¹"}
_CHAMBER_SYMBOLS = {"C1": "circle", "C2": "diamond"}
_TEXT_SECONDARY = "#52514e"
_SINGLE_SERIES_COLOR = "#3d3d3a"
PNG_WIDTH = 1000
PNG_SCALE = 2
_plotlyjs_sent = False


def settings() -> dict:
    return load_config(REPO_ROOT / "egcf.toml")


def processed_root() -> Path:
    return Path(settings()["processed_root"])


def raw_root() -> Path:
    return Path(settings()["raw_root"])


def _unit(unit: str) -> str:
    return _UNIT_LABELS.get(unit, unit)


def style(fig: go.Figure, title: str | None, height: int) -> go.Figure:
    """Report-wide layout: plotly_white, recessive grid, horizontal legend above the plots."""
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left"} if title else None,
        height=height,
        template="plotly_white",
        font={"size": 13, "color": _TEXT_SECONDARY},
        margin={"l": 80, "r": 30, "t": 110 if title else 70, "b": 50},
        legend={"orientation": "h", "x": 0, "xanchor": "left", "y": 1.02, "yanchor": "bottom"},
    )
    fig.update_annotations(font_size=13)
    return fig


def show(item: tuple[go.Figure, str] | None) -> None:
    """Display a (figure, caption) pair as one html + png bundle, then the caption.

    Quarto keeps the html (interactive) for html output and the png for typst.
    plotly.js goes into the first figure only: with embed-resources every
    figure would otherwise carry its own 4 MB copy.
    """
    global _plotlyjs_sent
    if item is None:
        return
    fig, caption = item
    title = fig.layout.title.text
    if title:
        title = re.sub(r"<br>\s*", ". ", title)
        caption = f"**{re.sub(r'<[^>]+>', '', title)}.** {caption}"
    entries = len({t.legendgroup or t.name for t in fig.data if t.showlegend is not False and t.name})
    rows = math.ceil(entries / 7) if fig.layout.showlegend is not False and entries > 1 else 0
    fig.update_layout(
        title=None,
        margin={"t": 30 + 24 * rows},
        legend={"orientation": "h", "x": 0, "xanchor": "left", "yref": "container", "y": 1, "yanchor": "top"},
    )
    fig.update_yaxes(exponentformat="power")
    html = fig.to_html(full_html=False, include_plotlyjs=not _plotlyjs_sent, config={"responsive": True})
    _plotlyjs_sent = True
    png = fig.to_image(format="png", width=PNG_WIDTH, scale=PNG_SCALE)
    display({"text/html": html, "image/png": base64.b64encode(png).decode()}, raw=True)
    display(Markdown(f"*{caption}*"))


def show_all(items: list[tuple[go.Figure, str]]) -> None:
    for item in items:
        show(item)


# ---------------------------------------------------------------- loading


def load_deployment(deployment_id: str) -> dict:
    """Saved tables for one deployment, Layer A clipped to the in-water window like Layers B-E."""
    path = processed_root() / deployment_id
    meta = db.load_processed_deployment(path)
    tables = db.clip_tables(db.load_all(path), meta["start"], meta["end"])
    for name in ("par_daily", "processing", "deployment"):
        tables[name] = db.load_table(path, name)
    return {"id": deployment_id, "meta": meta, "tables": tables, "processing": tables["processing"].row(0, named=True)}


def deployment_header_md(d: dict) -> str:
    m = d["meta"]
    position = f" ({m['lat']:.5f}, {m['lon']:.5f})" if m["lat"] is not None and m["lon"] is not None else ""
    lines = [
        (f"**{m['deployment_id']}**: {m['site']}{position}. In water {m['start']:%Y-%m-%d %H:%M} to "
        f"{m['end']:%Y-%m-%d %H:%M} UTC. C1: **{m['c1']}**, C2: **{m['c2']}**."),
    ]
    if m["comments"]:
        lines += ["", "::: {.callout-note title=\"Deployment notes\"}", m["comments"].strip().replace("\n", "  \n"), ":::"]
    return "\n".join(lines)


def deployments_table(deployments: list[dict]) -> pl.DataFrame:
    return pl.concat([d["tables"]["deployment"] for d in deployments], how="diagonal_relaxed").select(
        "deployment_id",
        "site",
        pl.col("start").dt.strftime("%Y-%m-%d %H:%M").alias("start (UTC)"),
        pl.col("end").dt.strftime("%Y-%m-%d %H:%M").alias("end (UTC)"),
        ((pl.col("end") - pl.col("start")).dt.total_minutes() / 60).round(1).alias("hours"),
        "c1",
        "c2",
        "test",
    )


def processing_table(deployments: list[dict]) -> pl.DataFrame:
    """One row per setting: the value if every deployment used the same one, else each deployment's value."""
    wide = pl.concat([d["tables"]["processing"] for d in deployments], how="diagonal_relaxed")
    rows = []
    for col in wide.columns:
        if col == "deployment_id":
            continue
        values = [_fmt(v) for v in wide[col].to_list()]
        if len(set(values)) == 1:
            value = values[0]
        else:
            value = "; ".join(f"{d}: {v}" for d, v in zip(wide["deployment_id"], values))
        rows.append({"setting": col, "value": value})
    return pl.DataFrame(rows)


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return _num(v, 4)
    return str(v)


def _num(v: float, sig: int = 3) -> str:
    """sig significant figures, in plain decimals between 1e-3 and 1e6 and scientific outside."""
    if math.isnan(v):
        return ""
    a = abs(v)
    if a == 0:
        return "0"
    if 1e-3 <= a < 1e6:
        digits = math.floor(math.log10(a)) + 1
        return f"{round(v, sig - digits):.{max(0, sig - digits)}f}"
    return f"{v:.{sig - 1}e}"


def md_table(df: pl.DataFrame, sig: int = 3) -> str:
    """A pipe table Quarto renders in both html and typst."""

    def cell(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return _num(v, sig)
        return str(v).replace("|", "\\|").replace("\n", " ")

    body = [[cell(v) for v in row] for row in df.iter_rows()]
    widths = [max([len(h), *(len(r[i]) for r in body)]) for i, h in enumerate(df.columns)]
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("-" * max(3, w) for w in widths) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


# ---------------------------------------------------------------- averaging


def minute_means(df: pl.DataFrame, cols: list[str], group: str | None = None) -> pl.DataFrame:
    """1-minute means of cols, with a null row after any gap over GAP_BREAK so lines break there."""
    keys = [group] if group else []
    out = (
        df.sort([*keys, "ts"])
        .group_by_dynamic("ts", every=AVERAGE_EVERY, group_by=group)
        .agg(pl.col(cols).mean())
    )
    return break_gaps(out, group)


def break_gaps(df: pl.DataFrame, group: str | None = None) -> pl.DataFrame:
    keys = [group] if group else []
    step = pl.col("ts").shift(-1) - pl.col("ts")
    if group:
        step = step.over(group)
    breaks = df.filter(step > GAP_BREAK).select(*keys, pl.col("ts") + pl.duration(seconds=1))
    return pl.concat([df, breaks], how="diagonal_relaxed").sort([*keys, "ts"])


def _line(df: pl.DataFrame, col: str, name: str, color: str | None = None, width: float = 1.5) -> go.Scatter:
    return go.Scatter(x=df["ts"], y=df[col], mode="lines", name=name, line={"color": color, "width": width})


def _linked(sections: list, title: str, spans: pl.DataFrame | None) -> go.Figure | None:
    """db.build_linked_figure in report style; single-trace panels need no legend entry."""
    fig = db.build_linked_figure(sections, title, spans)
    if fig is None:
        return None
    n = len([s for s in sections if s[1]])
    for trace in fig.data:
        if trace.name and trace.name.startswith("_"):
            trace.showlegend = False
    fig.update_layout(hovermode="x unified")
    return style(fig, title, PANEL_HEIGHT * n + 140)


def _single(df: pl.DataFrame, col: str, color: str = _SINGLE_SERIES_COLOR) -> list[go.Scatter]:
    return [_line(df, col, f"_{col}", color)]


# ---------------------------------------------------------------- deployment figures


def availability_fig(d: dict) -> tuple[go.Figure, str] | None:
    tables, meta = d["tables"], d["meta"]
    scores = qc.completeness(tables, start=meta["start"], end=meta["end"])
    segments = qc.coverage_segments(tables, start=meta["start"], end=meta["end"])
    if segments.is_empty():
        return None
    streams = scores["stream"].to_list()
    fig = db._availability_figure(segments, streams, meta["start"], meta["end"])
    style(fig, "Data availability", 110 + 24 * len(streams))
    fig.update_layout(margin={"t": 60})
    caption = (
        "Contiguous data runs per stream over the in-water window; blank space is a gap longer than the "
        "stream's gap threshold (qc.STREAM_CADENCE)."
    )
    return fig, caption


def status_figs(d: dict) -> list[tuple[go.Figure, str]]:
    tables = d["tables"]
    status, health = tables["status"], tables["system_health"]
    spans = db.active_chamber_spans(tables["valve"])
    out = []
    temp_cols = ["turbo_etemp_c", "turbo_btemp_c", "turbo_mtemp_c"]
    if status is not None and not status.is_empty():
        pressure = db.total_pressure_torr(status, d["processing"]["total_pressure_sensitivity_a_per_torr"])
        status = status.with_columns(
            pl.when(pressure["total_pressure_torr"] > 0).then(pressure["total_pressure_torr"]).alias("total_pressure_torr")
        )
        s = minute_means(status, ["turbo_speed_hz", "turbo_power_w", "total_pressure_torr", "pump_rpm", *temp_cols])
        sections = [
            ("Turbo speed (Hz)", _single(s, "turbo_speed_hz"), False, False),
            ("Turbo power (W)", _single(s, "turbo_power_w"), False, False),
            ("Total pressure (Torr)", _single(s, "total_pressure_torr"), True, True),
            ("Water pump (rpm)", _single(s, "pump_rpm"), False, False),
        ]
        fig = _linked(sections, "Vacuum system and sample pump", spans)
        out.append(
            (
                fig,
                ("Lander status (!: lines), 1-minute means. Total pressure is the RGA total-pressure current over the "
                "total-pressure sensitivity in processing.parquet; non-positive readings (gauge off) are left out "
                "before averaging. Background bands mark which chamber was being sampled."),
            )
        )
    sections = []
    if health is not None and not health.is_empty():
        h = minute_means(health, [c for c, _ in db._SYSTEM_HEALTH_PANELS])
        sections += [(label, _single(h, col), False, False) for col, label in db._SYSTEM_HEALTH_PANELS]
    if status is not None and not status.is_empty():
        sections.append(
            (
                "Turbo temperatures (°C)",
                [_line(s, c, c.removeprefix("turbo_").removesuffix("_c")) for c in temp_cols],
                False,
                False,
            )
        )
    if sections:
        fig = _linked(sections, "Power and temperatures", spans)
        source = "system health (surface events log) and " if health is not None and not health.is_empty() else ""
        out.append((fig, f"Supply and electronics from {source}lander status, 1-minute means."))
    return out


def rga_figs(d: dict) -> list[tuple[go.Figure, str]]:
    cycles = d["tables"]["egcf_chamber_cycles"]
    if cycles is None or cycles.is_empty():
        return []
    masses = db.discover_masses(cycles)
    colors = db.mass_color_map(masses)
    chambers = sorted(cycles["chamber"].unique().to_list())
    torr_traces, ratio_traces = [], []
    for chamber in chambers:
        c = cycles.filter(pl.col("chamber") == chamber)
        symbol = _CHAMBER_SYMBOLS.get(chamber, "square")
        ratios = db.rga_wide_ratio_to_mass(c, masses, ts_col="timestamp")
        for m in masses:
            marker = {"color": colors[m], "symbol": symbol, "size": 6}
            first = chamber == chambers[0]
            torr_traces.append(
                go.Scatter(
                    x=c["timestamp"], y=c[f"mass_{m}_torr"], mode="markers", marker=marker, name=f"m/z {m}",
                    legendgroup=str(m), showlegend=first,
                )
            )
            r = ratios.filter(pl.col("mass") == m)
            if not r.is_empty():
                ratio_traces.append(
                    go.Scatter(
                        x=r["ts"], y=r["ratio"], mode="markers", marker=marker, name=f"m/z {m}",
                        legendgroup=str(m), showlegend=False,
                    )
                )
    for chamber in chambers:
        torr_traces.append(
            go.Scatter(
                x=[None], y=[None], mode="markers", name=chamber,
                marker={"color": "#8a8984", "symbol": _CHAMBER_SYMBOLS.get(chamber, "square"), "size": 8},
            )
        )
    sections = [
        ("Partial pressure (Torr), chamber-cycle means", torr_traces, True, True),
        ("Ion current ratio to m/z 40 (Ar), chamber-cycle means", ratio_traces, False, True),
    ]
    fig = _linked(sections, "RGA", None)
    fig.update_layout(height=PANEL_HEIGHT * 2 * 1.4 + 160)
    sens = d["processing"]["partial_pressure_sensitivity_a_per_torr"]
    caption = (
        f"Layer C chamber-cycle means (one point per cycle, averaged over [cycle start + "
        f"{d['processing']['settle_offset_s']:g} s, next valve transition)); circles C1, diamonds C2. Partial pressure "
        f"uses one sensitivity for all masses ({sens * 1e3:.4g} mA/Torr). Ratios are of raw ion currents, not corrected "
        "for relative sensitivity."
    )
    return [(fig, caption)]


def water_figs(d: dict) -> list[tuple[go.Figure, str]]:
    tables = d["tables"]
    spans = db.active_chamber_spans(tables["valve"])
    out = []
    scalup = tables["scalup"]
    s = None
    if scalup is not None and not scalup.is_empty():
        panels = [
            ("temp_degc", "Temperature (°C)"),
            ("sal_psu", "Salinity (PSU)"),
            ("oxygen_mgl", "Oxygen (mg/L)"),
            ("ph", "pH"),
        ]
        s = minute_means(scalup, [c for c, _ in panels])
        sections = [(label, _single(s, col), False, False) for col, label in panels]
        out.append(
            (
                _linked(sections, "SCALUP sonde (sample line)", spans),
                ("SCALUP water quality on the sample line, 1-minute means. Background bands mark which chamber was being "
                "sampled; values in unshaded (flush) spans are not from either chamber."),
            )
        )
    sections = []
    hobo = tables["hobo_oxygen"]
    if hobo is not None and not hobo.is_empty():
        colors = db._hobo_location_colors(hobo)
        traces = [
            _line(break_gaps(hobo.filter(pl.col("location") == loc).sort("ts"), "location"), "oxygen_mgl", loc, color)
            for loc, color in colors.items()
        ]
        if s is not None:
            traces.append(_line(s, "oxygen_mgl", "SCALUP", db._SCALUP_OXYGEN_COLOR, width=1))
        sections.append(("Oxygen (mg/L)", traces, False, False))
        sections.append(
            (
                "HOBO temperature (°C)",
                [
                    go.Scatter(
                        x=g["ts"], y=g["temp_degc"], mode="lines", name=loc, line={"color": color, "width": 1.5},
                        legendgroup=loc, showlegend=False,
                    )
                    for loc, color in colors.items()
                    for g in [break_gaps(hobo.filter(pl.col("location") == loc).sort("ts"), "location")]
                ],
                False,
                False,
            )
        )
        for trace in sections[0][1]:
            trace.legendgroup = trace.name
        diff = db._hobo_mesocosm_diff_section(hobo, colors)
        if diff is not None:
            for trace in diff[1]:
                trace.update(showlegend=False, legendgroup=trace.name)
            sections.append(diff)
    par = tables["par"]
    if par is not None and not par["par_umol_m2_s"].drop_nulls().is_empty():
        sections.append(("PAR (µmol photons m⁻² s⁻¹)", _single(break_gaps(par.sort("ts")), "par_umol_m2_s"), False, False))
    if sections:
        caption = (
            "HOBO oxygen/temperature loggers at their native 1-minute logging interval, SCALUP oxygen (1-minute means) "
            "for comparison, mesocosm minus chamber oxygen (nearest-timestamp match), and ambient PAR from the Odyssey "
            "logger at its native interval."
        )
        if hobo is None or hobo.is_empty():
            caption = "Ambient PAR from the Odyssey logger at its native logging interval."
        out.append((_linked(sections, "Loggers: oxygen and light", spans), caption))
    return out


def _chamber_labels(df: pl.DataFrame) -> dict[str, str]:
    if "condition" not in df.columns:
        return {c: c for c in df["chamber"].unique().to_list()}
    return {c: f"{c} ({cond})" for c, cond in df.select("chamber", "condition").unique().iter_rows()}


def flux_over_time_fig(d: dict) -> tuple[go.Figure, str] | None:
    fluxes = d["tables"]["egcf_fluxes"]
    if fluxes is None or fluxes.is_empty():
        return None
    units = db.flux_variable_units(fluxes)
    chambers = sorted(fluxes["chamber"].unique().to_list())
    colors = db.chamber_color_map(chambers)
    labels = _chamber_labels(fluxes)
    sections = []
    for variable, label in FLUX_VARIABLES.items():
        g = fluxes.filter(pl.col("variable") == variable).sort("experiment_start")
        if g.is_empty():
            continue
        traces = []
        for chamber in chambers:
            gc = g.filter(pl.col("chamber") == chamber)
            traces.append(
                go.Scatter(
                    x=gc["experiment_start"], y=gc["output_value"], mode="lines+markers+text",
                    text=[str(e) for e in gc["experiment_number"]], textposition="top center",
                    textfont={"size": 9, "color": _TEXT_SECONDARY},
                    name=labels[chamber], legendgroup=chamber, showlegend=not sections,
                    line={"color": colors[chamber], "width": 1}, marker={"color": colors[chamber], "size": 7},
                    customdata=gc.select("n_points", "r2").to_numpy(),
                    hovertemplate="experiment %{text}<br>%{x}<br>flux %{y:.4g}<br>n=%{customdata[0]}, r²=%{customdata[1]:.3f}<extra></extra>",
                )
            )
        sections.append((f"{label} ({_unit(units[variable])})", traces, False, False, True))
    fig = _linked(sections, "Fluxes by experiment", None)
    fig.update_layout(hovermode="closest")
    caption = (
        "Layer D flux per experiment and chamber, plotted at experiment start and labelled with the experiment number; "
        "positive is out of the sediment into the chamber. Fit r² and n are in the table below."
    )
    return fig, caption


def flux_summary_table(d: dict) -> str:
    fluxes = d["tables"]["egcf_fluxes"]
    if fluxes is None or fluxes.is_empty():
        return "_No fluxes._"
    keys = ["experiment_number", "chamber", "experiment_start"]
    wide = fluxes.group_by(keys).agg(pl.col("n_points").max().alias("n")).sort("experiment_number", "chamber")
    for variable in FLUX_VARIABLES:
        g = fluxes.filter(pl.col("variable") == variable)
        if g.is_empty():
            continue
        wide = wide.join(
            g.select(
                *keys,
                pl.format(
                    "{} ({})",
                    pl.col("output_value").map_elements(_num, return_dtype=pl.Utf8),
                    pl.col("r2").map_elements(lambda v: "–" if v is None or math.isnan(v) else f"{v:.2f}", return_dtype=pl.Utf8),
                ).alias(f"{FLUX_SHORT[variable]} (r²)"),
            ),
            on=keys,
            how="left",
        )
    wide = wide.with_columns(pl.col("experiment_start").dt.strftime("%m-%d %H:%M")).rename(
        {"experiment_number": "exp", "experiment_start": "start (UTC)", "n": "n cycles"}
    )
    return md_table(wide)


def par_daily_table(d: dict) -> str | None:
    daily = d["tables"]["par_daily"]
    if daily is None or daily.is_empty():
        return None
    return md_table(
        daily.select(
            pl.col("date").cast(pl.Utf8).alias("date (UTC)"),
            pl.col("coverage").round(2),
            pl.col("dli_mol_m2_d").round(2).alias("DLI (mol m⁻² d⁻¹)"),
            pl.col("max_par_umol_m2_s").round(0).alias("max PAR (µmol m⁻² s⁻¹)"),
        ),
        4,
    )


def metabolism_fig(d: dict) -> tuple[go.Figure, str] | None:
    metabolism, pi_fit = d["tables"]["egcf_metabolism"], d["tables"]["egcf_pi_fit"]
    if metabolism is None:
        return None
    placed = metabolism.drop_nulls("par_chamber_umol_m2_s")
    if placed.is_empty():
        return None
    chambers = sorted(placed["chamber"].unique().to_list())
    colors = db.chamber_color_map(chambers)
    labels = _chamber_labels(placed)
    fig = go.Figure()
    for chamber in chambers:
        for used in (True, False):
            g = placed.filter((pl.col("chamber") == chamber) & (pl.col("used") == used))
            if g.is_empty():
                continue
            trace = db._metabolism_scatter(g, "o2_flux_mmol_m2_h", chamber, colors[chamber], used, True)
            trace.update(
                hovertext=trace.text,
                hovertemplate=trace.hovertemplate.replace("%{text}", "%{hovertext}"),
                text=[str(e) for e in g["experiment_number"]],
                mode="markers+text",
                textposition="top center",
                textfont={"size": 10, "color": _TEXT_SECONDARY},
                name=labels[chamber] if used else f"{labels[chamber]}, excluded",
            )
            fig.add_trace(trace)
    par_max = float(placed["par_chamber_umol_m2_s"].max())
    for trace in db._pi_curve_traces(pi_fit, par_max, colors):
        fig.add_trace(trace)
    fig.add_hline(y=0, line={"color": db._ZERO_LINE_COLOR, "width": 1})
    lo, hi = min(0.0, placed["o2_flux_mmol_m2_h"].min()), max(0.0, placed["o2_flux_mmol_m2_h"].max())
    pad = 0.15 * (hi - lo or 1.0)
    fig.update_yaxes(range=[lo - pad, hi + pad])
    fig.update_xaxes(title_text="Chamber PAR (µmol photons m⁻² s⁻¹)")
    fig.update_yaxes(title_text="O₂ flux (mmol m⁻² h⁻¹)")
    style(fig, "O₂ flux vs light, with Jassby–Platt P–I fit", 520)
    transmittance = d["processing"]["chamber_par_transmittance"]
    caption = (
        f"Each point is one experiment's O₂ flux against its mean chamber PAR (ambient × transmittance "
        f"{transmittance:g}), labelled with the experiment number. Hollow markers are excluded fluxes (reasons in "
        f"the metabolism table); dark is mean PAR below {d['processing']['dark_par_threshold_umol_m2_s']:g} "
        "µmol m⁻² s⁻¹. Lines are the per-chamber P–I fits; the y range is set by the points, so a fit that runs "
        "off-scale is clipped, not hidden."
    )
    return fig, caption


def metabolism_table(d: dict) -> str | None:
    m = d["tables"]["egcf_metabolism"]
    if m is None or m.is_empty():
        return None
    return md_table(
        m.sort("experiment_number", "chamber").select(
            pl.col("experiment_number").alias("exp"),
            "chamber",
            "period",
            pl.col("par_chamber_umol_m2_s").round(0).alias("PAR"),
            pl.col("o2_flux_mmol_m2_h").alias("O₂ flux"),
            pl.col("ncp_mmol_m2_h").alias("NCP"),
            pl.col("gpp_mmol_m2_h").alias("GPP"),
            pl.col("r2").round(2).alias("r²"),
            pl.when(pl.col("used")).then(pl.lit("")).otherwise(pl.col("excluded_reason")).alias("excluded"),
        )
    )


def pi_fit_warning(d: dict) -> str | None:
    """A callout naming any converged fit whose parameter SE exceeds the estimate, or None."""
    p = d["tables"]["egcf_pi_fit"]
    if p is None or p.is_empty():
        return None
    loose = p.filter(
        pl.col("converged")
        & (
            (pl.col("pmax_se") > pl.col("pmax_mmol_m2_h").abs())
            | (pl.col("alpha_se") > pl.col("alpha_mmol_m2_h_per_par").abs())
            | (pl.col("r_fit_se") > pl.col("r_fit_mmol_m2_h").abs())
        )
    )
    if loose.is_empty():
        return None
    detail = ", ".join(f"{c} ({n_light} light, {n_dark} dark points)" for c, n_light, n_dark in loose.select("chamber", "n_light", "n_dark").iter_rows())
    return (
        "::: {.callout-warning title=\"P–I fit not constrained\"}\n"
        f"The fit converged but at least one parameter's standard error exceeds its estimate for {detail}. "
        "Treat Pmax, α, R (fit) and Ik as unconstrained.\n:::"
    )


def pi_fit_table(d: dict) -> str | None:
    p = d["tables"]["egcf_pi_fit"]
    if p is None or p.is_empty():
        return None

    def pm(v: str, se: str) -> pl.Expr:
        return pl.format(
            "{} ± {}",
            pl.col(v).map_elements(_num, return_dtype=pl.Utf8),
            pl.col(se).map_elements(lambda x: _num(x, 2), return_dtype=pl.Utf8),
        )

    return md_table(
        p.select(
            "chamber",
            "condition",
            pl.format("{}/{}", "n_light", "n_dark").alias("n light/dark"),
            pm("pmax_mmol_m2_h", "pmax_se").alias("Pmax"),
            pm("alpha_mmol_m2_h_per_par", "alpha_se").alias("α"),
            pm("r_fit_mmol_m2_h", "r_fit_se").alias("R (fit)"),
            pl.col("r_dark_mmol_m2_h").alias("R (dark mean)"),
            pl.col("ik_umol_m2_s").round(0).alias("Ik"),
            pl.col("fit_r2").round(2).alias("fit r²"),
            "converged",
        )
    )


def cross_deployment_flux_fig(deployments: list[dict], labels: dict[str, str]) -> tuple[go.Figure, str]:
    """Flux distributions per deployment and chamber; every deployment is kept, flagged ones are labelled."""
    fluxes = pl.concat([d["tables"]["egcf_fluxes"] for d in deployments], how="diagonal_relaxed")
    units = db.flux_variable_units(fluxes)
    variables = ["oxygen", "n2_denitrification", "h_ion"]
    colors = db.chamber_color_map(fluxes["chamber"].unique().to_list())
    order = [labels[d["id"]] for d in deployments]
    fig = make_subplots(
        rows=len(variables), cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=[f"{FLUX_VARIABLES[v]} ({_unit(units[v])})" for v in variables],
    )
    for row, variable in enumerate(variables, start=1):
        g = fluxes.filter(pl.col("variable") == variable).with_columns(
            pl.col("deployment_id").replace_strict(labels).alias("label")
        )
        for chamber in sorted(colors):
            gc = g.filter(pl.col("chamber") == chamber)
            fig.add_trace(
                go.Box(
                    x=gc["label"], y=gc["output_value"], name=chamber, offsetgroup=chamber, legendgroup=chamber,
                    showlegend=row == 1, marker={"color": colors[chamber], "size": 5}, line={"width": 1.5},
                    boxpoints="all", jitter=0.4, pointpos=0, fillcolor="rgba(0,0,0,0)",
                    text=[f"experiment {e}" for e in gc["experiment_number"]],
                ),
                row=row, col=1,
            )
        fig.add_hline(y=0, line={"color": db._ZERO_LINE_COLOR, "width": 1}, row=row, col=1)
    fig.update_xaxes(categoryorder="array", categoryarray=order)
    fig.update_layout(boxmode="group")
    style(fig, "Fluxes across deployments", 3 * 260 + 160)
    caption = (
        "Every experiment's flux, by deployment and chamber (points), with box = quartiles. All deployments are "
        "shown, including flagged ones (see labels). The tank tests have no PAR record, so light and dark are mixed in them."
    )
    return fig, caption


def cross_deployment_table(deployments: list[dict], labels: dict[str, str]) -> str:
    fluxes = pl.concat([d["tables"]["egcf_fluxes"] for d in deployments], how="diagonal_relaxed")
    table = (
        fluxes.filter(pl.col("variable").is_in(["oxygen", "n2_denitrification", "h_ion"]))
        .group_by("deployment_id", "chamber", "variable")
        .agg(pl.col("output_value").median().alias("median"), pl.len().alias("n"))
        .with_columns(pl.col("variable").replace_strict(FLUX_VARIABLES))
        .pivot(index=["deployment_id", "chamber"], on="variable", values="median")
        .select("deployment_id", "chamber", *[v for v in FLUX_VARIABLES.values() if v != "Temperature change"])
    )
    order = {d["id"]: i for i, d in enumerate(deployments)}
    return md_table(
        table.with_columns(pl.col("deployment_id").replace_strict(order).alias("_o"))
        .sort("_o", "chamber")
        .with_columns(pl.col("deployment_id").replace_strict({k: v.replace("<br>", " ") for k, v in labels.items()}))
        .drop("_o")
        .rename({"deployment_id": "deployment"})
    )


# ---------------------------------------------------------------- turbo speed test


def load_turbo() -> dict:
    """The saved turbo test, with the stats tables rebuilt from rga_tagged/turbo_tagged.

    Only the tagged readings, speed changes and a few stats were written, and the
    saved per-segment stats predate the drift-robust RSD columns, so every stats
    table is recomputed with the pure functions in turbo_speed_analysis.py.
    """
    root = processed_root() / TURBO_TEST
    changes = pl.read_parquet(root / "speed_changes.parquet")
    turbo_tagged = pl.read_parquet(root / "turbo_tagged.parquet")
    rga_tagged = tsa.add_ratio(pl.read_parquet(root / "rga_tagged.parquet"), tsa.REF_MASS)
    settle_s = 60 * math.ceil(rga_tagged.filter("in_settle")["t_since_acq_on_s"].max() / 60)
    seg_keys = ["segment", "leg", "setpoint", "change_ts"]
    by_speed = tsa.mass_stats(rga_tagged, ["setpoint"])
    rsd_tests = tsa.rsd_speed_tests(rga_tagged)
    noise_tests = tsa.rsd_speed_tests(rga_tagged, relative=False)
    return {
        "settle_s": settle_s,
        "speed_changes": changes,
        "turbo_tagged": turbo_tagged,
        "rga_tagged": rga_tagged,
        "mass_stats_by_speed": by_speed,
        "mass_stats_by_segment": tsa.mass_stats(rga_tagged, seg_keys),
        "tp_by_speed": tsa.tp_stats(turbo_tagged, ["setpoint"]),
        "tp_by_segment": tsa.tp_stats(turbo_tagged, seg_keys),
        "rsd_speed_tests": rsd_tests,
        "noise_floor_summary": tsa.noise_floor_summary(by_speed, rsd_tests, noise_tests),
        "ratio_stats_by_segment": tsa.ratio_stats(rga_tagged, seg_keys, tsa.REF_MASS),
        "ratio_speed_tests": tsa.rsd_speed_tests(rga_tagged.filter(pl.col("mass") != tsa.REF_MASS), col="ratio"),
    }


def turbo_timeseries_static(t: dict) -> go.Figure:
    """Print version of tsa.fig_timeseries: 1-minute means as lines instead of ~10k WebGL markers."""
    turbo = t["turbo_tagged"].sort("ts")
    m = minute_means(turbo, ["tp_torr", "speed_hz"])
    changes = t["speed_changes"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(_line(m, "tp_torr", "TP", "#2a78d6"), row=1, col=1)
    fig.add_trace(_line(m, "speed_hz", "measured speed", "#2a78d6"), row=2, col=1)
    end = turbo["ts"].max()
    fig.add_trace(
        go.Scatter(
            x=[*changes["ts"].to_list(), end], y=[*changes["setpoint"].to_list(), changes["setpoint"][-1]],
            mode="lines", line={"shape": "hv", "width": 2, "color": "#eb6834"}, name="setpoint",
        ),
        row=3, col=1,
    )
    for ts, acq_on in changes.select("ts", "acq_on_ts").iter_rows():
        fig.add_vrect(x0=ts, x1=(acq_on or ts) + timedelta(seconds=t["settle_s"]), fillcolor="#8a8984", opacity=0.2, line_width=0)
    fig.update_yaxes(title_text="TP (Torr)", type="log", exponentformat="power", row=1, col=1)
    fig.update_yaxes(title_text="measured speed (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="setpoint (Hz)", row=3, col=1)
    fig.update_layout(showlegend=False)
    return style(fig, "Total pressure, measured speed and setpoint", 640)


def turbo_tp_table(t: dict) -> str:
    return md_table(
        t["tp_by_speed"].select(
            pl.col("setpoint").alias("setpoint (Hz)"),
            "n",
            pl.col("speed_hz_mean").round(0).alias("speed (Hz)"),
            pl.col("power_w_mean").round(1).alias("power (W)"),
            pl.col("tp_torr_mean").alias("TP (Torr)"),
            pl.col("tp_torr_std").alias("TP sd (Torr)"),
            pl.col("tp_rsd_pct").round(2).alias("TP RSD (%)"),
        )
    )


def turbo_noise_floor_table(t: dict) -> str:
    nf = t["noise_floor_summary"]
    return md_table(
        nf.select(
            "mass",
            pl.col("signal_lo_hi").alias("signal lo/hi"),
            pl.col("noise_lo_hi").alias("noise lo/hi"),
            pl.col("rsd_lo_hi").alias("RSD lo/hi"),
            pl.col("p_adj_rsd").alias("p (RSD)"),
            pl.col("p_adj_noise").alias("p (noise)"),
        )
    )


# ---------------------------------------------------------------- CDEM vs FC test


def load_cdem() -> dict:
    root = processed_root() / CDEM_TEST
    summary = pl.read_parquet(root / "cdem_fc_summary.parquet")
    _readings, events, _status = cfc.read_log(raw_root() / CDEM_LOG)
    return {
        "long": pl.read_parquet(root / "cdem_fc_rga_long.parquet"),
        "stats": pl.read_parquet(root / "cdem_fc_detector_stats.parquet"),
        "summary": summary,
        "total_pressure": pl.read_parquet(root / "total_pressure.parquet"),
        "events": events,
        "cdem_gain": float(summary["cdem_gain_applied"][0]),
        "sp_ma_per_torr": float(summary["sp_ma_per_torr"][0]),
    }


def cdem_figs(c: dict) -> list[tuple[go.Figure, str]]:
    pp = f"SP {c['sp_ma_per_torr']:.4g} mA/Torr; CDEM gain {c['cdem_gain']:.0f} applied to all masses"
    ts = cfc.fig_timeseries(c["long"], c["events"], "torr", "partial pressure (Torr)", "RGA partial pressure")
    ts.update_layout(title=None)
    by_det = cfc.fig_by_detector(c["long"], "torr", "pressure (Torr)", "RGA partial pressure", c["total_pressure"])
    by_det.update_layout(title=None)
    gain = cfc.fig_gain(c["summary"])
    gain.update_layout(title=None, height=420)
    rel = cfc.fig_relative_error(c["stats"])
    rel.update_layout(title=None, height=900)
    for fig in (ts, by_det, gain, rel):
        fig.update_layout(margin={"t": 70}, legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0})
    return [
        (
            ts,
            (f"Partial pressure per mass, FC (blue) vs CDEM (red), on surface receipt time ({pp}). Dotted lines are "
            "EMON/EMOFF/AON/AOFF control events; shaded spans are the stable analysis windows."),
        ),
        (
            by_det,
            (f"All masses per detector run ({pp}). Black: total pressure from the !: status lines during FC acquisition "
            "(it reads 0 while the CDEM is on). Shaded: analysis windows."),
        ),
        (gain, ("CDEM/FC current ratio per mass from stable-window means (± propagated SE). The runs are sequential, so "
         "this assumes the sample did not change between them.")),
        (rel, ("Relative error per mass and detector: scan-to-scan noise (MAD- and SD-based), stable-window RSD, and RSD "
         "of the ratio to m/z 40. Log scale.")),
    ]


def cdem_summary_table(c: dict) -> str:
    return md_table(
        c["summary"].select(
            "mass",
            pl.col("cdem_fc_gain").round(0).alias("CDEM/FC gain"),
            pl.col("cdem_fc_gain_se").round(1).alias("gain SE"),
            pl.col("fc_noise_mad_rsd_pct").alias("FC noise (%)"),
            pl.col("cdem_noise_mad_rsd_pct").alias("CDEM noise (%)"),
            pl.col("noise_improvement_fc_over_cdem").alias("FC/CDEM noise"),
            pl.col("fc_mean_stable_torr").alias("FC (Torr)"),
            pl.col("cdem_mean_stable_torr").alias("CDEM (Torr)"),
        )
    )
