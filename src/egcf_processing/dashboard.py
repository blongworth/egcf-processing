"""Read-only Streamlit dashboard over an already-processed data/processed/ directory.

Data-loading and transform helpers are kept free of Streamlit calls so they
can be unit tested directly; only the render_*/main functions touch `st`.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots

from egcf_processing.aggregate import (
    DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR,
    DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR,
    RAW_CURRENT_AMPS_PER_COUNT,
    aggregate_onto_windows,
    match_readings_to_windows,
)
from egcf_processing.combine import RGA_SCHEMA, SCALUP_SCHEMA, STATUS_SCHEMA, duration_cols_to_seconds
from egcf_processing.cycles import chamber_cycle_windows
from egcf_processing.flux import compute_fluxes, linear_fit
from egcf_processing.metabolism import jassby_platt
from egcf_processing.par import DEFAULT_MIN_DAY_COVERAGE, daily_max_trend, daily_par
from egcf_processing.pipeline import DEFAULT_SETTLE_OFFSET_S

TABLE_NAMES = [
    "status",
    "system_health",
    "rga",
    "scalup",
    "valve",
    "par",
    "egcf_rga_scans",
    "egcf_chamber_cycles",
    "egcf_fluxes",
    "egcf_metabolism",
    "egcf_pi_fit",
]

_MASS_COLOR_PALETTE = px.colors.qualitative.Plotly

ARGON_MASS = 40

_SCALUP_PANELS = [
    ("temp_degC", "Temperature (degC)"),
    ("sal_PSU", "Salinity (PSU)"),
    ("pressure_mbar", "Pressure (mbar)"),
    ("oxygen_mgL", "Oxygen (mg/L)"),
    ("pH", "pH"),
]

_STATUS_DIRECT_RAW_COLS = {
    "turbo_speed_hz": "turbo_speed_hz",
    "turbo_power_w": "turbo_power_w",
    "water_pump_rpm": "pump_rpm",
}


def load_table(data_dir: Path, name: str) -> pl.DataFrame | None:
    """Load {name}.parquet, falling back to {name}.csv; None if neither exists."""
    parquet_path = data_dir / f"{name}.parquet"
    if parquet_path.exists():
        return pl.read_parquet(parquet_path)
    csv_path = data_dir / f"{name}.csv"
    if csv_path.exists():
        return pl.read_csv(csv_path, try_parse_dates=True)
    return None


def with_elapsed_time_s(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize elapsed_time (Duration in parquet, float seconds in csv) to elapsed_time_s."""
    if "elapsed_time" not in df.columns:
        return df
    if df.schema["elapsed_time"].base_type() == pl.Duration:
        return df.with_columns(pl.col("elapsed_time").dt.total_seconds().alias("elapsed_time_s"))
    return df.with_columns(pl.col("elapsed_time").alias("elapsed_time_s"))


def load_all(data_dir: Path) -> dict[str, pl.DataFrame | None]:
    tables = {name: load_table(data_dir, name) for name in TABLE_NAMES}
    for name in ("egcf_rga_scans", "egcf_chamber_cycles"):
        if tables[name] is not None:
            tables[name] = with_elapsed_time_s(tables[name])
    return tables


def table_ts_col(df: pl.DataFrame) -> str | None:
    """Name of a table's time column: `timestamp` for Layer B/C, `ts` for Layer A."""
    for candidate in ("timestamp", "ts"):
        if candidate in df.columns:
            return candidate
    return None


def tables_time_bounds(tables: dict[str, pl.DataFrame | None]) -> tuple[datetime, datetime] | None:
    """Earliest and latest timestamp across every loaded table, or None if there is none."""
    stamps = []
    for df in tables.values():
        if df is None or df.is_empty():
            continue
        ts_col = table_ts_col(df)
        if ts_col is None:
            continue
        lo, hi = df[ts_col].min(), df[ts_col].max()
        if lo is not None and hi is not None:
            stamps.append((lo, hi))
    if not stamps:
        return None
    return min(lo for lo, _ in stamps), max(hi for _, hi in stamps)


def filter_tables_to_range(
    tables: dict[str, pl.DataFrame | None], start: datetime, end: datetime
) -> dict[str, pl.DataFrame | None]:
    """Restrict every timestamped table to [start, end], inclusive.

    Done once up front so each tab's plotting -- and the transforms feeding it
    (unit conversion, ratio joins, cycle-window detection) -- runs over the
    visible slice instead of the whole deployment.
    """
    filtered = {}
    for name, df in tables.items():
        ts_col = None if df is None else table_ts_col(df)
        filtered[name] = df if ts_col is None else df.filter(pl.col(ts_col).is_between(start, end))
    return filtered


TIME_RANGE_PRESETS: dict[str, timedelta | None] = {
    "All data": None,
    "Last hour": timedelta(hours=1),
    "Last 6 hours": timedelta(hours=6),
    "Last 24 hours": timedelta(days=1),
    "Last 7 days": timedelta(days=7),
}

CUSTOM_TIME_RANGE = "Custom"

_FINE_STEP = timedelta(minutes=1)


def preset_time_range(preset: str, lo: datetime, hi: datetime) -> tuple[datetime, datetime]:
    """Resolve a named preset to an absolute range, anchored at the *end* of the data.

    Anchoring at ``hi`` rather than "now" is what makes these useful on an
    already-recovered deployment, where the data is weeks old.
    """
    window = TIME_RANGE_PRESETS[preset]
    if window is None:
        return lo, hi
    return max(lo, hi - window), hi


def date_range_bounds(dates, lo: datetime, hi: datetime) -> tuple[datetime, datetime]:
    """Widen a selected date (or first/last of a date range) to whole-day bounds, clamped to the data.

    ``st.date_input`` in range mode returns a 1-tuple mid-selection, before the
    second date is picked -- that is treated as a single day, not an error.
    """
    selected = list(dates) if isinstance(dates, (list, tuple)) else [dates]
    if not selected:
        return lo, hi
    start = datetime.combine(selected[0], datetime.min.time())
    end = datetime.combine(selected[-1], datetime.max.time())
    return max(lo, start), min(hi, end)


def align_slider_bounds(lo: datetime, hi: datetime, step: timedelta) -> tuple[datetime, datetime]:
    """Round the upper bound up so it sits on a whole number of steps from the lower.

    st.slider only offers positions at ``min_value + k * step``, so unless the
    span is an exact multiple of the step the true maximum is unreachable --
    which is why the tail of a deployment could not be selected with the
    default (1-day) step. Overshooting past the last sample is harmless: the
    filter is inclusive and there is no data beyond it.
    """
    steps = max(-((lo - hi) // step), 1)
    return lo, lo + step * steps


def render_time_range_control(bounds: tuple[datetime, datetime]) -> tuple[datetime, datetime]:
    """Sidebar time-range picker: presets, plus a two-stage day + fine-time custom mode.

    A single slider over the whole deployment cannot resolve short windows (a
    39-day span is hours per pixel), so custom mode narrows by calendar day
    first and only then offers a minute-resolution slider *within* those days.
    """
    lo, hi = bounds
    st.header("Time range")
    st.caption("Applies to the Status and Measurements tabs; the Experiment Data tab has its own selector.")
    preset = st.selectbox("Preset", [*TIME_RANGE_PRESETS, CUSTOM_TIME_RANGE], key="time_range_preset")

    if preset != CUSTOM_TIME_RANGE:
        start, end = preset_time_range(preset, lo, hi)
    else:
        dates = st.date_input(
            "Days",
            value=(lo.date(), hi.date()),
            min_value=lo.date(),
            max_value=hi.date(),
            key="time_range_days",
        )
        day_lo, day_hi = date_range_bounds(dates, lo, hi)
        slider_lo, slider_hi = align_slider_bounds(day_lo, day_hi, _FINE_STEP)
        # Deliberately unkeyed: changing the day selection changes the widget's
        # min/max, and the reset back to the full selected span is what we want.
        start, end = st.slider(
            "Time within those days",
            min_value=slider_lo,
            max_value=slider_hi,
            value=(slider_lo, slider_hi),
            step=_FINE_STEP,
        )

    st.caption(f"{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}")
    return start, end


def discover_masses(df: pl.DataFrame) -> list[int]:
    """Discover mass ids from mass_{m}_{suffix} columns, sorted ascending."""
    return sorted({int(c.split("_")[1]) for c in df.columns if c.startswith("mass_")})


def mass_color_map(masses: list[int]) -> dict[int, str]:
    """Assign each mass id a fixed color, keyed by its position in the ascending mass list.

    This makes mass colors consistent across every RGA panel (full RGA data,
    RGA-cycle-averaged, chamber-cycle-averaged) regardless of which subset of
    masses is selected in any one panel's multiselect -- panel-local trace
    order would otherwise give the same mass a different color in each panel.
    """
    return {m: _MASS_COLOR_PALETTE[i % len(_MASS_COLOR_PALETTE)] for i, m in enumerate(sorted(masses))}


def chamber_color_map(chambers: list[str]) -> dict[str, str]:
    """Assign each chamber a fixed color, keyed by its position in the sorted list.

    Same rationale as mass_color_map: a given chamber must be the same color in
    every plot on the tab, regardless of which subset a particular plot shows.
    """
    return {c: _MASS_COLOR_PALETTE[i % len(_MASS_COLOR_PALETTE)] for i, c in enumerate(sorted(chambers))}


def flux_variable_units(fluxes: pl.DataFrame) -> dict[str, str]:
    """Map each flux variable to its output_unit (constant per variable)."""
    return dict(fluxes.select("variable", "output_unit").unique().iter_rows())


def rga_current_to_unit(current: pl.Expr, unit: str, sensitivity_a_per_torr: float) -> pl.Expr:
    """Convert a raw RGA ion-current expression to the requested display unit."""
    if unit == "raw":
        return current
    amps = current * RAW_CURRENT_AMPS_PER_COUNT
    if unit == "amps":
        return amps
    return amps / sensitivity_a_per_torr


def rga_full_ratio_to_mass(rga: pl.DataFrame, masses: list[int], reference_mass: int = ARGON_MASS) -> pl.DataFrame:
    """Ratio of each mass's raw current to reference_mass's, for the full (unaveraged) RGA table.

    The RGA scans one mass at a time, so different masses never share an exact
    timestamp -- each mass's readings are paired with the nearest-in-time
    reference_mass reading (join_asof, "nearest") rather than an exact match.
    The ratio is unit-invariant (raw counts, Amps, and Torr all share the same
    linear scale per reading, which cancels out), so this always uses raw
    current regardless of the unit selected elsewhere in the UI. Returns a
    long dataframe (ts, mass, ratio); reference_mass is excluded from the
    output, as are masses with no data or a zero reference reading.
    """
    schema = {"ts": pl.Datetime, "mass": pl.Int64, "ratio": pl.Float64}
    reference = (
        rga.filter(pl.col("mass") == reference_mass).select("ts", pl.col("current").alias("_ref_current")).sort("ts")
    )
    if reference.is_empty():
        return pl.DataFrame(schema=schema)
    frames = []
    for m in masses:
        if m == reference_mass:
            continue
        series = rga.filter(pl.col("mass") == m).select("ts", "current").sort("ts")
        if series.is_empty():
            continue
        joined = series.join_asof(reference, on="ts", strategy="nearest").filter(pl.col("_ref_current") != 0)
        if joined.is_empty():
            continue
        frames.append(
            joined.with_columns((pl.col("current") / pl.col("_ref_current")).alias("ratio"), mass=pl.lit(m)).select(
                "ts", "mass", "ratio"
            )
        )
    return pl.concat(frames) if frames else pl.DataFrame(schema=schema)


def rga_wide_ratio_to_mass(table: pl.DataFrame, masses: list[int], ts_col: str, reference_mass: int = ARGON_MASS) -> pl.DataFrame:
    """Ratio of each mass's averaged current to reference_mass's, for a wide per-cycle table.

    Always uses the raw *_avg columns since the ratio is unit-invariant (see
    rga_full_ratio_to_mass). Returns a long dataframe (ts, mass, ratio);
    reference_mass is excluded, as are masses with no data or a zero
    reference reading.
    """
    schema = {"ts": pl.Datetime, "mass": pl.Int64, "ratio": pl.Float64}
    ref_col = f"mass_{reference_mass}_avg"
    if ref_col not in table.columns:
        return pl.DataFrame(schema=schema)
    frames = []
    for m in masses:
        if m == reference_mass:
            continue
        col = f"mass_{m}_avg"
        if col not in table.columns:
            continue
        frame = (
            table.select(
                pl.col(ts_col).alias("ts"),
                pl.when(pl.col(ref_col) != 0).then(pl.col(col) / pl.col(ref_col)).alias("ratio"),
            )
            .drop_nulls("ratio")
            .with_columns(mass=pl.lit(m))
        )
        if not frame.is_empty():
            frames.append(frame.select("ts", "mass", "ratio"))
    return pl.concat(frames) if frames else pl.DataFrame(schema=schema)


def mass_to_argon_ratio_expr(mass: int, reference_mass: int = ARGON_MASS) -> pl.Expr:
    """Ratio of mass_{mass}_avg to mass_{reference_mass}_avg (Argon), null where Argon is zero.

    Always divides the raw *_avg columns since the ratio is unit-invariant
    (see rga_full_ratio_to_mass / rga_wide_ratio_to_mass) -- for a wide,
    already-aligned per-cycle table this is a plain column expression, no
    join needed.
    """
    reference = pl.col(f"mass_{reference_mass}_avg")
    return pl.when(reference != 0).then(pl.col(f"mass_{mass}_avg") / reference).otherwise(None)


def variable_value_expr(variable: str, is_mass_variable: bool) -> pl.Expr:
    """The "value" expression for a Variable selectbox choice on a wide cycle-averaged table.

    Mass variables are always the Argon-normalized ratio (see
    mass_to_argon_ratio_expr); everything else is just that column directly.
    """
    if is_mass_variable:
        return mass_to_argon_ratio_expr(int(variable.split("_")[1])).alias("value")
    return pl.col(variable).alias("value")


def experiment_rates(source: pl.DataFrame, variable: str, is_mass_variable: bool) -> pl.DataFrame:
    """Fit a per-(experiment, chamber) rate (see linear_fit) for one variable.

    ``source`` is a wide cycle-averaged table (elapsed_time_s, timestamp,
    experiment_number, chamber, plus the variable's own column(s)) covering
    every experiment, not just the one currently selected -- this is what
    lets the rate be plotted across the whole deployment. Returns a long
    dataframe (experiment_number, chamber, experiment_start, rate); an
    (experiment, chamber) pair with fewer than 2 valid points is omitted.
    """
    schema = {
        "experiment_number": pl.Int64,
        "chamber": pl.Utf8,
        "experiment_start": pl.Datetime,
        "rate": pl.Float64,
    }
    base = (
        source.select(
            "experiment_number",
            "chamber",
            "timestamp",
            (pl.col("elapsed_time_s") / 60).alias("elapsed_time_min"),
            variable_value_expr(variable, is_mass_variable),
        )
        .drop_nulls("value")
        .sort("timestamp")
    )
    if base.is_empty():
        return pl.DataFrame(schema=schema)
    rows = []
    for (exp_num, chamber), group in base.group_by(["experiment_number", "chamber"]):
        fit = linear_fit(group["elapsed_time_min"].to_list(), group["value"].to_list())
        if fit is None:
            continue
        slope, _ = fit
        rows.append(
            {
                "experiment_number": exp_num,
                "chamber": chamber,
                "experiment_start": group["timestamp"].min(),
                "rate": slope,
            }
        )
    return pl.DataFrame(rows, schema=schema).sort("experiment_start") if rows else pl.DataFrame(schema=schema)


def experiment_start_times(source: pl.DataFrame, ts_col: str) -> dict[int, datetime]:
    """Map each experiment_number in ``source`` to its real start time.

    Computed as ``(ts_col - elapsed_time).min()`` per experiment, same as the
    per-experiment subtitle -- ``source`` needs ``ts_col``, ``elapsed_time``
    (Duration), and ``experiment_number`` columns. If ``ts_col`` already has a
    settle offset baked in (e.g. Cycle averages' "timestamp", which is
    window_start = re_transition_ts + settle_offset_s), the caller must
    subtract that offset back out of the returned values themselves.
    """
    starts = source.group_by("experiment_number").agg((pl.col(ts_col) - pl.col("elapsed_time")).min().alias("start"))
    return dict(zip(starts["experiment_number"].to_list(), starts["start"].to_list()))


def attach_experiment_context(
    readings: pl.DataFrame,
    windows: pl.DataFrame,
    ts_col: str = "ts",
    settle_offset_s: float = 0.0,
) -> pl.DataFrame:
    """Attach chamber, experiment_number, per-reading elapsed_time, and a
    ``settled_out`` flag to raw readings.

    ``windows`` must be the (window_start, window_end, chamber, experiment_number,
    elapsed_time) table from cycles.chamber_cycle_windows(valve, settle_offset_s=0.0)
    -- settle_offset must be 0 so window_start is exactly the Re-transition
    timestamp (both so elapsed_time, the time from experiment start to that
    transition, has no settle offset baked in, making
    ``exp_start_ts = window_start - elapsed_time`` exact, and so
    ``settled_out`` below can measure time since the valve switch itself).
    Each reading's own elapsed_time is ``reading_ts - exp_start_ts`` (its own
    timestamp, not the window's), so readings within one cycle don't all
    collapse onto the same elapsed_time -- unlike the constant per-cycle
    elapsed_time already stored on a cycle-averaged table. Readings outside
    any window (a settle/flush gap, or before the first / after the last
    cycle) are dropped.

    ``settle_offset_s`` flags (via ``settled_out``) readings within that many
    seconds of the window's own start (the valve switch into this cycle),
    without removing them -- Full data is meant to show every reading, with
    settled-out ones greyed out rather than hidden. This mirrors the
    pipeline's time-based settle_offset_s (see cycles.chamber_cycle_windows)
    rather than a reading count, so the same settle value means the same
    thing whether it trims a live per-cycle average or greys out a scatter.
    """
    ctx_windows = windows.with_columns(exp_start=pl.col("window_start") - pl.col("elapsed_time"))
    matched = match_readings_to_windows(readings, ctx_windows, ts_col)
    matched = matched.with_columns(
        ((pl.col(ts_col) - pl.col("window_start")).dt.total_seconds() < settle_offset_s).alias("settled_out")
    )
    return matched.with_columns((pl.col(ts_col) - pl.col("exp_start")).alias("elapsed_time")).drop(
        "window_start", "window_end", "exp_start"
    )


def _mass_trace(ts: pl.Series, y: pl.Series, mass: int, mode: str, color_map: dict[int, str]) -> go.Scatter:
    color = color_map[mass]
    return go.Scatter(x=ts, y=y, mode=mode, name=f"mass {mass}", line={"color": color}, marker={"color": color})


def _mass_traces_from_long(long_df: pl.DataFrame, value_col: str, mode: str, color_map: dict[int, str]) -> list[go.Scatter]:
    return [
        _mass_trace(g["ts"], g[value_col], m, mode, color_map)
        for m in sorted(long_df["mass"].unique().to_list())
        for g in [long_df.filter(pl.col("mass") == m)]
    ]


def _empty_state(name: str) -> None:
    st.info(f"No {name} data available in this dataset.")


_CHAMBER_SHADE_OPACITY = 0.12

_CHAMBER_SPANS_SCHEMA = {"start": pl.Datetime, "end": pl.Datetime, "chamber": pl.Utf8}


def active_chamber_spans(valve: pl.DataFrame | None) -> pl.DataFrame:
    """[start, end) spans over which each chamber is the one being sampled.

    These are exactly Layer C's measurement cycles with no settling offset (a
    V: transition into (chamber, Re) up to the next transition), so the
    shading lines up with the windows the cycle averages are taken over. The
    Fl (flush) spans between them are left unshaded -- and note a chamber
    stays a sealed incubation across the whole experiment, so an unshaded gap
    means "not being sampled", not "not incubating".
    """
    if valve is None or valve.is_empty():
        return pl.DataFrame(schema=_CHAMBER_SPANS_SCHEMA)
    windows, _ = chamber_cycle_windows(valve, settle_offset_s=0.0)
    return windows.select(
        pl.col("window_start").alias("start"), pl.col("window_end").alias("end"), pl.col("chamber")
    )


def _shade_chamber_spans(fig: go.Figure, spans: pl.DataFrame) -> None:
    """Draw each span as one full-height band behind the traces, colored by chamber.

    One shape per span in paper-y coordinates rather than one per
    (span, subplot): every subplot matches the row-1 x axis, so a single band
    covers the whole stack -- which matters because a real deployment has
    hundreds of spans and Plotly renders every shape separately.
    """
    colors = chamber_color_map(spans["chamber"].unique().to_list())
    for start, end, chamber in spans.select("start", "end", "chamber").iter_rows():
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=start,
            x1=end,
            y0=0,
            y1=1,
            fillcolor=colors[chamber],
            opacity=_CHAMBER_SHADE_OPACITY,
            line_width=0,
            layer="below",
        )
    for chamber in sorted(colors):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"color": colors[chamber], "size": 10, "symbol": "square"},
                opacity=_CHAMBER_SHADE_OPACITY * 3,
                name=f"{chamber} active",
            )
        )


def _render_linked_timeseries(
    sections: list[tuple[str, list[go.Scatter], bool, bool]],
    title: str,
    zero_line: bool = False,
    chamber_spans: pl.DataFrame | None = None,
) -> None:
    """Render one subplot per section, stacked with a shared, zoom/pan-linked time axis.

    Each section's third element requests scientific-notation y-axis ticks,
    for the Amps/Torr panels whose magnitudes (~1e-8 to 1e-16) are unreadable
    in plain decimal. The fourth element requests a log-scale y-axis, for the
    RGA mass-current panels whose values span several orders of magnitude.

    ``zero_line`` draws a y=0 reference on every subplot -- for flux panels,
    where the sign carries the meaning (efflux above the line, uptake below)
    and the eye needs the crossing point.

    ``chamber_spans`` (see active_chamber_spans) shades the background by
    which chamber was being sampled at that time.
    """
    sections = [(label, traces, sci, log_y) for label, traces, sci, log_y in sections if traces]
    if not sections:
        return
    fig = make_subplots(rows=len(sections), cols=1, shared_xaxes=True, subplot_titles=[label for label, _, _, _ in sections])
    for i, (_label, traces, sci, log_y) in enumerate(sections, start=1):
        for trace in traces:
            fig.add_trace(trace, row=i, col=1)
        if sci:
            fig.update_yaxes(exponentformat="e", row=i, col=1)
        if log_y:
            fig.update_yaxes(type="log", row=i, col=1)
        if zero_line:
            fig.add_hline(y=0, line={"color": _ZERO_LINE_COLOR, "width": 1}, row=i, col=1)
    fig.update_xaxes(matches="x")
    if chamber_spans is not None and not chamber_spans.is_empty():
        _shade_chamber_spans(fig, chamber_spans)
    fig.update_layout(height=250 * len(sections), title=title)
    st.plotly_chart(fig, width="stretch")


_SYSTEM_HEALTH_PANELS = [
    ("voltage_v", "Supply voltage (V)"),
    ("current_a", "Supply current (A)"),
    ("teensy_temp_c", "Teensy temperature (degC)"),
]


def total_pressure_torr(status: pl.DataFrame, total_pressure_sensitivity: float) -> pl.DataFrame:
    """Convert the raw total-pressure counts in a status table to Amps, then Torr."""
    return status.select(
        "ts",
        (pl.col("raw_total_pressure_current") * RAW_CURRENT_AMPS_PER_COUNT).alias("total_pressure_amps"),
    ).with_columns((pl.col("total_pressure_amps") / total_pressure_sensitivity).alias("total_pressure_torr"))


def _chamber_shading_control(valve: pl.DataFrame | None, key: str) -> pl.DataFrame | None:
    """Offer the "shade by active chamber" toggle, returning the spans to shade.

    Returns None when there is no valve data to derive spans from (the toggle
    is not rendered at all in that case) or when the toggle is off.
    """
    spans = active_chamber_spans(valve)
    if spans.is_empty():
        return None
    if not st.checkbox("Shade by active chamber", value=False, key=key):
        return None
    return spans


def render_status_tab(tables: dict[str, pl.DataFrame | None], total_pressure_sensitivity: float) -> None:
    status = tables["status"]
    system_health = tables["system_health"]
    have_status = status is not None and not status.is_empty()
    have_system_health = system_health is not None and not system_health.is_empty()
    if not have_status and not have_system_health:
        _empty_state("status")
        return

    chamber_spans = _chamber_shading_control(tables["valve"], key="status_chamber_shading")

    sections: list[tuple[str, list[go.Scatter], bool, bool]] = []
    if have_status:
        sections += [
            (
                "Turbo speed (Hz)",
                [go.Scatter(x=status["ts"], y=status["turbo_speed_hz"], mode="lines", name="turbo_speed_hz")],
                False,
                False,
            ),
            (
                "Turbo power (W)",
                [go.Scatter(x=status["ts"], y=status["turbo_power_w"], mode="lines", name="turbo_power_w")],
                False,
                False,
            ),
        ]

        temp_cols = ["turbo_etemp_c", "turbo_btemp_c", "turbo_mtemp_c"]
        temp_long = status.select(["ts", *temp_cols]).unpivot(
            index="ts", on=temp_cols, variable_name="sensor", value_name="temp_c"
        )
        temp_traces = [
            go.Scatter(x=g["ts"], y=g["temp_c"], mode="lines", name=sensor)
            for sensor in temp_cols
            for g in [temp_long.filter(pl.col("sensor") == sensor)]
        ]
        sections.append(("Turbo temperatures (degC)", temp_traces, False, False))
        sections.append(
            (
                "Water pump (RPM)",
                [go.Scatter(x=status["ts"], y=status["pump_rpm"], mode="lines", name="pump_rpm")],
                False,
                False,
            )
        )

        if status["raw_total_pressure_current"].drop_nulls().is_empty():
            st.info("No total pressure data available in this dataset.")
        else:
            pressure = total_pressure_torr(status, total_pressure_sensitivity)
            sections.append(
                (
                    "Total pressure (Torr)",
                    [
                        go.Scatter(
                            x=pressure["ts"], y=pressure["total_pressure_torr"], mode="lines", name="total_pressure_torr"
                        )
                    ],
                    True,
                    False,
                )
            )

    if have_system_health:
        sections += [
            (
                label,
                [go.Scatter(x=system_health["ts"], y=system_health[col], mode="lines", name=col)],
                False,
                False,
            )
            for col, label in _SYSTEM_HEALTH_PANELS
        ]

    _render_linked_timeseries(sections, title="Status", chamber_spans=chamber_spans)


def _par_daily_sections(par: pl.DataFrame) -> list[tuple[str, list, bool, bool]]:
    """Daily light integral and daily max PAR panels, one bar/marker per UTC day.

    Computed from the (time-range-filtered) par table rather than loaded, so
    they follow the sidebar filter. Partial days (coverage below
    DEFAULT_MIN_DAY_COVERAGE) are drawn faded and left out of the trend: their
    DLI is short by construction, not dim. The trend line is the biofouling
    screen (see par.daily_max_trend).
    """
    daily = daily_par(par).drop_nulls("max_par_umol_m2_s")
    if daily.is_empty():
        return []
    noon = daily.with_columns(
        (pl.col("date").cast(pl.Datetime) + pl.duration(hours=12)).alias("noon"),
        (pl.col("coverage") >= DEFAULT_MIN_DAY_COVERAGE).alias("full"),
    )
    opacity = [1.0 if full else 0.35 for full in noon["full"]]
    hover = [
        f"{d}<br>coverage {c:.0%}" + ("" if full else " (partial day)")
        for d, c, full in zip(noon["date"], noon["coverage"], noon["full"])
    ]
    dli = go.Bar(
        x=noon["noon"],
        y=noon["dli_mol_m2_d"],
        width=[86_400_000 * 0.8] * noon.height,
        marker={"opacity": opacity},
        name="daily light integral",
        customdata=hover,
        hovertemplate="%{customdata}<br>DLI %{y:.2f} mol m⁻² d⁻¹<extra></extra>",
    )
    max_traces = [
        go.Scatter(
            x=noon["noon"],
            y=noon["max_par_umol_m2_s"],
            mode="markers",
            marker={"size": 10, "opacity": opacity},
            name="daily max PAR",
            text=hover,
            hovertemplate="%{text}<br>max %{y:.0f} µmol m⁻² s⁻¹<extra></extra>",
        )
    ]
    trend = daily_max_trend(daily)
    if trend is not None:
        full_noons = noon.filter(pl.col("full"))["noon"]
        days = [(t - full_noons.min()).days for t in full_noons]
        max_traces.append(
            go.Scatter(
                x=full_noons,
                y=[trend["intercept_umol_m2_s"] + trend["slope_umol_m2_s_per_day"] * d for d in days],
                mode="lines",
                line={"dash": "dash", "width": 2},
                name=f"trend {trend['pct_per_day']:+.1f}%/day ({trend['n_days']} full days)",
                hoverinfo="skip",
            )
        )
    return [
        ("Daily light integral (mol photons m⁻² d⁻¹)", [dli], False, False),
        ("Daily max PAR (µmol photons m⁻² s⁻¹) — biofouling screen", max_traces, False, False),
    ]


def render_measurements_tab(
    tables: dict[str, pl.DataFrame | None],
    partial_pressure_sensitivity: float,
) -> None:
    unit = st.radio("Unit", ["raw", "amps", "torr"], horizontal=True, key="measurements_unit")
    sci = unit in ("amps", "torr")
    chamber_spans = _chamber_shading_control(tables["valve"], key="measurements_chamber_shading")
    sections: list[tuple[str, list[go.Scatter], bool, bool]] = []

    rga = tables["rga"]
    cycles_table = tables["egcf_chamber_cycles"]
    have_full = rga is not None and not rga.is_empty()
    have_cycles = cycles_table is not None and not cycles_table.is_empty()

    all_masses: set[int] = set()
    if have_full:
        all_masses.update(rga["mass"].unique().to_list())
    if have_cycles:
        all_masses.update(discover_masses(cycles_table))
    color_map = mass_color_map(sorted(all_masses))

    if not have_full and not have_cycles:
        _empty_state("RGA")
    else:
        options = [label for label, available in [("Full RGA data", have_full), ("Chamber cycle averages", have_cycles)] if available]
        data_source = st.radio("RGA data source", options, horizontal=True, key="rga_data_source")

        if data_source == "Full RGA data":
            masses = sorted(rga["mass"].unique().to_list())
            selected = st.multiselect("Masses (RGA data)", masses, default=masses, key="rga_masses")
            filtered = rga.filter(pl.col("mass").is_in(selected)).with_columns(
                rga_current_to_unit(pl.col("current"), unit, partial_pressure_sensitivity).alias("value")
            )
            long_df = filtered.select(pl.col("ts"), pl.col("mass"), pl.col("value"))
            traces = _mass_traces_from_long(long_df, "value", "lines", color_map)
            sections.append((f"RGA data ({unit}, full)", traces, sci, True))

            ratio_long = rga_full_ratio_to_mass(rga, selected)
            if ratio_long.is_empty():
                st.info("No mass 40 data available to compute mass ratios.")
            else:
                ratio_traces = _mass_traces_from_long(ratio_long, "ratio", "lines", color_map)
                sections.append(("Masses / mass 40 (full)", ratio_traces, False, True))
        else:
            masses = discover_masses(cycles_table)
            selected = st.multiselect("Masses (chamber cycle averages)", masses, default=masses, key="rga_masses")
            suffix = unit if unit != "raw" else "avg"
            long_df = cycles_table.select(
                pl.col("timestamp").alias("ts"),
                *[pl.col(f"mass_{m}_{suffix}").alias(str(m)) for m in selected if f"mass_{m}_{suffix}" in cycles_table.columns],
            ).unpivot(index="ts", variable_name="mass", value_name="value")
            long_df = long_df.with_columns(pl.col("mass").cast(pl.Int64))
            traces = _mass_traces_from_long(long_df, "value", "markers", color_map)
            sections.append((f"RGA data ({unit}, chamber-cycle-averaged)", traces, sci, True))

            ratio_long = rga_wide_ratio_to_mass(cycles_table, selected, ts_col="timestamp")
            if ratio_long.is_empty():
                st.info("No mass 40 data available to compute mass ratios.")
            else:
                ratio_traces = _mass_traces_from_long(ratio_long, "ratio", "markers", color_map)
                sections.append(("Masses / mass 40 (chamber-cycle-averaged)", ratio_traces, False, True))

    scalup = tables["scalup"]
    if scalup is None or scalup.is_empty():
        _empty_state("scalup")
    else:
        cols_lower = {c.lower(): c for c in scalup.columns}
        for col, label in _SCALUP_PANELS:
            actual_col = cols_lower.get(col.lower())
            if actual_col is not None:
                sections.append(
                    (label, [go.Scatter(x=scalup["ts"], y=scalup[actual_col], mode="lines", name=col)], False, False)
                )

    par = tables["par"]
    if par is None or par.is_empty():
        _empty_state("PAR")
    elif par["par_umol_m2_s"].drop_nulls().is_empty():
        st.info("No PAR calibration matched this logger; plotting uncalibrated raw counts.")
        sections.append(
            (
                "PAR (raw counts, uncalibrated)",
                [go.Scatter(x=par["ts"], y=par["par_raw"], mode="lines", name="par_raw")],
                False,
                False,
            )
        )
    else:
        sections.append(
            (
                "PAR (µmol photons m⁻² s⁻¹)",
                [go.Scatter(x=par["ts"], y=par["par_umol_m2_s"], mode="lines", name="par_umol_m2_s")],
                False,
                False,
            )
        )
        sections += _par_daily_sections(par)

    _render_linked_timeseries(sections, title="Measurements", chamber_spans=chamber_spans)


def _elapsed_minutes_expr() -> pl.Expr:
    return (pl.col("elapsed_time").dt.total_seconds() / 60).alias("elapsed_time_min")


def _render_experiment_full_data(
    rga: pl.DataFrame | None,
    scalup: pl.DataFrame | None,
    status: pl.DataFrame | None,
    valve: pl.DataFrame,
    total_pressure_sensitivity: float,
) -> None:
    windows, _ = chamber_cycle_windows(valve, settle_offset_s=0.0)
    if windows.is_empty():
        st.info("No valid chamber cycles found in this dataset.")
        return

    start_by_exp = experiment_start_times(windows, "window_start")
    experiments = sorted(windows["experiment_number"].unique().to_list())
    experiment = st.selectbox(
        "Experiment",
        [str(e) for e in experiments],
        format_func=lambda e: f"{e} ({start_by_exp[int(e)]:%Y-%m-%d %H:%M:%S})",
        key="experiment_number",
    )
    exp_windows = windows.filter(pl.col("experiment_number") == int(experiment))
    experiment_start = start_by_exp[int(experiment)]

    settle_offset_s = st.slider(
        "Settling time after valve switch (s)",
        min_value=0,
        max_value=300,
        value=int(DEFAULT_SETTLE_OFFSET_S),
        key="settle_offset_s",
    )

    have_rga = rga is not None and not rga.is_empty()
    have_scalup = scalup is not None and not scalup.is_empty()
    have_status = status is not None and not status.is_empty()
    exp_rga = (
        attach_experiment_context(rga, exp_windows, ts_col="ts", settle_offset_s=settle_offset_s) if have_rga else None
    )
    exp_scalup = (
        attach_experiment_context(scalup, exp_windows, ts_col="ts", settle_offset_s=settle_offset_s)
        if have_scalup
        else None
    )
    exp_status = (
        attach_experiment_context(status, exp_windows, ts_col="ts", settle_offset_s=settle_offset_s)
        if have_status
        else None
    )

    has_argon = have_rga and ARGON_MASS in exp_rga["mass"].unique().to_list()
    masses = sorted(m for m in exp_rga["mass"].unique().to_list() if m != ARGON_MASS) if has_argon else []
    if have_rga and not has_argon:
        st.info(f"No mass {ARGON_MASS} (Argon) data available to normalize RGA masses.")
    mass_options = [f"mass_{m}" for m in masses]

    scalup_cols_lower = {c.lower(): c for c in scalup.columns} if have_scalup else {}
    scalup_options = [name for name, _ in _SCALUP_PANELS if name.lower() in scalup_cols_lower]
    status_options = [name for name in _STATUS_DIRECT_RAW_COLS if have_status]
    total_pressure_options = ["total_pressure_amps", "total_pressure_torr"] if have_status else []

    other_options = scalup_options + status_options + total_pressure_options
    if not mass_options and not other_options:
        st.info("No plottable variables available for this experiment.")
        return
    variable = st.selectbox("Variable", mass_options + other_options, key="experiment_variable")

    is_mass_variable = variable.startswith("mass_")
    if is_mass_variable:
        mass = int(variable.split("_")[1])
        ratio = rga_full_ratio_to_mass(exp_rga.select("ts", "mass", "current"), [mass])
        context = exp_rga.filter(pl.col("mass") == mass).select("ts", "chamber", "elapsed_time", "settled_out")
        plot_df = (
            ratio.join(context, on="ts", how="left")
            .with_columns(_elapsed_minutes_expr())
            .select("elapsed_time_min", "chamber", "settled_out", pl.col("ratio").alias("value"))
        )
        variable_label = f"{variable} / mass_{ARGON_MASS} (Argon-normalized)"
        col_is_sci = False
        download_df, download_name = exp_rga, "rga"
    elif variable in scalup_options:
        raw_col = scalup_cols_lower[variable.lower()]
        plot_df = exp_scalup.select(_elapsed_minutes_expr(), "chamber", "settled_out", pl.col(raw_col).alias("value"))
        variable_label = variable
        col_is_sci = False
        download_df, download_name = exp_scalup, "scalup"
    elif variable in _STATUS_DIRECT_RAW_COLS:
        raw_col = _STATUS_DIRECT_RAW_COLS[variable]
        plot_df = exp_status.select(_elapsed_minutes_expr(), "chamber", "settled_out", pl.col(raw_col).alias("value"))
        variable_label = variable
        col_is_sci = False
        download_df, download_name = exp_status, "status"
    else:
        amps = pl.col("raw_total_pressure_current") * RAW_CURRENT_AMPS_PER_COUNT
        value_expr = amps if variable == "total_pressure_amps" else (amps / total_pressure_sensitivity)
        plot_df = exp_status.select(_elapsed_minutes_expr(), "chamber", "settled_out", value_expr.alias("value"))
        variable_label = variable
        col_is_sci = True
        download_df, download_name = exp_status, "status"

    _render_experiment_plot(
        plot_df, experiment, variable_label, col_is_sci, is_mass_variable, experiment_start, settled_out_col="settled_out"
    )
    st.download_button(
        "Download this slice as CSV",
        duration_cols_to_seconds(download_df).write_csv(),
        file_name=f"experiment_{experiment}_{download_name}.csv",
        mime="text/csv",
    )


def _render_experiment_cycle_averages(
    rga: pl.DataFrame | None,
    scalup: pl.DataFrame | None,
    status: pl.DataFrame | None,
    valve: pl.DataFrame,
    total_pressure_sensitivity: float,
    chamber_volume_l: float,
    chamber_area_m2: float,
) -> None:
    settle_offset_s = st.slider(
        "Settling time after valve switch (s)",
        min_value=0,
        max_value=300,
        value=int(DEFAULT_SETTLE_OFFSET_S),
        key="settle_offset_s",
    )
    windows, _ = chamber_cycle_windows(valve, settle_offset_s=settle_offset_s)
    if windows.is_empty():
        st.info("No valid chamber cycles found in this dataset.")
        return

    table = with_elapsed_time_s(
        aggregate_onto_windows(
            windows,
            rga if rga is not None else pl.DataFrame(schema=RGA_SCHEMA),
            scalup if scalup is not None else pl.DataFrame(schema=SCALUP_SCHEMA),
            status if status is not None else pl.DataFrame(schema=STATUS_SCHEMA),
            DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR,
            total_pressure_sensitivity,
        )
    )

    with_experiment = table.filter(pl.col("experiment_number").is_not_null())
    if with_experiment.is_empty():
        st.info("No rows with a known experiment_number in this dataset.")
        return

    # "timestamp" is window_start, which has settle_offset_s baked in (unlike Full
    # data's windows, computed with settle_offset_s=0.0) -- subtract it back out so
    # these start times match Full data's exactly regardless of the slider.
    start_by_exp = {
        e: s - timedelta(seconds=settle_offset_s) for e, s in experiment_start_times(with_experiment, "timestamp").items()
    }
    experiments = sorted(with_experiment["experiment_number"].unique().to_list())
    experiment = st.selectbox(
        "Experiment",
        [str(e) for e in experiments],
        format_func=lambda e: f"{e} ({start_by_exp[int(e)]:%Y-%m-%d %H:%M:%S})",
        key="experiment_number",
    )
    exp_df = with_experiment.filter(pl.col("experiment_number").cast(pl.Utf8) == experiment)
    experiment_start = start_by_exp[int(experiment)]

    all_masses = discover_masses(exp_df)
    has_argon = f"mass_{ARGON_MASS}_avg" in exp_df.columns
    masses = [m for m in all_masses if m != ARGON_MASS] if has_argon else []
    if not has_argon and all_masses:
        st.info(f"No mass {ARGON_MASS} (Argon) data available to normalize RGA masses.")
    mass_options = [f"mass_{m}" for m in masses]
    other_options = [
        c
        for c in [
            "total_pressure_amps",
            "total_pressure_torr",
            "temp_degC",
            "sal_PSU",
            "pressure_mbar",
            "oxygen_mgL",
            "pH",
            "turbo_speed_hz",
            "turbo_power_w",
            "water_pump_rpm",
        ]
        if c in exp_df.columns
    ]
    if not mass_options and not other_options:
        st.info("No plottable variables available for this experiment.")
        return
    variable = st.selectbox("Variable", mass_options + other_options, key="experiment_variable")

    is_mass_variable = variable.startswith("mass_")
    plot_df = (
        exp_df.select("elapsed_time_s", "chamber", variable_value_expr(variable, is_mass_variable))
        .drop_nulls("value")
        .with_columns((pl.col("elapsed_time_s") / 60).alias("elapsed_time_min"))
    )
    if is_mass_variable:
        col_is_sci = False
        variable_label = f"{variable} / mass_{ARGON_MASS} (Argon-normalized)"
    else:
        col_is_sci = variable in ("total_pressure_amps", "total_pressure_torr")
        variable_label = variable

    fits = {}
    for chamber in sorted(plot_df["chamber"].unique().to_list()):
        fit = linear_fit(
            plot_df.filter(pl.col("chamber") == chamber)["elapsed_time_min"].to_list(),
            plot_df.filter(pl.col("chamber") == chamber)["value"].to_list(),
        )
        if fit is not None:
            fits[chamber] = fit

    _render_experiment_plot(plot_df, experiment, variable_label, col_is_sci, is_mass_variable, experiment_start, fits=fits)
    st.download_button(
        "Download this slice as CSV",
        duration_cols_to_seconds(exp_df).write_csv(),
        file_name=f"experiment_{experiment}_egcf_chamber_cycles.csv",
        mime="text/csv",
    )

    rates_df = experiment_rates(with_experiment, variable, is_mass_variable)
    _render_experiment_rates_plot(rates_df, variable_label, col_is_sci)

    _render_experiment_fluxes(with_experiment, experiment, chamber_volume_l, chamber_area_m2)


def _render_experiment_fluxes(
    with_experiment: pl.DataFrame,
    experiment: str,
    chamber_volume_l: float,
    chamber_area_m2: float,
) -> None:
    """Show benthic flux: the selected experiment, then the whole deployment.

    Independent of the Variable selectbox above -- these are a fixed set of
    quantities (see flux.compute_fluxes), not user-selected ones.

    The deployment-wide plot renders even when the *selected* experiment has
    no fittable cycle, so landing on a thin experiment (experiment 1 in the
    real corpus has one cycle per chamber) no longer looks like "no flux data"
    when fits exist elsewhere in the deployment.
    """
    st.subheader("Benthic flux")
    if chamber_volume_l <= 0 or chamber_area_m2 <= 0:
        st.info("Enter the chamber volume and sediment footprint area in the sidebar to compute flux.")
        return
    all_fluxes = compute_fluxes(with_experiment, chamber_volume_l, chamber_area_m2)
    if all_fluxes.is_empty():
        st.info("Not enough cycles in any experiment to fit a flux.")
        return

    this_exp = all_fluxes.filter(pl.col("experiment_number").cast(pl.Utf8) == experiment)
    if this_exp.is_empty():
        st.info("Not enough cycles in this experiment to fit a flux -- see the deployment-wide plot below.")
    else:
        _render_experiment_flux_chart(this_exp, experiment)
        st.dataframe(
            this_exp.select(
                "chamber", "variable", "output_value", "output_unit", "slope_native_per_min", "r2", "n_points"
            ),
            width="stretch",
        )

    _render_flux_over_time(all_fluxes)


def _render_experiment_flux_chart(exp_fluxes: pl.DataFrame, experiment: str) -> None:
    """Bar chart of one experiment's fluxes, one subplot per variable.

    One subplot per variable rather than one grouped bar chart, because the
    variables carry different units and magnitudes spanning four orders
    (oxygen ~1e4 umol m-2 h-1 next to h_ion ~1e0) -- on a shared axis
    everything but oxygen flattens to nothing.
    """
    variables = sorted(exp_fluxes["variable"].unique().to_list())
    units = flux_variable_units(exp_fluxes)
    chambers = sorted(exp_fluxes["chamber"].unique().to_list())
    colors = chamber_color_map(chambers)
    fig = make_subplots(rows=1, cols=len(variables), subplot_titles=[f"{v}<br><sub>{units[v]}</sub>" for v in variables])
    for col, variable in enumerate(variables, start=1):
        g = exp_fluxes.filter(pl.col("variable") == variable)
        for chamber in chambers:
            gc = g.filter(pl.col("chamber") == chamber)
            if gc.is_empty():
                continue
            fig.add_trace(
                go.Bar(
                    x=[chamber],
                    y=gc["output_value"],
                    name=chamber,
                    marker={"color": colors[chamber]},
                    legendgroup=chamber,
                    showlegend=col == 1,
                    hovertemplate=f"{chamber}<br>{variable}=%{{y:.4g}} {units[variable]}<extra></extra>",
                ),
                row=1,
                col=col,
            )
        fig.add_hline(y=0, line={"color": _ZERO_LINE_COLOR, "width": 1}, row=1, col=col)
    fig.update_layout(height=340, title=f"Experiment {experiment}: flux by variable", barmode="group")
    st.plotly_chart(fig, width="stretch")


def _render_flux_over_time(all_fluxes: pl.DataFrame) -> None:
    """Flux against experiment start across the whole deployment.

    One stacked, x-linked subplot per selected variable (see
    _render_experiment_flux_chart for why variables can't share an axis).
    Never log-scaled and never forced non-negative: a flux's sign is its
    meaning, and both directions are real.
    """
    variables = sorted(all_fluxes["variable"].unique().to_list())
    selected = st.multiselect("Variables (flux over time)", variables, default=variables, key="flux_variables")
    if not selected:
        st.info("Select at least one variable to plot flux over time.")
        return
    units = flux_variable_units(all_fluxes)
    chambers = sorted(all_fluxes["chamber"].unique().to_list())
    colors = chamber_color_map(chambers)
    sections: list[tuple[str, list[go.Scatter], bool, bool]] = []
    for variable in [v for v in variables if v in selected]:
        g = all_fluxes.filter(pl.col("variable") == variable).sort("experiment_start")
        traces = []
        for chamber in chambers:
            gc = g.filter(pl.col("chamber") == chamber)
            if gc.is_empty():
                continue
            traces.append(
                go.Scatter(
                    x=gc["experiment_start"],
                    y=gc["output_value"],
                    mode="lines+markers",
                    name=chamber,
                    line={"color": colors[chamber]},
                    marker={"color": colors[chamber]},
                    legendgroup=chamber,
                    showlegend=not sections,
                    # n and r2 belong in the hover: a 2-point fit always has
                    # r2=1.0, so the fit quality is only meaningful alongside n.
                    text=[
                        f"experiment {e} (n={n}, r²={r:.3f})"
                        for e, n, r in zip(gc["experiment_number"], gc["n_points"], gc["r2"])
                    ],
                    hovertemplate="%{text}<br>%{x}<br>flux=%{y:.4g}<extra></extra>",
                )
            )
        sections.append((f"{variable} ({units[variable]})", traces, False, False))
    _render_linked_timeseries(sections, title="Flux over time", zero_line=True)


_SETTLED_OUT_COLOR = "#B0B0B0"
_ZERO_LINE_COLOR = "#888888"


def _render_experiment_plot(
    plot_df: pl.DataFrame,
    experiment: str,
    variable_label: str,
    col_is_sci: bool,
    is_mass_variable: bool,
    experiment_start: datetime,
    settled_out_col: str | None = None,
    fits: dict[str, tuple[float, float]] | None = None,
) -> None:
    fig = go.Figure()
    chambers = sorted(plot_df["chamber"].unique().to_list())
    chamber_color = chamber_color_map(chambers)
    kept = plot_df.filter(~pl.col(settled_out_col)) if settled_out_col else plot_df
    for chamber in chambers:
        g = kept.filter(pl.col("chamber") == chamber)
        fig.add_trace(
            go.Scatter(
                x=g["elapsed_time_min"],
                y=g["value"],
                mode="lines+markers",
                name=chamber,
                line={"color": chamber_color[chamber]},
                marker={"color": chamber_color[chamber]},
            )
        )
    if settled_out_col:
        dropped = plot_df.filter(pl.col(settled_out_col))
        if not dropped.is_empty():
            fig.add_trace(
                go.Scatter(
                    x=dropped["elapsed_time_min"],
                    y=dropped["value"],
                    mode="markers",
                    name="dropped (settling)",
                    marker={"color": _SETTLED_OUT_COLOR},
                )
            )
    for chamber, (slope, intercept) in (fits or {}).items():
        g = kept.filter(pl.col("chamber") == chamber)
        if g.is_empty():
            continue
        x0, x1 = g["elapsed_time_min"].min(), g["elapsed_time_min"].max()
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[slope * x0 + intercept, slope * x1 + intercept],
                mode="lines",
                name=f"{chamber} fit ({slope:.3g}/min)",
                line={"color": chamber_color.get(chamber, "black"), "dash": "dash"},
            )
        )
    fig.update_layout(
        title={
            "text": f"Experiment {experiment}: {variable_label} vs elapsed time (min)"
            f"<br><sup>Started {experiment_start:%Y-%m-%d %H:%M:%S}</sup>"
        }
    )
    fig.update_xaxes(title="Elapsed time (min)")
    if col_is_sci:
        fig.update_yaxes(exponentformat="e")
    if is_mass_variable:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, width="stretch")


def _render_experiment_rates_plot(rates_df: pl.DataFrame, variable_label: str, col_is_sci: bool) -> None:
    if rates_df.is_empty():
        st.info("Not enough cycles in any experiment to fit a rate.")
        return
    fig = go.Figure()
    chambers = sorted(rates_df["chamber"].unique().to_list())
    chamber_color = chamber_color_map(chambers)
    for chamber in chambers:
        g = rates_df.filter(pl.col("chamber") == chamber)
        fig.add_trace(
            go.Scatter(
                x=g["experiment_start"],
                y=g["rate"],
                mode="lines+markers",
                name=chamber,
                line={"color": chamber_color[chamber]},
                marker={"color": chamber_color[chamber]},
                text=[f"experiment {e}" for e in g["experiment_number"]],
                hovertemplate="%{text}<br>%{x}<br>rate=%{y}<extra></extra>",
            )
        )
    fig.update_layout(title=f"{variable_label} rate per experiment")
    fig.update_xaxes(title="Experiment start")
    fig.update_yaxes(title=f"{variable_label} rate (per min)")
    if col_is_sci:
        fig.update_yaxes(exponentformat="e")
    st.plotly_chart(fig, width="stretch")


def render_experiment_tab(
    tables: dict[str, pl.DataFrame | None],
    total_pressure_sensitivity: float,
    chamber_volume_l: float = 0.0,
    chamber_area_m2: float = 0.0,
) -> None:
    rga = tables["rga"]
    scalup = tables["scalup"]
    status = tables["status"]
    valve = tables["valve"]

    if valve is None or valve.is_empty():
        _empty_state("experiment")
        return

    grain = st.radio("Grain", ["Full data", "Cycle averages"], horizontal=True, key="experiment_grain")

    if grain == "Full data":
        _render_experiment_full_data(rga, scalup, status, valve, total_pressure_sensitivity)
    else:
        _render_experiment_cycle_averages(
            rga, scalup, status, valve, total_pressure_sensitivity, chamber_volume_l, chamber_area_m2
        )


def _pi_curve_traces(pi_fit: pl.DataFrame | None, par_max: float, colors: dict[str, str]) -> list[go.Scatter]:
    if pi_fit is None or pi_fit.is_empty():
        return []
    grid = np.linspace(0.0, par_max * 1.05, 200)
    traces = []
    for row in pi_fit.filter(pl.col("converged")).iter_rows(named=True):
        chamber = row["chamber"]
        traces.append(
            go.Scatter(
                x=grid,
                y=jassby_platt(grid, row["pmax_umol_m2_h"], row["alpha_umol_m2_h_per_par"], row["r_fit_umol_m2_h"]),
                mode="lines",
                line={"color": colors.get(chamber), "width": 2},
                name=f"{chamber} fit (Ik={row['ik_umol_m2_s']:.0f})",
                legendgroup=chamber,
                hoverinfo="skip",
            )
        )
    return traces


def _metabolism_scatter(df: pl.DataFrame, y_col: str, chamber: str, color: str, used: bool, show_legend: bool) -> go.Scatter:
    return go.Scatter(
        x=df["par_chamber_umol_m2_s"],
        y=df[y_col],
        mode="markers",
        marker={
            "color": color if used else "rgba(0,0,0,0)",
            "size": 10,
            "line": {"color": color, "width": 2},
        },
        name=chamber if used else f"{chamber} excluded",
        legendgroup=chamber,
        showlegend=show_legend,
        text=[
            f"experiment {e} ({p})<br>{t:%Y-%m-%d %H:%M}<br>r²={'n/a' if r is None else f'{r:.2f}'}"
            + (f"<br>excluded: {x}" if x else "")
            for e, p, t, r, x in zip(df["experiment_number"], df["period"], df["experiment_start"], df["r2"], df["excluded_reason"])
        ],
        hovertemplate="%{text}<br>PAR %{x:.0f}<br>flux %{y:.4g}<extra></extra>",
    )


def render_metabolism_tab(tables: dict[str, pl.DataFrame | None]) -> None:
    """O2 (and H+) flux against experiment-mean PAR, with the per-chamber P-I fit.

    Reads the pipeline's egcf_metabolism / egcf_pi_fit / egcf_fluxes outputs
    rather than recomputing, so the chamber geometry, dark threshold and
    transmittance are whatever the pipeline was run with -- unlike the
    Experiment tab's live flux, the sidebar geometry doesn't apply here.
    """
    metabolism = tables["egcf_metabolism"]
    if metabolism is None or metabolism.is_empty():
        _empty_state("metabolism")
        return
    placed = metabolism.drop_nulls("par_chamber_umol_m2_s")
    if placed.is_empty():
        st.info("No O2 flux overlaps the PAR record, so there's nothing to place on a light axis.")
        return
    pi_fit = tables["egcf_pi_fit"]
    transmittance = pi_fit["chamber_par_transmittance"][0] if pi_fit is not None and not pi_fit.is_empty() else None
    par_basis = "ambient PAR at the logger" if transmittance in (None, 1.0) else f"PAR × chamber transmittance {transmittance:g}"
    st.caption(
        f"Each point is one experiment's O2 flux against its mean {par_basis}. Hollow markers are excluded "
        "fluxes (hover for the reason). Geometry, dark threshold and transmittance are fixed at processing time."
    )

    chambers = sorted(placed["chamber"].unique().to_list())
    colors = chamber_color_map(chambers)
    par_max = float(placed["par_chamber_umol_m2_s"].max())
    x_label = "PAR (µmol photons m⁻² s⁻¹)"

    fluxes = tables["egcf_fluxes"]
    h_ion = None
    if fluxes is not None and "par_mean_umol_m2_s" in fluxes.columns:
        h_ion = (
            fluxes.filter(pl.col("variable") == "h_ion")
            .select("experiment_number", "chamber", pl.col("output_value").alias("h_ion_flux"))
            .join(placed, on=["experiment_number", "chamber"], how="inner")
        )
        if h_ion.is_empty():
            h_ion = None

    rows = 2 if h_ion is not None else 1
    titles = ["O2 flux (µmol m⁻² h⁻¹)"] + (["H⁺ flux (µmol m⁻² h⁻¹) — should oppose O2"] if h_ion is not None else [])
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, subplot_titles=titles, vertical_spacing=0.08)
    for chamber in chambers:
        for used in (True, False):
            g = placed.filter((pl.col("chamber") == chamber) & (pl.col("used") == used))
            if not g.is_empty():
                fig.add_trace(_metabolism_scatter(g, "o2_flux_umol_m2_h", chamber, colors[chamber], used, True), row=1, col=1)
            if h_ion is not None:
                gh = h_ion.filter((pl.col("chamber") == chamber) & (pl.col("used") == used))
                if not gh.is_empty():
                    fig.add_trace(_metabolism_scatter(gh, "h_ion_flux", chamber, colors[chamber], used, False), row=2, col=1)
    for trace in _pi_curve_traces(pi_fit, par_max, colors):
        fig.add_trace(trace, row=1, col=1)
    for row in range(1, rows + 1):
        fig.add_hline(y=0, line={"color": _ZERO_LINE_COLOR, "width": 1}, row=row, col=1)
    fig.update_xaxes(title_text=x_label, row=rows, col=1)
    fig.update_layout(height=420 * rows, title="Metabolism vs light")
    st.plotly_chart(fig, width="stretch")

    if pi_fit is not None and not pi_fit.is_empty():
        st.subheader("P–I fit (Jassby–Platt)")
        st.caption(
            "NCP = Pmax·tanh(α·I/Pmax) − R, per chamber, over used light and dark fluxes. "
            "Fluxes in µmol O2 m⁻² h⁻¹; I and Ik in µmol photons m⁻² s⁻¹. R (dark mean) is the measured check on R (fit)."
        )
        st.dataframe(pi_fit, width="stretch", hide_index=True)

    light = metabolism.filter(pl.col("used") & (pl.col("period") == "light"))
    if not light.is_empty():
        with st.expander("Light incubations: NCP and GPP"):
            st.dataframe(
                light.select(
                    "experiment_number", "chamber", "experiment_start", "par_chamber_umol_m2_s", "ncp_umol_m2_h", "gpp_umol_m2_h", "r2"
                ),
                width="stretch",
                hide_index=True,
            )


def render_overview(tables: dict[str, pl.DataFrame | None]) -> None:
    with st.expander("Dataset overview"):
        for name, df in tables.items():
            if df is None:
                st.write(f"**{name}**: not found")
                continue
            ts_col = table_ts_col(df)
            time_range = ""
            if ts_col and not df.is_empty():
                time_range = f", {df[ts_col].min()} -> {df[ts_col].max()}"
            st.write(f"**{name}**: {df.height} rows{time_range}")


def main() -> None:
    st.set_page_config(page_title="EGFC Dashboard", layout="wide")
    st.title("EGFC Lander Dashboard")

    st.sidebar.header("Data source")
    data_dir_input = st.sidebar.text_input("Processed data directory", value="data/processed/surface")
    if st.sidebar.button("Reload data"):
        st.cache_data.clear()
    time_range_slot = st.sidebar.container()
    partial_pressure_sensitivity = st.sidebar.number_input(
        "Partial pressure sensitivity (A/Torr)",
        value=DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR,
        format="%.2e",
    )
    total_pressure_sensitivity = st.sidebar.number_input(
        "Total pressure sensitivity (A/Torr)",
        value=DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR,
        format="%.2e",
    )

    st.sidebar.header("Chamber geometry")
    st.sidebar.caption("Required to compute flux; there is no meaningful default.")
    chamber_volume_l = st.sidebar.number_input("Chamber volume (L)", value=0.0, min_value=0.0, format="%.3f")
    chamber_area_m2 = st.sidebar.number_input("Sediment footprint area (m^2)", value=0.0, min_value=0.0, format="%.4f")

    data_dir = Path(data_dir_input)
    if not data_dir.exists():
        st.error(f"Directory not found: {data_dir}")
        return

    tables = st.cache_data(load_all)(data_dir)
    render_overview(tables)

    plot_tables = tables
    bounds = tables_time_bounds(tables)
    if bounds is not None and bounds[0] < bounds[1]:
        with time_range_slot:
            start, end = render_time_range_control(bounds)
        plot_tables = filter_tables_to_range(tables, start, end)

    status_tab, measurements_tab, experiment_tab, metabolism_tab = st.tabs(
        ["Status", "Measurements", "Experiment Data", "Metabolism"]
    )
    with status_tab:
        render_status_tab(plot_tables, total_pressure_sensitivity)
    with measurements_tab:
        render_measurements_tab(plot_tables, partial_pressure_sensitivity)
    with experiment_tab:
        render_experiment_tab(tables, total_pressure_sensitivity, chamber_volume_l, chamber_area_m2)
    with metabolism_tab:
        render_metabolism_tab(tables)


if __name__ == "__main__":
    main()
