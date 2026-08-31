"""Read-only Streamlit dashboard over an already-processed data/processed/ directory.

Data-loading and transform helpers are kept free of Streamlit calls so they
can be unit tested directly; only the render_*/main functions touch `st`.
"""

from __future__ import annotations

from pathlib import Path

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
from egcf_processing.pipeline import DEFAULT_SETTLE_OFFSET_S

TABLE_NAMES = ["status", "rga", "scalup", "valve", "egcf_rga_scans", "egcf_chamber_cycles"]

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


def linear_fit(x: list[float], y: list[float]) -> tuple[float, float] | None:
    """Ordinary least-squares slope and intercept of y = slope * x + intercept.

    Used to turn a cycle-averaged time series into a rate (the slope) --
    e.g. an Argon-normalized mass ratio's rate of change per minute during
    one chamber incubation. Returns None with fewer than 2 points or if x
    has zero variance (an undefined/vertical fit), rather than raising.
    """
    pairs = [(xi, yi) for xi, yi in zip(x, y) if xi is not None and yi is not None]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((xi - mean_x) ** 2 for xi in xs)
    if var_x == 0:
        return None
    cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(xs, ys))
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    return slope, intercept


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


def _render_linked_timeseries(sections: list[tuple[str, list[go.Scatter], bool, bool]], title: str) -> None:
    """Render one subplot per section, stacked with a shared, zoom/pan-linked time axis.

    Each section's third element requests scientific-notation y-axis ticks,
    for the Amps/Torr panels whose magnitudes (~1e-8 to 1e-16) are unreadable
    in plain decimal. The fourth element requests a log-scale y-axis, for the
    RGA mass-current panels whose values span several orders of magnitude.
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
    fig.update_xaxes(matches="x")
    fig.update_layout(height=250 * len(sections), title=title)
    st.plotly_chart(fig, width="stretch")


def render_status_tab(tables: dict[str, pl.DataFrame | None], total_pressure_sensitivity: float) -> None:
    status = tables["status"]
    if status is None or status.is_empty():
        _empty_state("status")
        return

    sections: list[tuple[str, list[go.Scatter], bool, bool]] = [
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

    if status["raw_total_pressure_current"].drop_nulls().is_empty():
        st.info("No total pressure data available in this dataset.")
    else:
        pressure = status.select(
            "ts",
            (pl.col("raw_total_pressure_current") * RAW_CURRENT_AMPS_PER_COUNT).alias("total_pressure_amps"),
        ).with_columns((pl.col("total_pressure_amps") / total_pressure_sensitivity).alias("total_pressure_torr"))
        sections.append(
            (
                "Total pressure (Torr)",
                [go.Scatter(x=pressure["ts"], y=pressure["total_pressure_torr"], mode="lines", name="total_pressure_torr")],
                True,
                False,
            )
        )

    _render_linked_timeseries(sections, title="Status")


def render_measurements_tab(
    tables: dict[str, pl.DataFrame | None],
    partial_pressure_sensitivity: float,
) -> None:
    unit = st.radio("Unit", ["raw", "amps", "torr"], horizontal=True, key="measurements_unit")
    sci = unit in ("amps", "torr")
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

    _render_linked_timeseries(sections, title="Measurements")


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

    experiments = sorted(windows["experiment_number"].unique().to_list())
    experiment = st.selectbox("Experiment", [str(e) for e in experiments], key="experiment_number")
    exp_windows = windows.filter(pl.col("experiment_number") == int(experiment))

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

    _render_experiment_plot(plot_df, experiment, variable_label, col_is_sci, is_mass_variable, settled_out_col="settled_out")
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

    experiments = sorted(with_experiment["experiment_number"].unique().to_list())
    experiment = st.selectbox("Experiment", [str(e) for e in experiments], key="experiment_number")
    exp_df = with_experiment.filter(pl.col("experiment_number").cast(pl.Utf8) == experiment)

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

    _render_experiment_plot(plot_df, experiment, variable_label, col_is_sci, is_mass_variable, fits=fits)
    st.download_button(
        "Download this slice as CSV",
        duration_cols_to_seconds(exp_df).write_csv(),
        file_name=f"experiment_{experiment}_egcf_chamber_cycles.csv",
        mime="text/csv",
    )

    rates_df = experiment_rates(with_experiment, variable, is_mass_variable)
    _render_experiment_rates_plot(rates_df, variable_label, col_is_sci)


_SETTLED_OUT_COLOR = "#B0B0B0"


def _render_experiment_plot(
    plot_df: pl.DataFrame,
    experiment: str,
    variable_label: str,
    col_is_sci: bool,
    is_mass_variable: bool,
    settled_out_col: str | None = None,
    fits: dict[str, tuple[float, float]] | None = None,
) -> None:
    fig = go.Figure()
    chambers = sorted(plot_df["chamber"].unique().to_list())
    chamber_color = {c: _MASS_COLOR_PALETTE[i % len(_MASS_COLOR_PALETTE)] for i, c in enumerate(chambers)}
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
    fig.update_layout(title=f"Experiment {experiment}: {variable_label} vs elapsed time (min)")
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
    chamber_color = {c: _MASS_COLOR_PALETTE[i % len(_MASS_COLOR_PALETTE)] for i, c in enumerate(chambers)}
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
        _render_experiment_cycle_averages(rga, scalup, status, valve, total_pressure_sensitivity)


def render_overview(tables: dict[str, pl.DataFrame | None]) -> None:
    with st.expander("Dataset overview"):
        for name, df in tables.items():
            if df is None:
                st.write(f"**{name}**: not found")
                continue
            ts_col = "timestamp" if "timestamp" in df.columns else "ts" if "ts" in df.columns else None
            time_range = ""
            if ts_col and not df.is_empty():
                time_range = f", {df[ts_col].min()} -> {df[ts_col].max()}"
            st.write(f"**{name}**: {df.height} rows{time_range}")


def main() -> None:
    st.set_page_config(page_title="EGFC Dashboard", layout="wide")
    st.title("EGFC Lander Dashboard")

    st.sidebar.header("Data source")
    data_dir_input = st.sidebar.text_input("Processed data directory", value="data/processed")
    if st.sidebar.button("Reload data"):
        st.cache_data.clear()
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

    data_dir = Path(data_dir_input)
    if not data_dir.exists():
        st.error(f"Directory not found: {data_dir}")
        return

    tables = st.cache_data(load_all)(data_dir)
    render_overview(tables)

    status_tab, measurements_tab, experiment_tab = st.tabs(["Status", "Measurements", "Experiment Data"])
    with status_tab:
        render_status_tab(tables, total_pressure_sensitivity)
    with measurements_tab:
        render_measurements_tab(tables, partial_pressure_sensitivity)
    with experiment_tab:
        render_experiment_tab(tables, total_pressure_sensitivity)


if __name__ == "__main__":
    main()
