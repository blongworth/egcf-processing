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
)

TABLE_NAMES = ["status", "rga", "scalup", "valve", "egcf_rga_scans", "egcf_chamber_cycles"]

_SCALUP_PANELS = [
    ("temp_degC", "Temperature (degC)"),
    ("sal_PSU", "Salinity (PSU)"),
    ("pressure_mbar", "Pressure (mbar)"),
    ("oxygen_mgL", "Oxygen (mg/L)"),
    ("pH", "pH"),
]


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


def rga_current_to_unit(current: pl.Expr, unit: str, sensitivity_a_per_torr: float) -> pl.Expr:
    """Convert a raw RGA ion-current expression to the requested display unit."""
    if unit == "raw":
        return current
    amps = current * RAW_CURRENT_AMPS_PER_COUNT
    if unit == "amps":
        return amps
    return amps / sensitivity_a_per_torr


def _empty_state(name: str) -> None:
    st.info(f"No {name} data available in this dataset.")


def _render_linked_timeseries(sections: list[tuple[str, list[go.Scatter], bool]], title: str) -> None:
    """Render one subplot per section, stacked with a shared, zoom/pan-linked time axis.

    Each section's third element requests scientific-notation y-axis ticks,
    for the Amps/Torr panels whose magnitudes (~1e-8 to 1e-16) are unreadable
    in plain decimal.
    """
    sections = [(label, traces, sci) for label, traces, sci in sections if traces]
    if not sections:
        return
    fig = make_subplots(rows=len(sections), cols=1, shared_xaxes=True, subplot_titles=[label for label, _, _ in sections])
    for i, (_label, traces, sci) in enumerate(sections, start=1):
        for trace in traces:
            fig.add_trace(trace, row=i, col=1)
        if sci:
            fig.update_yaxes(exponentformat="e", row=i, col=1)
    fig.update_xaxes(matches="x")
    fig.update_layout(height=250 * len(sections), title=title)
    st.plotly_chart(fig, width="stretch")


def render_status_tab(tables: dict[str, pl.DataFrame | None], total_pressure_sensitivity: float) -> None:
    status = tables["status"]
    if status is None or status.is_empty():
        _empty_state("status")
        return

    sections: list[tuple[str, list[go.Scatter], bool]] = [
        ("Turbo speed (Hz)", [go.Scatter(x=status["ts"], y=status["turbo_speed_hz"], mode="lines", name="turbo_speed_hz")], False),
        ("Turbo power (W)", [go.Scatter(x=status["ts"], y=status["turbo_power_w"], mode="lines", name="turbo_power_w")], False),
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
    sections.append(("Turbo temperatures (degC)", temp_traces, False))

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
            )
        )

    _render_linked_timeseries(sections, title="Status")


def render_measurements_tab(
    tables: dict[str, pl.DataFrame | None],
    partial_pressure_sensitivity: float,
) -> None:
    unit = st.radio("Unit", ["raw", "amps", "torr"], horizontal=True, key="measurements_unit")
    sci = unit in ("amps", "torr")
    sections: list[tuple[str, list[go.Scatter], bool]] = []

    rga = tables["rga"]
    if rga is None or rga.is_empty():
        _empty_state("RGA")
    else:
        masses = sorted(rga["mass"].unique().to_list())
        selected = st.multiselect("Masses (full RGA data)", masses, default=masses, key="rga_full_masses")
        filtered = rga.filter(pl.col("mass").is_in(selected)).with_columns(
            rga_current_to_unit(pl.col("current"), unit, partial_pressure_sensitivity).alias("value")
        )
        traces = [
            go.Scatter(x=g["ts"], y=g["value"], mode="lines", name=f"mass {m}")
            for m in selected
            for g in [filtered.filter(pl.col("mass") == m)]
        ]
        sections.append((f"Full RGA data ({unit})", traces, sci))

    for key, label in [("egcf_rga_scans", "RGA-cycle-averaged data"), ("egcf_chamber_cycles", "Chamber-cycle-averaged data")]:
        table = tables[key]
        if table is None or table.is_empty():
            _empty_state(label)
            continue
        masses = discover_masses(table)
        selected = st.multiselect(f"Masses ({label})", masses, default=masses, key=f"{key}_masses")
        suffix = unit if unit != "raw" else "avg"
        traces = [
            go.Scatter(x=table["timestamp"], y=table[f"mass_{m}_{suffix}"], mode="lines", name=f"mass {m}")
            for m in selected
            if f"mass_{m}_{suffix}" in table.columns
        ]
        sections.append((f"{label} ({unit})", traces, sci))

    scalup = tables["scalup"]
    if scalup is None or scalup.is_empty():
        _empty_state("scalup")
    else:
        cols_lower = {c.lower(): c for c in scalup.columns}
        for col, label in _SCALUP_PANELS:
            actual_col = cols_lower.get(col.lower())
            if actual_col is not None:
                sections.append((label, [go.Scatter(x=scalup["ts"], y=scalup[actual_col], mode="lines", name=col)], False))

    _render_linked_timeseries(sections, title="Measurements")


def render_experiment_tab(
    tables: dict[str, pl.DataFrame | None],
    partial_pressure_sensitivity: float,
) -> None:
    grain = st.radio("Grain", ["RGA cycle", "Chamber cycle"], horizontal=True, key="experiment_grain")
    key = "egcf_rga_scans" if grain == "RGA cycle" else "egcf_chamber_cycles"
    table = tables[key]
    if table is None or table.is_empty():
        _empty_state(grain)
        return

    with_experiment = table.filter(pl.col("experiment_number").is_not_null())
    if with_experiment.is_empty():
        st.info("No rows with a known experiment_number in this dataset.")
        return

    experiments = sorted(with_experiment["experiment_number"].unique().to_list())
    experiment = st.selectbox("Experiment", [str(e) for e in experiments], key="experiment_number")
    exp_df = with_experiment.filter(pl.col("experiment_number").cast(pl.Utf8) == experiment)

    masses = discover_masses(exp_df)
    unit = st.radio("Mass unit", ["raw", "amps", "torr"], horizontal=True, key="experiment_unit")
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
    variable = st.selectbox("Variable", mass_options + other_options, key="experiment_variable")

    if variable.startswith("mass_"):
        suffix = unit if unit != "raw" else "avg"
        col = f"{variable}_{suffix}"
        col_is_sci = suffix in ("amps", "torr")
    else:
        col = variable
        col_is_sci = variable in ("total_pressure_amps", "total_pressure_torr")

    plot_df = exp_df.select("elapsed_time_s", "chamber", pl.col(col).alias("value"))
    fig = px.line(
        plot_df.to_pandas(),
        x="elapsed_time_s",
        y="value",
        color="chamber",
        markers=True,
        title=f"Experiment {experiment}: {variable} vs elapsed time",
    )
    if col_is_sci:
        fig.update_yaxes(exponentformat="e")
    st.plotly_chart(fig, width="stretch")

    st.download_button(
        "Download this slice as CSV",
        exp_df.write_csv(),
        file_name=f"experiment_{experiment}_{key}.csv",
        mime="text/csv",
    )


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
        render_experiment_tab(tables, partial_pressure_sensitivity)


if __name__ == "__main__":
    main()
