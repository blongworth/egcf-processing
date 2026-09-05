"""Layer D: benthic vertical flux from chamber-cycle concentration rate of change.

Flux = dC/dt * V / A, where dC/dt is the OLS slope of a cycle-averaged
concentration against elapsed time across one experiment's incubation
(egcf_chamber_cycles' cycle averages are that incubation's samples), V is
the chamber's enclosed water volume, and A is the sediment footprint area
enclosed by the chamber base -- both constant across chambers per the
project owner, passed in rather than hardcoded since no "nominal" chamber
size exists (unlike the RGA's nominal Faraday-cup sensitivity in
aggregate.py). Reported in umol m^-2 h^-1 (an explicit standing preference
over the more common mmol m^-2 d^-1 convention), except temp_degC, which is
a rate (degC h^-1) rather than a real flux -- there's no mass/energy
conservation quantity for temperature without water density and specific
heat capacity, out of scope here. This is a diagnostic (is the chamber
heating from internal electronics vs. tracking ambient tide), not a flux.

Sign convention: a rising concentration (efflux, sediment -> water) is
positive; a falling one (uptake into sediment, e.g. O2 consumption/SOD) is
negative -- reported signed, never forced positive.
"""

from __future__ import annotations

import math

import polars as pl

O2_UMOL_PER_MG = 1000 / 32  # mg/L -> umol/L, O2 molar mass 32 g/mol
H_ION_UMOL_PER_MOL = 1e6  # mol/L -> umol/L

# Fixed approximation, not a full T/S equation of state -- see AGENTS.md.
SEAWATER_DENSITY_KG_PER_L = 1.025

MIN_PER_HOUR = 60.0

_FLUX_SCHEMA = {
    "experiment_number": pl.Int64,
    "chamber": pl.Utf8,
    "experiment_start": pl.Datetime,
    "variable": pl.Utf8,
    "slope_native_per_min": pl.Float64,
    "r2": pl.Float64,
    "n_points": pl.Int64,
    "output_value": pl.Float64,
    "output_unit": pl.Utf8,
}

_SERIES_SCHEMA = {
    "experiment_number": pl.Int64,
    "chamber": pl.Utf8,
    "experiment_start": pl.Datetime,
    "slope": pl.Float64,
    "r2": pl.Float64,
    "n": pl.Int64,
}

# Hamme & Emerson (2004) Deep-Sea Research I 51:1517-1528, Table 4.
_AR_SOLUBILITY_COEFFS = {
    "A0": 2.79150,
    "A1": 3.17609,
    "A2": 4.13116,
    "A3": 4.90379,
    "B0": -6.96233e-3,
    "B1": -7.66670e-3,
    "B2": -1.16888e-2,
}


def linear_fit(x: list[float], y: list[float]) -> tuple[float, float] | None:
    """Ordinary least-squares slope and intercept of y = slope * x + intercept.

    Used to turn a cycle-averaged time series into a rate (the slope) --
    e.g. an Argon-normalized mass ratio's rate of change per minute during
    one chamber incubation. Returns None with fewer than 2 points or if x
    has zero variance (an undefined/vertical fit), rather than raising.
    """
    fit = ols_fit(x, y)
    return (fit["slope"], fit["intercept"]) if fit is not None else None


def ols_fit(x: list[float], y: list[float]) -> dict | None:
    """Like linear_fit, but also returns r2 and the point count n.

    Needed for QA on flux output (dashboard's plots only need the bare
    slope, so linear_fit's simpler two-tuple return stays untouched rather
    than changing every dashboard call site). r2 is 1.0 when y has zero
    variance and the fit is nonetheless exact (a flat line has no variance
    to explain, but explains it perfectly), rather than the conventional
    0/0 NaN.
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
    var_y = sum((yi - mean_y) ** 2 for yi in ys)
    r2 = 1.0 if var_y == 0 else (cov_xy**2) / (var_x * var_y)
    return {"slope": slope, "intercept": intercept, "r2": r2, "n": n}


def ar_solubility_umol_kg(temp_degc: float, sal_psu: float) -> float:
    """Argon solubility (umol/kg) at 1 atm total pressure, moist air.

    Hamme & Emerson (2004), Deep-Sea Research I 51:1517-1528, Table 4.
    NOTE: the coefficients above are transcribed from memory, not verified
    against the primary source in this session -- treat n2_denitrification
    flux as approximate until double-checked against the paper directly,
    the same "approximate until verified" treatment AGENTS.md already gives
    the RGA's nominal Faraday-cup sensitivity.
    """
    ts = math.log((298.15 - temp_degc) / (273.15 + temp_degc))
    c = _AR_SOLUBILITY_COEFFS
    ln_c = (
        c["A0"]
        + c["A1"] * ts
        + c["A2"] * ts**2
        + c["A3"] * ts**3
        + sal_psu * (c["B0"] + c["B1"] * ts + c["B2"] * ts**2)
    )
    return math.exp(ln_c)


def concentration_series(cycles: pl.DataFrame, value_col: str) -> pl.DataFrame:
    """Per-(experiment_number, chamber) OLS fit of value_col vs elapsed_time (minutes).

    Mirrors dashboard.experiment_rates() but Streamlit-free and without its
    Argon-ratio mass-variable branch (irrelevant to flux calculations).
    ``cycles`` must carry elapsed_time as a native Duration (true of both
    egcf_chamber_cycles as written by pipeline.run() and the dashboard's
    live-built cycle-averaged table -- neither ever round-trips through
    CSV before reaching this function). A group with fewer than 2 valid
    points is omitted, not an error.
    """
    base = (
        cycles.select(
            "experiment_number",
            "chamber",
            "timestamp",
            (pl.col("elapsed_time").dt.total_seconds() / 60).alias("elapsed_time_min"),
            pl.col(value_col).alias("value"),
        )
        .drop_nulls("value")
        .sort("timestamp")
    )
    if base.is_empty():
        return pl.DataFrame(schema=_SERIES_SCHEMA)
    rows = []
    for (exp_num, chamber), group in base.group_by(["experiment_number", "chamber"]):
        fit = ols_fit(group["elapsed_time_min"].to_list(), group["value"].to_list())
        if fit is None:
            continue
        rows.append(
            {
                "experiment_number": exp_num,
                "chamber": chamber,
                "experiment_start": group["timestamp"].min(),
                "slope": fit["slope"],
                "r2": fit["r2"],
                "n": fit["n"],
            }
        )
    return pl.DataFrame(rows, schema=_SERIES_SCHEMA).sort("experiment_start") if rows else pl.DataFrame(schema=_SERIES_SCHEMA)


def _flux_rows(
    series: pl.DataFrame,
    variable: str,
    native_to_umol_per_l: float,
    chamber_volume_l: float,
    chamber_area_m2: float,
) -> list[dict]:
    return [
        {
            "experiment_number": row["experiment_number"],
            "chamber": row["chamber"],
            "experiment_start": row["experiment_start"],
            "variable": variable,
            "slope_native_per_min": row["slope"],
            "r2": row["r2"],
            "n_points": row["n"],
            "output_value": row["slope"] * native_to_umol_per_l * chamber_volume_l / chamber_area_m2 * MIN_PER_HOUR,
            "output_unit": "umol m-2 h-1",
        }
        for row in series.iter_rows(named=True)
    ]


def _rate_rows(series: pl.DataFrame, variable: str, output_unit: str) -> list[dict]:
    return [
        {
            "experiment_number": row["experiment_number"],
            "chamber": row["chamber"],
            "experiment_start": row["experiment_start"],
            "variable": variable,
            "slope_native_per_min": row["slope"],
            "r2": row["r2"],
            "n_points": row["n"],
            "output_value": row["slope"] * MIN_PER_HOUR,
            "output_unit": output_unit,
        }
        for row in series.iter_rows(named=True)
    ]


def _n2_dissolved_umol_l_column(cycles: pl.DataFrame) -> pl.Series:
    temps = cycles["temp_degC"].to_list()
    sals = cycles["sal_PSU"].to_list()
    ratios = (cycles["mass_28_avg"] / cycles["mass_40_avg"]).to_list()
    values = [
        ratio * ar_solubility_umol_kg(t, s) * SEAWATER_DENSITY_KG_PER_L
        if ratio is not None and t is not None and s is not None
        else None
        for ratio, t, s in zip(ratios, temps, sals)
    ]
    return pl.Series("_n2_umol_l", values, dtype=pl.Float64)


def compute_fluxes(cycles: pl.DataFrame, chamber_volume_l: float, chamber_area_m2: float) -> pl.DataFrame:
    """Compute benthic flux (or, for temp_degC, a bare rate) per (experiment, chamber).

    ``cycles`` is an egcf_chamber_cycles-shaped table (or the dashboard's
    equivalent live-built one) covering every experiment, not just one --
    same convention as dashboard.experiment_rates(). Computes whichever of
    oxygen (from oxygen_mgL), h_ion (from pH), n2_denitrification (from the
    mass_28/mass_40 ratio and Ar solubility, N2:Ar method per Kana et al.
    1994) and temp_degC are available as columns; a variable whose source
    column(s) are absent is simply not included in the output, not an
    error. Calibrated flux for any other RGA mass (e.g. CO2 at mass 44)
    isn't implemented -- see AGENTS.md for why.
    """
    if chamber_volume_l <= 0 or chamber_area_m2 <= 0:
        raise ValueError("chamber_volume_l and chamber_area_m2 must both be positive")

    rows: list[dict] = []

    if "oxygen_mgL" in cycles.columns:
        series = concentration_series(cycles, "oxygen_mgL")
        rows += _flux_rows(series, "oxygen", O2_UMOL_PER_MG, chamber_volume_l, chamber_area_m2)

    if "pH" in cycles.columns:
        h_ion = cycles.with_columns(pl.lit(10.0).pow(-pl.col("pH")).alias("_h_ion_mol_l"))
        series = concentration_series(h_ion, "_h_ion_mol_l")
        rows += _flux_rows(series, "h_ion", H_ION_UMOL_PER_MOL, chamber_volume_l, chamber_area_m2)

    if {"mass_28_avg", "mass_40_avg", "temp_degC", "sal_PSU"} <= set(cycles.columns):
        n2 = cycles.filter(pl.col("mass_40_avg") != 0)
        series = concentration_series(n2.with_columns(_n2_dissolved_umol_l_column(n2)), "_n2_umol_l")
        rows += _flux_rows(series, "n2_denitrification", 1.0, chamber_volume_l, chamber_area_m2)

    if "temp_degC" in cycles.columns:
        series = concentration_series(cycles, "temp_degC")
        rows += _rate_rows(series, "temp_degC", "degC h-1")

    if not rows:
        return pl.DataFrame(schema=_FLUX_SCHEMA)
    return pl.DataFrame(rows, schema=_FLUX_SCHEMA).sort(["experiment_start", "chamber", "variable"])
