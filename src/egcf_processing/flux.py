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

MIN_PER_HOUR = 60.0

# Ratio of the RGA's mass-28 sensitivity to its mass-40 sensitivity. The raw
# ion-current ratio I28/I40 is NOT the molar N2/Ar ratio -- an RGA's
# transmission and ionization cross-section differ per mass -- so N2:Ar
# denitrification flux is only as accurate as this factor. 1.0 means
# "uncalibrated": the flux keeps its sign and shape but its magnitude is off
# by however far the true sensitivity ratio is from unity. Measure it with
# n2_ar_sensitivity_from_standard() against air-equilibrated water at known
# T/S and pass the result through the pipeline. In the real bench corpus the
# raw I28/I40 runs ~45 where equilibrated seawater should read ~37, so the
# real factor is materially different from 1.
DEFAULT_N2_AR_SENSITIVITY_RATIO = 1.0

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

# Hamme & Emerson (2004), Deep-Sea Research I 51:1517-1528, Table 4 --
# verified against the paper, including its own published check values
# (10 degC, S=35: Ar 13.4622, N2 500.885 umol/kg), which tests/test_flux.py
# asserts directly. Valid 0-30 degC and distilled water through seawater, so
# the full estuarine salinity range is in scope. Note the salinity dependence
# is a Setchenow relation fit at only two salinities (~0 and ~35), so
# intermediate estuarine salinities are interpolated rather than measured.
_SOLUBILITY_COEFFS = {
    "Ar": {
        "A0": 2.79150,
        "A1": 3.17609,
        "A2": 4.13116,
        "A3": 4.90379,
        "B0": -6.96233e-3,
        "B1": -7.66670e-3,
        "B2": -1.16888e-2,
    },
    "N2": {
        "A0": 6.42931,
        "A1": 2.92704,
        "A2": 4.32531,
        "A3": 4.69149,
        "B0": -7.44129e-3,
        "B1": -8.02566e-3,
        "B2": -1.46775e-2,
    },
}

# UNESCO/EOS-80 one-atmosphere International Equation of State (Millero &
# Poisson 1981), verified against the UNESCO Technical Paper in Marine
# Science No. 44 p.22 check values, which tests/test_flux.py asserts. Used
# instead of a fixed ~1.025 kg/L because that approximation is ~1.4% off at
# S=15 and ~2.6% off in fresh water -- a real error in an estuarine setting,
# where it feeds straight through to N2 flux magnitude.
_EOS80_PURE = (999.842594, 6.793952e-2, -9.095290e-3, 1.001685e-4, -1.120083e-6, 6.536332e-9)
_EOS80_A = (8.24493e-1, -4.0899e-3, 7.6438e-5, -8.2467e-7, 5.3875e-9)
_EOS80_B = (-5.72466e-3, 1.0227e-4, -1.6546e-6)
_EOS80_C = 4.8314e-4


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


def gas_solubility_umol_kg(gas: str, temp_degc: float, sal_psu: float) -> float:
    """Ar or N2 solubility (umol/kg) in equilibrium with moist air at 1 atm total pressure.

    Hamme & Emerson (2004) Equation 1:
        ln C = A0 + A1*Ts + A2*Ts^2 + A3*Ts^3 + S*(B0 + B1*Ts + B2*Ts^2)
        Ts   = ln((298.15 - t) / (273.15 + t))
    with t in degC and S the practical salinity (PSS).

    "1 atm total pressure" is the reference the coefficients are fit to, so
    this is the concentration the water would hold if last equilibrated with
    the atmosphere at exactly 1013.25 mbar. Real barometric pressure varies a
    few percent about that and scales the result nearly linearly; this
    function does not correct for it.
    """
    ts = math.log((298.15 - temp_degc) / (273.15 + temp_degc))
    c = _SOLUBILITY_COEFFS[gas]
    ln_c = (
        c["A0"]
        + c["A1"] * ts
        + c["A2"] * ts**2
        + c["A3"] * ts**3
        + sal_psu * (c["B0"] + c["B1"] * ts + c["B2"] * ts**2)
    )
    return math.exp(ln_c)


def ar_solubility_umol_kg(temp_degc: float, sal_psu: float) -> float:
    """Argon solubility (umol/kg); see gas_solubility_umol_kg."""
    return gas_solubility_umol_kg("Ar", temp_degc, sal_psu)


def seawater_density_kg_per_l(temp_degc: float, sal_psu: float) -> float:
    """Seawater density (kg/L) at one atmosphere, UNESCO/EOS-80.

    Converts the solubility functions' per-kg concentrations to the per-litre
    basis the chamber volume is expressed in. Pressure (depth) is ignored --
    at lander depths the compressibility correction is far smaller than the
    N2:Ar calibration uncertainty that dominates the flux.
    """
    t, s = temp_degc, sal_psu
    p = _EOS80_PURE
    rho_w = p[0] + p[1] * t + p[2] * t**2 + p[3] * t**3 + p[4] * t**4 + p[5] * t**5
    a = _EOS80_A
    coef_a = a[0] + a[1] * t + a[2] * t**2 + a[3] * t**3 + a[4] * t**4
    b = _EOS80_B
    coef_b = b[0] + b[1] * t + b[2] * t**2
    return (rho_w + coef_a * s + coef_b * s**1.5 + _EOS80_C * s**2) / 1000


def n2_ar_sensitivity_from_standard(raw_ratio: float, temp_degc: float, sal_psu: float) -> float:
    """Instrument mass-28/mass-40 sensitivity ratio from an air-equilibrated standard.

    Run water equilibrated with air at a known, stable temperature and
    salinity through the chamber, take the RGA's raw I28/I40 there, and this
    returns the factor to divide subsequent raw ratios by so they become true
    molar N2/Ar ratios. Pass the result as ``n2_ar_sensitivity_ratio``.
    """
    true_ratio = gas_solubility_umol_kg("N2", temp_degc, sal_psu) / gas_solubility_umol_kg("Ar", temp_degc, sal_psu)
    return raw_ratio / true_ratio


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


def _with_n2_dissolved_umol_l(cycles: pl.DataFrame, n2_ar_sensitivity_ratio: float) -> pl.DataFrame:
    """Add dissolved [N2] (umol/L) from the raw mass-28/mass-40 ion-current ratio.

    [N2] = (I28/I40 / k) * Ar_solubility(T0,S0) * density(T0,S0), where k is
    the instrument sensitivity ratio (see DEFAULT_N2_AR_SENSITIVITY_RATIO).
    Uses the raw *_avg counts rather than *_torr so the RGA's approximate
    nominal Faraday-cup sensitivity cancels in the ratio.

    T0/S0 are the conditions at each incubation's **first** cycle, not each
    cycle's own -- the chamber stays sealed for a whole experiment and is
    flushed only between experiments, so the enclosed water is a closed
    volume and inert Ar genuinely has one fixed concentration throughout.
    Recomputing the Ar term per cycle would let chamber temperature drift
    (Ar solubility moves about -2%/degC) masquerade as N2 production or
    consumption. Salinity cannot change in a sealed chamber either, so
    anchoring S also drops sonde noise out of the Ar term.
    """
    anchor = (
        cycles.drop_nulls(["temp_degC", "sal_PSU"])
        .sort("timestamp")
        .group_by(["experiment_number", "chamber"])
        .agg(pl.col("temp_degC").first().alias("_t0"), pl.col("sal_PSU").first().alias("_s0"))
    )
    joined = cycles.join(anchor, on=["experiment_number", "chamber"], how="left")
    ratios = (joined["mass_28_avg"] / joined["mass_40_avg"]).to_list()
    values = [
        (ratio / n2_ar_sensitivity_ratio) * ar_solubility_umol_kg(t0, s0) * seawater_density_kg_per_l(t0, s0)
        if ratio is not None and t0 is not None and s0 is not None
        else None
        for ratio, t0, s0 in zip(ratios, joined["_t0"].to_list(), joined["_s0"].to_list())
    ]
    return joined.with_columns(pl.Series("_n2_umol_l", values, dtype=pl.Float64)).drop("_t0", "_s0")


def compute_fluxes(
    cycles: pl.DataFrame,
    chamber_volume_l: float,
    chamber_area_m2: float,
    n2_ar_sensitivity_ratio: float = DEFAULT_N2_AR_SENSITIVITY_RATIO,
) -> pl.DataFrame:
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
    if n2_ar_sensitivity_ratio <= 0:
        raise ValueError("n2_ar_sensitivity_ratio must be positive")

    rows: list[dict] = []

    if "oxygen_mgL" in cycles.columns:
        series = concentration_series(cycles, "oxygen_mgL")
        rows += _flux_rows(series, "oxygen", O2_UMOL_PER_MG, chamber_volume_l, chamber_area_m2)

    if "pH" in cycles.columns:
        h_ion = cycles.with_columns(pl.lit(10.0).pow(-pl.col("pH")).alias("_h_ion_mol_l"))
        series = concentration_series(h_ion, "_h_ion_mol_l")
        rows += _flux_rows(series, "h_ion", H_ION_UMOL_PER_MOL, chamber_volume_l, chamber_area_m2)

    if {"mass_28_avg", "mass_40_avg", "temp_degC", "sal_PSU"} <= set(cycles.columns):
        n2 = _with_n2_dissolved_umol_l(cycles.filter(pl.col("mass_40_avg") != 0), n2_ar_sensitivity_ratio)
        series = concentration_series(n2, "_n2_umol_l")
        rows += _flux_rows(series, "n2_denitrification", 1.0, chamber_volume_l, chamber_area_m2)

    if "temp_degC" in cycles.columns:
        series = concentration_series(cycles, "temp_degC")
        rows += _rate_rows(series, "temp_degC", "degC h-1")

    if not rows:
        return pl.DataFrame(schema=_FLUX_SCHEMA)
    return pl.DataFrame(rows, schema=_FLUX_SCHEMA).sort(["experiment_start", "chamber", "variable"])
