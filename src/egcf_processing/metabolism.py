"""Layer E: light/dark O2 metabolism and a per-chamber P-I curve from Layer D.

Input is egcf_fluxes' ``oxygen`` rows, which carry the experiment's mean PAR
(see flux.attach_experiment_par). Each flux is classified dark or light by a
PAR threshold:

- dark: respiration, R = -mean(dark O2 flux) per chamber, so R is positive
  for net O2 uptake. Never forced positive -- a positive mean dark flux gives
  a negative R and is reported as-is.
- light: net community production, NCP = the O2 flux, and gross production
  GPP = NCP + R (assumes light respiration equals dark respiration).

I is ``par_chamber_umol_m2_s`` = the logger's ambient experiment-mean PAR
times ``chamber_par_transmittance`` (the fraction the chamber walls and lid
pass). It defaults to 1.0 -- unmeasured -- in which case every threshold and
fit parameter is relative to ambient light, not light at the sediment.

The P-I fit is Jassby & Platt (1976), NCP = Pmax*tanh(alpha*I/Pmax) - R, over
light and dark points together, per chamber. I is PAR in umol photons m^-2 s^-1
while fluxes are mmol O2 m^-2 h^-1, so alpha's unit is (mmol O2 m^-2 h^-1) per
(umol photons m^-2 s^-1) and Ik = Pmax/alpha is in PAR units.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import polars as pl
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

# Night-time PAR reads the calibration intercept (6.45 umol m^-2 s^-1 for
# sensor 1), not 0, and dusk/dawn experiments average a little above it.
DEFAULT_DARK_PAR_THRESHOLD_UMOL_M2_S = 20.0
DEFAULT_CHAMBER_PAR_TRANSMITTANCE = 1.0
DEFAULT_MIN_PAR_COVERAGE = 0.9
# No r2 filter by default: a flux near zero (e.g. at the compensation
# irradiance) is a flat line whose r2 is low by construction, so filtering
# on r2 would bias the P-I curve against exactly those points.
DEFAULT_MIN_R2 = 0.0
MIN_FIT_POINTS = 4

METABOLISM_SCHEMA = {
    "experiment_number": pl.Int64,
    "chamber": pl.Utf8,
    "experiment_start": pl.Datetime,
    "par_mean_umol_m2_s": pl.Float64,
    "par_integrated_mol_m2": pl.Float64,
    "par_coverage": pl.Float64,
    "par_chamber_umol_m2_s": pl.Float64,
    "o2_flux_mmol_m2_h": pl.Float64,
    "r2": pl.Float64,
    "period": pl.Utf8,
    "used": pl.Boolean,
    "excluded_reason": pl.Utf8,
    "ncp_mmol_m2_h": pl.Float64,
    "gpp_mmol_m2_h": pl.Float64,
}

PI_FIT_SCHEMA = {
    "chamber": pl.Utf8,
    "chamber_par_transmittance": pl.Float64,
    "n_points": pl.Int64,
    "n_light": pl.Int64,
    "n_dark": pl.Int64,
    "r_dark_mmol_m2_h": pl.Float64,
    "pmax_mmol_m2_h": pl.Float64,
    "pmax_se": pl.Float64,
    "alpha_mmol_m2_h_per_par": pl.Float64,
    "alpha_se": pl.Float64,
    "r_fit_mmol_m2_h": pl.Float64,
    "r_fit_se": pl.Float64,
    "ik_umol_m2_s": pl.Float64,
    "fit_r2": pl.Float64,
    "converged": pl.Boolean,
}


def jassby_platt(par: np.ndarray, pmax: float, alpha: float, r: float) -> np.ndarray:
    return pmax * np.tanh(alpha * par / pmax) - r


def classify_o2_fluxes(
    fluxes: pl.DataFrame,
    dark_par_threshold_umol_m2_s: float = DEFAULT_DARK_PAR_THRESHOLD_UMOL_M2_S,
    min_par_coverage: float = DEFAULT_MIN_PAR_COVERAGE,
    min_r2: float = DEFAULT_MIN_R2,
    chamber_par_transmittance: float = DEFAULT_CHAMBER_PAR_TRANSMITTANCE,
) -> pl.DataFrame:
    """One row per O2 flux: light/dark period, whether it's used, NCP and GPP.

    Excluded rows are kept with an ``excluded_reason`` rather than dropped.
    R for GPP is each chamber's mean over *used* dark rows.
    """
    if not 0 < chamber_par_transmittance <= 1:
        raise ValueError("chamber_par_transmittance must be in (0, 1]")
    o2 = fluxes.filter(pl.col("variable") == "oxygen")
    if o2.is_empty() or "par_mean_umol_m2_s" not in o2.columns:
        return pl.DataFrame(schema=METABOLISM_SCHEMA)
    par = pl.col("par_chamber_umol_m2_s")
    rows = o2.select(
        "experiment_number",
        "chamber",
        "experiment_start",
        "par_mean_umol_m2_s",
        "par_integrated_mol_m2",
        "par_coverage",
        (pl.col("par_mean_umol_m2_s") * chamber_par_transmittance).alias("par_chamber_umol_m2_s"),
        pl.col("output_value").alias("o2_flux_mmol_m2_h"),
        "r2",
    ).with_columns(
        pl.when(par.is_null())
        .then(None)
        .when(par < dark_par_threshold_umol_m2_s)
        .then(pl.lit("dark"))
        .otherwise(pl.lit("light"))
        .alias("period"),
        pl.when(par.is_null())
        .then(pl.lit("no PAR"))
        .when(pl.col("par_coverage") < min_par_coverage)
        .then(pl.lit("PAR coverage below minimum"))
        .when(pl.col("r2").is_null() | (pl.col("r2") < min_r2))
        .then(pl.lit("r2 below minimum"))
        .alias("excluded_reason"),
    ).with_columns(pl.col("excluded_reason").is_null().alias("used"))

    r_dark = (
        rows.filter(pl.col("used") & (pl.col("period") == "dark"))
        .group_by("chamber")
        .agg((-pl.col("o2_flux_mmol_m2_h").mean()).alias("_r_dark"))
    )
    is_light = pl.col("used") & (pl.col("period") == "light")
    return (
        rows.join(r_dark, on="chamber", how="left")
        .with_columns(
            pl.when(is_light).then(pl.col("o2_flux_mmol_m2_h")).alias("ncp_mmol_m2_h"),
            pl.when(is_light).then(pl.col("o2_flux_mmol_m2_h") + pl.col("_r_dark")).alias("gpp_mmol_m2_h"),
        )
        .select(list(METABOLISM_SCHEMA))
        .cast(METABOLISM_SCHEMA)
        .sort(["experiment_start", "chamber"])
    )


def _fit_chamber(chamber: str, group: pl.DataFrame, chamber_par_transmittance: float) -> dict:
    light = group.filter(pl.col("period") == "light")
    dark = group.filter(pl.col("period") == "dark")
    row = {k: None for k in PI_FIT_SCHEMA}
    row.update(
        chamber=chamber,
        chamber_par_transmittance=chamber_par_transmittance,
        n_points=group.height,
        n_light=light.height,
        n_dark=dark.height,
        r_dark_mmol_m2_h=-dark["o2_flux_mmol_m2_h"].mean() if dark.height else None,
        converged=False,
    )
    if group.height < MIN_FIT_POINTS or light.is_empty():
        logger.warning(
            "P-I fit for %s skipped: %d usable point(s), %d light (need >= %d, >= 1 light)",
            chamber,
            group.height,
            light.height,
            MIN_FIT_POINTS,
        )
        return row

    par = group["par_chamber_umol_m2_s"].to_numpy()
    flux = group["o2_flux_mmol_m2_h"].to_numpy()
    r0 = row["r_dark_mmol_m2_h"] if row["r_dark_mmol_m2_h"] is not None else -float(flux.min())
    pmax0 = max(float(flux.max()) + r0, 1.0)
    alpha0 = pmax0 / max(float(np.median(light["par_chamber_umol_m2_s"].to_numpy())), 1.0)
    try:
        popt, pcov = curve_fit(
            jassby_platt,
            par,
            flux,
            p0=[pmax0, alpha0, r0],
            bounds=([1e-9, 1e-9, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=10000,
        )
    except (RuntimeError, ValueError) as exc:
        logger.warning("P-I fit for %s did not converge: %s", chamber, exc)
        return row

    pmax, alpha, r = (float(v) for v in popt)
    se = [math.sqrt(v) if np.isfinite(v) and v >= 0 else None for v in np.diag(pcov)]
    residuals = flux - jassby_platt(par, *popt)
    ss_tot = float(((flux - flux.mean()) ** 2).sum())
    row.update(
        pmax_mmol_m2_h=pmax,
        pmax_se=se[0],
        alpha_mmol_m2_h_per_par=alpha,
        alpha_se=se[1],
        r_fit_mmol_m2_h=r,
        r_fit_se=se[2],
        ik_umol_m2_s=pmax / alpha,
        fit_r2=1 - float((residuals**2).sum()) / ss_tot if ss_tot > 0 else None,
        converged=True,
    )
    return row


def fit_pi_curves(
    metabolism: pl.DataFrame, chamber_par_transmittance: float = DEFAULT_CHAMBER_PAR_TRANSMITTANCE
) -> pl.DataFrame:
    """Per-chamber Jassby-Platt fit over the used rows of classify_o2_fluxes' output.

    ``chamber_par_transmittance`` must match the value classify_o2_fluxes was
    given; it's only recorded here, so each fit row says what light basis its
    alpha and Ik are on.
    """
    used = metabolism.filter(pl.col("used"))
    if used.is_empty():
        return pl.DataFrame(schema=PI_FIT_SCHEMA)
    rows = [
        _fit_chamber(chamber, group, chamber_par_transmittance) for (chamber,), group in used.group_by(["chamber"])
    ]
    return pl.DataFrame(rows, schema=PI_FIT_SCHEMA).sort("chamber")
