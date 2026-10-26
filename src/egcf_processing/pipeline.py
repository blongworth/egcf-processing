"""Orchestrates the full pipeline: discovery -> reader -> combine -> aggregate."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

import polars as pl

from egcf_processing import (
    aggregate,
    combine,
    cycles,
    deployment,
    discovery,
    events,
    flux,
    hobo,
    metabolism,
    par,
    reader,
    rga_scans,
)

logger = logging.getLogger(__name__)

DEFAULT_SETTLE_OFFSET_S = 60.0
DEFAULT_OUTPUT_FORMAT = "parquet"
DEFAULT_CHAMBER_VOLUME_L = 30.0
DEFAULT_CHAMBER_AREA_M2 = 0.30
DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR = aggregate.DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR
DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR = aggregate.DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR
DEFAULT_N2_AR_SENSITIVITY_RATIO = flux.DEFAULT_N2_AR_SENSITIVITY_RATIO
DEFAULT_PAR_CALIBRATIONS_PATH = par.DEFAULT_CALIBRATIONS_PATH
DEFAULT_PAR_TIME_OFFSET_H = 0.0
DEFAULT_DARK_PAR_THRESHOLD_UMOL_M2_S = metabolism.DEFAULT_DARK_PAR_THRESHOLD_UMOL_M2_S
DEFAULT_MIN_PAR_COVERAGE = metabolism.DEFAULT_MIN_PAR_COVERAGE
DEFAULT_METABOLISM_MIN_R2 = metabolism.DEFAULT_MIN_R2
DEFAULT_CHAMBER_PAR_TRANSMITTANCE = metabolism.DEFAULT_CHAMBER_PAR_TRANSMITTANCE

PROCESSING_DEFAULTS = {
    "chamber_area_m2": DEFAULT_CHAMBER_AREA_M2,
    "chamber_volume_l": DEFAULT_CHAMBER_VOLUME_L,
    "settle_offset_s": DEFAULT_SETTLE_OFFSET_S,
    "partial_pressure_sensitivity_a_per_torr": DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR,
    "total_pressure_sensitivity_a_per_torr": DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR,
    "n2_ar_sensitivity_ratio": DEFAULT_N2_AR_SENSITIVITY_RATIO,
    "par_time_offset_h": DEFAULT_PAR_TIME_OFFSET_H,
    "dark_par_threshold_umol_m2_s": DEFAULT_DARK_PAR_THRESHOLD_UMOL_M2_S,
    "min_par_coverage": DEFAULT_MIN_PAR_COVERAGE,
    "metabolism_min_r2": DEFAULT_METABOLISM_MIN_R2,
    "chamber_par_transmittance": DEFAULT_CHAMBER_PAR_TRANSMITTANCE,
}

# Every table run() writes, so a re-run can clear its own earlier output (in either format) first.
OUTPUT_TABLES = (
    "status",
    "rga",
    "scalup",
    "valve",
    "system_health",
    "par",
    "par_daily",
    "hobo_oxygen",
    "egcf_rga_scans",
    "egcf_chamber_cycles",
    "egcf_fluxes",
    "egcf_metabolism",
    "egcf_pi_fit",
    "deployment",
    "processing",
)


def clear_outputs(out_dir: Path) -> list[Path]:
    """Delete this pipeline's outputs from an earlier run of out_dir; any other file is left alone.

    Without this a format switch leaves the other format's files behind, and the
    dashboard (parquet first) would show the stale ones; so would a deployment.toml
    copy after its raw toml was removed.
    """
    names = [f"{name}.{ext}" for name in OUTPUT_TABLES for ext in ("parquet", "csv")]
    stale = [out_dir / name for name in [*names, deployment.DEPLOYMENT_FILENAME] if (out_dir / name).is_file()]
    for path in stale:
        path.unlink()
    return stale


def _tag(df: pl.DataFrame, meta: dict | None) -> pl.DataFrame:
    """Prefix a literal deployment_id column so concatenated deployments stay unambiguous."""
    if meta is None:
        return df
    return df.select(pl.lit(meta["deployment_id"]).alias("deployment_id"), pl.all())


def run(
    raw_dir: Path,
    out_dir: Path,
    chamber_volume_l: float | None,
    chamber_area_m2: float,
    settle_offset_s: float = DEFAULT_SETTLE_OFFSET_S,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    partial_pressure_sensitivity_a_per_torr: float = DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR,
    total_pressure_sensitivity_a_per_torr: float = DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR,
    n2_ar_sensitivity_ratio: float = DEFAULT_N2_AR_SENSITIVITY_RATIO,
    par_dir: Path | None = None,
    par_calibrations_path: Path = DEFAULT_PAR_CALIBRATIONS_PATH,
    par_time_offset_h: float = DEFAULT_PAR_TIME_OFFSET_H,
    par_start: datetime | None = None,
    par_end: datetime | None = None,
    hobo_dir: Path | None = None,
    dark_par_threshold_umol_m2_s: float = DEFAULT_DARK_PAR_THRESHOLD_UMOL_M2_S,
    min_par_coverage: float = DEFAULT_MIN_PAR_COVERAGE,
    metabolism_min_r2: float = DEFAULT_METABOLISM_MIN_R2,
    chamber_par_transmittance: float = DEFAULT_CHAMBER_PAR_TRANSMITTANCE,
    chamber_height_m: float | None = None,
) -> dict:
    """Run every layer. Chamber geometry is the area plus exactly one of volume or height."""
    geometry = flux.chamber_geometry(chamber_area_m2, chamber_volume_l, chamber_height_m)
    chamber_volume_l = geometry["chamber_volume_l"]
    logger.info(
        "chamber geometry: %g L over %g m2 = %g m mean water height (V/A)",
        chamber_volume_l,
        chamber_area_m2,
        geometry["chamber_height_m"],
    )
    if Path(out_dir).resolve() == Path(raw_dir).resolve():
        raise ValueError(f"out_dir must differ from raw_dir ({raw_dir}): clearing old output would touch raw files")
    meta = deployment.load_deployment(raw_dir)
    stale = clear_outputs(Path(out_dir))
    if stale:
        logger.info("removed %d output file(s) from an earlier run of %s", len(stale), out_dir)
    if meta is not None:
        combine.write_df(deployment.deployment_table(meta), out_dir, "deployment", output_format)
        used = {
            **geometry,
            "chamber_geometry_from": "volume" if chamber_height_m is None else "height",
            "settle_offset_s": settle_offset_s,
            "partial_pressure_sensitivity_a_per_torr": partial_pressure_sensitivity_a_per_torr,
            "total_pressure_sensitivity_a_per_torr": total_pressure_sensitivity_a_per_torr,
            "n2_ar_sensitivity_ratio": n2_ar_sensitivity_ratio,
            "par_time_offset_h": par_time_offset_h,
            "dark_par_threshold_umol_m2_s": dark_par_threshold_umol_m2_s,
            "min_par_coverage": min_par_coverage,
            "metabolism_min_r2": metabolism_min_r2,
            "chamber_par_transmittance": chamber_par_transmittance,
        }
        combine.write_df(deployment.processing_table(meta, used), out_dir, "processing", output_format)
        shutil.copyfile(raw_dir / deployment.DEPLOYMENT_FILENAME, out_dir / deployment.DEPLOYMENT_FILENAME)

    files = discovery.find_all_files(raw_dir)
    logger.info("found %d gems_*.txt/surface_*_lander.log file(s) under %s", len(files), raw_dir)

    records = reader.read_all(files)
    logger.info("parsed %d record(s)", len(records))

    event_files = discovery.find_surface_events_files(raw_dir)
    logger.info("found %d surface_*_events.log file(s) under %s", len(event_files), raw_dir)
    records.extend(events.read_all_events(event_files))

    tables = combine.build_tables(records)
    written = combine.write_tables({name: _tag(df, meta) for name, df in tables.items()}, out_dir, output_format)
    for name, path in written.items():
        logger.info("wrote %s (%d rows) -> %s", name, tables[name].height, path)

    par_search_dir = par_dir if par_dir is not None else raw_dir
    par_files = discovery.find_par_files(par_search_dir)
    logger.info("found %d Odyssey PAR export(s) under %s", len(par_files), par_search_dir)
    par_table = par.read_all_par(
        par_files, par.load_calibrations(par_calibrations_path), par_time_offset_h, par_start, par_end
    )
    par_path = combine.write_df(_tag(par_table, meta), out_dir, "par", output_format)
    logger.info("wrote par (%d rows) -> %s", par_table.height, par_path)
    par_daily = par.daily_par(par_table)
    par_daily_path = combine.write_df(_tag(par_daily, meta), out_dir, "par_daily", output_format)
    logger.info("wrote par_daily (%d rows) -> %s", par_daily.height, par_daily_path)
    trend = par.daily_max_trend(par.daily_par(deployment.clip(par_table, meta)))
    if trend is not None:
        logger.info(
            "daily max PAR trend over %d full day(s): %+.1f umol m-2 s-1 per day (%+.1f%%/day, r2=%.2f) -- "
            "a sustained decline suggests diffuser biofouling; cloudy days also lower it",
            trend["n_days"],
            trend["slope_umol_m2_s_per_day"],
            trend["pct_per_day"],
            trend["r2"],
        )

    hobo_search_dir = hobo_dir if hobo_dir is not None else raw_dir
    hobo_files = discovery.find_hobo_files(hobo_search_dir)
    logger.info("found %d HOBO oxygen logger export(s) under %s", len(hobo_files), hobo_search_dir)
    hobo_table = hobo.read_all_hobo(hobo_files)
    hobo_path = combine.write_df(_tag(hobo_table, meta), out_dir, "hobo_oxygen", output_format)
    logger.info("wrote hobo_oxygen (%d rows) -> %s", hobo_table.height, hobo_path)

    clipped = {name: deployment.clip(df, meta) for name, df in tables.items()}
    par_clipped = deployment.clip(par_table, meta)

    chamber_windows, cycle_stats = cycles.chamber_cycle_windows(clipped["valve"], settle_offset_s)
    logger.info(
        "chamber cycles: %d valid, %d dropped as too-short",
        cycle_stats["total_cycles"] - cycle_stats["dropped_too_short"],
        cycle_stats["dropped_too_short"],
    )

    scan_windows = rga_scans.rga_scan_windows(clipped["rga"])
    scan_windows = rga_scans.attach_chamber_context(scan_windows, chamber_windows)

    layer_b = aggregate.aggregate_onto_windows(
        scan_windows,
        clipped["rga"],
        clipped["scalup"],
        clipped["status"],
        partial_pressure_sensitivity_a_per_torr,
        total_pressure_sensitivity_a_per_torr,
        par=par_clipped,
    )
    layer_c = aggregate.aggregate_onto_windows(
        chamber_windows,
        clipped["rga"],
        clipped["scalup"],
        clipped["status"],
        partial_pressure_sensitivity_a_per_torr,
        total_pressure_sensitivity_a_per_torr,
        par=par_clipped,
    )

    layer_d = flux.compute_fluxes(layer_c, chamber_volume_l, chamber_area_m2, n2_ar_sensitivity_ratio)
    layer_d = flux.attach_experiment_par(
        layer_d, par_clipped, cycles.experiment_spans(chamber_windows, settle_offset_s)
    )
    if n2_ar_sensitivity_ratio == 1.0:
        logger.warning(
            "n2_ar_sensitivity_ratio is 1.0 (uncalibrated): N2:Ar flux magnitude is not quantitative -- "
            "measure it with flux.n2_ar_sensitivity_from_standard() against an air-equilibrated standard"
        )

    layer_e = metabolism.classify_o2_fluxes(
        layer_d, dark_par_threshold_umol_m2_s, min_par_coverage, metabolism_min_r2, chamber_par_transmittance
    )
    pi_fit = metabolism.fit_pi_curves(layer_e, chamber_par_transmittance)
    if chamber_par_transmittance == 1.0 and layer_e["used"].any():
        logger.warning(
            "chamber_par_transmittance is 1.0 (unmeasured): the dark threshold and P-I parameters are relative "
            "to ambient PAR at the logger, not PAR at the sediment inside the chamber"
        )

    if meta is not None:
        condition = deployment.chamber_condition_expr(meta)
        layer_d = layer_d.with_columns(condition)
        layer_e = layer_e.with_columns(condition)
        pi_fit = pi_fit.with_columns(condition)
    layer_b, layer_c, layer_d, layer_e, pi_fit = (_tag(df, meta) for df in (layer_b, layer_c, layer_d, layer_e, pi_fit))

    scans_path = combine.write_df(layer_b, out_dir, "egcf_rga_scans", output_format)
    cycles_path = combine.write_df(layer_c, out_dir, "egcf_chamber_cycles", output_format)
    fluxes_path = combine.write_df(layer_d, out_dir, "egcf_fluxes", output_format)
    logger.info("wrote rga_scans (%d rows) -> %s", layer_b.height, scans_path)
    logger.info("wrote chamber_cycles (%d rows) -> %s", layer_c.height, cycles_path)
    logger.info("wrote fluxes (%d rows) -> %s", layer_d.height, fluxes_path)
    metabolism_path = combine.write_df(layer_e, out_dir, "egcf_metabolism", output_format)
    pi_fit_path = combine.write_df(pi_fit, out_dir, "egcf_pi_fit", output_format)
    logger.info("wrote metabolism (%d rows) -> %s", layer_e.height, metabolism_path)
    logger.info("wrote pi_fit (%d rows) -> %s", pi_fit.height, pi_fit_path)

    return {
        "deployment_id": meta["deployment_id"] if meta is not None else None,
        "n_files": len(files),
        "n_records": len(records),
        "n_system_health_rows": tables["system_health"].height,
        "n_par_rows": par_table.height,
        "n_par_days": par_daily.height,
        "n_hobo_rows": hobo_table.height,
        "cycle_stats": cycle_stats,
        "layer_b_rows": layer_b.height,
        "layer_c_rows": layer_c.height,
        "layer_d_rows": layer_d.height,
        "layer_e_rows": layer_e.height,
        "pi_fit_rows": pi_fit.height,
    }
