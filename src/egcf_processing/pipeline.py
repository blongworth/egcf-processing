"""Orchestrates the full pipeline: discovery -> reader -> combine -> aggregate."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from egcf_processing import aggregate, combine, cycles, discovery, events, flux, metabolism, par, reader, rga_scans

logger = logging.getLogger(__name__)

DEFAULT_SETTLE_OFFSET_S = 60.0
DEFAULT_OUTPUT_FORMAT = "parquet"
DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR = aggregate.DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR
DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR = aggregate.DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR
DEFAULT_N2_AR_SENSITIVITY_RATIO = flux.DEFAULT_N2_AR_SENSITIVITY_RATIO
DEFAULT_PAR_CALIBRATIONS_PATH = par.DEFAULT_CALIBRATIONS_PATH
DEFAULT_PAR_TIME_OFFSET_H = 0.0
DEFAULT_DARK_PAR_THRESHOLD_UMOL_M2_S = metabolism.DEFAULT_DARK_PAR_THRESHOLD_UMOL_M2_S
DEFAULT_MIN_PAR_COVERAGE = metabolism.DEFAULT_MIN_PAR_COVERAGE
DEFAULT_METABOLISM_MIN_R2 = metabolism.DEFAULT_MIN_R2


def run(
    raw_dir: Path,
    out_dir: Path,
    chamber_volume_l: float,
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
    dark_par_threshold_umol_m2_s: float = DEFAULT_DARK_PAR_THRESHOLD_UMOL_M2_S,
    min_par_coverage: float = DEFAULT_MIN_PAR_COVERAGE,
    metabolism_min_r2: float = DEFAULT_METABOLISM_MIN_R2,
) -> dict:
    files = discovery.find_all_files(raw_dir)
    logger.info("found %d gems_*.txt/surface_*_lander.log file(s) under %s", len(files), raw_dir)

    records = reader.read_all(files)
    logger.info("parsed %d record(s)", len(records))

    event_files = discovery.find_surface_events_files(raw_dir)
    logger.info("found %d surface_*_events.log file(s) under %s", len(event_files), raw_dir)
    records.extend(events.read_all_events(event_files))

    tables = combine.build_tables(records)
    written = combine.write_tables(tables, out_dir, output_format)
    for name, path in written.items():
        logger.info("wrote %s (%d rows) -> %s", name, tables[name].height, path)

    par_search_dir = par_dir if par_dir is not None else raw_dir
    par_files = discovery.find_par_files(par_search_dir)
    logger.info("found %d Odyssey PAR export(s) under %s", len(par_files), par_search_dir)
    par_table = par.read_all_par(
        par_files, par.load_calibrations(par_calibrations_path), par_time_offset_h, par_start, par_end
    )
    par_path = combine.write_df(par_table, out_dir, "par", output_format)
    logger.info("wrote par (%d rows) -> %s", par_table.height, par_path)

    chamber_windows, cycle_stats = cycles.chamber_cycle_windows(tables["valve"], settle_offset_s)
    logger.info(
        "chamber cycles: %d valid, %d dropped as too-short",
        cycle_stats["total_cycles"] - cycle_stats["dropped_too_short"],
        cycle_stats["dropped_too_short"],
    )

    scan_windows = rga_scans.rga_scan_windows(tables["rga"])
    scan_windows = rga_scans.attach_chamber_context(scan_windows, chamber_windows)

    layer_b = aggregate.aggregate_onto_windows(
        scan_windows,
        tables["rga"],
        tables["scalup"],
        tables["status"],
        partial_pressure_sensitivity_a_per_torr,
        total_pressure_sensitivity_a_per_torr,
        par=par_table,
    )
    layer_c = aggregate.aggregate_onto_windows(
        chamber_windows,
        tables["rga"],
        tables["scalup"],
        tables["status"],
        partial_pressure_sensitivity_a_per_torr,
        total_pressure_sensitivity_a_per_torr,
        par=par_table,
    )

    layer_d = flux.compute_fluxes(layer_c, chamber_volume_l, chamber_area_m2, n2_ar_sensitivity_ratio)
    layer_d = flux.attach_experiment_par(
        layer_d, par_table, cycles.experiment_spans(chamber_windows, settle_offset_s)
    )
    if n2_ar_sensitivity_ratio == 1.0:
        logger.warning(
            "n2_ar_sensitivity_ratio is 1.0 (uncalibrated): N2:Ar flux magnitude is not quantitative -- "
            "measure it with flux.n2_ar_sensitivity_from_standard() against an air-equilibrated standard"
        )

    layer_e = metabolism.classify_o2_fluxes(layer_d, dark_par_threshold_umol_m2_s, min_par_coverage, metabolism_min_r2)
    pi_fit = metabolism.fit_pi_curves(layer_e)

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
        "n_files": len(files),
        "n_records": len(records),
        "n_system_health_rows": tables["system_health"].height,
        "n_par_rows": par_table.height,
        "cycle_stats": cycle_stats,
        "layer_b_rows": layer_b.height,
        "layer_c_rows": layer_c.height,
        "layer_d_rows": layer_d.height,
        "layer_e_rows": layer_e.height,
        "pi_fit_rows": pi_fit.height,
    }
