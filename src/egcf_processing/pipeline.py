"""Orchestrates the full pipeline: discovery -> reader -> combine -> aggregate."""

from __future__ import annotations

import logging
from pathlib import Path

from egcf_processing import aggregate, combine, cycles, discovery, flux, reader, rga_scans

logger = logging.getLogger(__name__)

DEFAULT_SETTLE_OFFSET_S = 60.0
DEFAULT_OUTPUT_FORMAT = "parquet"
DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR = aggregate.DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR
DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR = aggregate.DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR
DEFAULT_N2_AR_SENSITIVITY_RATIO = flux.DEFAULT_N2_AR_SENSITIVITY_RATIO


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
) -> dict:
    files = discovery.find_all_files(raw_dir)
    logger.info("found %d gems_*.txt/surface_*_lander.log file(s) under %s", len(files), raw_dir)

    records = reader.read_all(files)
    logger.info("parsed %d record(s)", len(records))

    tables = combine.build_tables(records)
    written = combine.write_tables(tables, out_dir, output_format)
    for name, path in written.items():
        logger.info("wrote %s (%d rows) -> %s", name, tables[name].height, path)

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
    )
    layer_c = aggregate.aggregate_onto_windows(
        chamber_windows,
        tables["rga"],
        tables["scalup"],
        tables["status"],
        partial_pressure_sensitivity_a_per_torr,
        total_pressure_sensitivity_a_per_torr,
    )

    layer_d = flux.compute_fluxes(layer_c, chamber_volume_l, chamber_area_m2, n2_ar_sensitivity_ratio)
    if n2_ar_sensitivity_ratio == 1.0:
        logger.warning(
            "n2_ar_sensitivity_ratio is 1.0 (uncalibrated): N2:Ar flux magnitude is not quantitative -- "
            "measure it with flux.n2_ar_sensitivity_from_standard() against an air-equilibrated standard"
        )

    scans_path = combine.write_df(layer_b, out_dir, "egcf_rga_scans", output_format)
    cycles_path = combine.write_df(layer_c, out_dir, "egcf_chamber_cycles", output_format)
    fluxes_path = combine.write_df(layer_d, out_dir, "egcf_fluxes", output_format)
    logger.info("wrote rga_scans (%d rows) -> %s", layer_b.height, scans_path)
    logger.info("wrote chamber_cycles (%d rows) -> %s", layer_c.height, cycles_path)
    logger.info("wrote fluxes (%d rows) -> %s", layer_d.height, fluxes_path)

    return {
        "n_files": len(files),
        "n_records": len(records),
        "cycle_stats": cycle_stats,
        "layer_b_rows": layer_b.height,
        "layer_c_rows": layer_c.height,
        "layer_d_rows": layer_d.height,
    }
