from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from egcf_processing.pipeline import (
    DEFAULT_CHAMBER_PAR_TRANSMITTANCE,
    DEFAULT_DARK_PAR_THRESHOLD_UMOL_M2_S,
    DEFAULT_METABOLISM_MIN_R2,
    DEFAULT_MIN_PAR_COVERAGE,
    DEFAULT_N2_AR_SENSITIVITY_RATIO,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_PAR_CALIBRATIONS_PATH,
    DEFAULT_PAR_TIME_OFFSET_H,
    DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR,
    DEFAULT_SETTLE_OFFSET_S,
    DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR,
    run,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="egcf-process")
    parser.add_argument("raw_dir", type=Path, help="Directory containing gems_*.txt files")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/processed"), help="Output directory for output files"
    )
    parser.add_argument(
        "--settle-offset-s",
        type=float,
        default=DEFAULT_SETTLE_OFFSET_S,
        help="Seconds to exclude from the start of each chamber cycle before averaging",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default=DEFAULT_OUTPUT_FORMAT,
        help="Output file format for all written tables",
    )
    parser.add_argument(
        "--chamber-volume-l",
        type=float,
        required=True,
        help="Chamber enclosed water volume in liters (same for C1 and C2), used to scale flux",
    )
    parser.add_argument(
        "--chamber-area-m2",
        type=float,
        required=True,
        help="Sediment footprint area enclosed by the chamber base in m^2 (same for C1 and C2)",
    )
    parser.add_argument(
        "--partial-pressure-sensitivity",
        type=float,
        default=DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR,
        help="RGA partial pressure sensitivity in A/Torr, used to convert per-mass ion current to Torr",
    )
    parser.add_argument(
        "--total-pressure-sensitivity",
        type=float,
        default=DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR,
        help="RGA total pressure sensitivity in A/Torr, used to convert total pressure current to Torr",
    )
    parser.add_argument(
        "--n2-ar-sensitivity-ratio",
        type=float,
        default=DEFAULT_N2_AR_SENSITIVITY_RATIO,
        help="RGA mass-28/mass-40 sensitivity ratio, from an air-equilibrated standard "
        "(see flux.n2_ar_sensitivity_from_standard). Leaving this at 1.0 makes N2:Ar flux "
        "non-quantitative in magnitude",
    )
    parser.add_argument(
        "--par-dir",
        type=Path,
        default=None,
        help="Directory to search for Odyssey PAR logger exports; defaults to raw_dir",
    )
    parser.add_argument(
        "--par-calibrations",
        type=Path,
        default=DEFAULT_PAR_CALIBRATIONS_PATH,
        help="CSV of Odyssey PAR calibrations (sensor_number, serial_number, cal_date, interval_s, slope, "
        "intercept); defaults to the bundled par_calibrations.csv",
    )
    parser.add_argument(
        "--par-time-offset-h",
        type=float,
        default=DEFAULT_PAR_TIME_OFFSET_H,
        help="Hours added to every PAR logger timestamp to bring it onto the lander's UTC clock",
    )
    parser.add_argument(
        "--par-start",
        type=datetime.fromisoformat,
        default=None,
        help="Drop PAR rows before this ISO datetime (after the time offset), e.g. the deployment start",
    )
    parser.add_argument(
        "--par-end",
        type=datetime.fromisoformat,
        default=None,
        help="Drop PAR rows at or after this ISO datetime (after the time offset), e.g. recovery",
    )
    parser.add_argument(
        "--dark-par-threshold",
        type=float,
        default=DEFAULT_DARK_PAR_THRESHOLD_UMOL_M2_S,
        help="Experiments with mean PAR below this (umol photons m^-2 s^-1) are dark (respiration)",
    )
    parser.add_argument(
        "--min-par-coverage",
        type=float,
        default=DEFAULT_MIN_PAR_COVERAGE,
        help="Minimum fraction of an experiment covered by PAR readings for its O2 flux to be used",
    )
    parser.add_argument(
        "--metabolism-min-r2",
        type=float,
        default=DEFAULT_METABOLISM_MIN_R2,
        help="Minimum O2 flux fit r2 to use it in metabolism/P-I (default 0: near-zero fluxes have low r2)",
    )
    parser.add_argument(
        "--chamber-par-transmittance",
        type=float,
        default=DEFAULT_CHAMBER_PAR_TRANSMITTANCE,
        help="Fraction of ambient PAR the chamber walls and lid transmit to the sediment, in (0, 1]; "
        "1.0 (default) means unmeasured, so metabolism/P-I use ambient PAR",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    run(
        args.raw_dir,
        args.out_dir,
        chamber_volume_l=args.chamber_volume_l,
        chamber_area_m2=args.chamber_area_m2,
        settle_offset_s=args.settle_offset_s,
        output_format=args.format,
        partial_pressure_sensitivity_a_per_torr=args.partial_pressure_sensitivity,
        total_pressure_sensitivity_a_per_torr=args.total_pressure_sensitivity,
        n2_ar_sensitivity_ratio=args.n2_ar_sensitivity_ratio,
        par_dir=args.par_dir,
        par_calibrations_path=args.par_calibrations,
        par_time_offset_h=args.par_time_offset_h,
        par_start=args.par_start,
        par_end=args.par_end,
        dark_par_threshold_umol_m2_s=args.dark_par_threshold,
        min_par_coverage=args.min_par_coverage,
        metabolism_min_r2=args.metabolism_min_r2,
        chamber_par_transmittance=args.chamber_par_transmittance,
    )


if __name__ == "__main__":
    main()
