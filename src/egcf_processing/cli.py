from __future__ import annotations

import argparse
import logging
from pathlib import Path

from egcf_processing.pipeline import (
    DEFAULT_N2_AR_SENSITIVITY_RATIO,
    DEFAULT_OUTPUT_FORMAT,
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
    )


if __name__ == "__main__":
    main()
