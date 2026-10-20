from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_OUT_ROOT = Path("data/processed")

from egcf_processing.config import CONFIG_FILENAME, load_config
from egcf_processing.deployment import load_deployment, resolve_processing
from egcf_processing.pipeline import (
    DEFAULT_CHAMBER_AREA_M2,
    DEFAULT_CHAMBER_PAR_TRANSMITTANCE,
    DEFAULT_CHAMBER_VOLUME_L,
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
    PROCESSING_DEFAULTS,
    run,
)

# Settings flags whose value resolves argument > deployment.toml > egcf.toml > default: dest -> setting key.
_PROCESSING_FLAGS = {
    "chamber_volume_l": "chamber_volume_l",
    "chamber_height_m": "chamber_height_m",
    "chamber_area_m2": "chamber_area_m2",
    "settle_offset_s": "settle_offset_s",
    "partial_pressure_sensitivity": "partial_pressure_sensitivity_a_per_torr",
    "total_pressure_sensitivity": "total_pressure_sensitivity_a_per_torr",
    "n2_ar_sensitivity_ratio": "n2_ar_sensitivity_ratio",
    "par_time_offset_h": "par_time_offset_h",
    "dark_par_threshold": "dark_par_threshold_umol_m2_s",
    "min_par_coverage": "min_par_coverage",
    "metabolism_min_r2": "metabolism_min_r2",
    "chamber_par_transmittance": "chamber_par_transmittance",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="egcf-process")
    parser.add_argument(
        "raw_dir",
        type=Path,
        help="Deployment directory (gems/surface logs, PAR, HOBO, optional deployment.toml), or a deployment id "
        "under raw_root from egcf.toml (default data/raw)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Settings file; defaults to $EGCF_CONFIG, else ./{CONFIG_FILENAME} if present",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to <processed_root>/<deployment_id> when raw_dir has a deployment.toml, "
        "else <processed_root> (processed_root from egcf.toml, default data/processed)",
    )
    parser.add_argument(
        "--settle-offset-s",
        type=float,
        default=None,
        help=f"Seconds to exclude from the start of each chamber cycle before averaging (default {DEFAULT_SETTLE_OFFSET_S:g})",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default=None,
        help=f"Output file format for all written tables (default {DEFAULT_OUTPUT_FORMAT})",
    )
    depth = parser.add_mutually_exclusive_group()
    depth.add_argument(
        "--chamber-volume-l",
        type=float,
        default=None,
        help="Chamber enclosed water volume in liters (same for C1 and C2), used to scale flux; overrides "
        f"[processing] in deployment.toml (default {DEFAULT_CHAMBER_VOLUME_L:g})",
    )
    depth.add_argument(
        "--chamber-height-m",
        type=float,
        default=None,
        help="Mean water height inside the chamber in m, instead of --chamber-volume-l (volume = height * area); "
        "overrides [processing] in deployment.toml",
    )
    parser.add_argument(
        "--chamber-area-m2",
        type=float,
        default=None,
        help="Sediment footprint area enclosed by the chamber base in m^2 (same for C1 and C2); overrides "
        f"[processing] in deployment.toml (default {DEFAULT_CHAMBER_AREA_M2:g})",
    )
    parser.add_argument(
        "--partial-pressure-sensitivity",
        type=float,
        default=None,
        help="RGA partial pressure sensitivity in A/Torr, used to convert per-mass ion current to Torr "
        f"(default {DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR:g})",
    )
    parser.add_argument(
        "--total-pressure-sensitivity",
        type=float,
        default=None,
        help="RGA total pressure sensitivity in A/Torr, used to convert total pressure current to Torr "
        f"(default {DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR:g})",
    )
    parser.add_argument(
        "--n2-ar-sensitivity-ratio",
        type=float,
        default=None,
        help="RGA mass-28/mass-40 sensitivity ratio, from an air-equilibrated standard "
        "(see flux.n2_ar_sensitivity_from_standard). Leaving this at 1.0 makes N2:Ar flux "
        f"non-quantitative in magnitude (default {DEFAULT_N2_AR_SENSITIVITY_RATIO:g})",
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
        default=None,
        help="CSV of Odyssey PAR calibrations (sensor_number, serial_number, cal_date, interval_s, slope, "
        "intercept); defaults to par_calibrations in egcf.toml, else the bundled par_calibrations.csv",
    )
    parser.add_argument(
        "--par-time-offset-h",
        type=float,
        default=None,
        help="Hours added to every PAR logger timestamp to bring it onto the lander's UTC clock "
        f"(default {DEFAULT_PAR_TIME_OFFSET_H:g})",
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
        "--hobo-dir",
        type=Path,
        default=None,
        help="Directory to search for HOBO oxygen logger exports; defaults to raw_dir",
    )
    parser.add_argument(
        "--dark-par-threshold",
        type=float,
        default=None,
        help="Experiments with mean PAR below this (umol photons m^-2 s^-1) are dark (respiration) "
        f"(default {DEFAULT_DARK_PAR_THRESHOLD_UMOL_M2_S:g})",
    )
    parser.add_argument(
        "--min-par-coverage",
        type=float,
        default=None,
        help="Minimum fraction of an experiment covered by PAR readings for its O2 flux to be used "
        f"(default {DEFAULT_MIN_PAR_COVERAGE:g})",
    )
    parser.add_argument(
        "--metabolism-min-r2",
        type=float,
        default=None,
        help="Minimum O2 flux fit r2 to use it in metabolism/P-I "
        f"(default {DEFAULT_METABOLISM_MIN_R2:g}: near-zero fluxes have low r2)",
    )
    parser.add_argument(
        "--chamber-par-transmittance",
        type=float,
        default=None,
        help="Fraction of ambient PAR the chamber walls and lid transmit to the sediment, in (0, 1]; "
        f"{DEFAULT_CHAMBER_PAR_TRANSMITTANCE:g} (default) means unmeasured, so metabolism/P-I use ambient PAR",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    settings = load_config(args.config)
    raw_dir = args.raw_dir
    raw_root = settings.get("raw_root", DEFAULT_RAW_ROOT)
    if not raw_dir.exists() and (raw_root / raw_dir).is_dir():
        raw_dir = raw_root / raw_dir
    out_root = settings.get("processed_root", DEFAULT_OUT_ROOT)
    meta = load_deployment(raw_dir)
    out_dir = args.out_dir
    if meta is None:
        out_dir = out_dir or out_root
        logger.info("no deployment.toml in %s: processing everything, untrimmed", raw_dir)
    else:
        out_dir = out_dir or out_root / meta["deployment_id"]
        logger.info(
            "deployment %s at %s, window [%s, %s) UTC",
            meta["deployment_id"],
            meta["site"],
            meta["start"],
            meta["end"] or "open",
        )
        if meta["test"]:
            logger.warning("*** %s is a TEST deployment (test = true) ***", meta["deployment_id"])
    processing = resolve_processing(
        meta,
        {key: getattr(args, dest) for dest, key in _PROCESSING_FLAGS.items()},
        PROCESSING_DEFAULTS,
        settings,
    )
    for key, (value, source) in processing.items():
        logger.info("%s = %g (from %s)", key, value, source)
    values = {key: value for key, (value, _) in processing.items()}
    run(
        raw_dir,
        out_dir,
        chamber_volume_l=values.pop("chamber_volume_l", None),
        chamber_height_m=values.pop("chamber_height_m", None),
        chamber_area_m2=values.pop("chamber_area_m2"),
        output_format=args.format or settings.get("output_format", DEFAULT_OUTPUT_FORMAT),
        par_dir=args.par_dir,
        par_calibrations_path=args.par_calibrations or settings.get("par_calibrations", DEFAULT_PAR_CALIBRATIONS_PATH),
        par_start=args.par_start,
        par_end=args.par_end,
        hobo_dir=args.hobo_dir,
        **values,
    )

if __name__ == "__main__":
    main()
