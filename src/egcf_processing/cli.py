from __future__ import annotations

import argparse
import logging
from pathlib import Path

from egcf_processing.pipeline import DEFAULT_OUTPUT_FORMAT, DEFAULT_SETTLE_OFFSET_S, run


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
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    run(args.raw_dir, args.out_dir, settle_offset_s=args.settle_offset_s, output_format=args.format)


if __name__ == "__main__":
    main()
