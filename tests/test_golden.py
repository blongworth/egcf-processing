"""Golden-file regression test: a small, real-format raw log fixture with a
checked-in expected output, so a refactor that silently changes parsing or
aggregation behavior is caught even if it doesn't break any of the more
targeted unit tests.

tests/fixtures/gems_gold_standard.txt is one experiment, two chamber cycles
(C1 then C2, 30s each), with two RGA masses, one detailed !: status line, both
real P: field-count eras (6-field and 7-field -- see AGENTS.md's "P: has two
field-count eras" gotcha), and enough RGA readings to form three complete
mass-scan cycles (two within C1, one within C2) -- small enough to verify
entirely by hand, but real gems_*.txt format end to end (not a hand-assembled
DataFrame), exercising the full pipeline: discovery -> reader -> lines ->
combine (Layer A) -> cycles/rga_scans + aggregate (Layers B and C).

Deliberately NOT included, since they don't produce output rows and are
already covered by targeted rejection-path unit tests in test_lines.py: the
old 4-field V: format, the undocumented PM: tag, and !:'s 2-field generic
fallback (all real in the raw corpora, but out of scope by design -- adding
them here would test skip-path behavior, not output correctness, and would
make the by-hand verification harder to follow for no benefit).
"""

from pathlib import Path

import polars as pl
import pytest

from egcf_processing.pipeline import run

FIXTURES = Path(__file__).parent / "fixtures"


def _assert_matches_expected(actual: pl.DataFrame, expected: pl.DataFrame) -> None:
    assert actual.columns == expected.columns
    assert actual.height == expected.height
    for col in expected.columns:
        actual_vals = actual[col].to_list()
        expected_vals = expected[col].to_list()
        if expected[col].dtype.is_numeric():
            assert actual_vals == pytest.approx(expected_vals, nan_ok=True)
        else:
            assert actual_vals == expected_vals


def test_gold_standard_chamber_cycles(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "gems_2026-01-01-00-00.txt").write_bytes((FIXTURES / "gems_gold_standard.txt").read_bytes())

    out_dir = tmp_path / "processed"
    run(raw_dir, out_dir, settle_offset_s=0, output_format="csv")

    actual = pl.read_csv(out_dir / "egcf_chamber_cycles.csv")
    expected = pl.read_csv(FIXTURES / "egcf_chamber_cycles_expected.csv")
    _assert_matches_expected(actual, expected)


def test_gold_standard_rga_scans(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "gems_2026-01-01-00-00.txt").write_bytes((FIXTURES / "gems_gold_standard.txt").read_bytes())

    out_dir = tmp_path / "processed"
    run(raw_dir, out_dir, settle_offset_s=0, output_format="csv")

    actual = pl.read_csv(out_dir / "egcf_rga_scans.csv")
    expected = pl.read_csv(FIXTURES / "egcf_rga_scans_expected.csv")
    _assert_matches_expected(actual, expected)
