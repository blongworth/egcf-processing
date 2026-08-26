# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**This repo has a detailed `AGENTS.md` — read it.** It covers the data-format gotchas, layer
architecture, and non-obvious implementation decisions in more depth than is repeated here. This
file is a shorter map to get oriented; `AGENTS.md` is the authoritative reference.

## What this is

A pipeline that parses raw log files from an eelgrass benthic flux chamber (EGFC) lander into
analysis-ready Parquet/CSV datasets. The lander runs unattended incubation experiments alternating
between two chambers (C1/C2), logging RGA mass-scan data, a SCALUP water-quality sonde, and
valve/chamber state. Two raw sources carry the same underlying payload grammar: SD-card recovery
files (`gems_YYYY-MM-DD-HH-MM.txt`) and near-real-time surface telemetry logs
(`surface_YYYY-MM-DD-HH-MM_lander.log`) — both are discovered and merged into the same pipeline.

## Commands

```
uv sync                            # install deps
uv run pytest -q                   # run the full test suite
uv run pytest tests/test_cycles.py # run a single test file
uv run pytest tests/test_cycles.py::test_name  # run a single test
uv run main.py <raw_dir> --out-dir <out_dir> [--settle-offset-s 60] [--format parquet|csv]
uv run streamlit run dashboard.py  # launch the read-only dashboard
```

No linter/formatter is configured. Match existing style: no comments unless something is genuinely
non-obvious, explicit polars schemas, functions over classes.

`data/` is gitignored — raw logs and processed output live there and are never committed.

## Architecture: three layers, one shared aggregator

```
src/egcf_processing/
  lines.py       # parse_line(payload) -> dict|None -- the core per-line grammar
  discovery.py   # find gems_*.txt + surface_*_lander.log files, merge by rotation ts, skip 0-byte
  reader.py      # read files in order (dispatch on filename), concatenate parsed records
  combine.py     # Layer A: build + write status/rga/scalup/valve tables
  rga_scans.py   # Layer B window boundaries: RGA scan-cycle detection
  cycles.py      # Layer C window boundaries: chamber-cycle + experiment numbering
  aggregate.py   # shared windowed aggregation used by both Layer B and C
  pipeline.py    # orchestrates the above; run(raw_dir, out_dir, settle_offset_s, output_format)
  cli.py         # argparse entry point
  dashboard.py   # Streamlit app (read-only viewer over data/processed)
main.py          # thin shim -> egcf_processing.cli.main
dashboard.py     # thin shim -> egcf_processing.dashboard
```

1. **Layer A (raw combined)** — every raw file (gems + surface) parsed and concatenated by tag into
   `status.parquet` (`!:`), `rga.parquet` (`R:`), `scalup.parquet` (`P:`), `valve.parquet` (`V:`).
   No aggregation. Written first; every later stage reads from these, not from raw files again.
2. **Layer B (`egcf_rga_scans`)** — one row per RGA mass-scan cycle, boundaries detected from the
   data itself (a "masses seen in this scan" set that resets on a repeat).
3. **Layer C (`egcf_chamber_cycles`)** — one row per chamber measurement cycle (`V:` transition into
   `(chamber, Re)` to the next transition), averaged over `[cycle_start + settle_offset, next_transition)`.

Layers B and C share `aggregate.aggregate_onto_windows()` — they differ only in which `windows`
table they're aggregated onto. If you need a third grain, add another window-boundary function and
call the same aggregator rather than duplicating averaging logic.

Cycle/scan boundary detection deliberately does **not** re-sort by timestamp — it trusts arrival
order from `reader.py` (file-rotation order, then in-file line order), so a non-monotonic embedded
timestamp becomes an invalid/too-short window rather than being silently sorted away. Only the
`join_asof` calls in `aggregate.py` sort, since that join requires it.

Surface log rotations ship as a pair, `surface_<ts>_lander.log` (payload wrapped in a
`<surface_receipt_ts> <payload>` envelope, discarded in favor of the lander-embedded `ts`) and
`surface_<ts>_events.log` (comms traffic, `iso8601 direction payload`, out of scope — never carries
an R:/V:/P:/!: payload).

See `AGENTS.md` for: the confirmed data-format divergences from the firmware README (`V:` format
change, `P:` field-count eras, `!:` never occurring in the gems corpus but present in the surface
corpus, why `PM:` is intentionally unparsed), the RGA raw-current-to-Amps/Torr conversion in
`aggregate.py`, CSV Duration-column handling, and dashboard test-harness notes
(`streamlit.testing.v1.AppTest`).
