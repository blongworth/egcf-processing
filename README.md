# egcf-processing

Processing pipeline for the eelgrass benthic flux chamber (EGFC) lander. Parses raw log files from
unattended incubation experiments into analysis-ready Parquet/CSV datasets, and includes a
read-only Streamlit dashboard for reviewing processed output.

The lander alternates between two chambers (C1/C2) and logs RGA mass-scan data, a SCALUP
water-quality sonde, and valve/chamber state. Two raw sources are supported and can be mixed
under the same `raw_dir`: SD-card recovery files (`gems_YYYY-MM-DD-HH-MM.txt`) and near-real-time
surface telemetry logs (`surface_YYYY-MM-DD-HH-MM_lander.log`).

## Install

```
uv sync
```

## Usage

```
uv run main.py <raw_dir> --out-dir <out_dir> [--settle-offset-s 60] [--format parquet|csv]
```

- `<raw_dir>`: directory (searched recursively) containing `gems_*.txt` and/or
  `surface_*_lander.log` files.
- `--out-dir`: output directory (default `data/processed`).
- `--settle-offset-s`: seconds excluded from the start of each chamber cycle before averaging (default 60).
- `--format`: `parquet` (default) or `csv`.
- `--partial-pressure-sensitivity` / `--total-pressure-sensitivity`: override the RGA A/Torr
  sensitivity used to convert ion current to pressure (defaults are nominal spec values, not this
  instrument's factory calibration).

This writes four combined raw tables (`status`, `rga`, `scalup`, `valve`) plus two derived,
aggregated tables: `egcf_rga_scans` (one row per RGA mass-scan cycle) and `egcf_chamber_cycles`
(one row per chamber measurement cycle, with experiment numbering and elapsed time).

## Dashboard

```
uv run streamlit run dashboard.py
```

A read-only viewer over an already-processed `data/processed`-style directory — it has no control
to trigger a pipeline run. Tabs: Status, Measurements (raw/Amps/Torr toggle), and Experiment Data
(per-experiment C1 vs C2 comparison).

## Development

```
uv run pytest -q
```

See `AGENTS.md` for data-format details, architecture notes, and implementation decisions.
