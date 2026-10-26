# egcf-processing

Processing pipeline for the eelgrass benthic flux chamber (EGFC) lander. It parses raw log files from
unattended incubation experiments into analysis-ready Parquet/CSV datasets, computes benthic fluxes
and O2 metabolism, and includes a read-only Streamlit dashboard for reviewing processed output.

The lander alternates between two chambers (C1/C2) and logs RGA mass-scan data, a SCALUP
water-quality sonde, and valve/chamber state. It has two raw sources, which can be mixed in one
deployment: SD-card recovery files (`gems_YYYY-MM-DD-HH-MM.txt`) and near-real-time surface
telemetry logs (`surface_YYYY-MM-DD-HH-MM_lander.log` / `_events.log`). Co-deployed Odyssey PAR
logger and HOBO oxygen logger CSV exports are read alongside them.

## Install

```
uv sync
```

## Settings and local defaults

Defaults for the CLI and dashboard live in `egcf.toml` at the repo root (committed). It holds the
data paths, output format, chamber geometry, settle offset, RGA sensitivities, PAR offset,
metabolism thresholds and dashboard defaults, each commented with its units.

To change defaults on your machine only, for example to keep data outside the repo, create
`egcf.local.toml` next to it. It is gitignored, uses the same sections and key names, and needs only
the keys you want to override:

```toml
# egcf.local.toml
[paths]
raw_root = "/Users/me/Documents/egcf/data/raw"             # deployments to process
processed_root = "/Users/me/Documents/egcf/data/processed" # output, and what the dashboard lists

[output]
output_format = "both"                                     # parquet | csv | both
```

Settings are applied in this order of precedence, highest first:
1. CLI flag
2. the deployment's `deployment.toml` `[processing]` table
3. `egcf.local.toml`
4. `egcf.toml`
5. code default

The CLI logs each value with where it came from. Relative paths are relative to the toml file.
Unknown keys and bad values are errors that name the file; TOML syntax errors also show the line.
Use `--config <file>` or `$EGCF_CONFIG` to load a different settings file; an `egcf.local.toml` next
to that file is still applied.

## Deployments

Raw data is organised one folder per deployment under `raw_root`. The folder name is the deployment
id. Each folder holds a hand-edited `deployment.toml` and that deployment's files; the files can sit
at any depth, and each type is recognised by its name pattern or content.
`deployment.toml.example` is a commented template:

```toml
site = "Hadley Harbor"
lat = 41.514359              # optional, decimal degrees
lon = -70.704184             # optional
start = 2026-10-13T20:42:50Z # in-water window [start, end); needs Z or +hh:mm, zero-padded dates
end   = 2026-10-15T18:43:00Z # optional, omit for an ongoing deployment
test  = false                # true hides it in the dashboard by default
c1 = "grass"                 # grass | sediment | none
c2 = "sediment"
comments = """Free-text field notes."""

[processing]                 # optional: override any egcf.toml processing setting for this deployment
chamber_volume_l = 30.0
```

A PAR or HOBO file that spans several deployments is copied into each deployment's folder; each
deployment trims it to its own window. A folder without a `deployment.toml` is processed as-is,
untrimmed, if you name it explicitly. Batch runs skip it.

## Usage

```
uv run main.py                          # every deployment below raw_root -> processed_root/<deployment_id>/
uv run main.py <deployment_id>          # one deployment under raw_root
uv run main.py <path/to/raw_dir>        # any folder; --out-dir to choose where output goes
uv run main.py --format both            # write CSV as well as Parquet
uv run main.py --help                   # every option, with its default
```

Every deployment toml is validated before anything is written. Options override the settings above:
- `--chamber-volume-l` or `--chamber-height-m`, plus `--chamber-area-m2` (defaults 30 L, 0.30 m²)
- `--settle-offset-s` (default 60)
- `--partial-pressure-sensitivity` / `--total-pressure-sensitivity` (A/Torr), `--n2-ar-sensitivity-ratio`
- `--par-time-offset-h`, `--par-start`/`--par-end`, `--par-calibrations`
- `--dark-par-threshold`, `--min-par-coverage`, `--metabolism-min-r2`, `--chamber-par-transmittance`

Output for each deployment, in `processed_root/<deployment_id>/`:

| Layer | Tables | Trimmed to window |
|---|---|---|
| A: combined raw | `status`, `rga`, `scalup`, `valve`, `system_health`, `par`, `par_daily`, `hobo_oxygen` | no |
| B: RGA scans | `egcf_rga_scans` (one row per mass-scan cycle) | yes |
| C: chamber cycles | `egcf_chamber_cycles` (one row per chamber measurement cycle) | yes |
| D: fluxes | `egcf_fluxes` (µmol m⁻² h⁻¹ per experiment, chamber, variable) | yes |
| E: metabolism | `egcf_metabolism` (light/dark O2, R, NCP, GPP), `egcf_pi_fit` (P–I curve per chamber) | yes |
| metadata | `deployment`, `processing` (parameters used), copy of `deployment.toml` | — |

Each table has a `deployment_id` column. Tables from Layer D onward also have a `condition` column
(grass/sediment), taken from the deployment's C1/C2 settings. A re-run first deletes the folder's
earlier tables in either format, but leaves other files alone.

## Dashboard

```
uv run streamlit run dashboard.py
```

A read-only viewer over `processed_root`; it cannot start a pipeline run. In the sidebar you pick a
deployment (test deployments are hidden unless ticked) and can narrow its time window. You can also
adjust the settling time, chamber geometry and RGA sensitivities; the Rates and Metabolism tabs
recompute live with those values. Tabs: Status, Measurements, Rates and Fluxes, and Metabolism.

## Development

```
uv run pytest -q
```

See `AGENTS.md` for data-format details, architecture notes, and implementation decisions.
