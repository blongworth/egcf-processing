# AGENTS.md

Guidance for AI coding agents working in this repo.

## What this is

A pipeline that parses raw log files from an eelgrass benthic flux chamber
(EGFC) lander into analysis-ready Parquet/CSV datasets. The lander runs
unattended incubation experiments alternating between two chambers (C1/C2),
logging RGA mass-scan data, a SCALUP water-quality sonde, valve/chamber
state, and (per the firmware spec) turbopump/pump status to SD-card files
named `gems_YYYY-MM-DD-HH-MM.txt`, rotating roughly every 4 hours.

Firmware reference: https://raw.githubusercontent.com/blongworth/egfc-firmware/refs/heads/main/README.md
— **treat this as a starting point, not ground truth.** The real data in
`data/raw/lander/` diverges from it in several confirmed ways (see below).
Always check actual sample files before assuming the README's format holds.

## Layout

```
src/egcf_processing/
  lines.py        # parse_line(payload) -> dict|None -- the core per-line grammar
  discovery.py     # find gems_*.txt + surface_*_lander.log files, parse rotation ts, skip 0-byte
  reader.py        # read files in order (dispatch by filename), concatenate parsed records
  combine.py       # Layer A: build + write status/rga/scalup/valve tables
  rga_scans.py     # Layer B window boundaries: RGA scan-cycle detection
  cycles.py        # Layer C window boundaries: chamber-cycle + experiment numbering
  aggregate.py     # shared windowed aggregation used by both Layer B and C
  pipeline.py      # orchestrates the above; run(raw_dir, out_dir, settle_offset_s, output_format)
  cli.py           # argparse entry point
main.py            # thin shim -> egcf_processing.cli.main
tests/             # one test file per module above, plus test_pipeline.py (end-to-end)
data/              # gitignored -- raw logs and processed output live here, never committed
```

## Two raw sources, one pipeline

`raw_dir` may contain `gems_*.txt` files (SD-card recovery, the original source), `surface_*_lander.log`
files (near-real-time telemetry relayed to a surface unit while the lander is still deployed), or both
mixed under arbitrary subdirectories — `discovery.find_all_files()` `rglob`s for both and merges them
into one file list ordered by each file's own rotation timestamp (embedded in its filename). `reader.read_file`
dispatches on the `surface_` filename prefix: gems lines are the bare payload; surface lines wrap the
*same* payload grammar in a `<surface_receipt_ts> <payload>` envelope (plus a `# ...` header, per
`surface-lander-log-v1`). The surface receipt timestamp is discarded, not treated as `ts` — every downstream
timeseries is keyed on the lander-embedded timestamp `parse_line` already extracts from the payload itself,
so gems and surface records are directly comparable/mergeable once parsed.

The surface log ships as a *pair* of files per rotation, `surface_<ts>_lander.log` and
`surface_<ts>_events.log`; only `_lander.log` is in scope. `_events.log` has a 3-field format
(`iso8601 direction payload`, e.g. `RX_CONSOLE VSTAT` / `TX_LANDER VSTAT`) recording console/lander
comms traffic, not measurements — it never contains an R:/V:/P:/!: payload in the real test corpus, so
`find_surface_files` excludes it by not matching the `_lander.log` glob (same exclusion-by-glob pattern
`find_gems_files` uses for legacy formats), rather than by explicit filtering.

**`!:` (status) occurs in the real surface corpus, unlike the gems corpus.** The "gems `!:` never
occurs" gotcha below is specific to the SD-card recovery data; `data/raw/surface/egcf_surface_test_data_2026-08-25/`
has thousands of real `!:` lines, so `turbo_speed_hz`/`turbo_power_w`/`raw_total_pressure_current`/
`pump_rpm` do get populated when processing surface data, unlike the gems-only case described below.

## Pipeline model (three layers)

1. **Layer A (raw combined)** — every raw file (gems + surface, see above) parsed and concatenated by
   tag into `status.parquet` (`!:`), `rga.parquet` (`R:`), `scalup.parquet` (`P:`), `valve.parquet` (`V:`).
   No aggregation. Written first; every later stage reads from these, not from raw files again.
2. **Layer B (`egcf_rga_scans`)** — one row per RGA mass-scan cycle (~10s pass through the
   configured mass list). Scan boundaries are detected from the data itself (a
   "masses seen in this scan" set that resets on a repeat), not from a fixed mass count/order.
3. **Layer C (`egcf_chamber_cycles`)** — one row per chamber measurement cycle (`V:` transition
   into `(chamber, Re)` to the next transition), averaged over `[cycle_start + settle_offset, next_transition)`.
   `experiment_number` increments when a `(C1, Re)` transition follows a `(C2, Fl)` transition
   since the last boundary; `elapsed_time` is time since that experiment's start.

Layers B and C share one aggregation function, `aggregate.aggregate_onto_windows()` — they differ
only in which `windows` table (window_start, window_end, chamber, experiment_number, elapsed_time)
they're aggregated onto. Don't duplicate the averaging logic if you need a third grain; add another
window-boundary function instead and call the same aggregator.

## Data format gotchas (confirmed against real files, not just the README)

- **Only `gems_YYYY-MM-DD-HH-MM.txt` is in scope.** `data_*.txt`, `*_test*.txt`, `*_smurp*.txt`,
  and `gems_pump_*.csv` are legacy/unrelated formats from earlier bench testing — excluded by
  `discovery.py`'s glob, not by explicit filtering. Many `gems_*.txt` files are 0 bytes; these are
  skipped.
- **`V:` format changed mid-testing.** Old format (4 fields, verbose state-machine names like
  `CHAMBER_TOGGLE`) is rejected by `lines.py` (returns `None`); only the new 3-field
  `V:<ts>,<C1/C2>,<Re/Fl/Unknown>` format is parsed. This means chamber/experiment context is only
  available from ~2026-08-10T19:52 onward in the current test corpus — large null stretches before
  that are correct, not a bug.
- **`P:` has two field-count eras**, both still in use in real data: 6 fields (no `pressure_mbar`)
  and 7 fields (matches the README). `lines.py` branches on field count, not on date.
- **`!:` (status) never occurs in the real gems (SD-card) test corpus** — it does occur in the surface
  corpus, see above. None of the README's other untimestamped tags (`TS,`, `TP,`, `PS,`, `VS,`, `S,`,
  `CFG,`, `ST,`, `RE,`, `OK,`/`ACK,`/`DONE,`/`ERR,`) occur in either corpus.
  Per explicit user direction, `turbo_speed_hz`, `turbo_power_w`, `total_pressure_amps`, and
  `water_pump_rpm` in Layers B/C are sourced **only** from the README-documented `!:` detailed-status
  row — not from `PM:`, an undocumented but real, timestamped pump-telemetry tag also present in the
  data. `PM:` is intentionally out of scope; don't "fix" this by wiring it in without checking with
  the user first, it was a deliberate choice, not an oversight.
- **Untimestamped tags inherit the most recent embedded timestamp** (`R:`/`V:`/`P:`/`!:`) seen
  earlier in the same file. This mechanism exists in the data-format understanding but isn't
  exercised by current code, since no untimestamped tag is currently parsed — don't build unused
  forward-fill machinery for it; only add it when a real untimestamped tag needs parsing.

## Key implementation decisions worth knowing before changing things

- **Cycle/scan boundary detection deliberately does NOT re-sort by timestamp** — it trusts the
  arrival order produced by `reader.py` (file-rotation order, then in-file line order). This is
  what lets `cycles.chamber_cycle_windows()` detect a non-monotonic embedded timestamp (a clock
  jump) as an invalid/too-short window instead of silently normalizing it away by sorting. Only the
  join_asof calls in `aggregate.py` explicitly sort (asof-join requires it) — that's a different,
  narrower concern from cycle-boundary detection.
- **CSV has no duration type.** `elapsed_time` is `Duration` in the parquet output; `combine.write_df`
  converts it to seconds (float) only for the CSV output path. Don't write Duration columns to CSV
  directly — polars raises `ComputeError`.
- **RGA mass columns (`mass_{m}_avg`) are discovered dynamically**, not hardcoded to the documented
  default list (`2,15,16,18,28,30,32,33,34,40,44`) — the configured mass list can change.
- **Raw RGA/status ion currents are converted to Amps and Torr in `aggregate.py`**, per the RGA
  RS-232 protocol documented in `manuals/RGAm.pdf`. `R:` mass currents and the status row's
  `raw_total_pressure_current` are raw integer counts in units of `1e-16 A` (`RAW_CURRENT_AMPS_PER_COUNT`);
  each mass gets `mass_{m}_avg` (raw count, unchanged, kept for backward compatibility),
  `mass_{m}_amps`, and `mass_{m}_torr` (amps divided by a partial-pressure sensitivity in A/Torr);
  total pressure gets `total_pressure_amps` and `total_pressure_torr` (same pattern, using a
  total-pressure sensitivity). The sensitivity defaults
  (`DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR` / `DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR`,
  both `2e-4`) come from the RGAm.pdf specifications table's nominal Faraday-cup sensitivity (measured
  with N2 @ 28 amu) — **not** this specific instrument's factory-calibrated `SP`/`ST` values, which
  aren't recoverable from the SD-card logs. Treat the Torr columns as approximate unless overridden
  with a measured sensitivity via `pipeline.run(...)`'s `partial_pressure_sensitivity_a_per_torr`/
  `total_pressure_sensitivity_a_per_torr` params or the CLI's `--partial-pressure-sensitivity`/
  `--total-pressure-sensitivity` flags.
- Both derived layers (B and C) always carry the full output schema even when a source table is
  empty (e.g. `status.parquet` is currently always 0 rows) — those columns are null, not absent. If
  you add a new source table, preserve this "schema present even when empty" behavior.

## Dashboard

`dashboard.py` (root shim, mirrors `main.py`) runs a Streamlit app defined in
`src/egcf_processing/dashboard.py`. It's a **read-only viewer** over an
already-processed `data/processed`-style directory (parquet, falling back to
csv per table if no parquet exists) — it has no control to trigger a
pipeline run itself, by design. Three tabs: Status (turbo speed/power/temp,
plus total pressure only if `status.parquet` has any non-null
`raw_total_pressure_current` — currently always empty against real data, so
this is normally a "no data" message, not a bug), Measurements (RGA mass
data plus scalup sonde data, all with a raw/Amps/Torr unit toggle reusing
`aggregate.py`'s conversion constants), and Experiment Data (per-experiment
C1-vs-C2 comparison of one RGA mass or other variable against elapsed time,
in minutes). The status tab's "current" plot is deliberately
`turbo_power_w` — there's no field literally named "current" in
`STATUS_SCHEMA` besides the pressure ion current, which already gets its own
plot; this was an explicit user choice, not a guess.

The Experiment Data tab's own "Grain" radio (`Full data` / `Cycle averages`)
picks between `_render_experiment_full_data` and
`_render_experiment_cycle_averages` -- both gated purely on the raw `valve`
table being present, since **both grains are computed live by the dashboard
from raw tables**, never by reading the pipeline's precomputed
`egcf_chamber_cycles.parquet`. This is deliberate: it lets the settling
period be adjusted interactively without re-running `egcf-process`.

Both grains share one "Settling time after valve switch (s)" slider
(`key="settle_offset_s"`, same widget key in both render functions so the
value persists across a grain switch, default `pipeline.DEFAULT_SETTLE_OFFSET_S`)
-- this is the *same* time-based settle_offset_s the pipeline itself uses
(cycles.chamber_cycle_windows), just recomputed live instead of fixed at
`egcf-process` run time.

`Cycle averages` computes chamber-cycle windows with the slider's value
directly (`chamber_cycle_windows(valve, settle_offset_s=slider_value)`,
exactly like the CLI would) and calls `aggregate.aggregate_onto_windows()` --
the exact same function the pipeline uses to build `egcf_chamber_cycles` --
against those windows, so its output has the identical schema pipeline
output always has (mass_{m}_avg/amps/torr, scalup/status columns, all
present even when a source table is empty or absent -- an absent raw table
is passed in as an empty DataFrame with the right schema).

`Full data` needs a different approach, because it must keep every reading
visible (greyed out, not dropped) rather than trim the window boundary
itself. It always computes windows with `settle_offset_s=0.0` -- **not**
the slider value -- so `window_start` stays exactly the Re-transition
timestamp regardless of the slider, and `attach_experiment_context()`
(built on `aggregate.match_readings_to_windows()` for the join, promoted
from a pipeline-private `_bucketize` to a public function specifically for
dashboard reuse) flags each reading's own `settled_out` as
`(reading_ts - window_start) < settle_offset_s` -- the slider value used
only for this comparison, not for the window itself. Each row stays its
own point (no averaging), and `elapsed_time` is computed from **its own
timestamp** minus the experiment start (`reading_ts - exp_start_ts`), not
copied from a per-cycle constant (that constant would put every reading in
a cycle at the same elapsed time, collapsing them onto one x-position).
`_render_experiment_plot()` renders settled-out points as a single grey
(`#B0B0B0`) "dropped (settling)" trace instead of hiding them, while kept
points still render per-chamber in their normal colors.

`Cycle averages` also fits a rate for the selected variable: `linear_fit()`
is a plain ordinary-least-squares slope/intercept over `(elapsed_time_min,
value)` (pure Python, no numpy dependency added for it), computed
per-chamber against the *currently selected experiment*'s points and drawn
as a dashed same-colored "`{chamber} fit ({slope:.3g}/min)`" line on the
main plot via `_render_experiment_plot()`'s `fits` param. A second chart,
`_render_experiment_rates_plot()`, repeats that fit for *every* experiment
(via `experiment_rates()`, grouping the full un-filtered `with_experiment`
table by `(experiment_number, chamber)`) and plots the resulting per-chamber
rate against each experiment's start time -- so a fouling/drift trend across
the whole deployment is visible at a glance, not just within one experiment.
Rates aren't restricted to non-negative or log-scale display (unlike the
value plots) since a rate can be positive or negative (production vs.
consumption). An (experiment, chamber) pair with fewer than 2 valid points
is silently omitted from both the fit overlay and the rates chart (a
one-point "fit" is undefined) rather than raising. `Full data` does not get
a fit overlay -- fitting is deliberately scoped to cycle-averaged points,
which are far less noisy than individual raw readings.

The single-experiment plot's title carries the experiment's real start time
as a `<br><sup>...</sup>` HTML subtitle (`Started YYYY-MM-DD HH:MM:SS`), and
the "Experiment" selectbox itself shows that same start time on every option
(`"{experiment_number} ({start:%Y-%m-%d %H:%M:%S})"` via `st.selectbox`'s
`format_func` -- the selectbox's returned *value* is still the plain
experiment-number string, only its on-screen label changes), so both grains
must compute the *same* start time per experiment regardless of the settling
slider. `experiment_start_times()` computes `(ts_col - elapsed_time).min()`
grouped by `experiment_number` in one pass, returning a `dict[int, datetime]`
both call sites use for the selectbox's `format_func` and (after picking one)
the subtitle, rather than two separate computations. `Full data`'s windows
are always settle_offset_s=0.0, so calling it with `ts_col="window_start"`
is exact. `Cycle averages`' windows have the slider's settle_offset_s baked
into `window_start` (`chamber_cycle_windows` sets
`window_start = re_transition_ts + settle_offset_s`), so calling it with
`ts_col="timestamp"` (window_start, renamed by `aggregate_onto_windows`)
leaves a constant `+ settle_offset_s` bias in every returned value that must
be subtracted back out explicitly -- confirmed to match `Full data`'s values
exactly regardless of the slider position.

Non-mass variables under `Full data` map onto raw columns with different
names/shapes than `aggregate_onto_windows`'s output: `_SCALUP_PANELS`
supplies the case-insensitive raw-scalup-column lookup (reusing the same
trick as the Measurements tab's scalup panels), `_STATUS_DIRECT_RAW_COLS`
renames `pump_rpm` to `water_pump_rpm` (the other two status fields are
already same-named), and `total_pressure_amps`/`total_pressure_torr` are
computed from `status.raw_total_pressure_current` via
`RAW_CURRENT_AMPS_PER_COUNT` and the total-pressure-sensitivity sidebar
control (which is why `render_experiment_tab` takes
`total_pressure_sensitivity`, not `partial_pressure_sensitivity` — the
latter is unused here since mass variables are always shown as a
unit-invariant Argon ratio, never as an Amps/Torr value). Both grains show
mass variables as Argon-normalized ratios (mass ÷ mass 40, via
`mass_to_argon_ratio_expr` for the wide cycle-averaged table or
`rga_full_ratio_to_mass`'s nearest-in-time join for the long raw table), on
a log-scale y-axis, and always plot elapsed time in minutes on the x-axis.
Mass 40 itself, and any raw/Amps/Torr unit choice for masses, are therefore
not options here — the ratio is unit-invariant, and Argon has nothing to
normalize against.

The Measurements tab's RGA panels are driven by one "RGA data source" radio
(`Full RGA data` / `Chamber cycle averages`, only offering a source that's
actually present in the loaded dataset) — there is deliberately no separate
RGA-cycle-averaged panel; that grain is only exposed via the Experiment Data
tab's own grain toggle. Full RGA data renders as lines; chamber-cycle
averages render as points (`mode="markers"`), since each point is one
already-averaged cycle rather than a continuous signal. Whichever source is
selected also drives a second panel, "Masses / mass 40", showing every other
mass's current ratioed to mass 40's — `rga_full_ratio_to_mass` pairs each
mass's full-resolution reading with the *nearest-in-time* mass-40 reading
(`join_asof`, since the RGA scans one mass at a time and different masses
never share an exact timestamp), while `rga_wide_ratio_to_mass` divides
same-row `mass_{m}_avg` columns directly for the already-aligned
chamber-cycle table. Both ratio helpers always use raw counts regardless of
the unit toggle, since a shared linear scale factor per reading cancels out
of any ratio. Every per-mass trace across both RGA panels (and the ratio
panel) is colored via `mass_color_map()`, built once from the union of
masses across the raw `rga` and `egcf_chamber_cycles` tables so a given mass
is the same color in every panel regardless of which subset a panel's own
multiselect shows — and every RGA panel uses a log-scale y-axis, since ion
currents span several orders of magnitude.

Data-loading/transform helpers (`load_table`, `with_elapsed_time_s`,
`discover_masses`, `rga_current_to_unit`, `mass_color_map`,
`rga_full_ratio_to_mass`, `rga_wide_ratio_to_mass`, `mass_to_argon_ratio_expr`,
`variable_value_expr`, `attach_experiment_context`, `linear_fit`,
`experiment_rates`, `experiment_start_times`) are kept free of Streamlit calls so
`tests/test_dashboard.py` can exercise them directly; only `render_*`/`main`
touch `st`. When testing interactive behavior by hand
instead of a browser, `streamlit.testing.v1.AppTest` runs the app headlessly
and surfaces exceptions from bad widget-state interactions (e.g. a selectbox
key reused across tables backed by different-typed columns) — this caught a
real dtype-comparison bug during development that a plain `streamlit run` +
manual click-through likely wouldn't have (this environment has no browser).

## Development

```
uv sync                 # install deps (polars, pytest dev group, streamlit, plotly)
uv run pytest -q        # run the test suite
uv run main.py <raw_dir> --out-dir <out_dir> [--settle-offset-s 60] [--format parquet|csv]
uv run streamlit run dashboard.py   # launch the dashboard
```

Test data lives in `data/raw/lander/egcf_lander_test_data_2026-08-24/` (gems SD-card format) and
`data/raw/surface/egcf_surface_test_data_2026-08-25/` (surface telemetry format), both gitignored,
local only. Both are real bench-test data with a compressed ~30s chamber-toggle cadence, not the
production ~15 min — the default `--settle-offset-s 60` will correctly drop most cycles as
too-short against them; use a smaller value (e.g. `5`) when validating against these datasets.

No linter/formatter is configured yet. Match the existing style (no comments unless something is
genuinely non-obvious, explicit polars schemas, functions over classes).
