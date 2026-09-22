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
  flux.py          # Layer D: benthic vertical flux from Layer C's cycle averages
  pipeline.py      # orchestrates the above; run(raw_dir, out_dir, chamber_volume_l, chamber_area_m2, ...)
  cli.py           # argparse entry point
main.py            # thin shim -> egcf_processing.cli.main
tests/             # one test file per module above (including test_cli.py), plus test_pipeline.py (end-to-end)
tests/test_golden.py    # end-to-end regression tests against checked-in fixtures (see below)
tests/fixtures/    # gems_gold_standard.txt (input) + egcf_{chamber_cycles,rga_scans}_expected.csv (expected output)
data/              # gitignored -- raw logs and processed output live here, never committed
```

`tests/test_golden.py` runs the real pipeline (discovery through Layers B and C)
against `tests/fixtures/gems_gold_standard.txt` -- one experiment, two 30s chamber
cycles (C1 then C2), two RGA masses spread across three complete mass-scan cycles
(two within C1, one within C2), one detailed `!:` status line, and both real `P:`
field-count eras (6-field and 7-field, both merged into the same window's scalup
average) -- small enough to verify entirely by hand -- and compares the resulting
`egcf_chamber_cycles.csv` and
`egcf_rga_scans.csv` against the checked-in `tests/fixtures/egcf_chamber_cycles_expected.csv`
/ `egcf_rga_scans_expected.csv` (numeric columns via `pytest.approx`, since CSV
float round-tripping isn't guaranteed byte-exact). Unlike the other end-to-end
tests in `test_pipeline.py`, which hand-assemble raw text as inline Python strings
and assert on individual computed values, this is a real on-disk input file and
real on-disk expected-output files, so it also catches unintended CSV
formatting/schema/column-order changes that inline value-assertions wouldn't. If
you intentionally change output shape or values, regenerate the fixtures by
running the pipeline against `gems_gold_standard.txt` and re-verifying the new
numbers by hand before overwriting the expected CSVs -- never regenerate them by
just accepting whatever the (possibly buggy) code produces.

`tests/test_cli.py` covers `cli.py`'s argparse wiring specifically -- that each
flag (`--settle-offset-s`, `--format`, `--partial-pressure-sensitivity`,
`--total-pressure-sensitivity`) actually reaches `pipeline.run()` and changes
its output accordingly, that the required `--chamber-volume-l`/`--chamber-area-m2`
flags are supplied (via the shared `GEOMETRY` list -- placeholder values that
exercise the flux arithmetic, not real EGFC dimensions), and that the argparse defaults match `pipeline`'s
`DEFAULT_*` constants. This is functional (real `main()` calls against a
tmp_path raw dir, real output files read back), not a mock of `pipeline.run`,
consistent with the rest of the suite's preference for exercising real code
over mocking.

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
`surface_<ts>_events.log`, and the two are read by *separate* discovery/reader paths. `_events.log`
has a 3-field format (`iso8601 direction payload`, e.g. `RX_CONSOLE VSTAT` / `TX_LANDER VSTAT`)
recording console/lander comms traffic, and never contains an R:/V:/P:/!: payload — so
`find_surface_files`/`find_all_files`/`reader` exclude it by not matching the `_lander.log` glob
(same exclusion-by-glob pattern `find_gems_files` uses for legacy formats), rather than by explicit
filtering.

It is not, however, wholly out of scope: `discovery.find_surface_events_files()` +
`events.read_all_events()` read it for exactly one line shape, the 10-second
`SYSTEM battery voltage=27.02V current=0.033A temp=42.5C` housekeeping line, emitted as an `SH`
record into `system_health.parquet` (Layer A). `events.py` is deliberately parallel to
`lines.py`/`reader.py` rather than folded into them, since both the envelope and the payload
grammar differ. **These records use the surface *receipt* timestamp as `ts`** — a deliberate
divergence from the rule stated just above (lander logs discard the receipt ts in favor of the
lander-embedded one), because the envelope ts is the only timestamp these lines have. Every other
events-log line (headers, `RX_CONSOLE`/`TX_LANDER` commands, `SYSTEM startup complete`, the
`battery voltage below threshold; sending OFF to lander` prose, garbled serial) parses to `None`
and is counted as a DEBUG-level skip, not warned about per line — non-data is the norm here.

**`!:` (status) occurs in the real surface corpus, unlike the gems corpus.** The "gems `!:` never
occurs" gotcha below is specific to the SD-card recovery data; `data/raw/surface/egcf_surface_test_data_2026-08-25/`
has thousands of real `!:` lines, so `turbo_speed_hz`/`turbo_power_w`/`raw_total_pressure_current`/
`pump_rpm` do get populated when processing surface data, unlike the gems-only case described below.

## Pipeline model (four layers)

1. **Layer A (raw combined)** — every raw file (gems + surface, see above) parsed and concatenated by
   tag into `status.parquet` (`!:`), `rga.parquet` (`R:`), `scalup.parquet` (`P:`), `valve.parquet` (`V:`),
   plus `system_health.parquet` (`SH`, from the surface events logs — `voltage_v`, `current_a`,
   `teensy_temp_c`). No aggregation. Written first; every later stage reads from these, not from raw
   files again. `system_health` is intentionally *not* aggregated onto cycle windows; if per-cycle mean
   voltage is wanted it rides `aggregate.aggregate_onto_windows()` as one more source table.

   **`system_health.teensy_temp_c` is not a battery temperature** despite riding the firmware's
   `battery` line prefix: the observed 42–60 °C at ~0.03 A is far too hot for a pack at that current,
   and is the Teensy die temperature. Observed ranges over the 2026-09-18 surface corpus (~44k rows,
   one row per 10 s): voltage 22.8–27.0 V (the supply/battery rail), current 0.032–0.040 A, temp
   42–60 °C. The corpus also contains `SYSTEM battery voltage below threshold; sending OFF to lander`
   and `low voltage shutdown confirmed by lander` events, so the voltage trace is the diagnostic for
   explaining deployment data gaps.
2. **Layer B (`egcf_rga_scans`)** — one row per RGA mass-scan cycle (~10s pass through the
   configured mass list). Scan boundaries are detected from the data itself (a
   "masses seen in this scan" set that resets on a repeat), not from a fixed mass count/order.
3. **Layer C (`egcf_chamber_cycles`)** — one row per chamber measurement cycle (`V:` transition
   into `(chamber, Re)` to the next transition), averaged over `[cycle_start + settle_offset, next_transition)`.
   `experiment_number` increments when a `(C1, Re)` transition follows a `(C2, Fl)` transition
   since the last boundary; `elapsed_time` is time since that experiment's start.
4. **Layer D (`egcf_fluxes`)** — one row per `(experiment_number, chamber, variable)`: the benthic
   vertical flux over one incubation, from the OLS slope of Layer C's cycle averages. See
   "Flux calculation" below.

Layers B and C share one aggregation function, `aggregate.aggregate_onto_windows()` — they differ
only in which `windows` table (window_start, window_end, chamber, experiment_number, elapsed_time)
they're aggregated onto. Don't duplicate the averaging logic if you need a third grain; add another
window-boundary function instead and call the same aggregator. Layer D is different in kind — it
consumes Layer C's output rather than raw readings, so it doesn't use the windowed aggregator at all.

## Flux calculation (`flux.py`, Layer D)

`Flux = dC/dt * V / A` — the OLS slope of a cycle-averaged concentration against `elapsed_time`
across one experiment's incubation (Layer C's cycle averages are that incubation's samples),
scaled by the chamber's enclosed water volume `V` and the sediment footprint area `A` enclosed by
its base. Both are **required** inputs with no default (`pipeline.run()` positional params,
`--chamber-volume-l` / `--chamber-area-m2` with `required=True`) — unlike the RGA's nominal
Faraday-cup sensitivity there is no meaningful "nominal" chamber size to fall back on, so a wrong
flux from a silent default would be worse than an argparse error. They're the same for C1 and C2
per the project owner. The dashboard's sidebar inputs default to `0.0`, which suppresses the flux
table with a prompt rather than computing a divide-by-zero.

Reported in **µmol m⁻² h⁻¹** — an explicit project preference over the more common
mmol m⁻² d⁻¹ convention. Sign is never forced positive: rising concentration = efflux
(sediment → water) = positive; falling = uptake (e.g. O2 consumption / SOD) = negative.

Variables computed, each only when its source column(s) exist on the input table (a missing source
means the row is absent from the output, never an error):

- **`oxygen`** — from `oxygen_mgL`, converted mg/L → µmol/L at O2's 32 g/mol molar mass.
- **`h_ion`** — from `10^(-pH)` (mol/L → µmol/L). Note this is a *raw H⁺* flux, which is
  unconventional; most benthic studies report total alkalinity flux instead. It's what was asked
  for — don't silently "correct" it to TA without checking, and don't add a TA calculation
  without the DIC/pCO2 second carbonate-system parameter it would need.
- **`n2_denitrification`** — N2:Ar ratio method (Kana et al. 1994):
  `[N2] = (mass_28_avg / mass_40_avg / k) * ar_solubility_umol_kg(temp_degC, sal_PSU) * density(T,S)`.
  Deliberately uses the **raw `_avg` counts, not `_torr`** — the ratio cancels the RGA's
  approximate nominal Faraday-cup sensitivity, which is the point of the method (Ar is a
  conservative tracer). Cycles with `mass_40_avg == 0` are dropped from the fit rather than
  producing an infinite ratio.

  `k` is `n2_ar_sensitivity_ratio` — the RGA's **mass-28/mass-40 sensitivity ratio**, which is
  *not* 1. An RGA's transmission and ionization cross-section differ per mass, so the raw
  I28/I40 is not the molar N2/Ar ratio: in the real bench corpus it runs ~45 where
  air-equilibrated seawater should read ~37 (Hamme & Emerson's own Table 3 measured 36.6–38.9).
  Measure `k` with `flux.n2_ar_sensitivity_from_standard(raw_ratio, T, S)` against
  air-equilibrated water at known T/S and pass it via `--n2-ar-sensitivity-ratio`. The default
  1.0 means *uncalibrated* — sign and shape of the flux are right, magnitude is not, and
  `pipeline.run()` logs a warning when it's left there.

  `[Ar]` is evaluated **once per incubation, at its first cycle's T and S** — not per cycle.
  Per the project owner, a chamber stays sealed for a whole experiment and is flushed only
  between experiments, so the enclosed water is a closed volume and inert Ar genuinely holds one
  fixed concentration throughout; the alternating `Re`/`Fl` transitions within an experiment
  select which chamber the RGA draws from, they don't re-flush the incubation. Recomputing the
  Ar term per cycle would let chamber temperature drift (Ar solubility moves ≈ −2%/°C)
  masquerade as N2 production or consumption. This is not hypothetical: against the real bench
  corpus it moved most incubations 1–20% and **flipped the sign** of experiment 16, the only
  warming one (+0.2 °C/h) and the one with the smallest real N2 signal. Salinity can't change in
  a sealed chamber either, so anchoring S also keeps sonde noise out of the Ar term.
  `_with_n2_dissolved_umol_l` does the anchoring per `(experiment_number, chamber)`; don't
  "simplify" it back to a per-row expression.
- **`temp_degC`** — reported as a **rate in °C/h, not a flux**, and not scaled by V/A. There's no
  mass/energy-conservation quantity for temperature without water density and specific heat
  capacity, which is out of scope. It rides in the same table (distinguished by `output_unit`) as
  an incubation QA signal — is the chamber heating from internal electronics vs. tracking ambient
  tide.

**Both physical constants are verified against their primary sources**, and
`tests/test_flux.py` asserts each against published check values rather than a plausibility
range — regenerate nothing here without re-checking those:

- `_SOLUBILITY_COEFFS` — Hamme & Emerson (2004), Deep-Sea Research I 51:1517–1528, Table 4,
  for both Ar and N2. Their Equation 1 is `ln C = A0 + A1·Ts + A2·Ts² + A3·Ts³ + S·(B0 + B1·Ts +
  B2·Ts²)` with `Ts = ln((298.15 − t)/(273.15 + t))`, giving µmol/kg in equilibrium with moist
  air at 1 atm **total** pressure. The paper's own check values at 10 °C, S=35 (Ar 13.4622,
  N2 500.885 µmol/kg) are the test. Valid 0–30 °C and distilled water through seawater, so the
  estuarine salinity range is in scope — though the salinity dependence is a Setchenow relation
  fit at only two salinities (~0 and ~35), so mid-estuarine values are interpolated.
- `seawater_density_kg_per_l` — UNESCO/EOS-80 one-atmosphere equation, tested against UNESCO
  Technical Paper in Marine Science No. 44 p.22. This replaced a fixed 1.025 kg/L, which is
  ~1.4% off at S=15 and ~2.6% off in fresh water — a real error in an estuary, feeding straight
  into N2 flux magnitude.

Not corrected for: **barometric pressure**. The solubilities are referenced to exactly 1 atm;
real sea-level pressure varies a few percent and scales `[Ar]` — hence the N2 flux — nearly
linearly. Correcting it properly needs the barometric pressure at the time the water last
equilibrated with the atmosphere, which a benthic lander can't observe.

`linear_fit()` lives here, not in `dashboard.py` — it was promoted so pipeline and dashboard share
one implementation; `dashboard.py` re-exports it, so existing imports from there still work.
`ols_fit()` is the same fit plus `r2`/`n` for output QA, kept separate so the dashboard's plotting
call sites don't have to unpack a dict for a slope they already had.

### Not implemented: calibrated flux for other RGA masses

Flux for any other scanned mass (CO2 at 44, etc.) needs one of two things this repo doesn't have:
(a) a real calibration curve from air-equilibrated water standards run across the deployment's
temperature range, replacing the nominal Faraday-cup sensitivity with a measured A/Torr and a
species-specific Henry's-law solubility, or (b) a ratio-to-Ar treatment like N2's, which only
works for a species whose solubility behavior can be tied to Ar's. Don't add a mass-44 flux column
by dividing `mass_44_torr` by a guessed solubility — the nominal sensitivity makes the absolute
Torr values approximate (see the RGA conversion note above), so the result would look
quantitative while being off by whatever the real SP/ST calibration factor is.

### Planned: PAR from a co-deployed Odyssey logger

An Odyssey submersible PAR logger is being co-deployed with the lander. **No parser is written
yet, deliberately** — no real Odyssey export exists in `data/raw/` to check the format against,
and this repo's standing rule is to read actual sample files rather than trust a spec (see the
firmware-README warning at the top). Don't write one from an assumed format; wait for a file.

Why it matters: without light, every O2 flux pools photosynthesis and respiration into a mean
that means little for an eelgrass bed. With it, dark incubations give respiration (R), light ones
give net community production (NCP = GPP − R), and GPP = NCP + |R|. It also cross-checks the
other variables — pH rises in light and falls in dark, so H⁺ flux should anticorrelate with PAR.

Agreed scope for Layer D once the data lands:

- **PAR as a covariate** — mean and integrated PAR per chamber cycle on every flux row.
- **Light/dark O2 partition** — classify each incubation by mean PAR; report R, NCP, GPP.
- **P–I curve fit** — NCP vs PAR across all incubations (Jassby & Platt tanh), yielding Pmax, α,
  saturation irradiance Ik, and dark R as the intercept. This is the real payoff of an
  unattended lander doing many incubations at many irradiances.
- Daily integrated metabolism was considered and **not** included in the initial scope.

Structurally the Odyssey is unlike anything in the pipeline today: a **separately-clocked,
separately-recovered logger**, not a payload in the lander's line grammar. Expected shape is a
new discovery/reader path feeding a Layer A `par` table, which then rides the *existing*
`aggregate.aggregate_onto_windows()` as one more source table (the case the "schema present even
when empty" note below already anticipates) — not a parallel aggregation path.

Instrument gotchas to handle explicitly, in rough order of how much damage each does:

- **Clock offset is the top risk.** The Odyssey's RTC is set by PC at launch and drifts over a
  multi-week deployment, and its software commonly writes **local time** while the lander runs
  UTC. A silent 15-minute misalignment puts dawn/dusk incubations at the wrong irradiance and
  quietly bends the whole P–I curve. Needed: launch time, timezone, and a recovery clock-check
  if one was taken. Treat the offset as an explicit input, never inferred.
- **Calibration state is unknown** — Odysseys log raw counts with per-unit calibration factors
  and real unit-to-unit variability. Support both: read raw counts and accept a calibration
  factor that defaults to pass-through, so an already-calibrated file works unchanged.
- **Biofouling is a drift, not noise.** A fouling diffuser reads progressively low and
  systematically bends P–I parameters across the deployment. Testable by checking whether
  clear-sky noon maxima decline monotonically over the record.
- **Chamber shading** — the logger sees ambient PAR; the enclosed sediment sees that minus what
  the chamber walls and lid block. Correcting it needs the chamber's transmittance.
- **Unit collision**: PAR is µmol photons m⁻² **s**⁻¹ while fluxes are µmol m⁻² **h**⁻¹. Name
  the columns so the two can't be confused.

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

The Status tab also plots `system_health.parquet` (supply voltage, supply current, Teensy
temperature) below the turbo panels. Its two halves are guarded independently — either table
alone renders, and the "No status data" empty state appears only when *both* are missing/empty.

Both the Status and Measurements tabs carry a "Shade by active chamber" checkbox
(`_chamber_shading_control`, keys `status_chamber_shading` / `measurements_chamber_shading`),
rendered only when `valve.parquet` yields at least one span and off by default. The spans come
from `active_chamber_spans()`, which is `cycles.chamber_cycle_windows(valve, settle_offset_s=0.0)`
— i.e. exactly the measurement cycles, so shading lines up with the windows cycle averages are
taken over; `Fl` spans stay unshaded. Remember this marks *which chamber is being sampled*, not
which is incubating: both chambers stay sealed for the whole experiment. `_shade_chamber_spans()`
draws **one shape per span in `yref="paper"` coordinates**, not one per (span, subplot): all
subplots match the row-1 x axis, so a single band covers the whole stack, and a real deployment
has ~1000 spans (× 7 panels would be ~7000 shapes for Plotly to render). Chamber colors come from
the same `chamber_color_map` the Experiment Data tab uses, and a no-data marker trace per chamber
supplies the legend entry.

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

Below the rate plot, `Cycle averages` renders a **Benthic flux** section
(`_render_experiment_fluxes`), calling the same `flux.compute_fluxes()` the
pipeline uses against the live-built cycle-averaged table. It's deliberately
independent of the Variable selectbox -- the flux quantities are a fixed set
(see "Flux calculation"), not user-selected -- and the whole section is
replaced by a prompt when the sidebar's chamber volume/area are still at their
`0.0` defaults. It has three parts:

1. `_render_experiment_flux_chart` -- grouped bars of the selected experiment's
   flux, **one subplot per variable**. Not one grouped bar chart: the variables
   carry different units and magnitudes spanning four orders (oxygen ~1e4
   µmol m⁻² h⁻¹ beside h_ion ~1e0), so on a shared axis everything but oxygen
   flattens to nothing.
2. The exact-numbers `st.dataframe` underneath.
3. `_render_flux_over_time` -- flux against experiment start for the **whole
   deployment**, with an `st.multiselect` (`key="flux_variables"`, defaults to
   every variable) choosing which variables to show. One stacked, x-linked
   subplot per selected variable, via the shared `_render_linked_timeseries`
   with its `zero_line=True` option, which draws y=0 on each panel because a
   flux's sign is its meaning (efflux above, uptake below). Never log-scaled
   and never clamped non-negative, for the same reason.

Parts 1 and 2 are skipped -- with an explanatory info message -- when the
*selected* experiment has too few cycles to fit, but **part 3 still renders**.
This matters against real data: experiment 1 in the surface corpus has one
cycle per chamber, so the old table-only version showed "not enough cycles"
and nothing else on load, even though 154 flux rows existed further into the
deployment. `n_points` and `r2` ride in part 3's hover text rather than the
axes, since a 2-point fit always has r²=1.0 and the number is only meaningful
next to n.

`chamber_color_map()` mirrors `mass_color_map()`'s rationale for chambers, and
is shared by all three of the tab's chamber-colored plots so C1/C2 keep one
color throughout.

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
uv run main.py <raw_dir> --out-dir <out_dir> --chamber-volume-l <L> --chamber-area-m2 <m2> \
    [--settle-offset-s 60] [--format parquet|csv]
uv run streamlit run dashboard.py   # launch the dashboard
```

Test data lives in `data/raw/lander/egcf_lander_test_data_2026-08-24/` (gems SD-card format) and
`data/raw/surface/egcf_surface_test_data_2026-08-25/` (surface telemetry format), both gitignored,
local only. Both are real bench-test data with a compressed ~30s chamber-toggle cadence, not the
production ~15 min — the default `--settle-offset-s 60` will correctly drop most cycles as
too-short against them; use a smaller value (e.g. `5`) when validating against these datasets.

No linter/formatter is configured yet. Match the existing style (no comments unless something is
genuinely non-obvious, explicit polars schemas, functions over classes).
