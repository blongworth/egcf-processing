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
  discovery.py     # find gems_*.txt + surface_*_lander.log files, parse rotation ts, skip 0-byte; find Odyssey PAR CSVs
  reader.py        # read files in order (dispatch by filename), concatenate parsed records
  combine.py       # Layer A: build + write status/rga/scalup/valve tables
  par.py           # Layer A par table: Odyssey PAR logger reader + calibration (par_calibrations.csv)
  rga_scans.py     # Layer B window boundaries: RGA scan-cycle detection
  cycles.py        # Layer C window boundaries: chamber-cycle + experiment numbering
  aggregate.py     # shared windowed aggregation used by both Layer B and C
  flux.py          # Layer D: benthic vertical flux from Layer C's cycle averages (+ per-experiment PAR)
  metabolism.py    # Layer E: light/dark O2 metabolism (R, NCP, GPP) + per-chamber Jassby-Platt P-I fit
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
(two within C1, one within C2), one detailed `!:` status line, and all three real `P:`
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

## Pipeline model (five layers)

1. **Layer A (raw combined)** — every raw file (gems + surface, see above) parsed and concatenated by
   tag into `status.parquet` (`!:`), `rga.parquet` (`R:`), `scalup.parquet` (`P:`), `valve.parquet` (`V:`),
   plus `system_health.parquet` (`SH`, from the surface events logs — `voltage_v`, `current_a`,
   `teensy_temp_c`). No aggregation. Written first; every later stage reads from these, not from raw
   files again. `par.parquet` (Odyssey PAR logger, see "PAR" below) is also Layer A, and unlike
   `system_health` it *is* aggregated onto Layers B/C. `system_health` is intentionally *not* aggregated onto cycle windows; if per-cycle mean
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
5. **Layer E (`egcf_metabolism`, `egcf_pi_fit`)** — Layer D's O2 fluxes classified light/dark by
   experiment-mean PAR, with R/NCP/GPP, plus a per-chamber P–I curve fit. See "Layer E" under the
   PAR section.

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

Reported in **mmol m⁻² h⁻¹** — still per hour, not the literature's more common mmol m⁻² d⁻¹
convention; only the molar-prefix scale changed from an earlier µmol m⁻² h⁻¹ convention. Sign is
never forced positive: rising concentration = efflux (sediment → water) = positive; falling =
uptake (e.g. O2 consumption / SOD) = negative.

Variables computed, each only when its source column(s) exist on the input table (a missing source
means the row is absent from the output, never an error):

- **`oxygen`** — from `oxygen_mgL`, converted mg/L → mmol/L at O2's 32 g/mol molar mass.
- **`h_ion`** — from `10^(-pH)` (mol/L → mmol/L). Note this is a *raw H⁺* flux, which is
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

### PAR from a co-deployed Odyssey logger (`par.py`)

An Odyssey integrating light logger (serial 50472) is deployed with the lander. It's a
**separately clocked, separately recovered logger**, not a payload in the lander's line grammar.
So `par.py` is its own reader, parallel to `events.py`. It feeds a Layer A `par` table
(`par.parquet`: `ts`, `scan_no`, `par_raw`, `par_umol_m2_s`, `serial_number`, `sensor_number`,
`cal_date`, `interval_s`). That table then rides the *existing* `aggregate.aggregate_onto_windows()`
through the `par=` keyword, adding the mean `par_umol_m2_s` and `par_raw` per window to Layers B
and C. There's no parallel aggregation path. With no PAR file, `par.parquet` is still written
empty with the typed schema, and the B/C PAR columns are null.

Why it matters: without light, every O2 flux pools photosynthesis and respiration into a mean
that means little for an eelgrass bed. Dark incubations give respiration (R), light ones give net
community production (NCP), and GPP = NCP + |R|. pH rises in light and falls in dark, so H⁺ flux
should anticorrelate with PAR.

**File identification is by content, not name.** `discovery.find_par_files()` `rglob`s
`--par-dir` (default: `raw_dir`) for `*.csv`/`*.CSV` and keeps files where `par.is_odyssey_export()` is true: the first line (BOM
stripped) starts with `Site Name` and the header has a `Logger Serial Number` line. This keeps
legacy CSVs like `gems_pump_*.csv` out. 0-byte files are skipped. The real export
(`data/raw/PAR/ESL-EGCF_011_001.CSV`) has a UTF-8 BOM and CRLF line endings, `key ,value` header
lines, two column-header rows (`Scan No ,...` / `...,RAW VALUE ,CALIBRATED VALUE,`), and data rows
like `1,18/09/2026 , 16:11:14,2223,2223`. The fields carry stray spaces. The date is **dd/mm/yyyy**.
In this file RAW == CALIBRATED, so it's uncalibrated.

**Calibration lives in `src/egcf_processing/par_calibrations.csv`, not in code.** It ships in the
package, and `--par-calibrations` points at a different file. Columns: `sensor_number`,
`serial_number`, `cal_date`, `interval_s`, `slope`, `intercept`, `notes`. The formula is ported
from CRISPEE's `loadPAR.m`: `par_umol_m2_s = max(0, slope·par_raw + intercept)`. The current
coefficients are CRISPEE's (a test deployment against a calibrated PME PAR logger).
`select_calibration()` looks only at rows for the file's serial and picks the latest `cal_date` on
or before the file's first timestamp. A blank `cal_date` sorts as the oldest, so a dated
calibration added later wins. If no row matches, you get a WARNING and `par_umol_m2_s` stays null.
`par_raw` is always kept.

A consequence of the loadPAR.m formula: with a positive intercept, night-time raw 0 reads as
`intercept` (6.45 µmol m⁻² s⁻¹ for sensor 1), not 0. The clamp only removes negatives. This is
faithful to the port, not a bug, but it biases dark-period means slightly high.

**The interval check.** The Odyssey is an *integrating* sensor, so raw counts depend on the
logging interval. `interval_s` in the calibration CSV is the interval the calibration was fit at.
The file's interval is the median `ts` spacing (300 s for the current file), recorded in
`par.interval_s`. If the calibration's `interval_s` is set and differs, you get a WARNING and
`par_umol_m2_s` stays null. Counts are **never rescaled silently**. If it's blank (as all rows are
now, since CRISPEE's interval isn't known), the calibration is applied with a WARNING that the
interval wasn't checked.

**Clock.** The logger's RTC is independent of the lander's. `--par-time-offset-h` (default 0,
loadPAR.m's `timeShift`) is added to every logger timestamp. `--par-start`/`--par-end` (ISO
datetimes, applied after the offset, half-open) trim to the deployment (loadPAR.m's
`startDate`/`endDate`). Nothing is trimmed by default. **Finding for the current file:** the
logger clock is probably already UTC. Hourly-mean raw counts peak at 16:00, first light is
~10:20–10:30 and last light ~22:50. All of these match Woods Hole solar noon, sunrise, and sunset
in UTC for mid-September. On a local-time (EDT) clock they would fall 4 h earlier. The code
**never infers** the offset. Keep it an explicit input and recheck it for every deployment. A
silent offset puts dawn/dusk incubations at the wrong irradiance and bends any P–I curve.

**Unit naming**: PAR is µmol photons m⁻² **s**⁻¹ (`par_umol_m2_s`) while fluxes are mmol m⁻²
**h**⁻¹. Keep the `_s` suffix on any derived PAR column so the two can't be confused.

The dashboard's Measurements tab plots `par_umol_m2_s` on the shared, linked time axis, with the
chamber shading, so light lines up against O2 and pH. If every calibrated value is null, it plots
`par_raw` as "PAR (raw counts, uncalibrated)" with an `st.info`.

**Daily light QC (`par.daily_par`, `par.daily_max_trend`).** `par_daily.parquet` has one row per
**UTC** day: `n_readings`, `coverage`, `dli_mol_m2_d` (the daily light integral) and
`max_par_umol_m2_s`.
- UTC days work here because UTC midnight is ~20:00 EDT, after Woods Hole sunset, so each UTC day
  holds one whole photoperiod. Recheck this for a deployment far from this longitude.
- `dli_mol_m2_d` is Σ(reading × interval) and is not extrapolated over gaps. `coverage` flags
  the partial first and last days.
- `daily_max_trend` is an OLS fit of daily max against day number, over full days only
  (`coverage` ≥ 0.9). It's the **biofouling screen**: a fouling diffuser reads progressively low.
  Cloudy days lower the daily max too, so a decline is a prompt to inspect, not proof of fouling.
  `pipeline.run()` logs it at INFO.
- The Measurements tab computes the same two quantities from the time-filtered `par` table, so
  they follow the sidebar filter. They appear as two more linked panels: DLI bars, and daily-max
  markers with the trend line. Partial days are faded and left out of the trend.

**Chamber shading (`--chamber-par-transmittance`).** The logger sees ambient PAR, while the
enclosed sediment sees that times the fraction the chamber walls and lid pass.
- The default is 1.0, meaning **unmeasured**. `pipeline.run()` then WARNs that the dark threshold
  and P–I parameters are relative to ambient light.
- The factor must be in (0, 1]. It applies only in Layer E: `egcf_metabolism.par_chamber_umol_m2_s`
  = ambient mean × transmittance. That value drives the light/dark split and the P–I fit, and
  `egcf_pi_fit.chamber_par_transmittance` records it.
- Layer A–D PAR columns stay **ambient**, because they describe the sensor.

**Dashboard Metabolism tab** (`render_metabolism_tab`). This tab reads the pipeline's
`egcf_metabolism`, `egcf_pi_fit` and `egcf_fluxes` outputs; it doesn't recompute them. Chamber
geometry, dark threshold and transmittance are therefore fixed at processing time, and the
sidebar geometry doesn't apply here, unlike the Experiment Data tab's live flux.
- The chart has two subplots sharing the PAR x-axis, each with its own y-axis, so it's never a
  dual axis. The top shows O2 flux, with the fitted Jassby–Platt curve for each converged
  chamber. The bottom shows H⁺ flux, which should run opposite to O2. It's omitted if there are
  no `h_ion` rows.
- Points use `par_chamber_umol_m2_s` and `chamber_color_map` colours. Excluded fluxes are hollow
  markers, with the reason in the hover. Fluxes with no PAR aren't placed.
- Below the chart: the P–I parameter table, and an expander with the light incubations' NCP/GPP.
- These three tables have no `ts`/`timestamp` column, so the sidebar time filter leaves them alone.

**PAR on Layer D.** Every `egcf_fluxes` row carries `par_mean_umol_m2_s`,
`par_integrated_mol_m2` and `par_coverage` for its experiment. They come from
`flux.attach_experiment_par()`, called in `pipeline.run()` after `compute_fluxes()`. The span is
the **whole experiment**, not individual cycles: the chambers stay sealed for the entire
experiment, so each flux reflects that span's light history. Both chambers share the span.
`cycles.experiment_spans()` defines it as running from the experiment's first `Re` transition to
the last surviving cycle's `window_end`. The start doesn't depend on `settle_offset_s`; the end can
lose a too-short last cycle, which is under 60 s at the default settle offset. On the real
deployment the spans are 3 h.

- `par_integrated_mol_m2` is Σ(calibrated reading × `interval_s`) / 1e6 over the readings in the
  span. It is **not extrapolated** over gaps.
- `par_coverage` is logged seconds / span seconds, capped at 1. It tells you how much of the span
  the integral represents.
- An experiment with no calibrated readings gets nulls, never zeros.

`compute_fluxes()` itself is unchanged, so the dashboard's live flux table doesn't carry these
columns. Only the written `egcf_fluxes` does.

**Layer E: metabolism and P–I (`metabolism.py`).** This layer reads Layer D's `oxygen` rows and
writes two tables.

`egcf_metabolism` has one row per O2 flux:
- `period` is `dark` when `par_mean_umol_m2_s` is below `--dark-par-threshold`, and `light`
  otherwise. The default threshold is 20. Night reads the 6.45 calibration intercept, and
  dusk/dawn experiments land a little above it.
- Excluded rows are **kept**, with `used = false` and an `excluded_reason`: `no PAR`,
  `PAR coverage below minimum` (`--min-par-coverage`, default 0.9), or `r2 below minimum`.
- R is −mean(used dark O2 flux) per chamber, so it's positive for uptake. It is **not** forced
  positive.
- On used light rows, `ncp_mmol_m2_h` is the O2 flux and `gpp_mmol_m2_h` = NCP + R. This assumes
  light respiration equals dark respiration.

`egcf_pi_fit` has one row per chamber. It holds a Jassby & Platt (1976) fit,
`NCP = Pmax·tanh(α·I/Pmax) − R`, over the used light and dark points together, done with
`scipy.optimize.curve_fit`. Pmax and α are bounded positive and R is unbounded. Columns: Pmax, α,
fitted R, each with a standard error from the covariance; Ik = Pmax/α; `fit_r2`; the dark-mean R
for comparison; and point counts. A chamber with fewer than 4 used points, or no light points,
gets a WARNING and `converged = false` with null parameters.

Units: I (PAR) is per **second** and the fluxes are per **hour**. So α is
`alpha_mmol_m2_h_per_par`, meaning (mmol O2 m⁻² h⁻¹) per (µmol photons m⁻² s⁻¹), and Ik is
`ik_umol_m2_s`, in PAR units.

**No r2 filter by default** (`--metabolism-min-r2 0`). A flux near zero, such as one at the
compensation irradiance, fits a flat line whose r2 is low by construction. Filtering on r2 would
remove exactly the points that pin the curve's low end.

First real result, 2026-09-18 to 09-21, 18 points per chamber, 9 light and 9 dark (values below are
in mmol m⁻² h⁻¹; recorded pre-unit-change against µmol m⁻² h⁻¹ and rescaled by 1/1000 here):
- C1: Pmax 1.560 ± 0.265, α 0.0041 ± 0.0016, R 0.169 ± 0.122 against a dark-mean R of 0.183,
  Ik 380, r² 0.74.
- C2: Pmax 3.477 ± 0.523, α 0.0118 ± 0.0041, R 0.573 ± 0.265 against a dark-mean R of 0.531,
  Ik 296, r² 0.77.

That's with placeholder geometry (4 L / 0.06 m²), so the magnitudes scale with the real V/A. The
fitted R agrees with the measured dark-mean R within its standard error, which is the sanity
check to repeat on each new deployment.

#### Suggested next steps (not implemented)

1. **Fill in `par_calibrations.csv`**: `interval_s` and `cal_date` for the CRISPEE calibration, and
   the serials for sensors 2 and 3.
2. **Measure the chamber PAR transmittance** and pass it with `--chamber-par-transmittance`.

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
- **`P:` has three field-count eras**, all still in use in real data: 6 fields (no
  `pressure_mbar`), 7 fields (matches the README), and 8 fields (adds a trailing `fieldMask`
  unsigned-int bitmask of which sensor groups reported, first seen 2026-09-23). In the 8-field
  era, any sensor value can come through as `NA` instead of a number when its mask bit is clear
  -- already handled by `_parse_float`'s null-token check. `lines.py` branches on field count, not
  on date.
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
pipeline run itself, by design. Four tabs: Status (turbo speed/power/temp, water
pump RPM, plus total pressure only if `status.parquet` has any non-null
`raw_total_pressure_current` — currently always empty against real data, so
this is normally a "no data" message, not a bug), Measurements (RGA mass
data plus scalup sonde data, all with a raw/Amps/Torr unit toggle reusing
`aggregate.py`'s conversion constants), and Experiment Data (per-experiment
C1-vs-C2 comparison of one RGA mass or other variable against elapsed time,
in minutes), and Metabolism (O2/H⁺ flux vs PAR with the P–I fit; see the PAR section). The status tab's "current" plot is deliberately
`turbo_power_w` — there's no field literally named "current" in
`STATUS_SCHEMA` besides the pressure ion current, which already gets its own
plot; this was an explicit user choice, not a guess.

The sidebar **"Time range"** control prefilters the loaded tables before any plotting. Its bounds
come from `tables_time_bounds()` (min/max over every loaded table's time column, `timestamp` for
Layer B/C and `ts` for Layer A -- see `table_ts_col`), and the whole control is skipped for a
dataset spanning a single instant. `filter_tables_to_range()` then slices every timestamped table
once, up front, and the *filtered* dict is what the Status and Measurements tabs receive -- so unit
conversion, ratio joins and cycle-window detection all run over the visible slice rather than the
whole deployment (a one-day window over the ~1.2M-row real corpus cuts a rerun from ~2.3 s to
~0.35 s).

Two deliberate exclusions: `render_overview` is given the *unfiltered* tables, since it describes
the dataset rather than the view; and so is the Experiment Data tab, because it derives
`experiment_number` live from the complete `valve` sequence and a truncated sequence would silently
renumber experiments. The control is created *after* the data loads (its bounds depend on it) but
rendered into a `st.sidebar.container()` reserved earlier, so it still appears directly below the
data-source controls.

`render_time_range_control()` is a **preset selectbox plus a two-stage custom mode**, and both
halves exist to fix specific failures of the single full-span range slider it replaced:

- `TIME_RANGE_PRESETS` (All data / Last hour / 6 hours / 24 hours / 7 days) resolve via
  `preset_time_range()`, anchored at the **end of the data**, not at "now" -- these are recovered
  deployments, weeks old, so a wall-clock anchor would always select nothing.
- Custom mode picks calendar days first (`st.date_input`) and only then offers a 1-minute-step
  slider *within* those days. A single slider across a 39-day deployment is hours per pixel, so
  short windows were undraggable. `date_range_bounds()` tolerates the 1-tuple `st.date_input`
  returns mid-selection (before the second date is picked), treating it as a single day.
- `align_slider_bounds()` rounds the slider's upper bound **up** to a whole number of steps.
  `st.slider` only offers positions at `min_value + k * step`, so unless the span is an exact
  multiple of the step the true maximum is unreachable -- with the default 1-day step this made
  the final partial day of a deployment impossible to select. Overshooting the last sample is
  harmless because the filter is inclusive.
- The fine slider is deliberately **unkeyed**: changing the day selection changes its min/max, and
  resetting to the full newly-selected span is the wanted behavior.

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
   carry different units and magnitudes spanning four orders (oxygen ~1e1
   mmol m⁻² h⁻¹ beside h_ion ~1e-3), so on a shared axis everything but oxygen
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
