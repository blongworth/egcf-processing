"""Interactive turbo-speed RGA performance test, built on turbo_speed_analysis.py.

    uv run marimo edit scripts/turbo_speed_notebook.py   # or `marimo run` for app view
    uv run python scripts/turbo_speed_notebook.py --raw-dir <dir> [--out-dir <dir>]   # headless
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Turbo speed test")


@app.cell
def _():
    import sys
    from datetime import timedelta
    from pathlib import Path

    import marimo as mo
    import polars as pl

    sys.path.insert(0, str(mo.notebook_dir()))
    import turbo_speed_analysis as tsa

    from egcf_processing.combine import build_tables, write_df
    from egcf_processing.discovery import find_all_files
    from egcf_processing.reader import read_all

    is_script = mo.app_meta().mode == "script"
    cli = mo.cli_args()
    return Path, build_tables, cli, find_all_files, is_script, mo, pl, read_all, timedelta, tsa, write_df


@app.cell
def _(cli, mo):
    raw_dir = mo.ui.text(value=cli.get("raw-dir") or "data/raw/tests", label="raw dir", full_width=True)
    commands_log = mo.ui.text(value=cli.get("commands-log") or "", label="commands log (optional)", full_width=True)
    settle_min = mo.ui.number(start=0, value=float(cli.get("settle-min") or 10), step=0.5, label="settle (min)")
    ack_window_s = mo.ui.number(start=0, value=float(cli.get("ack-window-s") or 120), step=10, label="ack window (s)")
    load_btn = mo.ui.run_button(label="Load", kind="success")
    mo.vstack([mo.hstack([raw_dir, commands_log], widths=[1, 1]), mo.hstack([settle_min, ack_window_s, load_btn], justify="start")])
    return ack_window_s, commands_log, load_btn, raw_dir, settle_min


@app.cell
def _(
    Path,
    ack_window_s,
    build_tables,
    commands_log,
    find_all_files,
    is_script,
    load_btn,
    mo,
    pl,
    raw_dir,
    read_all,
    tsa,
):
    mo.stop(not (load_btn.value or is_script), mo.md("Set the inputs and click **Load** to parse the raw logs."))
    _raw = Path(raw_dir.value)
    _cmd_log = Path(commands_log.value) if commands_log.value.strip() else None
    tables = build_tables(read_all(find_all_files(_raw)))
    commands, acks, aon_acks, ts_rows = tsa.scan_raw(_raw, _cmd_log)
    all_changes = tsa.add_acq_on(tsa.build_speed_changes(commands, acks, ack_window_s.value), aon_acks)
    _reading_ts = pl.concat([tables["rga"].select("ts"), tsa.build_turbo(ts_rows, tables["status"]).select("ts")])
    data_min = min(_reading_ts["ts"].min(), *all_changes["ts"].to_list())
    data_max = max(_reading_ts["ts"].max(), *all_changes["ts"].to_list())
    _summary = mo.md(
        f"{len(commands)} SPD command sightings, {len(acks)} OK,SPD acks, {len(aon_acks)} OK,AON acks "
        f"→ **{all_changes.height} speed changes**; {tables['rga'].height} R: readings, {len(ts_rows)} TS readings, "
        f"{data_min} → {data_max}"
    )
    _out = _summary
    if all_changes.is_empty():
        _out = mo.vstack([_summary, mo.callout(mo.md("No speed changes found; nothing to analyse."), kind="warn")])
    _out
    return acks, all_changes, commands, data_max, data_min, tables, ts_rows


@app.cell
def _(ack_window_s, all_changes, data_max, mo, pl, tables, timedelta, ts_rows, tsa):
    mo.stop(all_changes.is_empty())
    _gap = timedelta(seconds=2 * ack_window_s.value + 3600)
    _reading_ts = (
        pl.concat([tables["rga"].select("ts"), tsa.build_turbo(ts_rows, tables["status"]).select("ts")])
        .get_column("ts")
        .sort()
    )
    _change_ts = all_changes["ts"].to_list()
    _starts = [0] + [i for i in range(1, len(_change_ts)) if _change_ts[i] - _change_ts[i - 1] > _gap]
    _bounds = [*_starts[1:], len(_change_ts)]
    run_options = {f"all data ({len(_change_ts)} changes)": (_change_ts[0], data_max)}
    for _n, (_i, _j) in enumerate(zip(_starts, _bounds), start=1):
        _run_start = _change_ts[_i]
        _next = _change_ts[_j] if _j < len(_change_ts) else None
        _before_next = _reading_ts.filter(_reading_ts < _next) if _next is not None else _reading_ts
        _run_end = max(_before_next.max(), _change_ts[_j - 1]) if _before_next.len() else _change_ts[_j - 1]
        run_options[f"run {_n}: {_run_start:%Y-%m-%d %H:%M} → {_run_end:%Y-%m-%d %H:%M} ({_j - _i} changes)"] = (
            _run_start,
            _run_end,
        )
    run_picker = mo.ui.dropdown(options=run_options, value=next(iter(run_options)), label="test run")
    return (run_picker,)


@app.cell
def _(data_max, data_min, mo, run_picker):
    _default_start, _default_end = run_picker.value
    _opts = {"start": data_min, "stop": data_max, "precision": "second", "full_width": True}
    start_picker = mo.ui.datetime(value=_default_start, label="start (UTC)", **_opts)
    end_picker = mo.ui.datetime(value=_default_end, label="end (UTC)", **_opts)
    mo.hstack([run_picker, start_picker, end_picker], widths=[2, 1, 1], gap=2)
    return end_picker, start_picker


@app.cell
def _(all_changes, end_picker, mo, settle_min, start_picker, tables, ts_rows, tsa):
    mo.stop(start_picker.value is None or end_picker.value is None)
    settle_s = settle_min.value * 60
    outputs = tsa.analyze(
        tables,
        all_changes,
        ts_rows,
        start_picker.value,
        end_picker.value,
        settle_s,
        tsa.DEFAULT_PARTIAL_PRESSURE_SENSITIVITY_A_PER_TORR,
        tsa.DEFAULT_TOTAL_PRESSURE_SENSITIVITY_A_PER_TORR,
    )
    mo.stop(
        outputs["speed_changes"].is_empty(),
        mo.callout(mo.md("No speed changes in the selected window."), kind="warn"),
    )
    figs = tsa.build_figs(outputs, settle_s)
    ratio_figs = tsa.build_ratio_figs(outputs)
    noise_fig = tsa.fig_noise_vs_signal(outputs["mass_stats_by_segment"])
    return figs, noise_fig, outputs, ratio_figs, settle_s


@app.cell
def _(figs, mo, noise_fig, outputs, pl, ratio_figs, tsa):
    def _fmt(df):
        mapping = {}
        for col, dtype in df.schema.items():
            if not dtype.is_float():
                continue
            if "torr" in col or "amps" in col:
                mapping[col] = "{:.3e}"
            elif "rsd" in col or "speed" in col or col == "statistic" or col.endswith("_lo_hi"):
                mapping[col] = "{:.2f}"
            elif col.startswith("p_"):
                mapping[col] = "{:.3g}"
            elif "ratio" in col:
                mapping[col] = "{:.4g}"
        return mapping

    def _table(df, label):
        return mo.ui.table(df, format_mapping=_fmt(df), selection=None, label=label, page_size=20)

    _rsd = (
        outputs["mass_stats_by_speed"]
        .pivot(index="setpoint", on="mass", values="rsd_pct")
        .rename(lambda c: c if c == "setpoint" else f"rsd_m{c}")
    )
    _ratio_tab = (
        mo.vstack(
            [
                *ratio_figs,
                _table(outputs["ratio_speed_tests"], f"effect of setpoint on drift-robust RSD of signal / m{tsa.REF_MASS}"),
                _table(outputs["ratio_stats_by_speed"], f"signal / m{tsa.REF_MASS} by setpoint"),
            ]
        )
        if outputs["ratio_stats_by_segment"].height
        else mo.callout(mo.md(f"No mass {tsa.REF_MASS} readings in the selected window to normalise to."), kind="warn")
    )
    mo.ui.tabs(
        {
            "Plots": mo.vstack(figs),
            f"Normalised to m{tsa.REF_MASS}": _ratio_tab,
            "Noise vs signal": mo.vstack(
                [
                    noise_fig,
                    mo.md(
                        "If the speed effect works only through signal level, every point lies on the one model "
                        "curve whatever its colour; points at one setpoint sitting consistently off the curve "
                        "indicate a direct speed effect. In the table, `*_lo_hi` is the value at the lowest "
                        "setpoint over the highest. A **noise-floor-limited** mass shows `signal_lo_hi` > 1, "
                        "`noise_lo_hi` ≈ 1 (absolute-noise `p_adj_noise` not significant) and "
                        "`rsd_lo_hi` ≈ 1 / `signal_lo_hi`. A mass limited by **noise proportional to signal** "
                        "shows `noise_lo_hi` ≈ `signal_lo_hi` and `rsd_lo_hi` ≈ 1."
                    ),
                    _table(outputs["noise_floor_summary"], "noise floor check, per mass"),
                ]
            ),
            "Tables": mo.vstack(
                [
                    _table(outputs["speed_changes"], "speed changes"),
                    _table(outputs["tp_by_speed"], "total pressure by setpoint"),
                    _table(outputs["tp_by_segment"], "total pressure by segment"),
                    _table(outputs["mass_stats_by_speed"], "mass stats by setpoint"),
                    _table(outputs["mass_stats_by_segment"], "mass stats by segment"),
                    _table(outputs["rsd_speed_tests"], "effect of setpoint on drift-robust RSD"),
                ]
            ),
            "RSD pivot": mo.ui.table(
                _rsd,
                format_mapping={c: "{:.2f}" for c in _rsd.columns if c != "setpoint"},
                selection=None,
                label="RSD (%) by setpoint × mass, settle excluded",
            ),
        }
    )
    return


@app.cell
def _(cli, mo):
    out_dir = mo.ui.text(value=cli.get("out-dir") or "data/processed/turbo_speed", label="out dir", full_width=True)
    out_format = mo.ui.dropdown(options=["parquet", "csv"], value="parquet", label="format")
    export_btn = mo.ui.run_button(label="Export")
    mo.hstack([out_dir, out_format, export_btn], widths=[4, 1, 1])
    return export_btn, out_dir, out_format


@app.cell
def _(Path, cli, export_btn, figs, is_script, mo, noise_fig, out_dir, out_format, outputs, ratio_figs, tsa, write_df):
    mo.stop(not (export_btn.value or (is_script and cli.get("out-dir"))))
    _dir = Path(out_dir.value)
    for _name, _df in outputs.items():
        write_df(_df, _dir, _name, out_format.value)
    _report = _dir / "turbo_speed_report.html"
    tsa.write_report([*figs, noise_fig, *ratio_figs], _report)
    mo.md(f"Wrote {len(outputs)} tables and `{_report}`")
    return


if __name__ == "__main__":
    app.run()
