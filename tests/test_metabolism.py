import logging
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from egcf_processing.metabolism import (
    METABOLISM_SCHEMA,
    PI_FIT_SCHEMA,
    classify_o2_fluxes,
    fit_pi_curves,
    jassby_platt,
)


def _fluxes(par, o2, chamber="C1", coverage=None, r2=None, extra_variable=True):
    n = len(par)
    rows = pl.DataFrame(
        {
            "experiment_number": pl.Series(list(range(1, n + 1)), dtype=pl.Int64),
            "chamber": pl.Series([chamber] * n, dtype=pl.Utf8),
            "experiment_start": pl.Series(
                [datetime(2026, 9, 19) + timedelta(hours=4 * i) for i in range(n)], dtype=pl.Datetime
            ),
            "variable": pl.Series(["oxygen"] * n, dtype=pl.Utf8),
            "r2": pl.Series(r2 if r2 is not None else [0.9] * n, dtype=pl.Float64),
            "output_value": pl.Series(o2, dtype=pl.Float64),
            "par_mean_umol_m2_s": pl.Series(par, dtype=pl.Float64),
            "par_integrated_mol_m2": pl.Series([None if p is None else p * 10800 / 1e6 for p in par], dtype=pl.Float64),
            "par_coverage": pl.Series(coverage if coverage is not None else [1.0] * n, dtype=pl.Float64),
        }
    )
    if extra_variable:
        rows = pl.concat([rows, rows.with_columns(pl.lit("h_ion").alias("variable"))])
    return rows


def test_classify_splits_light_dark_by_threshold_and_computes_r_ncp_gpp():
    m = classify_o2_fluxes(_fluxes([6.5, 12.0, 500.0, 1000.0], [-100.0, -300.0, 800.0, 1200.0]))
    assert m.columns == list(METABOLISM_SCHEMA)
    assert m["period"].to_list() == ["dark", "dark", "light", "light"]
    assert m["used"].all()
    assert m["ncp_umol_m2_h"].to_list() == [None, None, 800.0, 1200.0]
    # R = -mean(dark) = 200, so GPP = NCP + 200.
    assert m["gpp_umol_m2_h"].to_list() == [None, None, 1000.0, 1400.0]


def test_classify_threshold_is_configurable():
    m = classify_o2_fluxes(_fluxes([6.5, 30.0], [-100.0, 50.0]), dark_par_threshold_umol_m2_s=50.0)
    assert m["period"].to_list() == ["dark", "dark"]


def test_classify_keeps_excluded_rows_with_a_reason():
    m = classify_o2_fluxes(
        _fluxes([None, 500.0, 600.0, 700.0], [1.0, 2.0, 3.0, 4.0], coverage=[None, 0.5, 1.0, 1.0], r2=[0.9, 0.9, 0.1, 0.9]),
        min_r2=0.5,
    )
    assert m["excluded_reason"].to_list() == ["no PAR", "PAR coverage below minimum", "r2 below minimum", None]
    assert m["used"].to_list() == [False, False, False, True]
    assert m["period"].to_list() == [None, "light", "light", "light"]
    assert m["ncp_umol_m2_h"].to_list() == [None, None, None, 4.0]


def test_classify_default_does_not_filter_on_r2():
    m = classify_o2_fluxes(_fluxes([500.0], [2.0], r2=[0.01]))
    assert m["used"].to_list() == [True]


def test_classify_does_not_force_r_positive():
    m = classify_o2_fluxes(_fluxes([6.5, 500.0], [50.0, 800.0]))
    # A positive mean dark flux gives R = -50; GPP = NCP + R is reported as-is.
    assert m["gpp_umol_m2_h"].to_list() == [None, 750.0]


def test_classify_gpp_null_without_dark_rows():
    m = classify_o2_fluxes(_fluxes([500.0, 900.0], [800.0, 1200.0]))
    assert m["gpp_umol_m2_h"].to_list() == [None, None]


def test_classify_empty_input_keeps_schema():
    m = classify_o2_fluxes(_fluxes([], []))
    assert m.is_empty()
    assert m.columns == list(METABOLISM_SCHEMA)


def test_fit_recovers_known_jassby_platt_parameters():
    par = np.array([6.5, 6.5, 8.0, 10.0, 50.0, 100.0, 200.0, 350.0, 500.0, 800.0, 1100.0, 1400.0])
    o2 = jassby_platt(par, 2000.0, 8.0, 300.0)
    fit = fit_pi_curves(classify_o2_fluxes(_fluxes(par.tolist(), o2.tolist())))
    row = fit.row(0, named=True)
    assert row["converged"]
    assert row["n_light"] == 8 and row["n_dark"] == 4
    assert row["pmax_umol_m2_h"] == pytest.approx(2000.0, rel=1e-4)
    assert row["alpha_umol_m2_h_per_par"] == pytest.approx(8.0, rel=1e-4)
    assert row["r_fit_umol_m2_h"] == pytest.approx(300.0, rel=1e-4)
    assert row["ik_umol_m2_s"] == pytest.approx(250.0, rel=1e-4)
    assert row["fit_r2"] == pytest.approx(1.0)


def test_fit_is_per_chamber():
    par = [6.5, 10.0, 200.0, 500.0, 1000.0]
    c1 = _fluxes(par, jassby_platt(np.array(par), 1000.0, 5.0, 100.0).tolist(), chamber="C1")
    c2 = _fluxes(par, jassby_platt(np.array(par), 3000.0, 10.0, 500.0).tolist(), chamber="C2")
    fit = fit_pi_curves(classify_o2_fluxes(pl.concat([c1, c2])))
    assert fit["chamber"].to_list() == ["C1", "C2"]
    assert fit["pmax_umol_m2_h"].to_list() == pytest.approx([1000.0, 3000.0], rel=1e-3)


def test_fit_skipped_with_too_few_points_warns(caplog):
    with caplog.at_level(logging.WARNING):
        fit = fit_pi_curves(classify_o2_fluxes(_fluxes([6.5, 500.0, 900.0], [-100.0, 800.0, 1200.0])))
    row = fit.row(0, named=True)
    assert not row["converged"]
    assert row["pmax_umol_m2_h"] is None
    assert row["r_dark_umol_m2_h"] == 100.0
    assert "P-I fit for C1 skipped" in caplog.text


def test_fit_empty_input_keeps_schema():
    fit = fit_pi_curves(classify_o2_fluxes(_fluxes([None], [1.0])))
    assert fit.is_empty()
    assert fit.columns == list(PI_FIT_SCHEMA)
