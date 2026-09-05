from datetime import datetime, timedelta

import polars as pl
import pytest

from egcf_processing.flux import (
    O2_UMOL_PER_MG,
    SEAWATER_DENSITY_KG_PER_L,
    ar_solubility_umol_kg,
    compute_fluxes,
    concentration_series,
    linear_fit,
    ols_fit,
)

VOLUME_L = 4.0
AREA_M2 = 0.06


def _cycles(**columns) -> pl.DataFrame:
    """Two C1 cycles 1 minute apart in experiment 1, plus whatever columns are given."""
    base = {
        "timestamp": [datetime(2026, 1, 1, 0, 0), datetime(2026, 1, 1, 0, 1)],
        "experiment_number": [1, 1],
        "elapsed_time": [timedelta(seconds=0), timedelta(seconds=60)],
        "chamber": ["C1", "C1"],
    }
    return pl.DataFrame({**base, **columns})


def test_linear_fit_exact_line():
    assert linear_fit([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0]) == (2.0, 1.0)


def test_linear_fit_none_with_fewer_than_two_points():
    assert linear_fit([1.0], [1.0]) is None
    assert linear_fit([], []) is None


def test_linear_fit_none_with_zero_x_variance():
    assert linear_fit([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) is None


def test_linear_fit_skips_null_pairs():
    assert linear_fit([0.0, 1.0, None, 2.0], [1.0, 3.0, 99.0, 5.0]) == (2.0, 1.0)


def test_ols_fit_returns_r2_and_n():
    fit = ols_fit([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0])
    assert fit["slope"] == pytest.approx(2.0)
    assert fit["intercept"] == pytest.approx(1.0)
    assert fit["r2"] == pytest.approx(1.0)
    assert fit["n"] == 4


def test_ols_fit_r2_is_one_for_a_flat_line():
    # Zero y-variance: nothing to explain, but the fit is exact -- 1.0, not a 0/0 NaN.
    fit = ols_fit([0.0, 1.0, 2.0], [5.0, 5.0, 5.0])
    assert fit["slope"] == pytest.approx(0.0)
    assert fit["r2"] == 1.0


def test_ols_fit_r2_below_one_for_scattered_points():
    fit = ols_fit([0.0, 1.0, 2.0, 3.0], [0.0, 3.0, 1.0, 4.0])
    assert 0.0 < fit["r2"] < 1.0


def test_ar_solubility_is_physically_plausible_and_decreases_with_temp_and_salinity():
    # Open-ocean Ar solubility sits around 13-17 umol/kg over normal T/S ranges.
    # TODO: verify the Hamme & Emerson (2004) Table 4 coefficients against the
    # paper directly and replace this range check with exact reference values.
    cold = ar_solubility_umol_kg(0.0, 35.0)
    warm = ar_solubility_umol_kg(25.0, 35.0)
    fresh = ar_solubility_umol_kg(10.0, 0.0)
    salty = ar_solubility_umol_kg(10.0, 35.0)
    assert 10.0 < warm < cold < 25.0
    assert salty < fresh


def test_concentration_series_fits_per_experiment_and_chamber():
    cycles = pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 1, 0, 0),
                datetime(2026, 1, 1, 0, 1),
                datetime(2026, 1, 1, 0, 0),
                datetime(2026, 1, 1, 0, 1),
            ],
            "experiment_number": [1, 1, 1, 1],
            "elapsed_time": [
                timedelta(seconds=0),
                timedelta(seconds=60),
                timedelta(seconds=0),
                timedelta(seconds=60),
            ],
            "chamber": ["C1", "C1", "C2", "C2"],
            "oxygen_mgL": [8.0, 7.0, 8.0, 9.0],
        }
    )
    series = concentration_series(cycles, "oxygen_mgL").sort("chamber")
    assert series["chamber"].to_list() == ["C1", "C2"]
    assert series["slope"].to_list() == pytest.approx([-1.0, 1.0])
    assert series["n"].to_list() == [2, 2]


def test_concentration_series_omits_groups_with_fewer_than_two_points():
    cycles = _cycles(oxygen_mgL=[8.0, None])
    assert concentration_series(cycles, "oxygen_mgL").is_empty()


def test_oxygen_flux_is_signed_and_scaled_by_volume_over_area():
    # 8.0 -> 7.0 mg/L over 1 min = -1 mg/L/min consumption (uptake, negative).
    fluxes = compute_fluxes(_cycles(oxygen_mgL=[8.0, 7.0]), VOLUME_L, AREA_M2)
    oxygen = fluxes.filter(pl.col("variable") == "oxygen")
    assert oxygen.height == 1
    assert oxygen["slope_native_per_min"][0] == pytest.approx(-1.0)
    assert oxygen["output_value"][0] == pytest.approx(-1.0 * O2_UMOL_PER_MG * VOLUME_L / AREA_M2 * 60)
    assert oxygen["output_unit"][0] == "umol m-2 h-1"


def test_h_ion_flux_from_ph():
    # pH 8.0 -> 7.0 means [H+] 1e-8 -> 1e-7 mol/L over 1 min.
    fluxes = compute_fluxes(_cycles(pH=[8.0, 7.0]), VOLUME_L, AREA_M2)
    h_ion = fluxes.filter(pl.col("variable") == "h_ion")
    expected_slope = 1e-7 - 1e-8
    assert h_ion["slope_native_per_min"][0] == pytest.approx(expected_slope)
    assert h_ion["output_value"][0] == pytest.approx(expected_slope * 1e6 * VOLUME_L / AREA_M2 * 60)


def test_temperature_is_a_rate_not_a_flux():
    fluxes = compute_fluxes(_cycles(temp_degC=[12.0, 12.5]), VOLUME_L, AREA_M2)
    temp = fluxes.filter(pl.col("variable") == "temp_degC")
    assert temp["output_unit"][0] == "degC h-1"
    # Not scaled by V/A -- 0.5 degC/min is 30 degC/h.
    assert temp["output_value"][0] == pytest.approx(30.0)


def test_n2_denitrification_flux_from_mass_28_to_40_ratio():
    cycles = _cycles(
        mass_28_avg=[100.0, 110.0],
        mass_40_avg=[1000.0, 1000.0],
        temp_degC=[10.0, 10.0],
        sal_PSU=[32.0, 32.0],
    )
    fluxes = compute_fluxes(cycles, VOLUME_L, AREA_M2)
    n2 = fluxes.filter(pl.col("variable") == "n2_denitrification")
    ar = ar_solubility_umol_kg(10.0, 32.0) * SEAWATER_DENSITY_KG_PER_L
    expected_slope = (0.11 - 0.10) * ar
    assert n2["slope_native_per_min"][0] == pytest.approx(expected_slope)
    assert n2["output_value"][0] == pytest.approx(expected_slope * VOLUME_L / AREA_M2 * 60)


def test_n2_omitted_when_argon_mass_is_absent():
    cycles = _cycles(mass_28_avg=[100.0, 110.0], temp_degC=[10.0, 10.0], sal_PSU=[32.0, 32.0])
    fluxes = compute_fluxes(cycles, VOLUME_L, AREA_M2)
    assert fluxes.filter(pl.col("variable") == "n2_denitrification").is_empty()
    assert not fluxes.filter(pl.col("variable") == "temp_degC").is_empty()


def test_zero_argon_cycles_are_dropped_from_the_n2_fit():
    cycles = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, 0, m) for m in range(3)],
            "experiment_number": [1, 1, 1],
            "elapsed_time": [timedelta(seconds=60 * m) for m in range(3)],
            "chamber": ["C1", "C1", "C1"],
            "mass_28_avg": [100.0, 999.0, 120.0],
            "mass_40_avg": [1000.0, 0.0, 1000.0],
            "temp_degC": [10.0, 10.0, 10.0],
            "sal_PSU": [32.0, 32.0, 32.0],
        }
    )
    n2 = compute_fluxes(cycles, VOLUME_L, AREA_M2).filter(pl.col("variable") == "n2_denitrification")
    ar = ar_solubility_umol_kg(10.0, 32.0) * SEAWATER_DENSITY_KG_PER_L
    # Ratio 0.10 at t=0 and 0.12 at t=2min, the zero-Argon cycle dropped entirely.
    assert n2["n_points"][0] == 2
    assert n2["slope_native_per_min"][0] == pytest.approx((0.12 - 0.10) * ar / 2)


def test_empty_cycles_returns_empty_table_with_full_schema():
    empty = pl.DataFrame(
        schema={
            "timestamp": pl.Datetime,
            "experiment_number": pl.Int64,
            "elapsed_time": pl.Duration,
            "chamber": pl.Utf8,
            "oxygen_mgL": pl.Float64,
        }
    )
    result = compute_fluxes(empty, VOLUME_L, AREA_M2)
    assert result.is_empty()
    assert "output_value" in result.columns
    assert "output_unit" in result.columns


def test_nonpositive_geometry_is_rejected():
    cycles = _cycles(oxygen_mgL=[8.0, 7.0])
    with pytest.raises(ValueError):
        compute_fluxes(cycles, 0.0, AREA_M2)
    with pytest.raises(ValueError):
        compute_fluxes(cycles, VOLUME_L, -1.0)
