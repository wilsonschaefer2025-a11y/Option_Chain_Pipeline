import numpy as np
import pandas as pd
import pytest

from src.pricing.smile import fit_volatility_smile, smile_iv


# ---------------------------------------------------------------------------
# fit_volatility_smile
# ---------------------------------------------------------------------------

def test_fit_volatility_smile_recovers_known_coefficients():
    # Construct IVs that exactly follow a known quadratic in log-moneyness,
    # so the fit should recover the same coefficients (up to float error)
    # rather than just checking the code against itself.
    S = 100
    true_coeffs = [0.5, -0.1, 0.25]  # c, b, a
    strikes = pd.Series([70, 85, 100, 115, 130, 150])
    log_moneyness = np.log(strikes / S)
    implied_vols = pd.Series(np.polyval(true_coeffs, log_moneyness))

    fitted = fit_volatility_smile(strikes, implied_vols, S)

    assert list(fitted) == pytest.approx(true_coeffs, abs=1e-8)


def test_fit_volatility_smile_drops_missing_points_before_fitting():
    S = 100
    true_coeffs = [0.5, -0.1, 0.25]
    strikes = pd.Series([70, 85, 100, 115, 130, 150, np.nan])
    log_moneyness = np.log(strikes / S)
    implied_vols = pd.Series(list(np.polyval(true_coeffs, log_moneyness[:-1])) + [np.nan])

    fitted = fit_volatility_smile(strikes, implied_vols, S)

    assert list(fitted) == pytest.approx(true_coeffs, abs=1e-8)


def test_fit_volatility_smile_raises_for_fewer_than_three_points():
    strikes = pd.Series([90, 100])
    implied_vols = pd.Series([0.25, 0.24])
    with pytest.raises(ValueError):
        fit_volatility_smile(strikes, implied_vols, S=100)


def test_fit_volatility_smile_works_with_nullable_float_dtype():
    # add_implied_volatility's implied_vol column is pandas' nullable
    # Float64, not plain numpy float64 -- make sure that actually works.
    S = 100
    strikes = pd.Series([90, 100, 110])
    implied_vols = pd.array([0.28, 0.25, 0.24], dtype="Float64")
    fitted = fit_volatility_smile(strikes, pd.Series(implied_vols), S)
    assert len(fitted) == 3


# ---------------------------------------------------------------------------
# smile_iv
# ---------------------------------------------------------------------------

def test_smile_iv_matches_fitted_points_at_their_own_strikes():
    S = 100
    true_coeffs = [0.5, -0.1, 0.25]
    strikes = pd.Series([70, 85, 100, 115, 130, 150])
    implied_vols = pd.Series(np.polyval(true_coeffs, np.log(strikes / S)))

    coeffs = fit_volatility_smile(strikes, implied_vols, S)

    for K, iv in zip(strikes, implied_vols):
        assert smile_iv(K, S, coeffs) == pytest.approx(iv, abs=1e-8)


def test_smile_iv_evaluates_at_an_unquoted_strike():
    # The whole point of fitting a curve -- evaluate it somewhere with
    # no clean quote of its own.
    S = 100
    true_coeffs = [0.5, -0.1, 0.25]
    strikes = pd.Series([70, 85, 100, 115, 130, 150])
    implied_vols = pd.Series(np.polyval(true_coeffs, np.log(strikes / S)))

    coeffs = fit_volatility_smile(strikes, implied_vols, S)

    unquoted_strike = 105
    expected = np.polyval(true_coeffs, np.log(unquoted_strike / S))
    assert smile_iv(unquoted_strike, S, coeffs) == pytest.approx(expected, abs=1e-8)


def test_smile_iv_accepts_an_array_of_strikes():
    S = 100
    true_coeffs = [0.5, -0.1, 0.25]
    strikes = pd.Series([70, 85, 100, 115, 130, 150])
    implied_vols = pd.Series(np.polyval(true_coeffs, np.log(strikes / S)))
    coeffs = fit_volatility_smile(strikes, implied_vols, S)

    result = smile_iv(np.array([90, 100, 110]), S, coeffs)

    assert len(result) == 3
