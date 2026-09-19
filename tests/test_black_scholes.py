import math

import pandas as pd
import pytest

from src.pricing.black_scholes import BS_call_price, BS_put_price, BS_price_series, delta, vega

#Test from claude, use at your own risk

# ---------------------------------------------------------------------------
# BS_call_price / BS_put_price
# ---------------------------------------------------------------------------

def _manual_price(S, K, T, r, sigma, option_type):
    # Independent reference calculation (same textbook formula, computed
    # separately here) so these tests aren't just checking the code
    # against itself.
    from scipy.stats import norm

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def test_call_price_matches_manual_calculation():
    result = BS_call_price(S=100, K=100, T=1, r=0.05, sigma=0.2)
    expected = _manual_price(100, 100, 1, 0.05, 0.2, "call")
    assert result == pytest.approx(expected)


def test_put_price_matches_manual_calculation():
    result = BS_put_price(S=100, K=100, T=1, r=0.05, sigma=0.2)
    expected = _manual_price(100, 100, 1, 0.05, 0.2, "put")
    assert result == pytest.approx(expected)


def test_put_call_parity_holds():
    # C - P == S - K*e^(-rT), independent of the specific pricing formula
    # -- catches a sign error that a single hardcoded reference value might not.
    S, K, T, r, sigma = 100, 95, 0.5, 0.03, 0.25
    c = BS_call_price(S, K, T, r, sigma)
    p = BS_put_price(S, K, T, r, sigma)
    assert c - p == pytest.approx(S - K * math.exp(-r * T))


def test_call_price_at_expiration_returns_intrinsic_value():
    assert BS_call_price(S=110, K=100, T=0, r=0.05, sigma=0.2) == pytest.approx(10.0)
    assert BS_call_price(S=90, K=100, T=0, r=0.05, sigma=0.2) == pytest.approx(0.0)


def test_put_price_at_expiration_returns_intrinsic_value():
    assert BS_put_price(S=90, K=100, T=0, r=0.05, sigma=0.2) == pytest.approx(10.0)
    assert BS_put_price(S=110, K=100, T=0, r=0.05, sigma=0.2) == pytest.approx(0.0)


def test_call_price_raises_for_non_positive_sigma():
    with pytest.raises(ValueError):
        BS_call_price(S=100, K=100, T=1, r=0.05, sigma=0.0)


def test_put_price_raises_for_non_positive_sigma():
    with pytest.raises(ValueError):
        BS_put_price(S=100, K=100, T=1, r=0.05, sigma=-0.1)


# ---------------------------------------------------------------------------
# vega
# ---------------------------------------------------------------------------

def test_vega_is_positive_for_valid_inputs():
    assert vega(S=100, K=100, T=1, r=0.05, sigma=0.2) > 0


def test_vega_matches_finite_difference_of_call_price():
    # dPrice/dSigma via central difference should match vega() closely --
    # verifies vega without hardcoding a separate reference value.
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    h = 1e-4
    numerical = (BS_call_price(S, K, T, r, sigma + h) - BS_call_price(S, K, T, r, sigma - h)) / (2 * h)
    assert vega(S, K, T, r, sigma) == pytest.approx(numerical, abs=1e-4)


def test_vega_returns_zero_at_expiration():
    assert vega(S=100, K=100, T=0, r=0.05, sigma=0.2) == 0.0


def test_vega_returns_zero_for_non_positive_sigma():
    assert vega(S=100, K=100, T=1, r=0.05, sigma=0.0) == 0.0


# ---------------------------------------------------------------------------
# delta
# ---------------------------------------------------------------------------

def test_call_delta_is_between_zero_and_one():
    assert 0 < delta(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call") < 1


def test_put_delta_is_between_negative_one_and_zero():
    assert -1 < delta(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put") < 0


def test_call_delta_minus_put_delta_equals_one():
    # Standard BS identity -- catches a sign error independent of any
    # single hardcoded reference value.
    call_delta = delta(S=100, K=95, T=0.5, r=0.03, sigma=0.25, option_type="call")
    put_delta = delta(S=100, K=95, T=0.5, r=0.03, sigma=0.25, option_type="put")
    assert call_delta - put_delta == pytest.approx(1.0)


def test_delta_matches_finite_difference_of_call_price():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    h = 1e-4
    numerical = (BS_call_price(S + h, K, T, r, sigma) - BS_call_price(S - h, K, T, r, sigma)) / (2 * h)
    assert delta(S, K, T, r, sigma, option_type="call") == pytest.approx(numerical, abs=1e-4)


def test_call_delta_at_expiration_returns_intrinsic_direction():
    assert delta(S=110, K=100, T=0, r=0.05, sigma=0.2, option_type="call") == 1.0
    assert delta(S=90, K=100, T=0, r=0.05, sigma=0.2, option_type="call") == 0.0


def test_put_delta_at_expiration_returns_intrinsic_direction():
    assert delta(S=90, K=100, T=0, r=0.05, sigma=0.2, option_type="put") == -1.0
    assert delta(S=110, K=100, T=0, r=0.05, sigma=0.2, option_type="put") == 0.0


def test_delta_raises_for_invalid_option_type():
    with pytest.raises(ValueError):
        delta(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="straddle")


def test_delta_raises_for_non_positive_sigma():
    with pytest.raises(ValueError):
        delta(S=100, K=100, T=1, r=0.05, sigma=0.0, option_type="call")


# ---------------------------------------------------------------------------
# BS_price_series
# ---------------------------------------------------------------------------

def test_BS_price_series_matches_BS_call_price_elementwise():
    S, T, r, sigma = 100, 1, 0.05, 0.2
    strikes = pd.Series([90, 100, 110])
    result = BS_price_series(strikes, S=S, r=r, T=T, sigma=sigma)
    expected = [BS_call_price(S, K, T, r, sigma) for K in strikes]
    assert list(result) == pytest.approx(expected)


def test_BS_price_series_works_for_puts():
    S, T, r, sigma = 100, 1, 0.05, 0.2
    strikes = pd.Series([90, 100, 110])
    result = BS_price_series(strikes, S=S, r=r, T=T, sigma=sigma, option_type="put")
    expected = [BS_put_price(S, K, T, r, sigma) for K in strikes]
    assert list(result) == pytest.approx(expected)


def test_BS_price_series_preserves_index():
    # A caller pricing a subset of a DataFrame (e.g. calls.loc[mask, "strike"])
    # needs the result indexed the same way to reassemble correctly.
    strikes = pd.Series([90, 110], index=[3, 7])
    result = BS_price_series(strikes, S=100, r=0.05, T=1, sigma=0.2)
    assert list(result.index) == [3, 7]


def test_BS_price_series_raises_for_invalid_option_type():
    with pytest.raises(ValueError):
        BS_price_series(pd.Series([100]), S=100, r=0.05, T=1, sigma=0.2, option_type="straddle")


def test_BS_price_series_raises_for_non_positive_sigma():
    with pytest.raises(ValueError):
        BS_price_series(pd.Series([100]), S=100, r=0.05, T=1, sigma=0.0)
