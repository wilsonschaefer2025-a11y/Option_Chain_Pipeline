import pytest

from src.pricing.black_scholes import BS_call_price, BS_put_price
from src.pricing.implied_vol import implied_volatility, is_low_confidence


# ---------------------------------------------------------------------------
# implied_volatility
# ---------------------------------------------------------------------------

def test_implied_volatility_recovers_known_sigma_for_call():
    S, K, T, r, true_sigma = 100, 100, 1, 0.05, 0.25
    price = BS_call_price(S, K, T, r, true_sigma)
    result = implied_volatility(price, S, K, T, r, option_type="call")
    assert result == pytest.approx(true_sigma, abs=1e-4)


def test_implied_volatility_recovers_known_sigma_for_put():
    S, K, T, r, true_sigma = 100, 105, 0.5, 0.03, 0.35
    price = BS_put_price(S, K, T, r, true_sigma)
    result = implied_volatility(price, S, K, T, r, option_type="put")
    assert result == pytest.approx(true_sigma, abs=1e-4)


def test_implied_volatility_falls_back_to_bisection_when_initial_guess_has_no_vega():
    # vega -> 0 as sigma -> 0 (for an ATM strike, d1 blows up), so an
    # initial_guess this small trips the "vega too small" break on the
    # very first Newton iteration, forcing the bisection fallback. Should
    # still recover the true sigma accurately once it does.
    S, K, T, r, true_sigma = 100, 100, 1, 0.05, 0.25
    price = BS_call_price(S, K, T, r, true_sigma)
    result = implied_volatility(price, S, K, T, r, option_type="call", initial_guess=1e-6)
    assert result == pytest.approx(true_sigma, abs=1e-4)


def test_implied_volatility_raises_for_invalid_option_type():
    with pytest.raises(ValueError):
        implied_volatility(10.0, S=100, K=100, T=1, r=0.05, option_type="straddle")


def test_implied_volatility_raises_when_time_to_expiration_is_non_positive():
    with pytest.raises(ValueError):
        implied_volatility(10.0, S=100, K=100, T=0, r=0.05, option_type="call")


def test_implied_volatility_raises_for_price_above_call_upper_bound():
    # A call can never be worth more than the spot price itself.
    with pytest.raises(ValueError):
        implied_volatility(150.0, S=100, K=100, T=1, r=0.05, option_type="call")


def test_implied_volatility_raises_for_price_below_call_lower_bound():
    # Below intrinsic-discounted lower bound -- no sigma reproduces this.
    with pytest.raises(ValueError):
        implied_volatility(0.0, S=150, K=100, T=1, r=0.05, option_type="call")


def test_implied_volatility_raises_clear_error_when_true_sigma_exceeds_search_range():
    # Force Newton to fail immediately (near-zero-vega initial_guess) so
    # this falls through to the bisection fallback, with a true sigma
    # (6.0) above MAX_SIGMA (5.0) -- brentq's bracket won't contain a
    # sign change. Should raise a clear, project-specific ValueError
    # instead of scipy's raw "f(a) and f(b) must have different signs".
    S, K, T, r, true_sigma = 100, 100, 0.01, 0.05, 6.0
    price = BS_put_price(S, K, T, r, true_sigma)
    with pytest.raises(ValueError, match="failed to converge"):
        implied_volatility(price, S, K, T, r, option_type="put", initial_guess=1e-6)


# ---------------------------------------------------------------------------
# is_low_confidence
# ---------------------------------------------------------------------------

def test_is_low_confidence_true_for_deep_otm_call():
    # Deep OTM, short-dated -- delta is essentially 0, well below the
    # LOW_CONFIDENCE_DELTA threshold.
    assert is_low_confidence(S=100, K=150, T=0.1, r=0.05, sigma=0.2, option_type="call") == True


def test_is_low_confidence_false_for_near_atm_call():
    # Near-ATM, one year out -- delta is comfortably above the threshold.
    assert is_low_confidence(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call") == False


def test_is_low_confidence_true_for_deep_itm_call():
    # Deep ITM, short-dated -- delta is essentially 1, same near-zero
    # vega problem as deep OTM but on the other side. Verifies the
    # ITM-side fix: |delta| > 1 - LOW_CONFIDENCE_DELTA also flags.
    assert is_low_confidence(S=150, K=100, T=0.05, r=0.05, sigma=0.2, option_type="call") == True
