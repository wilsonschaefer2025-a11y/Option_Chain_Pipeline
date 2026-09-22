import math
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.data.fetch_options import (
    fetch_option_chain,
    fetch_risk_free_rate,
    time_to_expiration,
    trading_days_since,
    historical_volatility,
    IRX_DAYS_TO_MATURITY,
)


# ---------------------------------------------------------------------------
# fetch_option_chain
# ---------------------------------------------------------------------------
# yf.Ticker is mocked throughout -- these tests never touch the network.

def _mock_ticker(history=None, options=(), calls=None, puts=None):
    ticker = MagicMock()
    ticker.history.return_value = history if history is not None else pd.DataFrame({"Close": [100.0]})
    ticker.options = options
    ticker.option_chain.return_value = SimpleNamespace(
        calls=calls if calls is not None else pd.DataFrame({"strike": [100]}),
        puts=puts if puts is not None else pd.DataFrame({"strike": [100]}),
    )
    return ticker


def test_fetch_option_chain_raises_for_empty_price_history(monkeypatch):
    ticker = _mock_ticker(history=pd.DataFrame({"Close": []}))
    monkeypatch.setattr("src.data.fetch_options.yf.Ticker", lambda symbol: ticker)

    with pytest.raises(ValueError):
        fetch_option_chain("FAKE")


def test_fetch_option_chain_raises_when_no_expirations_listed(monkeypatch):
    ticker = _mock_ticker(options=())
    monkeypatch.setattr("src.data.fetch_options.yf.Ticker", lambda symbol: ticker)

    with pytest.raises(ValueError):
        fetch_option_chain("FAKE")


def test_fetch_option_chain_uses_nearest_expiration_when_not_specified(monkeypatch):
    ticker = _mock_ticker(options=("2026-10-16", "2026-10-23"))
    monkeypatch.setattr("src.data.fetch_options.yf.Ticker", lambda symbol: ticker)

    _, _, _, expiration = fetch_option_chain("FAKE")

    assert expiration == "2026-10-16"
    ticker.option_chain.assert_called_once_with("2026-10-16")


def test_fetch_option_chain_uses_given_expiration_without_checking_options_list(monkeypatch):
    # ticker.options is deliberately empty -- if the function checked it
    # despite an expiration being explicitly given, this would raise.
    ticker = _mock_ticker(options=())
    monkeypatch.setattr("src.data.fetch_options.yf.Ticker", lambda symbol: ticker)

    _, _, _, expiration = fetch_option_chain("FAKE", expiration="2026-12-18")

    assert expiration == "2026-12-18"
    ticker.option_chain.assert_called_once_with("2026-12-18")


def test_fetch_option_chain_returns_spot_calls_and_puts(monkeypatch):
    calls_df = pd.DataFrame({"strike": [90, 100]})
    puts_df = pd.DataFrame({"strike": [90, 100]})
    ticker = _mock_ticker(
        history=pd.DataFrame({"Close": [123.45]}),
        options=("2026-10-16",),
        calls=calls_df,
        puts=puts_df,
    )
    monkeypatch.setattr("src.data.fetch_options.yf.Ticker", lambda symbol: ticker)

    spot, calls, puts, expiration = fetch_option_chain("FAKE")

    assert spot == pytest.approx(123.45)
    assert calls is calls_df
    assert puts is puts_df
    assert expiration == "2026-10-16"


# ---------------------------------------------------------------------------
# fetch_risk_free_rate
# ---------------------------------------------------------------------------

def test_fetch_risk_free_rate_raises_for_unsupported_ticker():
    # No mocking needed -- this is validated before any yfinance call.
    with pytest.raises(ValueError):
        fetch_risk_free_rate("^FAKE")


def test_fetch_risk_free_rate_raises_for_empty_price_history(monkeypatch):
    ticker = _mock_ticker(history=pd.DataFrame({"Close": []}))
    monkeypatch.setattr("src.data.fetch_options.yf.Ticker", lambda symbol: ticker)

    with pytest.raises(ValueError):
        fetch_risk_free_rate("^IRX")


def test_fetch_risk_free_rate_converts_irx_discount_rate_correctly(monkeypatch):
    raw_quote = 5.25
    ticker = _mock_ticker(history=pd.DataFrame({"Close": [raw_quote]}))
    monkeypatch.setattr("src.data.fetch_options.yf.Ticker", lambda symbol: ticker)

    result = fetch_risk_free_rate("^IRX")

    # Independent re-derivation of the discount-rate formula, so this
    # isn't just checking the code against itself.
    annual_rate = raw_quote / 100
    price_fraction = 1 - annual_rate * (IRX_DAYS_TO_MATURITY / 360)
    expected = -math.log(price_fraction) / (IRX_DAYS_TO_MATURITY / 365)
    assert result == pytest.approx(expected)


def test_fetch_risk_free_rate_converts_bey_rate_correctly(monkeypatch):
    raw_quote = 4.5
    ticker = _mock_ticker(history=pd.DataFrame({"Close": [raw_quote]}))
    monkeypatch.setattr("src.data.fetch_options.yf.Ticker", lambda symbol: ticker)

    result = fetch_risk_free_rate("^TNX")

    annual_rate = raw_quote / 100
    expected = 2 * math.log(1 + annual_rate / 2)
    assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# time_to_expiration
# ---------------------------------------------------------------------------
# No mocking -- these use real dates chosen to be unambiguous regardless
# of when the suite actually runs (a date in 2020 is always in the past;
# a date a decade out is always in the future).

def test_time_to_expiration_returns_zero_for_a_past_date():
    assert time_to_expiration("2020-01-01", "NYSE") == 0.0


def test_time_to_expiration_is_positive_for_a_future_date():
    assert time_to_expiration("2035-06-15", "NYSE") > 0.0


def test_time_to_expiration_increases_with_a_later_expiration():
    nearer = time_to_expiration("2035-06-15", "NYSE")
    farther = time_to_expiration("2036-06-15", "NYSE")
    assert farther > nearer


# The intraday branches need a fixed clock to be reachable at all -- without
# one, which of them runs depends on what day and time the suite happens to be
# invoked, so they went untested. reference_time makes them deterministic, the
# same way it already does for trading_days_since.
#
# 2026-09-22 is a normal Tuesday session (09:30-16:00 ET), so 12:45 is its
# exact midpoint. Assertions are written as relationships between reference
# times rather than hardcoded fractions, so they don't encode how many
# sessions a particular year happens to have.

def _ny(stamp):
    return pd.Timestamp(stamp, tz="America/New_York")


def test_time_to_expiration_counts_today_in_full_before_the_open():
    # Nothing of the session has elapsed yet, so a same-day expiration still
    # has the whole day left -- identical to standing at the opening bell.
    before_open = time_to_expiration("2026-09-22", "NYSE", reference_time=_ny("2026-09-22 08:00"))
    at_open = time_to_expiration("2026-09-22", "NYSE", reference_time=_ny("2026-09-22 09:30"))
    assert before_open > 0
    assert before_open == pytest.approx(at_open)


def test_time_to_expiration_is_half_a_session_at_the_midpoint_of_the_day():
    before_open = time_to_expiration("2026-09-22", "NYSE", reference_time=_ny("2026-09-22 08:00"))
    midday = time_to_expiration("2026-09-22", "NYSE", reference_time=_ny("2026-09-22 12:45"))
    assert midday == pytest.approx(before_open / 2)


def test_time_to_expiration_is_zero_from_the_close_of_expiration_day():
    assert time_to_expiration("2026-09-22", "NYSE", reference_time=_ny("2026-09-22 16:00")) == 0.0
    assert time_to_expiration("2026-09-22", "NYSE", reference_time=_ny("2026-09-22 18:00")) == 0.0


def test_time_to_expiration_decreases_as_the_session_progresses():
    stamps = ["2026-09-22 09:30", "2026-09-22 11:00", "2026-09-22 12:45", "2026-09-22 15:00"]
    values = [time_to_expiration("2026-09-22", "NYSE", reference_time=_ny(s)) for s in stamps]
    assert values == sorted(values, reverse=True)


def test_time_to_expiration_honors_an_early_close():
    # 2026-11-27 (the day after Thanksgiving) closes at 13:00 ET, not 16:00.
    # Its midpoint is therefore 11:15, and the session is already over at
    # 13:00 -- a hardcoded 16:00 close would report ~46% of the day still
    # remaining there instead of nothing.
    before_open = time_to_expiration("2026-11-27", "NYSE", reference_time=_ny("2026-11-27 08:00"))
    midday = time_to_expiration("2026-11-27", "NYSE", reference_time=_ny("2026-11-27 11:15"))
    assert midday == pytest.approx(before_open / 2)
    assert time_to_expiration("2026-11-27", "NYSE", reference_time=_ny("2026-11-27 13:00")) == 0.0


def test_time_to_expiration_adds_no_fraction_of_today_on_a_weekend():
    # 2026-09-19/20 are a Saturday and Sunday, so neither contributes any
    # part of a session -- both should leave exactly Monday's session. A
    # Friday pre-open reference leaves Friday plus Monday, i.e. twice as much.
    saturday = time_to_expiration("2026-09-21", "NYSE", reference_time=_ny("2026-09-19 12:00"))
    sunday = time_to_expiration("2026-09-21", "NYSE", reference_time=_ny("2026-09-20 12:00"))
    friday_pre_open = time_to_expiration("2026-09-21", "NYSE", reference_time=_ny("2026-09-18 08:00"))
    assert saturday == pytest.approx(sunday)
    assert friday_pre_open == pytest.approx(2 * saturday)


def test_time_to_expiration_reads_a_naive_reference_time_as_new_york_time():
    # Matches trading_days_since's handling -- a naive stamp is the
    # exchange's own clock, not UTC, which would shift it by four hours and
    # land outside the session entirely.
    naive = time_to_expiration("2026-09-22", "NYSE", reference_time=datetime(2026, 9, 22, 12, 45))
    aware = time_to_expiration("2026-09-22", "NYSE", reference_time=_ny("2026-09-22 12:45"))
    assert naive == pytest.approx(aware)


def test_time_to_expiration_converts_an_aware_reference_time_to_new_york():
    # 16:45 UTC is 12:45 ET, so this must agree with the midpoint above.
    utc = time_to_expiration("2026-09-22", "NYSE", reference_time=pd.Timestamp("2026-09-22 16:45", tz="UTC"))
    aware = time_to_expiration("2026-09-22", "NYSE", reference_time=_ny("2026-09-22 12:45"))
    assert utc == pytest.approx(aware)


def test_time_to_expiration_returns_zero_for_a_past_date_against_a_fixed_clock():
    assert time_to_expiration("2026-09-18", "NYSE", reference_time=_ny("2026-09-22 12:00")) == 0.0


# ---------------------------------------------------------------------------
# trading_days_since
# ---------------------------------------------------------------------------
# reference_time makes these fully deterministic without any mocking.

def test_trading_days_since_returns_zero_for_same_day():
    reference = pd.Timestamp("2026-09-22 16:00:00", tz="America/New_York")  # Tuesday
    dates = pd.Series([pd.Timestamp("2026-09-22 10:00:00", tz="America/New_York")])
    result = trading_days_since(dates, "NYSE", reference_time=reference)
    assert result.iloc[0] == 0


def test_trading_days_since_returns_one_for_the_previous_trading_day():
    reference = pd.Timestamp("2026-09-22 16:00:00", tz="America/New_York")  # Tuesday
    dates = pd.Series([pd.Timestamp("2026-09-21 16:00:00", tz="America/New_York")])  # Monday
    result = trading_days_since(dates, "NYSE", reference_time=reference)
    assert result.iloc[0] == 1


def test_trading_days_since_does_not_count_a_weekend_as_a_session():
    friday_close = pd.Timestamp("2026-09-18 16:00:00", tz="America/New_York")
    monday_open = pd.Timestamp("2026-09-21 09:30:00", tz="America/New_York")
    dates = pd.Series([friday_close])
    result = trading_days_since(dates, "NYSE", reference_time=monday_open)
    assert result.iloc[0] == 1


def test_trading_days_since_returns_nan_for_a_missing_date():
    reference = pd.Timestamp("2026-09-22 16:00:00", tz="America/New_York")
    dates = pd.Series([pd.NaT])
    result = trading_days_since(dates, "NYSE", reference_time=reference)
    assert pd.isna(result.iloc[0])


# ---------------------------------------------------------------------------
# historical_volatility
# ---------------------------------------------------------------------------

def test_historical_volatility_raises_for_insufficient_history(monkeypatch):
    ticker = _mock_ticker(history=pd.DataFrame({"Close": [100.0]}))  # only 1 row
    monkeypatch.setattr("src.data.fetch_options.yf.Ticker", lambda symbol: ticker)

    with pytest.raises(ValueError):
        historical_volatility("FAKE")


def test_historical_volatility_matches_manual_zero_mean_rms_calculation(monkeypatch):
    # Prices constructed from known log returns, so the expected result
    # can be computed independently rather than re-deriving the same
    # formula the code uses.
    log_returns = np.array([0.02, -0.01, 0.015, -0.005])
    prices = [100.0]
    for lr in log_returns:
        prices.append(prices[-1] * math.exp(lr))

    ticker = _mock_ticker(history=pd.DataFrame({"Close": prices}))
    monkeypatch.setattr("src.data.fetch_options.yf.Ticker", lambda symbol: ticker)

    result = historical_volatility("FAKE")

    expected_daily = math.sqrt(np.mean(log_returns ** 2))
    expected = expected_daily * math.sqrt(252)
    assert result == pytest.approx(expected)


def test_historical_volatility_passes_period_through_to_yfinance(monkeypatch):
    ticker = _mock_ticker(history=pd.DataFrame({"Close": [100.0, 101.0, 99.0]}))
    monkeypatch.setattr("src.data.fetch_options.yf.Ticker", lambda symbol: ticker)

    historical_volatility("FAKE", period="3mo")

    ticker.history.assert_called_once_with(period="3mo")
