from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.data.cleaning_options import (
    add_mid_price,
    apply_spread_tolerance,
    apply_staleness_flag,
    filter_no_arbitrage,
    filter_strike_monotonicity,
    filter_strike_convexity,
    filter_liquidity,
    filter_contract_sanity,
)


# ---------------------------------------------------------------------------
# add_mid_price
# ---------------------------------------------------------------------------

def test_add_mid_price_uses_bid_ask_average_when_both_sides_quoted():
    calls = pd.DataFrame({"bid": [1.0], "ask": [1.2], "lastPrice": [1.05]})
    result = add_mid_price(calls)
    assert result["mid_price"].iloc[0] == pytest.approx(1.1)
    assert result["unpriceable"].iloc[0] == False
    assert result["has_live_quote"].iloc[0] == True


def test_add_mid_price_treats_zero_bid_with_live_ask_as_valid_quote():
    # bid=0, ask>0 is a legitimate one-sided quote, not missing data.
    calls = pd.DataFrame({"bid": [0.0], "ask": [1.2], "lastPrice": [0.9]})
    result = add_mid_price(calls)
    assert result["mid_price"].iloc[0] == pytest.approx(0.6)
    assert result["unpriceable"].iloc[0] == False
    assert result["has_live_quote"].iloc[0] == True


def test_add_mid_price_falls_back_to_last_price_when_no_live_quote():
    calls = pd.DataFrame({"bid": [0.0], "ask": [0.0], "lastPrice": [0.5]})
    result = add_mid_price(calls)
    assert result["mid_price"].iloc[0] == pytest.approx(0.5)
    assert result["unpriceable"].iloc[0] == False
    assert result["has_live_quote"].iloc[0] == False


def test_add_mid_price_flags_unpriceable_when_nothing_available():
    calls = pd.DataFrame({"bid": [0.0], "ask": [0.0], "lastPrice": [0.0]})
    result = add_mid_price(calls)
    assert result["mid_price"].iloc[0] == 0
    assert result["unpriceable"].iloc[0] == True
    assert result["has_live_quote"].iloc[0] == False


def test_add_mid_price_does_not_mutate_input():
    calls = pd.DataFrame({"bid": [1.0], "ask": [1.2], "lastPrice": [1.05]})
    add_mid_price(calls)
    assert "mid_price" not in calls.columns


# ---------------------------------------------------------------------------
# apply_spread_tolerance
# ---------------------------------------------------------------------------

def test_apply_spread_tolerance_uses_uncapped_half_spread_for_live_rows():
    # A very wide relative spread (100%) should NOT be capped -- tolerance
    # is simply half the real observed spread, however wide.
    calls = pd.DataFrame({
        "bid": [1.0], "ask": [3.0], "mid_price": [2.0], "has_live_quote": [True],
    })
    result = apply_spread_tolerance(calls)
    assert result["tolerance"].iloc[0] == pytest.approx(1.0)


def test_apply_spread_tolerance_gives_no_live_quote_rows_zero_tolerance_and_flags_them():
    # No real bid/ask spread to read for a lastPrice-fallback row -- gets
    # tolerance=0 (no synthetic cushion) but is flagged via
    # no_live_quote_flag so filter_liquidity can act on it. Also covers
    # the fully unpriceable case (mid_price == 0) without raising on the
    # relative_spread division.
    calls = pd.DataFrame({
        "bid": [0.0], "ask": [0.0], "mid_price": [0.0], "has_live_quote": [False],
    })
    result = apply_spread_tolerance(calls)
    assert result["tolerance"].iloc[0] == 0
    assert result["no_live_quote_flag"].iloc[0] == True
    assert result["wide_spread_flag"].iloc[0] == False


def test_apply_spread_tolerance_flags_top_percentile_as_wide():
    calls = pd.DataFrame({
        "bid":  [1.00, 1.00, 1.00],
        "ask":  [1.01, 1.06, 1.30],
        "mid_price": [1.005, 1.03, 1.15],
        "has_live_quote": [True, True, True],
    })
    # percentile=0.0 -> threshold is the minimum relative spread (row 0's
    # own value), so only rows strictly wider than the tightest quote in
    # the chain are flagged -- unambiguous, no interpolation to reason about.
    result = apply_spread_tolerance(calls, wide_spread_percentile=0.0)
    assert list(result["wide_spread_flag"]) == [False, True, True]


def test_apply_spread_tolerance_percentile_threshold_ignores_no_live_quote_rows():
    # A no-live-quote row's "relative spread" is a meaningless 0/mid (its
    # raw spread is always 0-0). If it leaked into the percentile
    # calculation it would drag the threshold down and cause live rows to
    # be spuriously flagged wide. It shouldn't.
    calls = pd.DataFrame({
        "bid":            [1.00,  1.00,  0.0],
        "ask":            [1.05,  1.20,  0.0],
        "mid_price":      [1.025, 1.10,  5.0],
        "has_live_quote": [True,  True,  False],
    })
    result = apply_spread_tolerance(calls, wide_spread_percentile=0.0)
    # If the fallback row's 0 relative spread had been included as the
    # new minimum, row 0 (relative spread ~0.049) would be wrongly flagged.
    assert result["wide_spread_flag"].iloc[0] == False


def test_apply_spread_tolerance_flags_crossed_market_and_gives_zero_tolerance():
    # bid > ask -- a bad/stale snapshot, not a real market. Left
    # unhandled, raw_spread would be negative and so would tolerance,
    # corrupting every downstream comparison that adds it as a cushion.
    calls = pd.DataFrame({
        "bid": [5.10], "ask": [5.00], "mid_price": [5.05], "has_live_quote": [True],
    })
    result = apply_spread_tolerance(calls)
    assert result["crossed_market_flag"].iloc[0] == True
    assert result["tolerance"].iloc[0] == 0
    assert result["wide_spread_flag"].iloc[0] == False


def test_apply_spread_tolerance_percentile_threshold_ignores_crossed_market_rows():
    # Same concern as the no-live-quote exclusion test above: a crossed
    # row's relative spread is negative, and if it leaked into the
    # percentile pool it would drag the threshold down and cause a
    # perfectly normal live row to be spuriously flagged wide.
    calls = pd.DataFrame({
        "bid":            [1.00,  1.00,  5.10],
        "ask":            [1.05,  1.20,  5.00],
        "mid_price":      [1.025, 1.10,  5.05],
        "has_live_quote": [True,  True,  True],
    })
    result = apply_spread_tolerance(calls, wide_spread_percentile=0.0)
    assert result["wide_spread_flag"].iloc[0] == False


def test_apply_spread_tolerance_does_not_flag_tick_bound_spread_as_wide():
    # bid=0.01/ask=0.02 is a ~67% relative spread, but it's exactly the
    # minimum tick width -- no market maker could tighten it further, so
    # it shouldn't be flagged regardless of how "wide" it looks in raw
    # relative terms.
    calls = pd.DataFrame({
        "bid": [0.01], "ask": [0.02], "mid_price": [0.015], "has_live_quote": [True],
    })
    result = apply_spread_tolerance(calls, wide_spread_percentile=0.0)
    assert result["wide_spread_flag"].iloc[0] == False


def test_apply_spread_tolerance_tick_bound_rows_do_not_mask_a_genuinely_wide_spread():
    # Three tick-bound cheap contracts (spread == min_tick, ~67% relative
    # spread purely from tick size) alongside one contract with a real,
    # non-tick-driven wide spread. Under plain relative-spread pooling,
    # the tick-bound noise would dominate the percentile and could mask
    # the genuinely bad spread; excess-spread pooling should not --
    # the tick-bound rows contribute ~0 excess and the real outlier
    # still gets caught.
    calls = pd.DataFrame({
        "bid":       [0.01,  0.01,  0.01,  5.00],
        "ask":       [0.02,  0.02,  0.02,  5.50],
        "mid_price": [0.015, 0.015, 0.015, 5.25],
        "has_live_quote": [True, True, True, True],
    })
    result = apply_spread_tolerance(calls, wide_spread_percentile=0.90)
    assert list(result["wide_spread_flag"]) == [False, False, False, True]


def test_apply_spread_tolerance_tick_size_does_not_affect_tolerance():
    # The tick-size policy should only change what counts as "wide" --
    # the actual no-arbitrage tolerance is still the full real spread
    # regardless of why it exists.
    calls = pd.DataFrame({
        "bid": [1.00], "ask": [1.06], "mid_price": [1.03], "has_live_quote": [True],
    })
    result_default = apply_spread_tolerance(calls, low_tick=0.01)
    result_wide_tick = apply_spread_tolerance(calls, low_tick=0.05)
    assert result_default["tolerance"].iloc[0] == result_wide_tick["tolerance"].iloc[0] == pytest.approx(0.03)


def test_apply_spread_tolerance_uses_nickel_tick_at_or_above_threshold():
    # Same $0.05 raw spread, but one contract is priced under $3 (penny
    # tick applies -- $0.04 of that spread is "excess") and one is priced
    # at/above $3 (nickel tick applies -- none of it is excess).
    calls = pd.DataFrame({
        "bid":       [2.00, 5.00],
        "ask":       [2.05, 5.05],
        "mid_price": [2.025, 5.025],
        "has_live_quote": [True, True],
    })
    result = apply_spread_tolerance(calls, wide_spread_percentile=0.0)
    # Below $3: excess = 0.05 - 0.01 = 0.04 (nonzero).
    # At/above $3: excess = 0.05 - 0.05 = 0 (tick-bound, never flagged).
    assert result["wide_spread_flag"].iloc[0] == True
    assert result["wide_spread_flag"].iloc[1] == False


# ---------------------------------------------------------------------------
# filter_no_arbitrage
# ---------------------------------------------------------------------------

def test_filter_no_arbitrage_flags_price_below_lower_bound():
    # Deep ITM call (strike well under spot) quoted far too cheap.
    calls = pd.DataFrame({
        "strike": [80.0],
        "bid": [5.0], "ask": [5.5], "lastPrice": [5.25],
    })
    result = filter_no_arbitrage(calls, S=100.0, r=0.05, T=0.5)
    assert result["no_arb_violation"].iloc[0] == True
    assert result["lower_bound"].iloc[0] > result["mid_price"].iloc[0]


def test_filter_no_arbitrage_allows_price_within_bounds():
    # OTM call priced sensibly -- should not be flagged.
    calls = pd.DataFrame({
        "strike": [110.0],
        "bid": [2.9], "ask": [3.1], "lastPrice": [3.0],
    })
    result = filter_no_arbitrage(calls, S=100.0, r=0.05, T=0.5)
    assert result["no_arb_violation"].iloc[0] == False
    assert result["lower_bound"].iloc[0] == pytest.approx(0.0)
    assert result["upper_bound"].iloc[0] == pytest.approx(100.0)


def test_filter_no_arbitrage_flags_price_above_spot():
    # A call can never be worth more than the underlying itself.
    calls = pd.DataFrame({
        "strike": [50.0],
        "bid": [150.0], "ask": [152.0], "lastPrice": [151.0],
    })
    result = filter_no_arbitrage(calls, S=100.0, r=0.05, T=0.5)
    assert result["no_arb_violation"].iloc[0] == True


def test_filter_no_arbitrage_does_not_mutate_input():
    calls = pd.DataFrame({
        "strike": [110.0],
        "bid": [2.9], "ask": [3.1], "lastPrice": [3.0],
    })
    filter_no_arbitrage(calls, S=100.0, r=0.05, T=0.5)
    assert "mid_price" not in calls.columns


# ---------------------------------------------------------------------------
# filter_strike_monotonicity
# ---------------------------------------------------------------------------

def test_filter_strike_monotonicity_flags_violation_beyond_tolerance():
    calls = pd.DataFrame({
        "strike": [90.0, 100.0],
        "bid": [10.00, 10.50], "ask": [10.02, 10.52], "lastPrice": [10.01, 10.51],
    })
    result = filter_strike_monotonicity(calls)
    # tolerance=0.01 each -> combined 0.02; 10.51 > 10.01+0.02 -- a real violation
    assert result["monotonicity_violation"].iloc[0] == True


def test_filter_strike_monotonicity_does_not_flag_violation_within_tolerance():
    calls = pd.DataFrame({
        "strike": [90.0, 100.0],
        "bid": [10.00, 10.06], "ask": [10.10, 10.16], "lastPrice": [10.05, 10.11],
    })
    result = filter_strike_monotonicity(calls)
    # tolerance=0.05 each -> combined 0.10; 10.11 > 10.05+0.10=10.15 is False
    assert result["monotonicity_violation"].iloc[0] == False


def test_filter_strike_monotonicity_allows_normal_decreasing_prices():
    calls = pd.DataFrame({
        "strike": [90.0, 100.0, 110.0],
        "bid": [10.00, 5.00, 2.00], "ask": [10.02, 5.02, 2.02],
        "lastPrice": [10.01, 5.01, 2.01],
    })
    result = filter_strike_monotonicity(calls)
    assert list(result["monotonicity_violation"]) == [False, False, False]


def test_filter_strike_monotonicity_sorts_unsorted_input_by_strike():
    calls = pd.DataFrame({
        "strike": [110.0, 90.0, 100.0],
        "bid": [2.00, 10.00, 5.00], "ask": [2.02, 10.02, 5.02],
        "lastPrice": [2.01, 10.01, 5.01],
    })
    result = filter_strike_monotonicity(calls)
    assert list(result["strike"]) == [90.0, 100.0, 110.0]
    assert list(result["monotonicity_violation"]) == [False, False, False]


def test_filter_strike_monotonicity_reuses_existing_mid_price_and_tolerance():
    # bid/ask/lastPrice deliberately absent -- if this tried to recompute
    # mid_price/tolerance instead of reusing what's already there, it
    # would KeyError.
    calls = pd.DataFrame({
        "strike": [90.0, 100.0],
        "mid_price": [10.01, 10.51],
        "tolerance": [0.01, 0.01],
    })
    result = filter_strike_monotonicity(calls)
    assert result["monotonicity_violation"].iloc[0] == True


def test_filter_strike_monotonicity_does_not_mutate_input():
    calls = pd.DataFrame({
        "strike": [90.0, 100.0],
        "bid": [10.00, 10.50], "ask": [10.02, 10.52], "lastPrice": [10.01, 10.51],
    })
    filter_strike_monotonicity(calls)
    assert "mid_price" not in calls.columns


# ---------------------------------------------------------------------------
# filter_strike_convexity
# ---------------------------------------------------------------------------

def test_filter_strike_convexity_flags_violation_beyond_tolerance():
    # Evenly spaced strikes: chord at K2=100 is (10.00+2.00)/2=6.00.
    # mid_price=8.00 at K2 is well above that -- a real convexity violation.
    calls = pd.DataFrame({
        "strike": [90.0, 100.0, 110.0],
        "bid": [9.99, 7.99, 1.99], "ask": [10.01, 8.01, 2.01],
        "lastPrice": [10.00, 8.00, 2.00],
    })
    result = filter_strike_convexity(calls)
    assert result["convexity_violation"].iloc[1] == True


def test_filter_strike_convexity_allows_normal_convex_prices():
    # Typical decreasing, convex call price curve -- chord at K2=100 is
    # (10.00+2.00)/2=6.00, and mid_price=5.00 sits comfortably below it.
    calls = pd.DataFrame({
        "strike": [90.0, 100.0, 110.0],
        "bid": [9.99, 4.99, 1.99], "ask": [10.01, 5.01, 2.01],
        "lastPrice": [10.00, 5.00, 2.00],
    })
    result = filter_strike_convexity(calls)
    assert list(result["convexity_violation"]) == [False, False, False]


def test_filter_strike_convexity_does_not_flag_violation_within_tolerance():
    # chord=6.00, combined_tolerance=0.15; mid_price=6.05 is 0.05 above
    # the chord -- a real but noise-sized gap, not a genuine violation.
    calls = pd.DataFrame({
        "strike": [90.0, 100.0, 110.0],
        "bid": [9.95, 6.00, 1.95], "ask": [10.05, 6.10, 2.05],
        "lastPrice": [10.00, 6.05, 2.00],
    })
    result = filter_strike_convexity(calls)
    assert result["convexity_violation"].iloc[1] == False


def test_filter_strike_convexity_uses_distance_weighted_chord_for_uneven_spacing():
    # Strikes are NOT evenly spaced (90, 95, 110) -- the correct chord at
    # K2=95 weights the nearer strike (90) more heavily:
    #   w = (110-95)/(110-90) = 0.75 -> chord = 0.75*10.00 + 0.25*2.00 = 8.00
    # mid_price=7.90 sits just under that true chord, so this should NOT
    # be flagged. A naive equal-weight average would wrongly compute
    # chord=(10.00+2.00)/2=6.00 and flag 7.90 as a violation -- this test
    # fails under that naive (wrong) implementation.
    calls = pd.DataFrame({
        "strike": [90.0, 95.0, 110.0],
        "bid": [9.99, 7.89, 1.99], "ask": [10.01, 7.91, 2.01],
        "lastPrice": [10.00, 7.90, 2.00],
    })
    result = filter_strike_convexity(calls)
    assert result["convexity_violation"].iloc[1] == False


def test_filter_strike_convexity_sorts_unsorted_input_by_strike():
    calls = pd.DataFrame({
        "strike": [110.0, 90.0, 100.0],
        "bid": [1.99, 9.99, 4.99], "ask": [2.01, 10.01, 5.01],
        "lastPrice": [2.00, 10.00, 5.00],
    })
    result = filter_strike_convexity(calls)
    assert list(result["strike"]) == [90.0, 100.0, 110.0]
    assert list(result["convexity_violation"]) == [False, False, False]


def test_filter_strike_convexity_reuses_existing_mid_price_and_tolerance():
    # bid/ask/lastPrice deliberately absent -- if this tried to recompute
    # mid_price/tolerance instead of reusing what's already there, it
    # would KeyError.
    calls = pd.DataFrame({
        "strike": [90.0, 100.0, 110.0],
        "mid_price": [10.00, 8.00, 2.00],
        "tolerance": [0.01, 0.01, 0.01],
    })
    result = filter_strike_convexity(calls)
    assert result["convexity_violation"].iloc[1] == True


def test_filter_strike_convexity_does_not_mutate_input():
    calls = pd.DataFrame({
        "strike": [90.0, 100.0, 110.0],
        "bid": [9.99, 4.99, 1.99], "ask": [10.01, 5.01, 2.01],
        "lastPrice": [10.00, 5.00, 2.00],
    })
    filter_strike_convexity(calls)
    assert "mid_price" not in calls.columns


# ---------------------------------------------------------------------------
# filter_liquidity
# ---------------------------------------------------------------------------

def test_filter_liquidity_flags_lowest_open_interest_and_volume():
    calls = pd.DataFrame({
        "openInterest": [2, 5, 10],
        "volume": [100, 50, 20],
        # Pre-supplied so this test stays isolated to OI/volume behavior.
        "wide_spread_flag": [False, False, False],
        "no_live_quote_flag": [False, False, False],
        "crossed_market_flag": [False, False, False],
        "stale_quote_flag": [False, False, False],
    })
    # percentile=0.0 -> threshold is just the minimum of each column,
    # so the expected flags are unambiguous (no interpolation).
    result = filter_liquidity(calls, percentile=0.0)
    assert list(result["illiquid_flag"]) == [True, False, True]


def test_filter_liquidity_does_not_flag_contract_above_both_thresholds():
    calls = pd.DataFrame({
        "openInterest": [1, 1000],
        "volume": [1, 500],
        "wide_spread_flag": [False, False],
        "no_live_quote_flag": [False, False],
        "crossed_market_flag": [False, False],
        "stale_quote_flag": [False, False],
    })
    result = filter_liquidity(calls, percentile=0.10)
    assert result["illiquid_flag"].iloc[1] == False


def test_filter_liquidity_does_not_mutate_input():
    calls = pd.DataFrame({
        "openInterest": [2, 5, 10], "volume": [100, 50, 20],
        "wide_spread_flag": [False, False, False],
        "no_live_quote_flag": [False, False, False],
        "crossed_market_flag": [False, False, False],
        "stale_quote_flag": [False, False, False],
    })
    filter_liquidity(calls)
    assert "illiquid_flag" not in calls.columns


def test_filter_liquidity_treats_missing_open_interest_and_volume_as_illiquid():
    # Missing data (NaN) is at least as strong an illiquidity signal as a
    # low reported value, so it should never silently pass the threshold.
    # A third, clearly-liquid row keeps the quantile threshold from
    # collapsing onto the single non-NaN value (which would trivially
    # flag everything via "<= threshold").
    calls = pd.DataFrame({
        "openInterest": [np.nan, 5, 1000],
        "volume": [np.nan, 5, 500],
        "wide_spread_flag": [False, False, False],
        "no_live_quote_flag": [False, False, False],
        "crossed_market_flag": [False, False, False],
        "stale_quote_flag": [False, False, False],
    })
    result = filter_liquidity(calls, percentile=0.10)
    assert result["illiquid_flag"].iloc[0] == True
    assert result["illiquid_flag"].iloc[2] == False


def test_filter_liquidity_reuses_existing_spread_flags_without_recomputing():
    # bid/ask/lastPrice/lastTradeDate are deliberately absent -- if
    # filter_liquidity tried to recompute wide_spread_flag/no_live_quote_flag/
    # crossed_market_flag/stale_quote_flag instead of reusing what's
    # already there, this would KeyError.
    calls = pd.DataFrame({
        "openInterest": [5, 1000, 2000],
        "volume": [5, 1000, 2000],
        "wide_spread_flag": [False, True, False],
        "no_live_quote_flag": [False, False, False],
        "crossed_market_flag": [False, False, False],
        "stale_quote_flag": [False, False, False],
    })
    result = filter_liquidity(calls, percentile=0.10)
    assert result["illiquid_flag"].iloc[1] == True   # flagged via spread alone
    assert result["illiquid_flag"].iloc[2] == False  # clean on every signal


def test_filter_liquidity_computes_spread_and_staleness_flags_when_not_already_present():
    # No wide_spread_flag/no_live_quote_flag/stale_quote_flag columns
    # supplied -- filter_liquidity should compute them itself so it also
    # works standalone on a raw chain.
    now = pd.Timestamp("2026-01-15 20:00:00", tz="UTC")
    calls = pd.DataFrame({
        "bid": [0.0, 1.00], "ask": [0.0, 1.01], "lastPrice": [0.0, 1.005],
        "lastTradeDate": [now - pd.Timedelta(days=10), now],
        # High OI/volume on the no-quote row, low on the well-quoted row,
        # so illiquid_flag on row 0 can only be explained by the missing
        # live quote, not by openInterest/volume.
        "openInterest": [5000, 10],
        "volume": [5000, 10],
    })
    result = filter_liquidity(calls, reference_time=now)
    assert result["no_live_quote_flag"].iloc[0] == True
    assert result["stale_quote_flag"].iloc[0] == True
    assert result["illiquid_flag"].iloc[0] == True


def test_filter_liquidity_sets_low_oi_flag_and_low_volume_flag_independently():
    # low_oi_flag/low_volume_flag are the persisted, per-signal breakout of
    # illiquid_flag -- row 0 is thin on OI only, row 1 thin on volume only,
    # so each flag should be independently traceable to its own signal.
    calls = pd.DataFrame({
        "openInterest": [1, 1000, 1000],
        "volume": [1000, 1, 1000],
        "wide_spread_flag": [False, False, False],
        "no_live_quote_flag": [False, False, False],
        "crossed_market_flag": [False, False, False],
        "stale_quote_flag": [False, False, False],
    })
    result = filter_liquidity(calls, percentile=0.10)
    assert list(result["low_oi_flag"]) == [True, False, False]
    assert list(result["low_volume_flag"]) == [False, True, False]
    assert list(result["illiquid_flag"]) == [True, True, False]


def test_filter_liquidity_low_oi_flag_and_low_volume_flag_treat_missing_as_flagged():
    # Same missing-data-is-illiquid rule as illiquid_flag itself, but
    # checked on the individual flags so callers relying on just one of
    # them still get the conservative NaN-handling behavior.
    calls = pd.DataFrame({
        "openInterest": [np.nan, 1000],
        "volume": [np.nan, 1000],
        "wide_spread_flag": [False, False],
        "no_live_quote_flag": [False, False],
        "crossed_market_flag": [False, False],
        "stale_quote_flag": [False, False],
    })
    result = filter_liquidity(calls, percentile=0.10)
    assert result["low_oi_flag"].iloc[0] == True
    assert result["low_volume_flag"].iloc[0] == True


# ---------------------------------------------------------------------------
# apply_staleness_flag
# ---------------------------------------------------------------------------

def test_apply_staleness_flag_flags_old_last_trade_date():
    # Friday close to the following Tuesday: Monday's session was
    # genuinely skipped without a trade -- 2 sessions elapsed, stale.
    friday_close = pd.Timestamp("2026-09-18 16:00:00", tz="America/New_York")
    tuesday = pd.Timestamp("2026-09-22 09:30:00", tz="America/New_York")
    calls = pd.DataFrame({"lastTradeDate": [friday_close]})
    result = apply_staleness_flag(calls, max_age_trading_days=1.0, reference_time=tuesday)
    assert result["stale_quote_flag"].iloc[0] == True


def test_apply_staleness_flag_does_not_flag_recent_last_trade_date():
    now = pd.Timestamp("2026-01-15 20:00:00", tz="UTC")
    calls = pd.DataFrame({"lastTradeDate": [now - pd.Timedelta(hours=2)]})
    result = apply_staleness_flag(calls, max_age_trading_days=1.0, reference_time=now)
    assert result["stale_quote_flag"].iloc[0] == False


def test_apply_staleness_flag_does_not_flag_across_a_normal_weekend():
    # The exact bug this fix addresses: a contract that traded right at
    # Friday's close and hasn't traded again by Monday's open is not
    # stale -- only 1 trading session (Monday) has occurred since, and
    # the market being closed over the weekend isn't a data problem.
    friday_close = pd.Timestamp("2026-09-18 16:00:00", tz="America/New_York")
    monday_open = pd.Timestamp("2026-09-21 09:30:00", tz="America/New_York")
    calls = pd.DataFrame({"lastTradeDate": [friday_close]})
    result = apply_staleness_flag(calls, max_age_trading_days=1.0, reference_time=monday_open)
    assert result["stale_quote_flag"].iloc[0] == False


def test_apply_staleness_flag_treats_missing_last_trade_date_as_stale():
    now = pd.Timestamp("2026-01-15 20:00:00", tz="UTC")
    calls = pd.DataFrame({"lastTradeDate": [pd.NaT]})
    result = apply_staleness_flag(calls, reference_time=now)
    assert result["stale_quote_flag"].iloc[0] == True


def test_apply_staleness_flag_handles_naive_and_tz_aware_timestamps_consistently():
    # lastTradeDate coming back tz-naive (no tz info) shouldn't error or
    # silently shift the age calculation -- both should agree once
    # reference_time is given as a plain (naive) value too.
    reference_naive = datetime(2026, 9, 21, 20, 0, 0)
    calls = pd.DataFrame({"lastTradeDate": [pd.Timestamp("2026-09-21 19:00:00")]})
    result = apply_staleness_flag(calls, max_age_trading_days=1.0, reference_time=reference_naive)
    assert result["stale_quote_flag"].iloc[0] == False


# ---------------------------------------------------------------------------
# filter_contract_sanity
# ---------------------------------------------------------------------------

def test_filter_contract_sanity_flags_non_positive_strike():
    calls = pd.DataFrame({"strike": [0.0, -5.0, 100.0]})
    result = filter_contract_sanity(calls)
    assert list(result["invalid_strike_flag"]) == [True, True, False]


def test_filter_contract_sanity_flags_missing_strike():
    calls = pd.DataFrame({"strike": [np.nan, 100.0]})
    result = filter_contract_sanity(calls)
    assert list(result["invalid_strike_flag"]) == [True, False]


def test_filter_contract_sanity_flags_duplicate_strikes():
    calls = pd.DataFrame({"strike": [90.0, 100.0, 100.0, 110.0]})
    result = filter_contract_sanity(calls)
    assert list(result["duplicate_strike_flag"]) == [False, True, True, False]


def test_filter_contract_sanity_does_not_flag_unique_valid_strikes():
    calls = pd.DataFrame({"strike": [90.0, 100.0, 110.0]})
    result = filter_contract_sanity(calls)
    assert list(result["invalid_strike_flag"]) == [False, False, False]
    assert list(result["duplicate_strike_flag"]) == [False, False, False]


def test_filter_contract_sanity_does_not_double_flag_invalid_strike_as_duplicate():
    # Two rows both missing a strike are each individually invalid, but
    # that's not a meaningful "duplicate real strike" -- shouldn't also
    # trip duplicate_strike_flag.
    calls = pd.DataFrame({"strike": [np.nan, np.nan, 100.0]})
    result = filter_contract_sanity(calls)
    assert list(result["invalid_strike_flag"]) == [True, True, False]
    assert list(result["duplicate_strike_flag"]) == [False, False, False]


def test_filter_contract_sanity_flags_nonstandard_contract_size():
    calls = pd.DataFrame({"strike": [90.0, 100.0], "contractSize": [100, 127]})
    result = filter_contract_sanity(calls)
    assert list(result["nonstandard_contract_size_flag"]) == [False, True]


def test_filter_contract_sanity_handles_missing_contract_size_column():
    # No contractSize column at all -- shouldn't KeyError, and can't
    # determine standardness, so nothing is flagged.
    calls = pd.DataFrame({"strike": [90.0, 100.0]})
    result = filter_contract_sanity(calls)
    assert list(result["nonstandard_contract_size_flag"]) == [False, False]


def test_filter_contract_sanity_does_not_mutate_input():
    calls = pd.DataFrame({"strike": [90.0, 100.0], "contractSize": [100, 100]})
    filter_contract_sanity(calls)
    assert "invalid_strike_flag" not in calls.columns
