import math
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
    add_implied_volatility,
    add_theoretical_price,
    filter_put_call_parity,
    implied_spot,
)
from src.pricing.black_scholes import BS_call_price, BS_put_price


#Tests are currently set up by Claude. Looking into implementing more rigorous tests 
#in the future. 
#Use tests at your own risk

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


def test_add_mid_price_treats_a_missing_bid_as_no_live_quote_not_a_zero_bid():
    # Live chains really do carry NaN bid/ask. NaN is not the same as 0:
    # there's no mid to compute, so this must fall back to lastPrice rather
    # than averaging into NaN. Previously (NaN > 0) was False but
    # (ask > 0) was True, so the row was called live and got mid=NaN.
    calls = pd.DataFrame({"bid": [np.nan], "ask": [1.2], "lastPrice": [1.1]})
    result = add_mid_price(calls)
    assert result["has_live_quote"].iloc[0] == False
    assert result["mid_price"].iloc[0] == pytest.approx(1.1)
    assert result["unpriceable"].iloc[0] == False


def test_add_mid_price_flags_unpriceable_when_quotes_and_last_price_are_all_missing():
    # Nothing to price against at all. Without an isna() check, NaN == 0 is
    # False, so the row claimed to be priceable and was handed to the IV
    # solver with a NaN price.
    calls = pd.DataFrame({"bid": [np.nan], "ask": [np.nan], "lastPrice": [np.nan]})
    result = add_mid_price(calls)
    assert pd.isna(result["mid_price"].iloc[0])
    assert result["unpriceable"].iloc[0] == True


def test_add_mid_price_flags_unpriceable_when_a_missing_quote_has_no_last_price():
    calls = pd.DataFrame({"bid": [np.nan], "ask": [1.2], "lastPrice": [np.nan]})
    result = add_mid_price(calls)
    assert result["has_live_quote"].iloc[0] == False
    assert result["unpriceable"].iloc[0] == True


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


def test_apply_spread_tolerance_gives_a_missing_quote_zero_tolerance_not_nan():
    # A NaN tolerance is worse than a wrong one: it makes every downstream
    # comparison that adds it as a cushion evaluate False, silently
    # disabling the check instead of failing loudly.
    calls = pd.DataFrame({"bid": [np.nan], "ask": [1.2], "lastPrice": [1.1]})
    result = apply_spread_tolerance(calls)
    assert result["tolerance"].iloc[0] == 0
    assert result["no_live_quote_flag"].iloc[0] == True
    assert result["crossed_market_flag"].iloc[0] == False


def test_apply_spread_tolerance_percentile_threshold_ignores_missing_quote_rows():
    # Same concern as the no-live-quote and crossed-market exclusions: a
    # NaN row has no meaningful relative spread and must not reach the
    # quantile pool.
    calls = pd.DataFrame({
        "bid":       [1.00,  1.00,  np.nan],
        "ask":       [1.05,  1.20,  5.00],
        "lastPrice": [1.025, 1.10,  5.0],
    })
    result = apply_spread_tolerance(calls, wide_spread_percentile=0.0)
    assert result["wide_spread_flag"].iloc[0] == False
    assert result["wide_spread_flag"].iloc[2] == False


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


def test_filter_no_arbitrage_uses_put_bounds_when_option_type_is_put():
    # Put upper bound is the discounted strike, not S -- a put priced
    # above K*e^(-rT) is a violation even though it'd be fine for a call.
    S, r, T = 100.0, 0.05, 0.5
    K = 120.0
    discounted_strike = K * math.exp(-r * T)
    puts = pd.DataFrame({
        "strike": [K],
        "bid": [discounted_strike + 1.0], "ask": [discounted_strike + 1.2],
        "lastPrice": [discounted_strike + 1.1],
    })
    result = filter_no_arbitrage(puts, S=S, r=r, T=T, option_type="put")
    assert result["upper_bound"].iloc[0] == pytest.approx(discounted_strike)
    assert result["no_arb_violation"].iloc[0] == True


def test_filter_no_arbitrage_allows_a_sensible_put_price():
    S, r, T = 100.0, 0.05, 0.5
    puts = pd.DataFrame({
        "strike": [90.0],
        "bid": [1.9], "ask": [2.1], "lastPrice": [2.0],
    })
    result = filter_no_arbitrage(puts, S=S, r=r, T=T, option_type="put")
    assert result["no_arb_violation"].iloc[0] == False


def test_filter_no_arbitrage_reuses_existing_mid_price_and_tolerance():
    # bid/ask/lastPrice deliberately absent -- same contract every other
    # filter in this module honors. filter_no_arbitrage used to recompute
    # unconditionally and would KeyError on 'bid' here, making it the one
    # function that couldn't accept an already-cleaned frame.
    calls = pd.DataFrame({
        "strike": [110.0], "mid_price": [3.0], "tolerance": [0.1],
    })
    result = filter_no_arbitrage(calls, S=100.0, r=0.05, T=0.5)
    assert result["no_arb_violation"].iloc[0] == False
    assert result["mid_price"].iloc[0] == pytest.approx(3.0)
    assert result["tolerance"].iloc[0] == pytest.approx(0.1)


def test_filter_no_arbitrage_raises_for_invalid_option_type():
    calls = pd.DataFrame({"strike": [100.0], "bid": [2.9], "ask": [3.1], "lastPrice": [3.0]})
    with pytest.raises(ValueError):
        filter_no_arbitrage(calls, S=100.0, r=0.05, T=0.5, option_type="straddle")


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


def test_filter_strike_monotonicity_still_checks_a_neighbour_of_a_missing_quote():
    # Regression test for the worst consequence of the NaN handling: a NaN
    # tolerance made both this row's comparison AND its neighbour's
    # evaluate False, so a genuine violation next to a partially-quoted
    # contract disappeared. Here K=90 falls back to lastPrice 10.01 with
    # tolerance 0, so 10.51 > 10.01 + 0.01 is caught.
    calls = pd.DataFrame({
        "strike": [90.0, 100.0],
        "bid": [np.nan, 10.50], "ask": [10.02, 10.52], "lastPrice": [10.01, 10.51],
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


def test_filter_strike_monotonicity_flags_a_decreasing_put_price_as_a_violation():
    # Puts should be non-decreasing in strike -- a lower strike priced
    # *higher* than a higher strike is a violation for puts (the
    # opposite direction from calls).
    puts = pd.DataFrame({
        "strike": [90.0, 100.0],
        "mid_price": [10.51, 10.01],
        "tolerance": [0.01, 0.01],
    })
    result = filter_strike_monotonicity(puts, option_type="put")
    assert result["monotonicity_violation"].iloc[0] == True


def test_filter_strike_monotonicity_allows_normal_increasing_put_prices():
    puts = pd.DataFrame({
        "strike": [90.0, 100.0, 110.0],
        "mid_price": [2.01, 5.01, 10.01],
        "tolerance": [0.01, 0.01, 0.01],
    })
    result = filter_strike_monotonicity(puts, option_type="put")
    assert list(result["monotonicity_violation"]) == [False, False, False]


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


def test_filter_strike_convexity_allows_real_bs_put_prices_unchanged():
    # Convexity holds identically for puts (verified via BS_put_price
    # directly, no direction flip needed unlike filter_no_arbitrage/
    # filter_strike_monotonicity) -- real BS put prices across strikes
    # should never trip this filter with no option_type param at all.
    S, T, r, sigma = 100, 0.5, 0.05, 0.25
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    mid_prices = [BS_put_price(S, K, T, r, sigma) for K in strikes]
    puts = pd.DataFrame({"strike": strikes, "mid_price": mid_prices, "tolerance": [0.0] * 5})
    result = filter_strike_convexity(puts)
    assert result["convexity_violation"].sum() == 0


def test_filter_strike_convexity_does_not_flag_violation_within_tolerance():
    # chord=6.00, weighted combined_tolerance=0.10; mid_price=6.05 is 0.05
    # above the chord -- a real but noise-sized gap, not a genuine
    # violation. Confirmed from the other direction: selling K2 at its bid
    # (6.00) doesn't cover the wings at their asks (6.05), so there's no
    # executable trade here.
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


def test_filter_strike_convexity_flags_a_butterfly_executable_at_quoted_prices():
    # Sell one K2 at its bid (6.08) and buy the wings at their asks
    # (0.5*10.05 + 0.5*2.05 = 6.05): a $0.03/share riskless credit, or $3
    # per contract, executable at the quoted prices.
    #
    # The unweighted tolerance (tol1 + tol2 + tol3 = 0.15) put the
    # threshold at 6.15 and let this through. Weighting the wings the same
    # way the chord weights their prices gives 0.10, catching it.
    calls = pd.DataFrame({
        "strike":    [90.0,  100.0, 110.0],
        "bid":       [9.95,  6.08,  1.95],
        "ask":       [10.05, 6.18,  2.05],
        "lastPrice": [10.00, 6.13,  2.00],
    })
    result = filter_strike_convexity(calls)
    assert result["convexity_violation"].iloc[1] == True


def test_filter_strike_convexity_distance_weights_the_tolerances_not_just_the_prices():
    # Unevenly spaced (90, 95, 110) -> w = 0.75, so K1's tolerance counts
    # for 0.75 and the far wing K3's for only 0.25. Here K3 is quoted very
    # wide (1.50 x 2.50, tolerance 0.50); counting that in full, as the
    # unweighted version did, put the threshold at 8.56 and hid the
    # violation. Weighted it's 8.18, and mid_price 8.25 is a genuine
    # breach -- bid 8.20 > 0.75*10.01 + 0.25*2.50 = 8.13 is executable.
    calls = pd.DataFrame({
        "strike":    [90.0,  95.0,  110.0],
        "bid":       [9.99,  8.20,  1.50],
        "ask":       [10.01, 8.30,  2.50],
        "lastPrice": [10.00, 8.25,  2.00],
    })
    result = filter_strike_convexity(calls)
    assert result["convexity_violation"].iloc[1] == True


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


def test_filter_liquidity_flags_a_missing_quote_row_as_illiquid():
    # A contract the feed gave no bid/ask for is at least as illiquid as
    # one quoted 0x0. High OI/volume on that row so the flag can only come
    # from the missing quote, not from the liquidity thresholds.
    now = pd.Timestamp("2026-01-15 20:00:00", tz="UTC")
    calls = pd.DataFrame({
        "bid": [np.nan, 1.00], "ask": [1.20, 1.01], "lastPrice": [1.10, 1.005],
        "lastTradeDate": [now, now],
        "openInterest": [5000, 10],
        "volume": [5000, 10],
    })
    result = filter_liquidity(calls, reference_time=now)
    assert result["no_live_quote_flag"].iloc[0] == True
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
    # "REGULAR" is what Yahoo/yfinance actually put in this column for a
    # standard contract -- it's a string enum, not a share count. The
    # second value just stands in for "anything that isn't REGULAR"; no
    # non-REGULAR value has been observed in live data, so this pins the
    # comparison rather than claiming a real Yahoo encoding.
    calls = pd.DataFrame({"strike": [90.0, 100.0], "contractSize": ["REGULAR", "NONSTANDARD"]})
    result = filter_contract_sanity(calls)
    assert list(result["nonstandard_contract_size_flag"]) == [False, True]


def test_filter_contract_sanity_does_not_flag_a_real_all_regular_chain():
    # Regression test for the bug this replaced: comparing contractSize to
    # the integer 100 marked every row of a live chain nonstandard, since
    # yfinance reports the string "REGULAR". Verified against live data --
    # 34/34 AAPL contracts were falsely flagged.
    calls = pd.DataFrame({"strike": [90.0, 100.0, 110.0], "contractSize": ["REGULAR"] * 3})
    result = filter_contract_sanity(calls)
    assert result["nonstandard_contract_size_flag"].sum() == 0


def test_filter_contract_sanity_handles_missing_contract_size_column():
    # No contractSize column at all -- shouldn't KeyError, and can't
    # determine standardness, so nothing is flagged.
    calls = pd.DataFrame({"strike": [90.0, 100.0]})
    result = filter_contract_sanity(calls)
    assert list(result["nonstandard_contract_size_flag"]) == [False, False]


def test_filter_contract_sanity_does_not_mutate_input():
    calls = pd.DataFrame({"strike": [90.0, 100.0], "contractSize": ["REGULAR", "REGULAR"]})
    filter_contract_sanity(calls)
    assert "invalid_strike_flag" not in calls.columns


# ---------------------------------------------------------------------------
# add_implied_volatility
# ---------------------------------------------------------------------------

def test_add_implied_volatility_recovers_known_sigma_from_mid_price():
    S, K, T, r, true_sigma = 100, 100, 1, 0.05, 0.25
    price = BS_call_price(S, K, T, r, true_sigma)
    calls = pd.DataFrame({"strike": [K], "mid_price": [price]})
    result = add_implied_volatility(calls, S=S, r=r, T=T)
    assert result["implied_vol"].iloc[0] == pytest.approx(true_sigma, abs=1e-4)
    assert result["low_confidence_iv_flag"].iloc[0] == False


def test_add_implied_volatility_works_for_puts():
    S, K, T, r, true_sigma = 100, 105, 0.5, 0.03, 0.3
    price = BS_put_price(S, K, T, r, true_sigma)
    calls = pd.DataFrame({"strike": [K], "mid_price": [price]})
    result = add_implied_volatility(calls, S=S, r=r, T=T, option_type="put")
    assert result["implied_vol"].iloc[0] == pytest.approx(true_sigma, abs=1e-4)


def test_add_implied_volatility_skips_unpriceable_rows():
    # bid/ask/lastPrice all 0 -> add_mid_price computes mid_price=0 and
    # flags unpriceable=True.
    #
    # The strike matters. For this OTM contract the no-arbitrage lower
    # bound is 0, so price=0 sits *inside* the solver's valid range and
    # implied_volatility() returns a bogus sigma (~0.25) instead of
    # raising -- the try/except never fires. So this row is skipped only
    # because the unpriceable flag is checked explicitly, which is what
    # this test pins down. An ATM strike would raise on the bounds check
    # and pass even with that flag check removed.
    calls = pd.DataFrame({
        "strike": [150], "bid": [0.0], "ask": [0.0], "lastPrice": [0.0],
    })
    result = add_implied_volatility(calls, S=100, r=0.05, T=0.1)
    assert pd.isna(result["implied_vol"].iloc[0])
    assert pd.isna(result["low_confidence_iv_flag"].iloc[0])


def test_add_implied_volatility_skips_invalid_strike_flag_rows():
    calls = pd.DataFrame({
        "strike": [-5], "mid_price": [10.0], "invalid_strike_flag": [True],
    })
    result = add_implied_volatility(calls, S=100, r=0.05, T=1)
    assert pd.isna(result["implied_vol"].iloc[0])


def test_add_implied_volatility_skips_no_arb_violation_rows():
    calls = pd.DataFrame({
        "strike": [100], "mid_price": [10.0], "no_arb_violation": [True],
    })
    result = add_implied_volatility(calls, S=100, r=0.05, T=1)
    assert pd.isna(result["implied_vol"].iloc[0])


def test_add_implied_volatility_handles_a_mix_of_skipped_and_solvable_rows():
    # One solvable row and one skipped row in the same chain -- the skip
    # shouldn't affect the other row's result.
    S, r, T, true_sigma = 100, 0.05, 1, 0.3
    good_price = BS_call_price(S, 100, T, r, true_sigma)
    calls = pd.DataFrame({
        "strike": [100, 90],
        "mid_price": [good_price, 5.0],
        "unpriceable": [False, True],
    })
    result = add_implied_volatility(calls, S=S, r=r, T=T)
    assert result["implied_vol"].iloc[0] == pytest.approx(true_sigma, abs=1e-4)
    assert pd.isna(result["implied_vol"].iloc[1])


def test_add_implied_volatility_flags_low_confidence_for_deep_otm():
    S, K, T, r, true_sigma = 100, 150, 0.1, 0.05, 0.2
    price = BS_call_price(S, K, T, r, true_sigma)
    calls = pd.DataFrame({"strike": [K], "mid_price": [price]})
    result = add_implied_volatility(calls, S=S, r=r, T=T)
    assert result["low_confidence_iv_flag"].iloc[0] == True


def test_add_implied_volatility_computes_mid_price_when_missing():
    # No mid_price column supplied -- should compute it via add_mid_price
    # so this also works standalone on a raw chain.
    S, K, T, r, true_sigma = 100, 100, 1, 0.05, 0.25
    price = BS_call_price(S, K, T, r, true_sigma)
    calls = pd.DataFrame({
        "strike": [K], "bid": [price - 0.01], "ask": [price + 0.01], "lastPrice": [price],
    })
    result = add_implied_volatility(calls, S=S, r=r, T=T)
    assert "mid_price" in result.columns
    assert result["implied_vol"].iloc[0] == pytest.approx(true_sigma, abs=1e-4)


def test_add_implied_volatility_skips_rows_with_a_missing_quote_and_no_last_price():
    # The row has nothing to price against, so it must be skipped rather
    # than handed to the solver. It previously reached implied_volatility()
    # with a NaN price because unpriceable was computed as (NaN == 0).
    calls = pd.DataFrame({
        "strike": [100], "bid": [np.nan], "ask": [np.nan], "lastPrice": [np.nan],
    })
    result = add_implied_volatility(calls, S=100, r=0.05, T=1)
    assert pd.isna(result["implied_vol"].iloc[0])
    assert pd.isna(result["low_confidence_iv_flag"].iloc[0])


def test_add_implied_volatility_skips_nonstandard_contract_size_rows():
    # Both rows carry an identical, perfectly solvable price -- the only
    # difference is the flag, so a skip is the only thing that can explain
    # the NA. An adjusted contract delivers something other than 100 shares
    # of the underlying, so any sigma solved from its price is built on the
    # wrong deliverable and would go on to pollute the smile fit.
    S, K, T, r, true_sigma = 100, 100, 1, 0.05, 0.25
    price = BS_call_price(S, K, T, r, true_sigma)
    calls = pd.DataFrame({
        "strike": [K, K],
        "mid_price": [price, price],
        "nonstandard_contract_size_flag": [False, True],
    })
    result = add_implied_volatility(calls, S=S, r=r, T=T)
    assert result["implied_vol"].iloc[0] == pytest.approx(true_sigma, abs=1e-4)
    assert pd.isna(result["implied_vol"].iloc[1])
    assert pd.isna(result["low_confidence_iv_flag"].iloc[1])


def test_add_implied_volatility_skips_a_contract_sanity_flagged_adjusted_contract():
    # Same skip, but with the flag arriving the way it actually does --
    # from filter_contract_sanity reading contractSize off the chain --
    # rather than being hand-set.
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.25
    price = BS_call_price(S, K, T, r, sigma)
    calls = filter_contract_sanity(pd.DataFrame({
        "strike": [K], "mid_price": [price], "contractSize": ["NONSTANDARD"],
    }))
    assert calls["nonstandard_contract_size_flag"].iloc[0] == True
    result = add_implied_volatility(calls, S=S, r=r, T=T)
    assert pd.isna(result["implied_vol"].iloc[0])


def test_add_implied_volatility_still_solves_a_standard_regular_contract():
    # Control for the two above: the same path with contractSize "REGULAR"
    # must still solve, so the skip can't be silently swallowing everything.
    S, K, T, r, true_sigma = 100, 100, 1, 0.05, 0.25
    price = BS_call_price(S, K, T, r, true_sigma)
    calls = filter_contract_sanity(pd.DataFrame({
        "strike": [K], "mid_price": [price], "contractSize": ["REGULAR"],
    }))
    result = add_implied_volatility(calls, S=S, r=r, T=T)
    assert result["implied_vol"].iloc[0] == pytest.approx(true_sigma, abs=1e-4)


def test_add_implied_volatility_does_not_mutate_input():
    S, K, T, r, true_sigma = 100, 100, 1, 0.05, 0.25
    price = BS_call_price(S, K, T, r, true_sigma)
    calls = pd.DataFrame({"strike": [K], "mid_price": [price]})
    add_implied_volatility(calls, S=S, r=r, T=T)
    assert "implied_vol" not in calls.columns


# ---------------------------------------------------------------------------
# add_theoretical_price
# ---------------------------------------------------------------------------

def test_add_theoretical_price_matches_bs_call_price_directly():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.25
    calls = pd.DataFrame({"strike": [K], "mid_price": [12.0]})
    result = add_theoretical_price(calls, S=S, r=r, T=T, sigma=sigma)
    assert result["bs_price"].iloc[0] == pytest.approx(BS_call_price(S, K, T, r, sigma))


def test_add_theoretical_price_works_for_puts():
    S, K, T, r, sigma = 100, 105, 0.5, 0.03, 0.3
    calls = pd.DataFrame({"strike": [K], "mid_price": [10.0]})
    result = add_theoretical_price(calls, S=S, r=r, T=T, sigma=sigma, option_type="put")
    assert result["bs_price"].iloc[0] == pytest.approx(BS_put_price(S, K, T, r, sigma))


def test_add_theoretical_price_computes_price_diff_against_mid_price():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.25
    bs_price = BS_call_price(S, K, T, r, sigma)
    calls = pd.DataFrame({"strike": [K], "mid_price": [bs_price + 2.0]})
    result = add_theoretical_price(calls, S=S, r=r, T=T, sigma=sigma)
    # Market price is $2 above the model price -- rich relative to sigma.
    assert result["price_diff"].iloc[0] == pytest.approx(2.0)


def test_add_theoretical_price_still_prices_unpriceable_and_no_arb_violation_rows():
    # Unlike add_implied_volatility, this doesn't depend on the row's own
    # price being trustworthy -- sigma is external -- so these rows
    # should still get a theoretical price rather than being skipped.
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.25
    calls = pd.DataFrame({
        "strike": [K, K],
        "mid_price": [0.0, 999.0],
        "unpriceable": [True, False],
        "no_arb_violation": [False, True],
    })
    result = add_theoretical_price(calls, S=S, r=r, T=T, sigma=sigma)
    expected = BS_call_price(S, K, T, r, sigma)
    assert result["bs_price"].iloc[0] == pytest.approx(expected)
    assert result["bs_price"].iloc[1] == pytest.approx(expected)


def test_add_theoretical_price_skips_invalid_strike_flag_rows():
    calls = pd.DataFrame({
        "strike": [-5], "mid_price": [10.0], "invalid_strike_flag": [True],
    })
    result = add_theoretical_price(calls, S=100, r=0.05, T=1, sigma=0.25)
    assert pd.isna(result["bs_price"].iloc[0])
    assert pd.isna(result["price_diff"].iloc[0])


def test_add_theoretical_price_computes_mid_price_when_missing():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.25
    calls = pd.DataFrame({
        "strike": [K], "bid": [11.9], "ask": [12.1], "lastPrice": [12.0],
    })
    result = add_theoretical_price(calls, S=S, r=r, T=T, sigma=sigma)
    assert "mid_price" in result.columns
    assert result["price_diff"].iloc[0] == pytest.approx(12.0 - result["bs_price"].iloc[0])


def test_add_theoretical_price_raises_for_invalid_option_type():
    calls = pd.DataFrame({"strike": [100], "mid_price": [10.0]})
    with pytest.raises(ValueError):
        add_theoretical_price(calls, S=100, r=0.05, T=1, sigma=0.25, option_type="straddle")


def test_add_theoretical_price_raises_for_non_positive_sigma():
    calls = pd.DataFrame({"strike": [100], "mid_price": [10.0]})
    with pytest.raises(ValueError):
        add_theoretical_price(calls, S=100, r=0.05, T=1, sigma=0.0)


def test_add_theoretical_price_validates_sigma_even_when_every_row_is_skipped():
    # BS_price_series (where sigma is normally validated) never actually
    # gets called when every row is flagged invalid_strike_flag -- sigma
    # must still be validated unconditionally, not just as a side effect
    # of pricing at least one row.
    calls = pd.DataFrame({
        "strike": [-5], "mid_price": [10.0], "invalid_strike_flag": [True],
    })
    with pytest.raises(ValueError):
        add_theoretical_price(calls, S=100, r=0.05, T=1, sigma=0.0)


def test_add_theoretical_price_does_not_mutate_input():
    calls = pd.DataFrame({"strike": [100], "mid_price": [10.0]})
    add_theoretical_price(calls, S=100, r=0.05, T=1, sigma=0.25)
    assert "bs_price" not in calls.columns


# ---------------------------------------------------------------------------
# filter_put_call_parity
# ---------------------------------------------------------------------------

def test_filter_put_call_parity_holds_for_real_bs_prices():
    # Parity holds exactly by construction for real BS call/put prices at
    # the same S/K/T/r/sigma, so this should never flag a violation.
    S, T, r, sigma = 100, 0.5, 0.05, 0.25
    strikes = [90, 100, 110]
    calls = pd.DataFrame({
        "strike": strikes,
        "mid_price": [BS_call_price(S, K, T, r, sigma) for K in strikes],
        "tolerance": [1e-9] * 3,
    })
    puts = pd.DataFrame({
        "strike": strikes,
        "mid_price": [BS_put_price(S, K, T, r, sigma) for K in strikes],
        "tolerance": [1e-9] * 3,
    })
    result = filter_put_call_parity(calls, puts, S=S, r=r, T=T)
    assert result["parity_violation"].sum() == 0
    assert list(result["parity_residual"]) == pytest.approx([0.0, 0.0, 0.0], abs=1e-8)


def test_filter_put_call_parity_flags_violation_beyond_tolerance():
    S, r, T = 100.0, 0.05, 0.5
    K = 100.0
    discounted_strike = K * math.exp(-r * T)
    # Real C-P should equal S - discounted_strike; make it off by $5.
    calls = pd.DataFrame({"strike": [K], "mid_price": [10.0 + 5.0], "tolerance": [0.01]})
    puts = pd.DataFrame({"strike": [K], "mid_price": [10.0 - (S - discounted_strike)], "tolerance": [0.01]})
    result = filter_put_call_parity(calls, puts, S=S, r=r, T=T)
    assert result["parity_violation"].iloc[0] == True


def test_filter_put_call_parity_does_not_flag_violation_within_tolerance():
    S, r, T = 100.0, 0.05, 0.5
    K = 100.0
    discounted_strike = K * math.exp(-r * T)
    call_price = 12.0
    # put price chosen so C - P is off from parity by only $0.02, inside
    # the combined $0.05 tolerance.
    put_price = call_price - (S - discounted_strike) + 0.02
    calls = pd.DataFrame({"strike": [K], "mid_price": [call_price], "tolerance": [0.025]})
    puts = pd.DataFrame({"strike": [K], "mid_price": [put_price], "tolerance": [0.025]})
    result = filter_put_call_parity(calls, puts, S=S, r=r, T=T)
    assert result["parity_violation"].iloc[0] == False


def test_filter_put_call_parity_only_matches_common_strikes():
    calls = pd.DataFrame({"strike": [90.0, 100.0, 110.0], "mid_price": [12.0, 8.0, 5.0], "tolerance": [0.0] * 3})
    puts = pd.DataFrame({"strike": [100.0, 110.0, 120.0], "mid_price": [3.0, 6.0, 10.0], "tolerance": [0.0] * 3})
    result = filter_put_call_parity(calls, puts, S=100.0, r=0.05, T=0.5)
    assert list(result["strike"]) == [100.0, 110.0]


def test_filter_put_call_parity_skips_unpriceable_rows_with_na_not_false():
    calls = pd.DataFrame({
        "strike": [100.0], "mid_price": [8.0], "tolerance": [0.0], "unpriceable": [True],
    })
    puts = pd.DataFrame({
        "strike": [100.0], "mid_price": [3.0], "tolerance": [0.0], "unpriceable": [False],
    })
    result = filter_put_call_parity(calls, puts, S=100.0, r=0.05, T=0.5)
    assert pd.isna(result["parity_violation"].iloc[0])
    assert pd.isna(result["parity_residual"].iloc[0])


def test_filter_put_call_parity_skips_invalid_strike_flag_rows():
    calls = pd.DataFrame({
        "strike": [-5.0], "mid_price": [8.0], "tolerance": [0.0], "invalid_strike_flag": [True],
    })
    puts = pd.DataFrame({
        "strike": [-5.0], "mid_price": [3.0], "tolerance": [0.0], "invalid_strike_flag": [True],
    })
    result = filter_put_call_parity(calls, puts, S=100.0, r=0.05, T=0.5)
    assert pd.isna(result["parity_violation"].iloc[0])


def test_filter_put_call_parity_computes_mid_price_when_missing():
    S, T, r, sigma = 100, 0.5, 0.05, 0.25
    K = 100
    call_price = BS_call_price(S, K, T, r, sigma)
    put_price = BS_put_price(S, K, T, r, sigma)
    calls = pd.DataFrame({"strike": [K], "bid": [call_price - 0.01], "ask": [call_price + 0.01], "lastPrice": [call_price]})
    puts = pd.DataFrame({"strike": [K], "bid": [put_price - 0.01], "ask": [put_price + 0.01], "lastPrice": [put_price]})
    result = filter_put_call_parity(calls, puts, S=S, r=r, T=T)
    assert result["parity_violation"].iloc[0] == False


def test_filter_put_call_parity_returns_one_row_per_duplicated_strike():
    # Without deduplication pd.merge pairs every call against every put at
    # a duplicated strike -- two of each produced four rows, checking the
    # same contract repeatedly and breaking the documented
    # one-row-per-matched-strike contract.
    calls = pd.DataFrame({"strike": [100.0, 100.0], "mid_price": [8.0, 9.0], "tolerance": [0.0, 0.0]})
    puts = pd.DataFrame({"strike": [100.0, 100.0], "mid_price": [3.0, 4.0], "tolerance": [0.0, 0.0]})
    result = filter_put_call_parity(calls, puts, S=100.0, r=0.05, T=0.5)
    assert len(result) == 1


def test_filter_put_call_parity_skips_duplicated_strikes_with_na_not_false():
    # A duplicated strike has no single counterparty contract to check
    # against, so it's unverifiable rather than clean -- same NA treatment
    # as an unpriceable row, not a False "no violation here".
    calls = pd.DataFrame({"strike": [100.0, 100.0], "mid_price": [8.0, 9.0], "tolerance": [0.0, 0.0]})
    puts = pd.DataFrame({"strike": [100.0, 100.0], "mid_price": [3.0, 4.0], "tolerance": [0.0, 0.0]})
    result = filter_put_call_parity(calls, puts, S=100.0, r=0.05, T=0.5)
    assert pd.isna(result["parity_violation"].iloc[0])
    assert pd.isna(result["parity_residual"].iloc[0])


def test_filter_put_call_parity_duplicate_on_one_side_only_still_skips():
    # Ambiguity on either side is enough -- the call chain here is clean,
    # but there's no way to know which of the two puts to match it to.
    calls = pd.DataFrame({"strike": [100.0], "mid_price": [8.0], "tolerance": [0.0]})
    puts = pd.DataFrame({"strike": [100.0, 100.0], "mid_price": [3.0, 4.0], "tolerance": [0.0, 0.0]})
    result = filter_put_call_parity(calls, puts, S=100.0, r=0.05, T=0.5)
    assert len(result) == 1
    assert pd.isna(result["parity_violation"].iloc[0])


def test_filter_put_call_parity_does_not_skip_clean_strikes_alongside_a_duplicate():
    # The duplicate shouldn't contaminate unrelated strikes -- K=90 has one
    # call and one put and must still be checked normally.
    S, T, r, sigma = 100, 0.5, 0.05, 0.25
    calls = pd.DataFrame({
        "strike": [90.0, 100.0, 100.0],
        "mid_price": [BS_call_price(S, 90, T, r, sigma), 8.0, 9.0],
        "tolerance": [1e-9, 0.0, 0.0],
    })
    puts = pd.DataFrame({
        "strike": [90.0, 100.0, 100.0],
        "mid_price": [BS_put_price(S, 90, T, r, sigma), 3.0, 4.0],
        "tolerance": [1e-9, 0.0, 0.0],
    })
    result = filter_put_call_parity(calls, puts, S=S, r=r, T=T).set_index("strike")
    assert result.loc[90.0, "parity_violation"] == False
    assert pd.isna(result.loc[100.0, "parity_violation"])


def test_filter_put_call_parity_reuses_existing_duplicate_strike_flag():
    # filter_contract_sanity has already run here and excludes invalid
    # strikes from its duplicate definition -- this should defer to that
    # column rather than recomputing a subtly different rule.
    calls = pd.DataFrame({
        "strike": [100.0], "mid_price": [8.0], "tolerance": [0.0],
        "duplicate_strike_flag": [True],
    })
    puts = pd.DataFrame({
        "strike": [100.0], "mid_price": [3.0], "tolerance": [0.0],
        "duplicate_strike_flag": [False],
    })
    result = filter_put_call_parity(calls, puts, S=100.0, r=0.05, T=0.5)
    assert pd.isna(result["parity_violation"].iloc[0])


def test_filter_put_call_parity_honors_an_unpriceable_put_flagged_on_one_side_only():
    # The calls frame arrives pre-cleaned (mid_price/tolerance already
    # present, so add_mid_price never runs and no 'unpriceable' column is
    # created), while the puts frame arrives raw and gains one. pd.merge
    # only suffixes columns present on both sides, so the flag survives
    # under its bare name -- and was being ignored, reporting a contract
    # with no price at all as a genuine parity violation.
    calls = pd.DataFrame({"strike": [100.0], "mid_price": [8.0], "tolerance": [0.0]})
    puts = pd.DataFrame({"strike": [100.0], "bid": [0.0], "ask": [0.0], "lastPrice": [0.0]})
    result = filter_put_call_parity(calls, puts, S=100.0, r=0.05, T=0.5)
    assert pd.isna(result["parity_violation"].iloc[0])
    assert pd.isna(result["parity_residual"].iloc[0])


def test_filter_put_call_parity_honors_an_unpriceable_call_flagged_on_one_side_only():
    # Mirror of the above with the raw/pre-cleaned sides swapped, so the
    # bare flag comes from the calls frame instead.
    calls = pd.DataFrame({"strike": [100.0], "bid": [0.0], "ask": [0.0], "lastPrice": [0.0]})
    puts = pd.DataFrame({"strike": [100.0], "mid_price": [3.0], "tolerance": [0.0]})
    result = filter_put_call_parity(calls, puts, S=100.0, r=0.05, T=0.5)
    assert pd.isna(result["parity_violation"].iloc[0])
    assert pd.isna(result["parity_residual"].iloc[0])


def test_filter_put_call_parity_skips_a_nonstandard_contract_on_either_side():
    # An adjusted contract and a standard one at the same strike aren't a
    # parity pair -- they don't deliver the same thing -- so the identity
    # doesn't apply and the row is unverifiable rather than violating.
    # Flags arrive via filter_contract_sanity's fallback, as in real use.
    def pair(call_size, put_size):
        calls = pd.DataFrame({
            "strike": [100.0], "mid_price": [8.0], "tolerance": [0.0],
            "contractSize": [call_size],
        })
        puts = pd.DataFrame({
            "strike": [100.0], "mid_price": [3.0], "tolerance": [0.0],
            "contractSize": [put_size],
        })
        return filter_put_call_parity(calls, puts, S=100.0, r=0.05, T=0.5)

    assert pd.isna(pair("NONSTANDARD", "REGULAR")["parity_violation"].iloc[0])
    assert pd.isna(pair("REGULAR", "NONSTANDARD")["parity_violation"].iloc[0])
    # Control: two standard contracts are still checked normally.
    assert pair("REGULAR", "REGULAR")["parity_violation"].iloc[0] is not pd.NA


def test_filter_put_call_parity_does_not_mutate_input():
    calls = pd.DataFrame({"strike": [100.0], "mid_price": [8.0], "tolerance": [0.0]})
    puts = pd.DataFrame({"strike": [100.0], "mid_price": [3.0], "tolerance": [0.0]})
    filter_put_call_parity(calls, puts, S=100.0, r=0.05, T=0.5)
    assert "parity_violation" not in calls.columns
    assert "parity_violation" not in puts.columns


# ---------------------------------------------------------------------------
# implied_spot
# ---------------------------------------------------------------------------

def _chain_priced_at(true_S, strikes, T=0.5, r=0.05, sigma=0.25, tolerance=0.01):
    """Calls/puts generated from a known spot, so parity holds exactly at it."""
    calls = pd.DataFrame({
        "strike": strikes,
        "mid_price": [BS_call_price(true_S, K, T, r, sigma) for K in strikes],
        "tolerance": [tolerance] * len(strikes),
    })
    puts = pd.DataFrame({
        "strike": strikes,
        "mid_price": [BS_put_price(true_S, K, T, r, sigma) for K in strikes],
        "tolerance": [tolerance] * len(strikes),
    })
    return calls, puts


def test_implied_spot_recovers_the_spot_the_chain_was_priced_at():
    # The chain is quoted against 100 but we hand the pipeline a stale 101,
    # which is exactly the yfinance endpoint-mismatch case. Parity should
    # recover the 100 the option market is actually using.
    true_S, stale_S, r, T = 100.0, 101.0, 0.05, 0.5
    calls, puts = _chain_priced_at(true_S, [90.0, 95.0, 100.0, 105.0, 110.0], T=T, r=r)
    parity = filter_put_call_parity(calls, puts, S=stale_S, r=r, T=T)
    assert implied_spot(parity, stale_S) == pytest.approx(true_S, abs=1e-8)


def test_implied_spot_returns_the_quoted_spot_when_parity_already_holds():
    # No endpoint mismatch -> nothing to correct, so this must be a no-op
    # rather than drifting the spot on a clean chain.
    S, r, T = 100.0, 0.05, 0.5
    calls, puts = _chain_priced_at(S, [90.0, 100.0, 110.0], T=T, r=r)
    parity = filter_put_call_parity(calls, puts, S=S, r=r, T=T)
    assert implied_spot(parity, S) == pytest.approx(S, abs=1e-8)


def test_implied_spot_median_resists_a_single_bad_strike():
    # One badly mispriced put (a wide deep-wing quote) shouldn't move the
    # estimate -- this is why the median is used rather than the mean,
    # which would be dragged roughly 1/5th of the way to the outlier.
    true_S, r, T = 100.0, 0.05, 0.5
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    calls, puts = _chain_priced_at(true_S, strikes, T=T, r=r)
    puts.loc[4, "mid_price"] += 5.0          # corrupt one strike badly
    parity = filter_put_call_parity(calls, puts, S=true_S, r=r, T=T)

    assert implied_spot(parity, true_S) == pytest.approx(true_S, abs=1e-8)
    # the mean would not have survived it
    mean_based = true_S + parity["parity_residual"].dropna().astype(float).mean()
    assert abs(mean_based - true_S) > 0.5


def test_implied_spot_ignores_strikes_parity_could_not_check():
    # Rows filter_put_call_parity skipped (unpriceable here) come back as NA
    # and must not be counted as a zero residual, which would pull the
    # estimate back toward the stale quoted spot.
    true_S, stale_S, r, T = 100.0, 101.0, 0.05, 0.5
    strikes = [90.0, 95.0, 100.0, 105.0]
    calls, puts = _chain_priced_at(true_S, strikes, T=T, r=r)
    calls["unpriceable"] = [False, False, False, True]
    puts["unpriceable"] = [False, False, False, False]
    parity = filter_put_call_parity(calls, puts, S=stale_S, r=r, T=T)

    assert parity["parity_residual"].isna().sum() == 1
    assert implied_spot(parity, stale_S) == pytest.approx(true_S, abs=1e-8)


def test_implied_spot_raises_when_too_few_strikes_are_checkable():
    S, r, T = 100.0, 0.05, 0.5
    calls, puts = _chain_priced_at(S, [100.0, 105.0], T=T, r=r)
    parity = filter_put_call_parity(calls, puts, S=S, r=r, T=T)
    with pytest.raises(ValueError, match="at least 3 checkable strikes"):
        implied_spot(parity, S)


def test_implied_spot_min_strikes_is_configurable():
    S, r, T = 100.0, 0.05, 0.5
    calls, puts = _chain_priced_at(S, [100.0, 105.0], T=T, r=r)
    parity = filter_put_call_parity(calls, puts, S=S, r=r, T=T)
    assert implied_spot(parity, S, min_strikes=2) == pytest.approx(S, abs=1e-8)


def test_implied_spot_returns_the_dividend_adjusted_spot_on_a_dividend_payer():
    # Parity with a dividend before expiration is really
    #     C - P = S - PV(D) - K*e^(-rT)
    # so what comes back is S - PV(D), not the traded share price. That is
    # the quantity a q=0 Black-Scholes wants (the standard escrowed-dividend
    # adjustment), so downstream pricing stays consistent -- but the number
    # sits below the traded price, and the gap it implies against the quoted
    # spot is quote-timing plus PV(D) rather than quote-timing alone.
    S_true, D, t, T, r, sigma = 100.0, 2.00, 0.25, 0.5, 0.05, 0.25
    S_adj = S_true - D * math.exp(-r * t)

    # Real European prices on a dividend payer are BS prices off S_adj.
    calls, puts = _chain_priced_at(
        S_adj, [90.0, 95.0, 100.0, 105.0, 110.0], T=T, r=r, sigma=sigma, tolerance=1e-9
    )
    parity = filter_put_call_parity(calls, puts, S=S_true, r=r, T=T)

    assert implied_spot(parity, S_true) == pytest.approx(S_adj, abs=1e-8)
    assert implied_spot(parity, S_true) < S_true
