import numpy as np
import pandas as pd

from src.data.fetch_options import trading_days_since


def add_mid_price(calls: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a mid_price column: (bid+ask)/2 when at least one side has 
    a live quote. (Note: bid=0 with ask>0 is considered a live quote).
    Falls back to lastPrice if both bid and ask equal zero. 
    Flags rows as unpriceable when the resulting mid_price is zero.
    (No live quote on either side and no recent trade price)
    """
    calls = calls.copy()

    has_valid_quote = (calls["bid"] > 0) | (calls["ask"] > 0)

    calls["mid_price"] = np.where(
        has_valid_quote,
        (calls["bid"] + calls["ask"]) / 2,
        calls["lastPrice"]
    )

    calls["unpriceable"] = calls["mid_price"] == 0
    calls["has_live_quote"] = has_valid_quote

    return calls

def apply_spread_tolerance(
    calls,
    wide_spread_percentile: float = 0.90,
    low_tick: float = 0.01,
    high_tick: float = 0.05,
    tick_threshold: float = 3.0,
):
    """
    Adds a no-arbitrage tolerance column.
    Additionally adds quote-quality flags: wide_spread_flag, no_live_quote flag,
    crossed_market_flag. These flags are used by filter_liquidity
    
    tolerance is calculated as half the real, uncapped bid-ask spread,
    for live, uncrossed quotes. (This rarely filters out options). 

    For options with no-live-quote and options with crossed_market flag
    (bid>ask) the tolerance was set to 0. 
    (This is particually important for crossed options as they would return
    a negative tolerance)

    wide_spread_flag and its percentile threshold are based on execess spread:
    raw spread minus contract's tick size, not the raw spread. This means
    a cheap contract does not accidently get flagged as having a large 
    spread due to it's minimum increment. 
    Tick size follows CBOE's Penny Program: pennies below tick_threshold=$3
    and nickles at/above $3 calculated using mid-price.

    Note: the tolerance calculation itself keeps the raw spread
    as the minimum increment size for these contracts still impacts
    uncertainty. 

    No_live_quote and crossed rows were also excluded from 
    the percentile pool since they would skew the threshold.

    Reuses mid_price/has_live_quote (via add_mid_price if missing), so
    this also works standalone on a raw chain.

    The wide_spread_percentile is semi-arbitrary (follows other conventions 
    established through the cleaning filters) with more research and 
    testing being looked into in the future. 
    """
    calls = calls.copy()

    if "mid_price" not in calls.columns or "has_live_quote" not in calls.columns:
        calls = add_mid_price(calls)

    live = calls["has_live_quote"]

    raw_spread = calls["ask"] - calls["bid"]
    crossed_market_flag = raw_spread < 0
    calls["crossed_market_flag"] = crossed_market_flag

    valid_spread = live & ~crossed_market_flag

    #Avoids dividing by zero
    safe_mid = calls["mid_price"].replace(0, np.nan)

    # CBOE Penny Program: pennies below tick_threshold, nickels at/above it.
    min_tick = np.where(calls["mid_price"] < tick_threshold, low_tick, high_tick)

    #How much wider the spread is than the required minimum tick. 
    excess_spread = np.maximum(raw_spread - min_tick, 0)
    relative_excess_spread = excess_spread / safe_mid

    spread_threshold = relative_excess_spread[valid_spread].quantile(wide_spread_percentile)
    calls["wide_spread_flag"] = valid_spread & (relative_excess_spread > spread_threshold)
    calls["no_live_quote_flag"] = ~live

    calls["tolerance"] = np.where(valid_spread, raw_spread / 2, 0.0)

    return calls


def filter_no_arbitrage(calls: pd.DataFrame, S: float, r: float, T: float) -> pd.DataFrame:
    """
    Flags no-arbitrage violations, allowing a tolerance (calculated in last function)
    to absorb some market noise instead of flagging economincally meaningless 
    small violations. Rows with no live quote get zero tolerance. 

    T is the time to expiration in years taken from time_to_expiration. 
    This must be computed before passing into the function. 
    This function does not calculate T or grab it from somewhere else.
    
    """

    calls = calls.copy()
    calls = add_mid_price(calls)
    calls = apply_spread_tolerance(calls)

    lower_bound = np.maximum(S - calls["strike"] * np.exp(-r * T), 0)
    upper_bound = S

    calls["lower_bound"] = lower_bound
    calls["upper_bound"] = upper_bound


    calls["no_arb_violation"] = (
        (calls["mid_price"] < lower_bound - calls["tolerance"]) |
        (calls["mid_price"] > upper_bound + calls["tolerance"])
    )

    return calls

def filter_strike_monotonicity(calls: pd.DataFrame) -> pd.DataFrame:
    """
    Flags call prices that violate strike monotonicity:
    If a strike K1<K2 (assuming same expiration), then C(K1) should 
    never be less then C(K2).

    Note: Unlike other functions this function is chain-wide not 
    row-wise. This is because it needs every strike for one
    expiration, sorted, to compare each contract against its 
    immediate neighbor. Without the sorting, the function 
    could compare non-adjacent strikes. The returned 
    DataFrame is sorted by strike (ascending), 
    which may reorder rows relative to the input. 

    Reuses mid_price and tolerance. This function also uses tolerance
    to avoid flagging meaningessly small violations. 

    """
    calls = calls.copy()

    if "mid_price" not in calls.columns or "tolerance" not in calls.columns:
        calls = add_mid_price(calls)
        calls = apply_spread_tolerance(calls)

    calls = calls.sort_values("strike").reset_index(drop=True)

    next_price = calls["mid_price"].shift(-1)
    next_tolerance = calls["tolerance"].shift(-1)
    combined_tolerance = calls["tolerance"] + next_tolerance

    calls["monotonicity_violation"] = next_price > calls["mid_price"] + combined_tolerance

    return calls

def filter_strike_convexity(calls: pd.DataFrame) -> pd.DataFrame:
    """
    Flags call prices that violate strike convexity:
    Uses the general convexity condition so it can handle unequally-spaced 
    options:
        For any three strikes K1 < K2 < K3 (same underlying/expiration)
        C(K2) <= w*C(K1) + (1-w)*C(K3) + tolerance, 
        where w = (K3-K2)/(K3-K1)
        Note: tolerance is not part of the textbook formula but was added
        to reduce options for being eliminated for extremely small violations
        
    The logic behind the formula can be found by reading into 
    the general strike convexity condition (does not assume equally-spaced)
    as the code (besides tolerance) follows textbook logic and equations.

    This function is chain-wide, not row-wise, similar to filter_strike_monotoncity.
    This means it needs every strike for one expiration, sorted, to form
    consecutive triplets. Run it on the full chain. The returned Dataframe
    is sorted by strike (ascending). The first and last strikes can't form
    a full triplet and are never flagged.

    Reuses mid_price/tolerance (via add_mid_price/apply_spread_tolerance
    if not already present). A violation is only flagged beyond the
    combined quoted tolerance of all three contracts involved, same
    reasoning as filter_strike_monotonicity.
    """
    calls = calls.copy()

    if "mid_price" not in calls.columns or "tolerance" not in calls.columns:
        calls = add_mid_price(calls)
        calls = apply_spread_tolerance(calls)

    calls = calls.sort_values("strike").reset_index(drop=True)

    K1, K2, K3 = calls["strike"].shift(1), calls["strike"], calls["strike"].shift(-1)
    C1, C2, C3 = calls["mid_price"].shift(1), calls["mid_price"], calls["mid_price"].shift(-1)
    tol1, tol3 = calls["tolerance"].shift(1), calls["tolerance"].shift(-1)

    #Avoids dividing by zero on duplicate/degenerate strikes
    strike_span = (K3 - K1).replace(0, np.nan)
    weight_1 = (K3 - K2) / strike_span
    chord_value = weight_1 * C1 + (1 - weight_1) * C3

    combined_tolerance = tol1 + calls["tolerance"] + tol3

    calls["convexity_violation"] = C2 > chord_value + combined_tolerance

    return calls

def apply_staleness_flag(
    calls: pd.DataFrame,
    max_age_trading_days: float = 1.0,
    stock_exchange: str = "NYSE",
    reference_time=None,
) -> pd.DataFrame:
    """
    Flags contracts who weren't traded during a full session 
    using the trading_days_since function from fetch_options. 
    
    A live bid/ask doesn't make this redundant: a contract can have a
    quoted market and still not have traded recently, so this is an
    independent signal from has_live_quote/no_live_quote_flag.

    Missing lastTradeDate is treated as stale (can't verify freshness).
    reference_time defaults to now (UTC) if not given; pass a fixed value
    for deterministic testing.
    """
    calls = calls.copy()

    sessions_elapsed = trading_days_since(calls["lastTradeDate"], stock_exchange, reference_time)

    calls["stale_quote_flag"] = sessions_elapsed.isna() | (sessions_elapsed > max_age_trading_days)

    return calls

def filter_liquidity(
    calls: pd.DataFrame,
    percentile: float = 0.10,
    max_age_trading_days: float = 1.0,
    stock_exchange: str = "NYSE",
    reference_time=None,
) -> pd.DataFrame:
    """
    This function flags contracts as illiquid based on:
    open interest, volume and quote quality. If any of these
    conditions are violated it is flagged as illiquid:
    -openInterest or volume in the bottom 'percentile' of the day's 
        chain
    -violated wide_spread_flag, no_live_flag, crossed_market_flag
    -stale_quote_flag
    
    Each of these violations is also stored as its own flag, so callers
    can filter on a specific signal instead of only the combined one.
    

    

    None of these conditions are no-arbitrage signals. This function
    helps clean up the options before they're inputed into the 
    the no-arbitrage function so they do not mess with it's tolerance.

    # NOTE: "max_age_days" below is now max_age_trading_days -- this
    # function forwards it to apply_staleness_flag's trading-day-aware
    # check instead of a raw calendar-day one. Left the rest of this
    # paragraph as-is rather than rewritten.

    Flags contracts as illiquid based on open interest, volume, and quote
    quality -- OR logic across all signals, so a contract needs only one
    thin signal to be flagged (stricter than AND):
      - openInterest or volume in the bottom `percentile` of that day's chain
      - wide_spread_flag / no_live_quote_flag / crossed_market_flag (bid/ask
        quote quality)
      - stale_quote_flag (lastTradeDate older than max_age_days)
    None of these are no-arbitrage signals, which is why they're consumed
    here rather than affecting filter_no_arbitrage's tolerance.
    Relative (percentile-based) thresholds, not absolute counts, so this
    adapts across tickers/expirations without hardcoded cutoffs.
    Missing openInterest/volume (NaN) is treated as illiquid rather than
    silently passing the threshold check, since missing data is at least
    as strong a thin-liquidity signal as a low reported value.
    If wide_spread_flag/no_live_quote_flag/crossed_market_flag/
    stale_quote_flag are already present (e.g. this DataFrame came from
    filter_no_arbitrage), they're reused unchanged; otherwise this
    computes them itself via add_mid_price/apply_spread_tolerance/
    apply_staleness_flag, so filter_liquidity also works standalone on a
    raw option chain.
    Does not drop rows -- downstream consumers decide whether to exclude
    illiquid contracts for their specific use case.
    """

    calls = calls.copy()

    if (
        "wide_spread_flag" not in calls.columns
        or "no_live_quote_flag" not in calls.columns
        or "crossed_market_flag" not in calls.columns
    ):
        calls = add_mid_price(calls)
        calls = apply_spread_tolerance(calls)

    if "stale_quote_flag" not in calls.columns:
        calls = apply_staleness_flag(
            calls,
            max_age_trading_days=max_age_trading_days,
            stock_exchange=stock_exchange,
            reference_time=reference_time,
        )

    oi_threshold = calls["openInterest"].quantile(percentile)
    volume_threshold = calls["volume"].quantile(percentile)

    low_oi = calls["openInterest"].isna() | (calls["openInterest"] <= oi_threshold)
    low_volume = calls["volume"].isna() | (calls["volume"] <= volume_threshold)

    calls["low_oi_flag"] = low_oi
    calls["low_volume_flag"] = low_volume

    calls["illiquid_flag"] = (
        low_oi
        | low_volume
        | calls["wide_spread_flag"]
        | calls["no_live_quote_flag"]
        | calls["crossed_market_flag"]
        | calls["stale_quote_flag"]
    )

    return calls

def filter_contract_sanity(calls: pd.DataFrame, expected_contract_size: int = 100) -> pd.DataFrame:
    """
    Flags basic structural problems with a contract row, as opposed to
    problems with its quoted price (unlike every other filter in this
    module):
      - invalid_strike_flag: strike is missing or <= 0.
      - duplicate_strike_flag: the same strike appears more than once in
        the chain. Chain-wide like the strike-arbitrage filters (needs
        every row to detect a duplicate), though unlike those it doesn't
        need sorting. Rows already flagged invalid_strike_flag are
        excluded here, so a problem is only ever reported once per row.
      - nonstandard_contract_size_flag: contractSize isn't the standard
        100-shares-per-contract. This happens for "adjusted" contracts
        surviving certain corporate actions (special dividends,
        spin-offs, some splits) that keep trading alongside new standard
        contracts at the same strikes/expiration. It matters because
        filter_no_arbitrage's bounds (S - K*e^(-rT) <= C <= S) implicitly
        assume a 100-share multiplier -- an adjusted contract can look
        like a bogus arbitrage violation when it's actually just priced
        for a different number of shares. If contractSize isn't present
        on the input at all, this flag is set to False for every row
        (can't be determined) rather than raising.

    Does not drop rows -- same philosophy as the rest of this module.
    """
    calls = calls.copy()

    calls["invalid_strike_flag"] = calls["strike"].isna() | (calls["strike"] <= 0)
    calls["duplicate_strike_flag"] = (
        calls["strike"].duplicated(keep=False) & ~calls["invalid_strike_flag"]
    )

    if "contractSize" in calls.columns:
        calls["nonstandard_contract_size_flag"] = calls["contractSize"] != expected_contract_size
    else:
        calls["nonstandard_contract_size_flag"] = False

    return calls
