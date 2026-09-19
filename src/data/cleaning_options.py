import numpy as np
import pandas as pd

from src.data.fetch_options import trading_days_since
from src.pricing.implied_vol import implied_volatility, is_low_confidence
from src.pricing.black_scholes import BS_price_series


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


def filter_no_arbitrage(calls: pd.DataFrame, S: float, r: float, T: float, option_type: str = "call") -> pd.DataFrame:
    """
    Flags no-arbitrage violations, allowing a tolerance (calculated in last function)
    to absorb some market noise instead of flagging economincally meaningless
    small violations. Rows with no live quote get zero tolerance.

    T is the time to expiration in years taken from time_to_expiration.
    This must be computed before passing into the function.
    This function does not calculate T or grab it from somewhere else.

    option_type selects which no-arbitrage bounds apply: calls use
    max(S - K*e^(-rT), 0) <= C <= S; puts use max(K*e^(-rT) - S, 0)
    <= P <= K*e^(-rT). 
    
    Note: The "calls" parameter name is kept for consistency
    as this was originally built for calls but this also works
    with a puts DataFrame via option_type="put".
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    calls = calls.copy()
    calls = add_mid_price(calls)
    calls = apply_spread_tolerance(calls)

    discounted_strike = calls["strike"] * np.exp(-r * T)

    if option_type == "call":
        lower_bound = np.maximum(S - discounted_strike, 0)
        upper_bound = S
    else:
        lower_bound = np.maximum(discounted_strike - S, 0)
        upper_bound = discounted_strike

    calls["lower_bound"] = lower_bound
    calls["upper_bound"] = upper_bound


    calls["no_arb_violation"] = (
        (calls["mid_price"] < lower_bound - calls["tolerance"]) |
        (calls["mid_price"] > upper_bound + calls["tolerance"])
    )

    return calls

def filter_strike_monotonicity(calls: pd.DataFrame, option_type: str = "call") -> pd.DataFrame:
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

    option_type controls which direction counts as a violation. The
    call case above holds: K1<K2 => C(K1) >= C(K2), non-increasing in
    strike. Puts are the opposite: P is non-decreasing in strike
    (K1<K2 => P(K1) <= P(K2)), since the right to sell at a higher
    strike is worth more, not less.
    """

    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    calls = calls.copy()

    if "mid_price" not in calls.columns or "tolerance" not in calls.columns:
        calls = add_mid_price(calls)
        calls = apply_spread_tolerance(calls)

    calls = calls.sort_values("strike").reset_index(drop=True)

    next_price = calls["mid_price"].shift(-1)
    next_tolerance = calls["tolerance"].shift(-1)
    combined_tolerance = calls["tolerance"] + next_tolerance

    if option_type == "call":
        calls["monotonicity_violation"] = next_price > calls["mid_price"] + combined_tolerance
    else:
        calls["monotonicity_violation"] = next_price < calls["mid_price"] - combined_tolerance

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
    
    None of these are no-arbitrage signals, which is why they're consumed
    here rather than affecting filter_no_arbitrage's tolerance.
    
    Missing openInterest/volume (NaN) is treated as illiquid
    
    If wide_spread_flag/no_live_quote_flag/crossed_market_flag/
    stale_quote_flag are already present (e.g. this DataFrame came from
    filter_no_arbitrage), they're reused. Otherwise this function
    recomputes them.
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
        the chain. Rows already flagged invalid_strike_flag are
        excluded here, so a problem is only ever reported once per row.
      - nonstandard_contract_size_flag: contractSize isn't the standard
        100-shares-per-contract. This happens for "adjusted" contracts
        surviving certain corporate actions (special dividends,
        spin-offs, some splits) that keep trading alongside new standard
        contracts at the same strikes/expiration. It matters because
        filter_no_arbitrage's bounds (S - K*e^(-rT) <= C <= S) implicitly
        assume a 100-share multiplier. If contractSize isn't present
        on the input at all, this flag is set to False for every row
        as it can't be determined.

    Does not drop rows, just flags them.
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

def add_implied_volatility(calls: pd.DataFrame, S: float, r: float, T: float, option_type: str = "call") -> pd.DataFrame:
    """
    Adds implied_vol and low_confidence_iv_flag columns by solving each
    row's mid_price for its Black-Scholes implied volatility (see
    src/pricing/implied_vol.py).

    Skips rows already flagged unpriceable, invalid_strike_flag, or
    no_arb_violation. This is particularly important because unpriceable
    options won't automatically raise an error causing implied_volatility()
    to generate a misleading sigma. Skipped rows, and rows where 
    implied_volatility() raises get NA in both new columns. 

    Reuses mid_price (via add_mid_price if missing), so this also works
    standalone on a raw chain.

    T is the time to expiration in years,
    Note: It must be computed via time_to_expiration before calling this. 
   
    S/r are also constant across the whole chain 
    (one spot price, one risk-free rate), same as filter_no_arbitrage.
    """
    calls = calls.copy()

    if "mid_price" not in calls.columns:
        calls = add_mid_price(calls)

    skip = pd.Series(False, index=calls.index)
    for flag_col in ("unpriceable", "invalid_strike_flag", "no_arb_violation"):
        if flag_col in calls.columns:
            skip = skip | calls[flag_col]

    implied_vols = pd.array([pd.NA] * len(calls), dtype="Float64")
    low_confidence = pd.array([pd.NA] * len(calls), dtype="boolean")

    for pos in range(len(calls)):
        if skip.iloc[pos]:
            continue

        strike = calls["strike"].iloc[pos]
        price = calls["mid_price"].iloc[pos]

        try:
            sigma = implied_volatility(price, S, strike, T, r, option_type=option_type)
        except ValueError:
            continue

        implied_vols[pos] = sigma
        low_confidence[pos] = is_low_confidence(S, strike, T, r, sigma, option_type=option_type)

    calls["implied_vol"] = implied_vols
    calls["low_confidence_iv_flag"] = low_confidence

    return calls

def add_theoretical_price(calls: pd.DataFrame, S: float, r: float, T: float, sigma: float, option_type: str = "call") -> pd.DataFrame:
    """
    Adds a bs_price column (the Black-Scholes theoretical price) for each
    row, using an externally-supllied sigma (e.g. historical volatility()
    from fetch_options.py). Do not use the contract's own solved
    implied_vol as it will create circular logic. 

    Also adds price_diff = mid_price - bs_price (reuses mid_price, via
    add_mid_price if missing). If positive means the market is pricing
    this contract above the theoretical price and vice versa for
    negative. 

    Only skips rows flagged invalid_strike_flag. Unlike 
    add_implied_volatility, this function doesn't depend on
    a row's own price being trustworthy as sigma comes from 
    outside the chain so unpriceable/no_arb_violation rows 
    still get a theoritical price. If filter_contract_sanity
    hasn't been run (no invalid_strike_flag column), 
    a bad strike will raise instead of being skipped. 

    The actual Black-Scholes math is done by BS_price_series 
    function. This was done so BS_price_series
    stays a pure pricing-layer function without need
    for cleaning-specific flags or mid_price. It's also
    important as it avoids circular imports. 

    option_type/sigma are validated here unconditionally (not just
    inside BS_price_series) so a bad value still raises even if every
    row happens to be skipped and BS_price_series is never actually
    called.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    calls = calls.copy()

    if "mid_price" not in calls.columns:
        calls = add_mid_price(calls)

    skip = calls["invalid_strike_flag"] if "invalid_strike_flag" in calls.columns else pd.Series(False, index=calls.index)
    priceable = ~skip

    bs_prices = pd.array([pd.NA] * len(calls), dtype="Float64")
    if priceable.any():
        priced = BS_price_series(calls.loc[priceable, "strike"], S, r, T, sigma, option_type)
        bs_prices[priceable.to_numpy()] = pd.array(priced, dtype="Float64")

    calls["bs_price"] = bs_prices
    calls["price_diff"] = calls["mid_price"] - calls["bs_price"]

    return calls

def filter_put_call_parity(calls: pd.DataFrame, puts: pd.DataFrame, S: float, r: float, T: float) -> pd.DataFrame:
    """
    Cross-checks calls against puts at matching strikes using put-call
    parity: C - P == S - K*e^(-rT). Unlike every other filter in this
    module, this is a pure no-arbitrage identity -- it needs no
    volatility assumption at all, since it just compares the two sides
    of the market against each other directly rather than against a
    theoretical price.

    Reuses mid_price/tolerance on both calls and puts (via
    add_mid_price/apply_spread_tolerance if missing). Matches strikes
    present in both chains (inner join) -- a strike only listed as a
    call or only as a put can't be checked, and is silently excluded
    rather than flagged.

    Skips (NA, not False) rows where either side is already flagged
    unpriceable or invalid_strike_flag, if those columns are present --
    same reasoning as add_implied_volatility: an unpriceable/invalid
    quote makes the comparison meaningless, not informative, so it
    shouldn't read as "checked and clean."

    parity_violation flags |residual| beyond the combined tolerance of
    both sides (call tolerance + put tolerance) -- same "flag beyond
    combined tolerance" pattern as filter_strike_monotonicity/convexity.

    Returns a new DataFrame, one row per matched strike -- not calls or
    puts modified in place, since this produces new joint information
    rather than augmenting either existing chain on its own.
    """
    calls = calls.copy()
    puts = puts.copy()

    if "mid_price" not in calls.columns or "tolerance" not in calls.columns:
        calls = add_mid_price(calls)
        calls = apply_spread_tolerance(calls)
    if "mid_price" not in puts.columns or "tolerance" not in puts.columns:
        puts = add_mid_price(puts)
        puts = apply_spread_tolerance(puts)

    keep_cols = ["strike", "mid_price", "tolerance", "unpriceable", "invalid_strike_flag"]
    call_cols = [c for c in keep_cols if c in calls.columns]
    put_cols = [c for c in keep_cols if c in puts.columns]

    merged = pd.merge(calls[call_cols], puts[put_cols], on="strike", suffixes=("_call", "_put"))

    skip = pd.Series(False, index=merged.index)
    for flag_col in ("unpriceable", "invalid_strike_flag"):
        if f"{flag_col}_call" in merged.columns:
            skip = skip | merged[f"{flag_col}_call"]
        if f"{flag_col}_put" in merged.columns:
            skip = skip | merged[f"{flag_col}_put"]

    checkable = ~skip

    residuals = pd.array([pd.NA] * len(merged), dtype="Float64")
    violations = pd.array([pd.NA] * len(merged), dtype="boolean")

    if checkable.any():
        discounted_strike = merged.loc[checkable, "strike"] * np.exp(-r * T)
        parity_rhs = S - discounted_strike
        parity_lhs = merged.loc[checkable, "mid_price_call"] - merged.loc[checkable, "mid_price_put"]
        residual = parity_lhs - parity_rhs

        combined_tolerance = merged.loc[checkable, "tolerance_call"] + merged.loc[checkable, "tolerance_put"]
        violation = residual.abs() > combined_tolerance

        residuals[checkable.to_numpy()] = pd.array(residual, dtype="Float64")
        violations[checkable.to_numpy()] = pd.array(violation, dtype="boolean")

    merged["parity_residual"] = residuals
    merged["parity_violation"] = violations

    return merged
