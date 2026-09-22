import math
import numpy as np
import yfinance as yf
import pandas as pd
import pandas_market_calendars as mcal
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

def fetch_option_chain(ticker_symbol: str, expiration: str = None):
    """
    Fetches underlying price and option chain for one ticker/expiration.
    If expiration is None, use the nearest available expiration.
    
    Returns: (spot_price: float, calls: pd.DataFrame, puts: pd.DataFrame, expiration: str)
    """
    ticker = yf.Ticker(ticker_symbol) 

    # --- 1. Spot price ---
    hist = ticker.history(period="1d")
    if hist.empty:
        raise ValueError(f"No price history returned for '{ticker_symbol}' — check the ticker symbol.")
    spot_price = hist["Close"].iloc[-1]

    # --- 2. Expiration date ---
    if expiration is None:
        if not ticker.options:
            raise ValueError(f"'{ticker_symbol}' has no listed option expirations.")
        expiration = ticker.options[0]  # Finds nearest expiration (the list is sorted)

    # --- 3. Option chain ---
    chain = ticker.option_chain(expiration) 
    calls = chain.calls 
    puts = chain.puts



    return spot_price, calls, puts, expiration 


#For each ticker:
# Divisor turns raw Yahoo quote into a decimal annual rate 
#"discount": signals to use T-bill discount rate (^IRX)
#"bey": signals to use bond-equivalent yield and semiannual compounding (^FVX, ^TNX, ^TYX)


TREASURY_RATE_CONFIG = {
    "^IRX": (100, "discount"),   # 13-week T-bill, quoted directly in percent (e.g. 5.25 -> 5.25%)
    "^FVX": (100, "bey"),        # 5-year yield, quoted directly in percent (verified against Treasury.gov 2026-09-11)
    "^TNX": (100, "bey"),        # 10-year yield, quoted directly in percent
    "^TYX": (100, "bey"),        # 30-year yield, quoted directly in percent
}

# ^IRX tracks the 13-week (91-day) T-bill; needed to convert its discount
# rate into a continuously-compounded rate.
IRX_DAYS_TO_MATURITY = 91 

#Note for IRX_DAYS_TO_MATURITY
#The IRX_DAYS_TO_MATURITY is an approximation as yfinance doesn't expose the exact maturity date.
# The range for this maturity can fluctuate by +-9 in the most extreme cases with the largest normal fluctuation
#being +-7 days. Given the math behind the discount rate, even a swing of +-7 moves the resulting
# continuously-compounded rate by only a few basis points. This difference is negligible for BS pricing.


def fetch_risk_free_rate(ticker_symbol: str = "^IRX") -> float:
    """
    Fetches an annualized, continuously-compounded risk-free rate from yfinance's
    treasury ticker.

    The default choice is ^IRX (13-week T-bill).
    Only tickers in TREASURY_RATE_CONFIG are supported as they 
    have different quoting conventions causing them to need
    different conversions to continuous compounding.
    """
    if ticker_symbol not in TREASURY_RATE_CONFIG:
        raise ValueError(
            f"Unsupported risk-free rate ticker '{ticker_symbol}'. "
            f"Supported tickers: {sorted(TREASURY_RATE_CONFIG)}"
        )

    divisor, convention = TREASURY_RATE_CONFIG[ticker_symbol]

    ticker = yf.Ticker(ticker_symbol)

    hist = ticker.history(period="1d")
    if hist.empty:
        raise ValueError(f"No price history returned for '{ticker_symbol}' — check the ticker symbol.")
    raw_quote = hist["Close"].iloc[-1]

    annual_rate = raw_quote / divisor

    if convention == "discount":
        # Bank-discount rate -> price fraction -> continuously-compounded
        # rate over the bill's own maturity, annualized on actual/365.
        price_fraction = 1 - annual_rate * (IRX_DAYS_TO_MATURITY / 360)
        risk_free_rate = -math.log(price_fraction) / (IRX_DAYS_TO_MATURITY / 365)
    else:
        # Semiannual-compounding bond-equivalent yield -> continuous.
        risk_free_rate = 2 * math.log(1 + annual_rate / 2)

    return risk_free_rate



def time_to_expiration(expiration: str, stock_exchange: str, reference_time=None) -> float:
    """
    Takes an expiration date and a stock exchange calendar and calculates
    the fraction of a trading year remaining until expiration.

    Uses mcal to filter out weekends, holidays, and early closes.


    reference_time defaults to now (New York time). If you want
    to do deterministic testing pass a fixed value. Naive values are
    read as New York time and aware ones are converted to it. 
    """
    stock_exchange_loc = mcal.get_calendar(stock_exchange)
    tz = ZoneInfo("America/New_York")

    if reference_time is None:
        reference_time = datetime.now(tz)
    now = pd.Timestamp(reference_time)
    now = now.tz_localize(tz) if now.tzinfo is None else now.tz_convert(tz)

    exp_date = date.fromisoformat(expiration)

    if exp_date < now.date():
        return 0.0

    # Look up today's actual session instead of assuming a fixed
    # 9:30-16:00 window. This returns nothing for weekends/holidays
    # and the correct hours for early-close days.
    today_schedule = stock_exchange_loc.schedule(start_date=now.date(), end_date=now.date())

    if today_schedule.empty:
        fraction_of_today = 0.0
    else:
        market_open = today_schedule["market_open"].iloc[0].to_pydatetime()
        market_close = today_schedule["market_close"].iloc[0].to_pydatetime()

        if now < market_open:
            fraction_of_today = 1.0
        elif now > market_close:
            fraction_of_today = 0.0
        else:
            fraction_of_today = (market_close - now) / (market_close - market_open)

    # Schedule from tomorrow through expiration -- today is handled separately
    # via fraction_of_today so it isn't double-counted as a full day.
    tomorrow = now.date() + timedelta(days=1)

    if tomorrow <= exp_date:
        schedule = stock_exchange_loc.schedule(start_date = tomorrow, end_date = exp_date)
        remaining_full_days = len(schedule.index)
    else: 
        remaining_full_days = 0
    

    trading_days = remaining_full_days + fraction_of_today 


    trading_days_per_year_schedule = stock_exchange_loc.schedule(
        start_date=date(now.year, 1, 1),
        end_date=date(now.year + 1, 1, 1)
    )

    trading_days_per_year = len(trading_days_per_year_schedule.index)

    T = trading_days / trading_days_per_year

    return T


def trading_days_since(dates: pd.Series, stock_exchange: str, reference_time=None) -> pd.Series:
    """
    For each date in 'dates', returns the number of trading sessions 
    that have occured after that trading day till today. 
    
    If the date inputed is today the function returns 0, if the 
    date has no trading sessions inbetween date and today
    (e.g. yesterday) the function returns 1. If there are 
    full trading sessions between the date and today the function 
    returns 2+.

    This function is similar to time_to_expiration but works
    in the opposite direction. 

    Note: this function does not account for the exact time
    of today's session (e.g. doesn't account for hours elapsed
    in todays session).

    NaT/missing dates return NaN; the caller decides how to treat that
    case (e.g. apply_staleness_flag treats it as maximally stale).
    """
    stock_exchange_loc = mcal.get_calendar(stock_exchange)
    tz = ZoneInfo("America/New_York")

    if reference_time is None:
        reference_time = datetime.now(tz)
    reference_time = pd.Timestamp(reference_time)
    reference_time = (
        reference_time.tz_localize(tz) if reference_time.tzinfo is None
        else reference_time.tz_convert(tz)
    )
    today = reference_time.date()

    dates = pd.to_datetime(dates, utc=True)

    def sessions_elapsed(last_trade):
        if pd.isna(last_trade):
            return np.nan
        last_trade_date = last_trade.tz_convert(tz).date()
        if last_trade_date >= today:
            return 0
        schedule = stock_exchange_loc.schedule(
            start_date=last_trade_date + timedelta(days=1), end_date=today
        )
        return len(schedule.index)

    return dates.apply(sessions_elapsed)


def historical_volatility(ticker_symbol: str, period: str = "1y") -> float:
    """
    Computes the annualized realized (historical) volatility for a ticker
    using its daily price movements. 
    
    Note: This sigma is independent of every option's price in the chain 
    so it can be used to price a chain without circularly reusing a 
    contract's own price against itself.

    period is a yfinance history period string (e.g "1y"). The function
    annualizes daily log-return volatility using the standard 
    252-trading-days-per-year-convention.

    Note: The function returns a single flat number. This means
    the function can't capture volatility smile/skew seen in
    real option chains. Thus be cautious when using this result
    as feeding this one sigma into add_theoretical_price for 
    every strike will misprice that skew. To get a per-strike 
    sigma instead, use each contract's own add_implied_volatility()
    result directly, or fit a curve across those same results with
    fit_volatility_smile()/smile_iv() (src/pricing/smile.py) to also
    cover strikes add_implied_volatility couldn't solve cleanly.
    It should be noted that Neither plugs directly into 
    add_theoretical_price, though. 

    Another note: period has no awareness of T. It's a static lookback 
    regardless of what expiration you're pricing. This is important to
    consider as volatility clusters in time so the default of "1y"
    can run into issues for a short-dated option. 

    Uses the zero-mean RMS convention (sqrt(mean(log_returns**2))), 
    not sample standard deviation, since daily drift is assumed to
    be ~0 over short horizons. The two usually differ only slightly
    but sample std increasingly understates true dispersion the stronger
    a stock's sustained drift is relative to its daily volatility. 
    """
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period=period)
    if len(hist) < 2:
        raise ValueError(
            f"Not enough price history for '{ticker_symbol}' over period={period!r} "
            f"to compute historical volatility."
        )

    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    daily_volatility = np.sqrt(np.mean(log_returns ** 2))

    return daily_volatility * math.sqrt(252)

