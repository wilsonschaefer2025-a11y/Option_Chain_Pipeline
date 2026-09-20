import math

import pandas as pd
from scipy.stats import norm


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def BS_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes price of a European call.
    Assumes no dividend yield.

    At/after expiration (T <= 0) returns the intrinsic value directly,
    since d1/d2 are undefined when T == 0.
    """
    if S <= 0 or K <= 0:
        raise ValueError(f"S and K must be positive, got S={S}, K={K}.")
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def BS_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes price of a European put (same conventions as BS_call_price).
    """
    if S <= 0 or K <= 0:
        raise ValueError(f"S and K must be positive, got S={S}, K={K}.")
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def BS_price_series(strikes: pd.Series, S: float, r: float, T: float, sigma: float, option_type: str = "call") -> pd.Series:
    """
    Vectorized Black-Scholes price across a Series of strikes, using one
    shared S/r/T/sigma for all of them.

    Validates option_type/sigma once up front rather than per element.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    price_fn = BS_call_price if option_type == "call" else BS_put_price
    return strikes.apply(lambda K: price_fn(S, K, T, r, sigma))


def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """
    Black-Scholes delta: d(price)/d(S).

    Used by implied_vol.py's is_low_confidence() to flag deep OTM/near-
    expiry results -- delta (not vega) is the standard moneyness-based
    measure the Duarte/Jones/Wang low-confidence threshold is defined on.

    At/after expiration (T <= 0) returns the intrinsic-value direction:
    1.0/-1.0 if in the money, 0.0 otherwise.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if S <= 0 or K <= 0:
        raise ValueError(f"S and K must be positive, got S={S}, K={K}.")
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes vega: d(price)/d(sigma). Identical for calls and puts.
    Used to calculate implied volatility.

    Returns 0.0 at/after expiration or for non-positive sigma, since
    price is flat with respect to sigma there.
    """
    if S <= 0 or K <= 0:
        raise ValueError(f"S and K must be positive, got S={S}, K={K}.")
    if T <= 0 or sigma <= 0:
        return 0.0

    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * math.sqrt(T)
