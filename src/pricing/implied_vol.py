import math

from scipy.optimize import brentq

from src.pricing.black_scholes import BS_call_price, BS_put_price, delta, vega

# Upper bound for the bisection fallback's search bracket -- 500% annualized
# vol comfortably covers single-name equities and most crypto-adjacent names.
MAX_SIGMA = 5.0
MIN_SIGMA = 1e-6

# Duarte, Jones & Wang (2024, JF) flag options with |delta| below this as
# too deep OTM/near-expiry for their IV to be trustworthy. 
# See is_low_confidence for more explanation
LOW_CONFIDENCE_DELTA = 0.15


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    initial_guess: float = 0.5,
    price_tol: float = 1e-6,
    sigma_tol: float = 1e-6,
    max_iterations: int = 50,
) -> float:
    """
    Solves for the Black-Scholes sigma that reproduces the given price, using
    the contract's observed S/K/T/r.

    Tries Newton-Raphson first (faster, uses vega), falling back to
    bisection (scipy's brentq) if Newton fails to converge. This is because
    Newton can stall near-zero vega or step into non-positive sigma for deep
    ITM/OTM or near-expiry contracts, where bisection is slower but
    always converges given a valid bracket.

    Raises ValueError if T <= 0 (volatility is undefined at/after
    expiration) or if 'price' falls outside this contract's no-arbitrage
    bounds, since no sigma could reproduce a price outside them.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if T <= 0:
        raise ValueError("T must be positive. Implied volatility is undefined at/after expiration.")

    price_fn = BS_call_price if option_type == "call" else BS_put_price
    discounted_strike = K * math.exp(-r * T)

    if option_type == "call":
        lower_bound = max(S - discounted_strike, 0.0)
        upper_bound = S
    else:
        lower_bound = max(discounted_strike - S, 0.0)
        upper_bound = discounted_strike

    if not (lower_bound - price_tol <= price <= upper_bound + price_tol):
        raise ValueError(
            f"price {price} is outside this contract's no-arbitrage bounds "
            f"[{lower_bound}, {upper_bound}]: no sigma can reproduce it."
        )

    sigma = initial_guess
    for _ in range(max_iterations):
        diff = price_fn(S, K, T, r, sigma) - price

        if abs(diff) < price_tol:
            return sigma

        # NOTE: low vega makes the sigma we get back less trustworthy. When vega is this small
        # (deep OTM/near-expiry contracts) a lot of different sigmas give
        # almost the same price, so it is hard to get an accurate calculation. 
        # Duarte, Jones & Wang (2024, JF) deal with this by
        # filtering out delta<0.15 options. is_low_confidence() below
        # checks for this same thing, but it's opt-in. Callers have to
        # call it themselves after getting a sigma back.
        v = vega(S, K, T, r, sigma)
        if v < 1e-8:
            break  # vega too small meaning Newton step would blow up, fall back to bisection

        sigma -= diff / v

        if sigma <= 0:
            break  # stepped into an invalid region, fall back to bisection

    def error(sigma):
        return price_fn(S, K, T, r, sigma) - price

    # Newton doesn't respect MAX_SIGMA, as it can find a root above 5.0 on
    # its own. This fallback only runs when Newton failed early (e.g. a
    # near-zero-vega starting point), so if the true IV also happens to be
    # above MAX_SIGMA, brentq's bracket won't contain a sign change and it
    # raises a raw, confusing ValueError. Catch that and re-raise with a
    # clear, project-specific message instead.
    try:
        return brentq(error, MIN_SIGMA, MAX_SIGMA, xtol=sigma_tol)
    except ValueError as e:
        raise ValueError(
            f"implied_volatility failed to converge for price={price} "
            f"(S={S}, K={K}, T={T}, r={r}, option_type={option_type!r}): "
            f"the true implied volatility may lie outside the solver's "
            f"search range [{MIN_SIGMA}, {MAX_SIGMA}]."
        ) from e


def is_low_confidence(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> bool:
    """
    Flags a solved implied volatility as low-confidence when the
    contract's |delta| < LOW_CONFIDENCE_DELTA, following Duarte, Jones &
    Wang (2024, JF). Deep OTM/near-expiry contracts have such small
    vega that many different sigmas fit the observed price almost
    equally well making it hard to trust the calculated sigma.

    Meant to be checked by the caller after solving

    # Note: It also flags the ITM side now (|delta| > 1 - LOW_CONFIDENCE_DELTA),
    # as deep ITM can run into similar issues as deep OTM

    """
    d = abs(delta(S, K, T, r, sigma, option_type))
    return d < LOW_CONFIDENCE_DELTA or d > 1 - LOW_CONFIDENCE_DELTA
