import numpy as np
import pandas as pd


def fit_volatility_smile(strikes: pd.Series, implied_vols: pd.Series, S: float) -> np.ndarray:
    """
    Fits a smooth IV curve across strikes for one expiration, using a
    quadratic in log-moneyness: IV(K) = a + b*k + c*k^2, where
    k = log(K/S). This is the standard shortcut for capturing a
    chain's smile/skew shape without a full parametric model
    (e.g. SVI) which may be implemented later but is currently
    beyond the scope of this project. 
    Note: c>0 means smile-shaped (OMT/ITM trade above ATM),
    b != 0 means skewed (tilted toward one side).

    Rows with a missing strike or. implied_vol are dropped 
    before fitting. Another thing to note is that callers
    should filter out untrustworthy points first. (This could
    be done with low_confidence_ivflag rows from add_implied)volatility)
    before calling this function. This is important as noisy or
    untrustworhty IVs could pull the fit off. 

    Riase ValueError if fewer than 3 valid points remain (the 
    quadratic fit would be meaningless at that point). 

    Fits via ordinary least squares (numpy.polyfit). The returned
    coefficients [c, b, a] follow numpy.polyfit's own convention
    (highest degree first), and can be passed directly into smile_iv()
    to evaluate the fitted curve at any strike.
    """
    valid = strikes.notna() & implied_vols.notna()
    strikes = strikes[valid]
    implied_vols = implied_vols[valid]

    if len(strikes) < 3:
        raise ValueError(
            f"Need at least 3 valid (strike, implied_vol) points to fit a "
            f"quadratic smile curve, got {len(strikes)}."
        )

    log_moneyness = np.log(strikes.to_numpy(dtype=float) / S)
    return np.polyfit(log_moneyness, implied_vols.to_numpy(dtype=float), deg=2)


def smile_iv(strike, S: float, coeffs: np.ndarray):
    """
    Evaluates a fitted smile curve (from fit_volatility_smile) at a
    given strike. This function works for any strike, including
    ones with no clean quote of their own. strike accepts either a
    single value or a whole array/Series, since both log and polyval
    apply elementwise on their own.
    """
    log_moneyness = np.log(np.asarray(strike, dtype=float) / S)
    return np.polyval(coeffs, log_moneyness)
