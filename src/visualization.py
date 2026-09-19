import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- this is a CLI tool, don't assume a display exists
import matplotlib.pyplot as plt
import numpy as np

from src.pricing.smile import smile_iv


#Visulizations done entirely by Claude. Will look into adding stuff 
#and revamping this in the future

def plot_smile(chain, S: float, coeffs, ax=None):
    """
    Plots implied vol against strike: a scatter of the chain's own
    solved implied_vol points (from add_implied_volatility), split into
    trustworthy vs. low-confidence markers, plus the fitted curve (from
    fit_volatility_smile) as a smooth line over the same strike range.

    Only visualizes data that already exists elsewhere in the pipeline
    -- doesn't compute anything new about the chain itself.
    """
    if ax is None:
        _, ax = plt.subplots()

    solved = chain["implied_vol"].notna()
    low_conf = solved & chain["low_confidence_iv_flag"].fillna(True)
    trustworthy = solved & ~chain["low_confidence_iv_flag"].fillna(True)

    ax.scatter(
        chain.loc[trustworthy, "strike"], chain.loc[trustworthy, "implied_vol"],
        color="tab:blue", label="solved IV (trustworthy)", zorder=3,
    )
    ax.scatter(
        chain.loc[low_conf, "strike"], chain.loc[low_conf, "implied_vol"],
        color="tab:orange", marker="x", label="solved IV (low-confidence)", zorder=3,
    )

    strike_range = np.linspace(chain["strike"].min(), chain["strike"].max(), 200)
    ax.plot(strike_range, smile_iv(strike_range, S, coeffs), color="black", linestyle="--", label="fitted smile")

    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility")
    ax.legend()
    return ax


def plot_price_comparison(chain, ax=None):
    """
    Plots market mid_price against theoretical bs_price (from
    add_theoretical_price) across strike, for whichever rows have both.
    Only visualizes existing columns -- no new computation.
    """
    if ax is None:
        _, ax = plt.subplots()

    priced = chain["bs_price"].notna()

    ax.plot(chain.loc[priced, "strike"], chain.loc[priced, "mid_price"], marker="o", label="market mid_price")
    ax.plot(chain.loc[priced, "strike"], chain.loc[priced, "bs_price"], marker="o", label="theoretical bs_price")

    ax.set_xlabel("Strike")
    ax.set_ylabel("Price")
    ax.legend()
    return ax
