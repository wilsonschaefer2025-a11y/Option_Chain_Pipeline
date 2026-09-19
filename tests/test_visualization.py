import pandas as pd
import pytest

from src.pricing.black_scholes import BS_call_price
from src.pricing.smile import fit_volatility_smile
from src.visualization import plot_smile, plot_price_comparison


# ---------------------------------------------------------------------------
# Smoke tests only -- checking a plot "looks right" isn't meaningfully
# testable the way the rest of this project's correctness checks are.
# These just confirm the functions run without error on real-shaped data
# and produce an actual populated Axes, not that the chart is correct.
# ---------------------------------------------------------------------------

def _sample_chain(S=100, K_range=(80, 120, 5)):
    strikes = list(range(*K_range))
    true_sigma = 0.25
    prices = [BS_call_price(S, K, 0.5, 0.05, true_sigma) for K in strikes]
    chain = pd.DataFrame({
        "strike": strikes,
        "mid_price": prices,
        "implied_vol": [true_sigma] * len(strikes),
        "low_confidence_iv_flag": [False] * len(strikes),
        "bs_price": prices,
    })
    return chain, S


def test_plot_smile_runs_without_error_and_populates_axes():
    chain, S = _sample_chain()
    coeffs = fit_volatility_smile(chain["strike"], chain["implied_vol"], S)
    ax = plot_smile(chain, S, coeffs)
    assert len(ax.collections) > 0  # the scatter points were actually drawn
    assert len(ax.lines) > 0        # the fitted curve line was actually drawn


def test_plot_smile_handles_a_mix_of_confidence_levels():
    chain, S = _sample_chain()
    chain.loc[0, "low_confidence_iv_flag"] = True
    coeffs = fit_volatility_smile(chain["strike"], chain["implied_vol"], S)
    ax = plot_smile(chain, S, coeffs)
    assert len(ax.collections) == 2  # trustworthy + low-confidence scatter groups


def test_plot_price_comparison_runs_without_error_and_populates_axes():
    chain, _ = _sample_chain()
    ax = plot_price_comparison(chain)
    assert len(ax.lines) == 2  # market mid_price line + theoretical bs_price line


def test_plot_price_comparison_skips_rows_with_no_theoretical_price():
    chain, _ = _sample_chain()
    chain.loc[0, "bs_price"] = pd.NA
    ax = plot_price_comparison(chain)
    # Should still run cleanly and plot the remaining priced rows.
    assert len(ax.lines[0].get_xdata()) == len(chain) - 1
