import argparse
import os
import sys

from src.data.fetch_options import (
    fetch_option_chain,
    fetch_risk_free_rate,
    time_to_expiration,
    historical_volatility,
)
from src.data.cleaning_options import (
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
from src.pricing.smile import fit_volatility_smile
from src.visualization import plot_smile, plot_price_comparison

#Note: Currently built entirely by Claude. The main thing that's running through here 
#is implied_spot has some stuff that more heavily relies on it here and 
# 
# Looking at revamping myself later but my main focus
#of the project was the backend of the pipeline and less so the presentation. 


def clean_chain(chain, S, r, T, sigma, option_type):
    chain = filter_no_arbitrage(chain, S, r, T, option_type=option_type)
    chain = filter_strike_monotonicity(chain, option_type=option_type)
    chain = filter_strike_convexity(chain)
    chain = filter_liquidity(chain)
    chain = filter_contract_sanity(chain)
    chain = add_implied_volatility(chain, S, r, T, option_type=option_type)
    chain = add_theoretical_price(chain, S, r, T, sigma, option_type=option_type)
    return chain


def summarize(name, chain):
    n = len(chain)
    solved = chain["implied_vol"].notna()

    lines = [
        f"--- {name} ({n} contracts) ---",
        f"  no-arbitrage violations: {chain['no_arb_violation'].mean():.1%}",
        f"  illiquid: {chain['illiquid_flag'].mean():.1%}",
        f"  IV solved: {solved.mean():.1%} ({solved.sum()}/{n})",
    ]
    if solved.any():
        low_conf = chain.loc[solved, "low_confidence_iv_flag"]
        lines.append(f"  low-confidence among solved: {low_conf.mean():.1%}")
        lines.append(f"  median implied vol: {chain.loc[solved, 'implied_vol'].median():.4f}")
    return "\n".join(lines)


def print_smile_fit(name, chain, S):
    """Returns the fitted coefficients (or None if skipped) so main() can reuse them for plotting."""
    clean = chain["implied_vol"].notna() & ~chain["low_confidence_iv_flag"].fillna(True)
    if clean.sum() < 3:
        print(f"{name} smile fit: skipped ({clean.sum()} clean points, need at least 3)")
        return None
    coeffs = fit_volatility_smile(chain.loc[clean, "strike"], chain.loc[clean, "implied_vol"], S)
    c, b, a = coeffs
    print(f"{name} smile fit (from {clean.sum()} clean points): a={a:.4f} b={b:.4f} c={c:.4f}")
    return coeffs


def main():
    parser = argparse.ArgumentParser(
        description="Fetch, clean, and price a live option chain end-to-end."
    )
    parser.add_argument("ticker", help="Underlying ticker symbol, e.g. AAPL")
    parser.add_argument("--expiration", default=None, help="Expiration date (YYYY-MM-DD); defaults to nearest")
    parser.add_argument("--exchange", default="NYSE", help="Trading calendar to use (default: NYSE)")
    parser.add_argument("--period", default="1y", help="Lookback period for historical_volatility (default: 1y)")
    parser.add_argument(
        "--quoted-spot", action="store_true",
        help="Use yfinance's quoted spot as-is instead of the spot inferred from put-call parity",
    )
    parser.add_argument("--save", metavar="DIR", default=None, help="Save full cleaned calls/puts/parity tables as CSV to this directory")
    parser.add_argument("--plot", metavar="DIR", default=None, help="Save smile and price-comparison charts as PNGs to this directory")
    args = parser.parse_args()

    try:
        S, calls, puts, expiration = fetch_option_chain(args.ticker, expiration=args.expiration)
        r = fetch_risk_free_rate()
        T = time_to_expiration(expiration, args.exchange)
        sigma = historical_volatility(args.ticker, period=args.period)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # The chain and the spot quote come from different endpoints and aren't
    # sampled together, so the chain is often priced against a spot that
    # differs from the one fetched beside it. Recover the market's own spot
    # from put-call parity before anything downstream uses S -- every
    # implied vol depends on it, not just the parity check.
    quoted_S = S
    spot_gap = None
    if not args.quoted_spot:
        try:
            S = implied_spot(filter_put_call_parity(calls, puts, quoted_S, r, T), quoted_S)
            spot_gap = S - quoted_S
        except (ValueError, KeyError) as e:
            print(f"Warning: using the quoted spot -- {e}", file=sys.stderr)

    print(
        f"{args.ticker} -- spot=${S:.2f}  expiration={expiration}  T={T:.4f}yr  "
        f"risk-free rate={r:.2%}  realized vol({args.period})={sigma:.2%}"
    )
    if spot_gap is not None:
        print(
            f"  spot quoted ${quoted_S:.2f}, implied by parity ${S:.2f} "
            f"({spot_gap:+.2f}) -- non-simultaneous quotes, plus PV(dividends) "
            f"for any ex-date before expiration"
        )
    print()

    calls = clean_chain(calls, S, r, T, sigma, option_type="call")
    puts = clean_chain(puts, S, r, T, sigma, option_type="put")

    print(summarize("Calls", calls))
    print()
    print(summarize("Puts", puts))
    print()

    parity = filter_put_call_parity(calls, puts, S, r, T)
    checked = parity["parity_violation"].notna()
    print(f"--- Put-Call Parity ({checked.sum()} matched strikes checked) ---")
    if checked.any():
        residuals = parity.loc[checked, "parity_residual"].astype(float)
        print(f"  violations: {parity.loc[checked, 'parity_violation'].mean():.1%}")
        # With the spot inferred from these same residuals their median is ~0
        # by construction, so the spread around it is the real signal: how
        # far individual strikes stray from the chain's own consensus.
        print(f"  residual spread: median {residuals.median():+.3f}, std {residuals.std():.3f}")
    print()

    call_coeffs = print_smile_fit("Calls", calls, S)
    put_coeffs = print_smile_fit("Puts", puts, S)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        calls.to_csv(os.path.join(args.save, f"{args.ticker}_calls.csv"), index=False)
        puts.to_csv(os.path.join(args.save, f"{args.ticker}_puts.csv"), index=False)
        parity.to_csv(os.path.join(args.save, f"{args.ticker}_parity.csv"), index=False)
        print(f"\nSaved full tables to {args.save}/")

    if args.plot:
        os.makedirs(args.plot, exist_ok=True)
        for name, chain, coeffs in [("calls", calls, call_coeffs), ("puts", puts, put_coeffs)]:
            plot_price_comparison(chain).figure.savefig(
                os.path.join(args.plot, f"{args.ticker}_{name}_price_comparison.png")
            )
            if coeffs is not None:
                plot_smile(chain, S, coeffs).figure.savefig(
                    os.path.join(args.plot, f"{args.ticker}_{name}_smile.png")
                )
        print(f"\nSaved charts to {args.plot}/")


if __name__ == "__main__":
    main()
