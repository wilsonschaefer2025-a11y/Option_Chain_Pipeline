# Option_Chain_Pipeline

An options analytics pipeline: fetches live option chains, flags data-quality issues, prices contracts using Black-Scholes, solves implied volatility with a confidence check, fits a volatility smile and cross-validates calls against puts.

Also offers visualizations and reports for results. 
Only needs one CLI command to operate.

## What it does

Given a ticker, the pipeline:

1. Fetches the live option chain, spot price, and risk-free rate (yfinance)
2. Infers the chain's true spot price from put-call parity, correcting for
   non-simultaneous quotes between the chain and the spot feed
3. Flags (not drops) data-quality issues: no-arbitrage violations, strike
   monotonicity/convexity breaks, wide spreads, stale or illiquid quotes,
   invalid/duplicate strikes
4. Solves implied volatility per contract, flagging results that fall in a
   mathematically unreliable region (deep ITM/OTM)
5. Prices each contract against realized volatility for comparison
6. Fits a volatility smile to the clean, trustworthy points
7. Cross-checks calls against puts via put-call parity
8. Reports a summary and (optionally) saves charts and full data tables

## Data and methodology

The pricing formulas are Black-Scholes. The majority of the 
work of this pipeline is fetching data and cleaning that 
data.  

### Data 

All data comes from yfinace

Inputs and Sources:

Spot "S": Latest close from `history(period="1d")`. Replaced with 
implied_spot (spot inferred from the parity).

Option Chain: `option_chain(expiration)` pulls the nearest expiry 
by default

Risk-free rate `r`: `^IRX` (13-week T-bill) is the default. ^FVX`/`^TNX`/`^TYX`
are supported by fetch_risk_free_rate but are not appart of the CLI  

Time to expiry `T`: Trading sessions left (today's remaining fraction + full sessions to expiry) ÷ sessions in the year. This is done using pandas_market_calendars

Realized vol `σ`: Zero-mean RMS of daily log returns × √252 over `--period`


A couple things to note: 
-`contractSize` is a string in yfinance ("REGULAR)
and not an integer (e.g. 100)
-`bid`/`ask` can be NaN. In that case midprice cannot be 
calculated and the price falls back to `lastPrice`. 

### Spot Correction 

It was discovered that the spot and chain came from different endpoints and are 
not sampled together. This showed up in data as the bulk of the parity
residuals having the same sign. Other causes were ruled out (dividends and 
interest rates) as the gap did not grow from 1 to 30 day expirations as expected
with these two cases. To solve this issue an implied_spot calculation was implemented.

`implied_spot` inverts put-call parity (`S = C - P + K·e^(-rT)`) at each clean
matched strike and takes the median. The result replaces the quoted spot
everywhere downstream. `--quoted-spot` disables this.

### No-arbitrage screens

Contracts are valued at the mid price, with a tolerance of half the spread. The
tolerance is zero for missing or crossed quotes. This logic works sense 
`mid ± half-spread` is equivalent to `bid`/`ask` meaning each screen flags
a contract only when a trade could be made at the quoted prices. 

Screen and Flags:

Lower bound: `ask < max(S - Ke^(-rT), 0)` 
Upper bound: `bid > S`
Monotonicity:  `bid(K2) > ask(K1)`, `K1 < K2`
Convexity: `bid(K2) > w·ask(K1) + (1-w)·ask(K3)`
Put-call parity:  `bid(C) - ask(P) > S - Ke^(-rT)` or `ask(C) - bid(P) < S - Ke^(-rT)`

### Liquidity Flags

`illiquid_flag` combines these checks, and each one is also kept as its own
column:

- Open interest or volume in the chain's bottom 10%
- Spread in the top 10%, measured beyond CBOE's minimum tick ($0.01 under $3,
  $0.05 at or above), so tick-bound cheap contracts aren't penalized
- No live quote, or a crossed quote
- No trade since before the previous session

Missing or non-positive strikes, duplicate strikes, and non-standard contract
sizes are flagged separately.

### Implied volatility and smile

The IV is solved using Newton-Raphson. If this fails it falls back to 
Brent's method. Unpriceable contracts and those violating arbitrage 
are skipped. Using the logic established by Duarte, Jones and Wang (2024 *JF*),
a solved IV is flagged low-confidence when `|Δ| < 0.15` or `|Δ| > 0.85`,
since the vega there is too small to pin down σ.

The theoretical prices use the realized volatility and not the contract's own
IV. This is used to avoid circular comparisons. The smile, `IV = a + b·k + c·k²`
with `k = ln(K/S)`, is fitted by least squares to high-confidence IVs only.


## Installation

```bash
git clone https://github.com/wilsonschaefer2025-a11y/Option_Chain_Pipeline.git
cd Option_Chain_Pipeline
pip install -r requirements.txt
```

## Usage

```bash
python main.py AAPL
python main.py TSLA --expiration 2026-11-20 --period 3mo
python main.py AAPL --save output/ --plot charts/
python main.py --help
```

| Flag | Description |
|---|---|
| `--expiration` | Expiration date (YYYY-MM-DD); defaults to nearest |
| `--exchange` | Trading calendar to use (default: NYSE) |
| `--period` | Lookback period for realized volatility (default: 1y) |
| `--quoted-spot` | Use yfinance's quoted spot as-is instead of the parity-inferred one |
| `--save DIR` | Save full cleaned calls/puts/parity tables as CSV |
| `--plot DIR` | Save smile and price-comparison charts as PNGs |

## Example output

```
AAPL -- spot=$338.83  expiration=2026-11-20  T=0.1753yr  risk-free rate=4.06%  realized vol(1y)=24.69%
  spot quoted $338.98, implied by parity $338.83 (-0.15) -- non-simultaneous quotes, plus PV(dividends) for any ex-date before expiration

--- Calls (83 contracts) ---
  no-arbitrage violations: 3.6%
  illiquid: 47.0%
  IV solved: 94.0% (78/83)
  low-confidence among solved: 79.5%
  median implied vol: 0.3428

--- Puts (61 contracts) ---
  no-arbitrage violations: 0.0%
  illiquid: 44.3%
  IV solved: 100.0% (61/61)
  low-confidence among solved: 72.1%
  median implied vol: 0.3888

--- Put-Call Parity (58 matched strikes checked) ---
  violations: 1.7%
  residual spread: median -0.000, std 0.431

Calls smile fit (from 16 clean points): a=0.2524 b=-0.1996 c=0.5062
Puts smile fit (from 17 clean points): a=0.2518 b=-0.1304 c=1.5037
```

After the spot correction the residual *median* is ~0 by construction, so the *std*
is the number worth reading.

With `--plot`, the same run also produces the fitted smile:

![AAPL implied volatility smile](docs/AAPL_calls_smile.png)

The colour split is the confidence flag. Trustworthy points (blue) cluster near the
money; low-confidence points (orange) fan into the wings, where vega collapses. The
curve is fitted to the blue points only, then extrapolated across all strikes.

## Project structure

```
src/data/fetch_options.py       # live data ingestion (yfinance)
src/data/cleaning_options.py    # data-quality flagging pipeline
src/pricing/black_scholes.py    # pricing formulas & Greeks
src/pricing/implied_vol.py      # IV solver + confidence flagging
src/pricing/smile.py            # volatility smile fitting
src/visualization.py            # chart generation
main.py                         # CLI entry point
tests/                          # 215 tests
```

**Couple important notes**

The pipeline only flags and does not drop data. This means
none of the data is removed by the cleaning functions. Instead 
each check writes a boolean column. This allows the user to decide
which flags to drop from the data and which to keep. 

Time-to-expiration is measured in trading sessions, using `pandas_market_calendars`. 
It also gives partial credit for the current  session and for early closes (e.g. the day after Thanksgiving). 


## Running tests

```bash
pytest -q          # 215 passed
```

Correctness is checked against independent references rather than against the code
itself: Greeks by finite difference, textbook identities (put-call parity,
`Δcall - Δput = 1`), known-sigma round trips, and a 3,000-case randomized equivalence
check on the arbitrage screens. Coverage is 99% across `src/` and `main.py`.

## Known limitations

- No dividend yield (`q=0`) in the pricing model
- European-style pricing assumption vs. real American-style equity options
- `implied_spot` returns `S - PV(D)` on a dividend payer, not the traded share price.
  That is the right input for a `q=0` model (the escrowed-dividend adjustment), but the
  number sits below the share price and the reported gap includes `PV(D)`
- American early exercise turns the parity equality into an inequality band, so
  `implied_spot` reads slightly low, most so at deep ITM strikes. The median mitigates
  this, since early-exercise premium concentrates in the wings
- Uniform chain-wide parity violations are absorbed into the inferred spot rather than
  flagged, since a stale spot and a genuine chain-wide mispricing are indistinguishable
  from inside the chain. `--quoted-spot` gives the uncorrected view
- Single-expiration scope, with no multi-expiration term structure or full vol
  surface yet
- The confidence flag is delta-only, so it doesn't catch the separate ATM-near-expiry
  low-vega case
- `historical_volatility` is a single flat number and can't capture smile/skew on its
  own; that's what the fitted smile is for
