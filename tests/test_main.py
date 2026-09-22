import sys

import pandas as pd
import pytest

import main as cli
from src.pricing.black_scholes import BS_call_price, BS_put_price


# ---------------------------------------------------------------------------
# main.py's own logic: the pipeline orchestration, the two report builders,
# and the end-to-end run. Only the four network calls are stubbed -- the real
# cleaning/pricing stack runs underneath, so these exercise main.py's wiring
# rather than re-testing the layers below it.
# ---------------------------------------------------------------------------

S, T, R, SIGMA = 100.0, 0.5, 0.05, 0.25
STRIKES = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0]


def _raw_chain(option_type, spot=S, strikes=STRIKES):
    """A raw yfinance-shaped chain priced from a known sigma."""
    price_fn = BS_call_price if option_type == "call" else BS_put_price
    mids = [price_fn(spot, K, T, R, SIGMA) for K in strikes]
    now = pd.Timestamp.now(tz="UTC")  # a trade "now" is never stale
    return pd.DataFrame({
        "strike": strikes,
        "bid": [m - 0.02 for m in mids],
        "ask": [m + 0.02 for m in mids],
        "lastPrice": mids,
        "lastTradeDate": [now] * len(strikes),
        # varied so the 10th-percentile liquidity thresholds are meaningful
        "volume": [10 * (i + 1) for i in range(len(strikes))],
        "openInterest": [100 * (i + 1) for i in range(len(strikes))],
        "contractSize": ["REGULAR"] * len(strikes),
    })


def _stub_market(monkeypatch, quoted_spot=S, chain_spot=S, strikes=STRIKES):
    """
    Replaces every network call main() makes with deterministic data.

    quoted_spot and chain_spot are separable on purpose: the chain is priced
    off chain_spot while fetch returns quoted_spot, which is exactly the
    endpoint mismatch implied_spot exists to correct.
    """
    calls = _raw_chain("call", spot=chain_spot, strikes=strikes)
    puts = _raw_chain("put", spot=chain_spot, strikes=strikes)
    monkeypatch.setattr(
        cli, "fetch_option_chain",
        lambda ticker, expiration=None: (quoted_spot, calls.copy(), puts.copy(), "2026-12-18"),
    )
    monkeypatch.setattr(cli, "fetch_risk_free_rate", lambda: R)
    monkeypatch.setattr(cli, "time_to_expiration", lambda expiration, exchange: T)
    monkeypatch.setattr(cli, "historical_volatility", lambda ticker, period="1y": SIGMA)


def _run(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    cli.main()
    return capsys.readouterr()


# ---------------------------------------------------------------------------
# clean_chain
# ---------------------------------------------------------------------------

def test_clean_chain_produces_every_column_the_rest_of_main_reads():
    # summarize, print_smile_fit and the plotting helpers all index into the
    # result by name, so a renamed or dropped column breaks the CLI without
    # breaking any single cleaning function's own tests.
    chain = cli.clean_chain(_raw_chain("call"), S, R, T, SIGMA, option_type="call")
    for column in (
        "strike", "mid_price", "no_arb_violation", "illiquid_flag",
        "implied_vol", "low_confidence_iv_flag", "bs_price", "price_diff",
    ):
        assert column in chain.columns


def test_clean_chain_recovers_the_sigma_the_chain_was_priced_with():
    # End-to-end check that the stages compose: quotes in, correct IV out.
    chain = cli.clean_chain(_raw_chain("call"), S, R, T, SIGMA, option_type="call")
    solved = chain["implied_vol"].notna()
    assert solved.all()
    assert chain.loc[solved, "implied_vol"].median() == pytest.approx(SIGMA, abs=1e-4)


def test_clean_chain_returns_rows_sorted_by_strike():
    # filter_strike_monotonicity/convexity sort internally, so the rows come
    # back reordered relative to the input -- the plots depend on this.
    shuffled = _raw_chain("call").sample(frac=1, random_state=0)
    chain = cli.clean_chain(shuffled, S, R, T, SIGMA, option_type="call")
    assert list(chain["strike"]) == sorted(chain["strike"])


def test_clean_chain_works_for_puts():
    chain = cli.clean_chain(_raw_chain("put"), S, R, T, SIGMA, option_type="put")
    solved = chain["implied_vol"].notna()
    assert solved.any()
    assert chain.loc[solved, "implied_vol"].median() == pytest.approx(SIGMA, abs=1e-4)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_summarize_reports_the_contract_count_and_solved_rate():
    chain = cli.clean_chain(_raw_chain("call"), S, R, T, SIGMA, option_type="call")
    out = cli.summarize("Calls", chain)
    assert f"Calls ({len(chain)} contracts)" in out
    assert "IV solved: 100.0%" in out
    assert "median implied vol: 0.2500" in out


def test_summarize_omits_the_iv_lines_when_nothing_solved():
    # The `if solved.any()` branch: a chain with no solvable row must still
    # produce a report rather than dividing by zero on an empty selection.
    chain = cli.clean_chain(_raw_chain("call"), S, R, T, SIGMA, option_type="call")
    # nullable Float64, matching what add_implied_volatility actually emits
    chain["implied_vol"] = pd.array([pd.NA] * len(chain), dtype="Float64")
    out = cli.summarize("Calls", chain)
    assert "IV solved: 0.0%" in out
    assert "median implied vol" not in out
    assert "low-confidence among solved" not in out


# ---------------------------------------------------------------------------
# print_smile_fit
# ---------------------------------------------------------------------------

def test_print_smile_fit_returns_coefficients_for_a_clean_chain():
    chain = cli.clean_chain(_raw_chain("call"), S, R, T, SIGMA, option_type="call")
    coeffs = cli.print_smile_fit("Calls", chain, S)
    assert coeffs is not None
    assert len(coeffs) == 3


def test_print_smile_fit_returns_none_and_says_skipped_when_too_few_clean_points(capsys):
    chain = cli.clean_chain(_raw_chain("call"), S, R, T, SIGMA, option_type="call")
    chain["low_confidence_iv_flag"] = True  # nothing trustworthy left
    assert cli.print_smile_fit("Calls", chain, S) is None
    assert "skipped" in capsys.readouterr().out


def test_print_smile_fit_treats_a_missing_confidence_flag_as_low_confidence():
    # The fillna(True) guard: a row whose flag is NA was never successfully
    # solved, so it must not be counted as a trustworthy smile point.
    # Built as nullable boolean, which is the dtype add_implied_volatility
    # actually produces -- assigning bare pd.NA would make it object dtype
    # and exercise a path the pipeline never generates.
    chain = cli.clean_chain(_raw_chain("call"), S, R, T, SIGMA, option_type="call")
    chain["low_confidence_iv_flag"] = pd.array([pd.NA] * len(chain), dtype="boolean")
    assert cli.print_smile_fit("Calls", chain, S) is None


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def test_main_runs_end_to_end_and_reports_the_headline_sections(monkeypatch, capsys):
    _stub_market(monkeypatch)
    out = _run(monkeypatch, capsys, "FAKE").out
    assert "FAKE -- spot=$" in out
    assert "--- Calls (9 contracts) ---" in out
    assert "--- Puts (9 contracts) ---" in out
    assert "--- Put-Call Parity (9 matched strikes checked) ---" in out
    assert "Calls smile fit" in out
    assert "Puts smile fit" in out


def test_main_corrects_a_stale_quoted_spot_using_parity(monkeypatch, capsys):
    # The chain is quoted against 100 while fetch reports 101 -- the exact
    # endpoint mismatch. main() should price off the recovered 100 and say so.
    _stub_market(monkeypatch, quoted_spot=101.0, chain_spot=100.0)
    out = _run(monkeypatch, capsys, "FAKE").out
    assert "spot quoted $101.00, implied by parity $100.00 (-1.00)" in out
    assert "FAKE -- spot=$100.00" in out


def test_main_quoted_spot_flag_skips_the_inference(monkeypatch, capsys):
    _stub_market(monkeypatch, quoted_spot=101.0, chain_spot=100.0)
    out = _run(monkeypatch, capsys, "FAKE", "--quoted-spot").out
    assert "FAKE -- spot=$101.00" in out
    assert "implied by parity" not in out


def test_main_falls_back_to_the_quoted_spot_when_too_few_strikes_are_checkable(monkeypatch, capsys):
    # implied_spot needs 3 checkable strikes; with 2 it raises and main()
    # must warn and carry on with the quoted spot rather than dying.
    _stub_market(monkeypatch, quoted_spot=101.0, chain_spot=100.0, strikes=[95.0, 105.0])
    result = _run(monkeypatch, capsys, "FAKE")
    assert "Warning: using the quoted spot" in result.err
    assert "FAKE -- spot=$101.00" in result.out
    assert "implied by parity" not in result.out


def test_main_exits_nonzero_when_the_fetch_fails(monkeypatch, capsys):
    def boom(ticker, expiration=None):
        raise ValueError("No price history returned for 'NOPE'")

    _stub_market(monkeypatch)
    monkeypatch.setattr(cli, "fetch_option_chain", boom)
    monkeypatch.setattr(sys, "argv", ["main.py", "NOPE"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 1
    assert "No price history returned" in capsys.readouterr().err


def test_main_save_writes_the_three_csv_tables(monkeypatch, capsys, tmp_path):
    _stub_market(monkeypatch)
    out = _run(monkeypatch, capsys, "FAKE", "--save", str(tmp_path)).out
    written = sorted(p.name for p in tmp_path.glob("*.csv"))
    assert written == ["FAKE_calls.csv", "FAKE_parity.csv", "FAKE_puts.csv"]
    assert len(pd.read_csv(tmp_path / "FAKE_calls.csv")) == len(STRIKES)
    assert "Saved full tables to" in out


def test_main_plot_writes_smile_and_comparison_charts(monkeypatch, capsys, tmp_path):
    _stub_market(monkeypatch)
    out = _run(monkeypatch, capsys, "FAKE", "--plot", str(tmp_path)).out
    written = sorted(p.name for p in tmp_path.glob("*.png"))
    assert written == [
        "FAKE_calls_price_comparison.png",
        "FAKE_calls_smile.png",
        "FAKE_puts_price_comparison.png",
        "FAKE_puts_smile.png",
    ]
    assert "Saved charts to" in out


def test_main_plot_skips_the_smile_chart_when_no_fit_was_possible(monkeypatch, capsys, tmp_path):
    # print_smile_fit returns None when there aren't 3 clean points, and the
    # plotting loop has to tolerate that rather than passing None as coeffs.
    _stub_market(monkeypatch, strikes=[95.0, 100.0])
    out = _run(monkeypatch, capsys, "FAKE", "--plot", str(tmp_path)).out
    written = sorted(p.name for p in tmp_path.glob("*.png"))
    assert written == ["FAKE_calls_price_comparison.png", "FAKE_puts_price_comparison.png"]
    assert "Saved charts to" in out
