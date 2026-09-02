"""
data_fetcher.py
---------------
Responsible for one thing only: downloading clean price data.
This is your "data layer" — the rest of the app never touches yfinance directly,
it just calls these functions.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

from cache import cached
from yfinance_utils import with_yfinance_retry, YFinanceRateLimitError, TickerNotFoundError

DAY = 60 * 60 * 24

# A rate-limited or otherwise failed yfinance call can still return a
# DataFrame instead of raising — just a near-empty one. Below this many rows
# there isn't enough data for a real analysis, so don't cache it: better to
# retry on the next request than serve that failure for a full day.
MIN_ROWS_TO_CACHE = 5


# ─── Main function ────────────────────────────────────────────────────────────

@cached(ttl=DAY, prefix="fetch_closing_prices", should_cache=lambda df: len(df) >= MIN_ROWS_TO_CACHE)
def fetch_closing_prices(
    tickers: list[str],
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Download daily adjusted closing prices for a list of tickers.

    Parameters
    ----------
    tickers    : e.g. ["AAPL", "MSFT", "TSLA"]
    start_date : "YYYY-MM-DD"  e.g. "2022-01-01"
    end_date   : "YYYY-MM-DD"  — defaults to today if not provided

    Returns
    -------
    A DataFrame where:
      - each column is a ticker
      - each row is a trading day
      - values are the adjusted closing price in USD

    Example output:
              AAPL    MSFT    TSLA
    Date
    2022-01-03  182.01  336.32  399.93
    2022-01-04  179.70  329.91  383.20
    ...
    """

    # Default end date to today
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    # Normalise tickers — strip whitespace, force uppercase
    # so "aapl " and "AAPL" both work
    tickers = [t.strip().upper() for t in tickers]

    print(f"Fetching data for: {tickers}")
    print(f"Period: {start_date} -> {end_date}")

    # yf.download() is the core call.
    # auto_adjust=True means we get adjusted prices (corrected for
    # stock splits and dividends) — always use this for return calculations.
    raw = with_yfinance_retry(
        lambda: yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,   # suppress the yfinance download bar
        ),
        tickers,
    )

    # yf.download returns a multi-level column index like:
    #   ("Close", "AAPL"), ("Close", "MSFT"), ("Open", "AAPL"), ...
    # We only want the "Close" level.
    prices = raw["Close"]

    # If only one ticker was passed, yfinance returns a plain Series.
    # Wrap it back into a DataFrame so the rest of the code always
    # gets the same shape regardless of input size.
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    # A ticker can come back missing entirely, or present but all-NaN,
    # without yf.download() ever raising — no YFRateLimitError was thrown
    # anywhere above, so this isn't a rate-limit case. It means the symbol
    # itself doesn't exist or isn't fetchable (e.g. "SPX" instead of
    # "^GSPC") — catch that here instead of letting it crash downstream in
    # the maths layer, and instead of misreporting it as rate limiting.
    failed_tickers = [
        t for t in tickers
        if t not in prices.columns or prices[t].isna().all()
    ]
    if failed_tickers:
        # Every requested ticker failing together, or "^GSPC" specifically
        # failing, can't be "the user typed a bad symbol": ^GSPC is a
        # hardcoded constant fetch_with_benchmark appends itself, never user
        # input, so it failing at all is proof this is a fetch-level failure
        # (most likely yf.download() silently swallowing a rate-limit/IP
        # block instead of raising YFRateLimitError — retried above, but not
        # every failure mode raises that specific exception). Same logic for
        # every ticker in the request failing at once: vanishingly unlikely
        # to all be typos simultaneously. Don't misreport either as a bad
        # ticker — surface it as the retryable, honest rate-limit error.
        if "^GSPC" in failed_tickers or len(failed_tickers) == len(tickers):
            raise YFinanceRateLimitError(failed_tickers)
        raise TickerNotFoundError(failed_tickers)

    # Drop any rows where ALL tickers have NaN (e.g. market holidays).
    # Keep rows where at least one ticker has data.
    prices = prices.dropna(how="all")

    return prices


# ─── Validation helper ────────────────────────────────────────────────────────

def validate_tickers(tickers: list[str]) -> tuple[list[str], list[str]]:
    """
    Check which tickers are valid before fetching a full date range.
    Returns two lists: (valid_tickers, invalid_tickers).

    How it works: fetches just the last 5 days of data.
    If a ticker returns empty data, it's invalid (delisted, misspelled, etc.)

    Goes through with_yfinance_retry so a transient rate-limit blip doesn't
    get a genuinely valid ticker wrongly bucketed as invalid — matches
    fetch_closing_prices' retry-then-distinguish handling. There's no
    message field in this function's (valid, invalid) contract to report
    "still rate-limited after retry" separately from "not found", so both
    still land in `invalid` here; the accurate distinction lives in
    fetch_closing_prices, which is what actually surfaces an error message
    to the user.
    """
    valid, invalid = [], []

    # Use a short recent window — fast to fetch, just for validation
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")

    for ticker in tickers:
        ticker = ticker.strip().upper()
        try:
            test = with_yfinance_retry(
                lambda: yf.download(ticker, start=start, end=end,
                                     auto_adjust=True, progress=False),
                [ticker],
            )
            if test.empty:
                invalid.append(ticker)
            else:
                valid.append(ticker)
        except Exception:
            invalid.append(ticker)

    return valid, invalid


# ─── Convenience wrapper ──────────────────────────────────────────────────────

@cached(ttl=DAY, prefix="fetch_with_benchmark", should_cache=lambda result: len(result[0]) >= MIN_ROWS_TO_CACHE)
def fetch_with_benchmark(
    tickers: list[str],
    start_date: str,
    end_date: str | None = None,
    benchmark: str = "^GSPC",   # S&P 500
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch portfolio prices AND benchmark prices in one call.
    Returns (portfolio_prices, benchmark_prices) as separate DataFrames.

    Having the benchmark separate keeps your stats functions clean —
    they only ever receive portfolio data, and you compare against
    benchmark data at the chart/display layer.
    """
    all_tickers = tickers + [benchmark]
    all_prices = fetch_closing_prices(all_tickers, start_date, end_date)

    # Split back out
    portfolio_prices = all_prices[tickers]
    benchmark_prices = all_prices[[benchmark]]

    return portfolio_prices, benchmark_prices


# ─── Quick test — run this file directly to check it works ───────────────────

if __name__ == "__main__":

    # Test 1: basic fetch
    prices = fetch_closing_prices(
        tickers=["AAPL", "MSFT", "GOOGL"],
        start_date="2023-01-01",
    )
    print("\n--- Closing prices (first 5 rows) ---")
    print(prices.head())
    print(f"\nShape: {prices.shape}  ({prices.shape[0]} trading days x {prices.shape[1]} tickers)")

    # Test 2: validation
    print("\n--- Ticker validation ---")
    valid, invalid = validate_tickers(["AAPL", "FAKEXYZ", "TSLA"])
    print(f"Valid:   {valid}")
    print(f"Invalid: {invalid}")

    # Test 3: fetch with benchmark
    print("\n--- Portfolio + benchmark ---")
    port, bench = fetch_with_benchmark(
        tickers=["AAPL", "MSFT"],
        start_date="2023-01-01",
    )
    print("Portfolio columns:", port.columns.tolist())
    print("Benchmark columns:", bench.columns.tolist())