"""
data_fetcher.py
---------------
Responsible for one thing only: downloading clean price data.
This is your "data layer" — the rest of the app never touches yfinance directly,
it just calls these functions.
"""

import time

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from yfinance.exceptions import YFRateLimitError

from cache import cached

DAY = 60 * 60 * 24

# yf.download() can raise YFRateLimitError outright, or — more insidiously —
# return normally with one ticker's column silently missing or all-NaN.
# Render's IP gets rate-limited by Yahoo more aggressively than local dev,
# so this needs its own retry + a clean, specific error.
RATE_LIMIT_RETRIES  = 1
RATE_LIMIT_BACKOFF_SECONDS = 2


class YFinanceRateLimitError(Exception):
    """
    Raised when yfinance can't get real data for one or more requested
    tickers — either yf.download() raised YFRateLimitError directly, or it
    returned a DataFrame missing (or all-NaN for) one of the tickers we
    asked for. Both usually mean Yahoo Finance is rate-limiting this
    server's IP.
    """
    def __init__(self, failed_tickers: list[str]):
        self.failed_tickers = failed_tickers
        super().__init__(
            f"Unable to fetch data for: {', '.join(failed_tickers)}. "
            "Yahoo Finance may be rate-limiting this server."
        )

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
    print(f"Period: {start_date} → {end_date}")

    # yf.download() is the core call.
    # auto_adjust=True means we get adjusted prices (corrected for
    # stock splits and dividends) — always use this for return calculations.
    #
    # Yahoo rate-limits are transient — retry once after a short backoff
    # before giving up, since a single retry often succeeds.
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            raw = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,   # suppress the yfinance download bar
            )
            break
        except YFRateLimitError:
            if attempt < RATE_LIMIT_RETRIES:
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                continue
            raise YFinanceRateLimitError(tickers)

    # yf.download returns a multi-level column index like:
    #   ("Close", "AAPL"), ("Close", "MSFT"), ("Open", "AAPL"), ...
    # We only want the "Close" level.
    prices = raw["Close"]

    # If only one ticker was passed, yfinance returns a plain Series.
    # Wrap it back into a DataFrame so the rest of the code always
    # gets the same shape regardless of input size.
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    # A rate-limited ticker can come back missing entirely, or present but
    # all-NaN, without yf.download() ever raising. Catch that here instead
    # of letting it crash downstream in the maths layer.
    failed_tickers = [
        t for t in tickers
        if t not in prices.columns or prices[t].isna().all()
    ]
    if failed_tickers:
        raise YFinanceRateLimitError(failed_tickers)

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
    """
    valid, invalid = [], []

    # Use a short recent window — fast to fetch, just for validation
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")

    for ticker in tickers:
        ticker = ticker.strip().upper()
        try:
            test = yf.download(ticker, start=start, end=end,
                               auto_adjust=True, progress=False)
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
    print(f"\nShape: {prices.shape}  ({prices.shape[0]} trading days × {prices.shape[1]} tickers)")

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