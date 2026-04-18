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


# ─── Main function ────────────────────────────────────────────────────────────

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
    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,   # suppress the yfinance download bar
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