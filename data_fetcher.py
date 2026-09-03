"""
data_fetcher.py
---------------
Responsible for one thing only: downloading clean price data.
This is your "data layer" — the rest of the app never touches Twelve Data
directly, it just calls these functions.

Migrated off yfinance to Twelve Data (Sept 2026) — see twelvedata_utils.py
for why, and for the transport-layer retry/rate-limit/not-found handling
this module builds on. Function signatures and return shapes below are
unchanged from the yfinance version: every call site (api.py) works as-is.

^GSPC (Yahoo's S&P 500 index symbol) has no free-tier Twelve Data
equivalent — the raw index is paywalled behind a paid plan there. The
benchmark default below is SPY (the S&P 500 ETF) instead: same vendor as
everything else, no literal-index precision lost that matters for a
"portfolio vs. the market" comparison chart.
"""

import pandas as pd
from datetime import datetime, timedelta

from cache import cached
from twelvedata_utils import (
    time_series_batch, TwelveDataRateLimitError, TwelveDataAuthError, TickerNotFoundError,
)

DAY = 60 * 60 * 24

# A rate-limited or otherwise failed fetch can still return a small/partial
# result rather than raising — below this many rows there isn't enough
# data for a real analysis, so don't cache it: better to retry on the next
# request than serve that failure for a full day.
MIN_ROWS_TO_CACHE = 5


# ─── Batch response → DataFrame ───────────────────────────────────────────────

def _prices_from_batch(batch: dict, tickers: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Turn a twelvedata_utils.time_series_batch() result into a closing-price
    DataFrame, splitting any failures into (not_found, ambiguous):

      - not_found: Twelve Data explicitly returned a 404 "not found" for
        this symbol — real evidence the ticker itself is bad.
      - ambiguous: anything else (a 429, an unrecognised error, or a
        200 with zero rows) — Twelve Data didn't say the symbol is
        invalid, so this isn't proof of that; treated as a retryable
        fetch problem instead. See twelvedata_utils.py's module docstring.
    """
    series = {}
    not_found, ambiguous = [], []

    for t in tickers:
        entry = batch.get(t)
        if entry is None or entry.get("status") != "ok":
            message = (entry or {}).get("message", "") or ""
            code = (entry or {}).get("code")
            if code == 404 and "not found" in message.lower():
                not_found.append(t)
            else:
                ambiguous.append(t)
            continue

        values = entry.get("values") or []
        if not values:
            # Twelve Data reported success but returned zero rows — real,
            # if rare (e.g. no trading data anywhere in the requested
            # window). Twelve Data didn't say "not found" here, so this
            # isn't confirmed-invalid — bucket with the ambiguous cases.
            ambiguous.append(t)
            continue

        series[t] = pd.Series(
            {pd.Timestamp(v["datetime"]): float(v["close"]) for v in values},
            name=t,
        )

    prices = pd.concat(series.values(), axis=1).sort_index() if series else pd.DataFrame()
    return prices, not_found, ambiguous


# ─── Main function ────────────────────────────────────────────────────────────

@cached(ttl=DAY, prefix="fetch_closing_prices", should_cache=lambda df: len(df) >= MIN_ROWS_TO_CACHE)
def fetch_closing_prices(
    tickers: list[str],
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Download daily adjusted (split + dividend) closing prices for a list of
    tickers via Twelve Data.

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

    batch = time_series_batch(tickers, start_date=start_date, end_date=end_date)
    prices, not_found, ambiguous = _prices_from_batch(batch, tickers)

    if not_found or ambiguous:
        # Any ambiguity at all — a symbol Twelve Data didn't explicitly
        # confirm as invalid — gets treated as a retryable fetch problem,
        # not a bad ticker. Only symbols Twelve Data explicitly confirmed
        # don't exist get reported as not-found, and only once every
        # failed symbol got that same explicit confirmation.
        if ambiguous:
            raise TwelveDataRateLimitError(ambiguous + not_found)
        raise TickerNotFoundError(not_found)

    # Drop any rows where ALL tickers have NaN (e.g. market holidays).
    # Keep rows where at least one ticker has data.
    prices = prices.dropna(how="all")

    return prices


# ─── Validation helper ────────────────────────────────────────────────────────

def validate_tickers(tickers: list[str]) -> tuple[list[str], list[str]]:
    """
    Check which tickers are valid before fetching a full date range.
    Returns two lists: (valid_tickers, invalid_tickers).

    How it works: fetches just the last 7 days of data for every ticker in
    a single batched Twelve Data call (not one call per ticker — every
    call here spends real, shared daily quota, unlike yfinance's
    effectively-unlimited free access).

    A whole-request failure (rate limit, misconfigured key) can't confirm
    anything about any individual ticker, so — matching this function's
    pre-migration behaviour — every ticker lands in `invalid` rather than
    silently reporting nothing. There's no message field in this
    function's (valid, invalid) contract to separate "still rate-limited"
    from "not found"; that accurate distinction lives in
    fetch_closing_prices, which is what actually surfaces an error
    message to the user.
    """
    tickers = [t.strip().upper() for t in tickers]

    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        batch = time_series_batch(tickers, start_date=start, end_date=end)
    except (TwelveDataRateLimitError, TwelveDataAuthError):
        return [], tickers

    valid, invalid = [], []
    for t in tickers:
        entry = batch.get(t)
        if entry and entry.get("status") == "ok" and entry.get("values"):
            valid.append(t)
        else:
            invalid.append(t)

    return valid, invalid


# ─── Convenience wrapper ──────────────────────────────────────────────────────

@cached(ttl=DAY, prefix="fetch_with_benchmark", should_cache=lambda result: len(result[0]) >= MIN_ROWS_TO_CACHE)
def fetch_with_benchmark(
    tickers: list[str],
    start_date: str,
    end_date: str | None = None,
    benchmark: str = "SPY",   # S&P 500 proxy — see module docstring
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
