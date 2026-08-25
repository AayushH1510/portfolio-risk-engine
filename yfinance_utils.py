"""
yfinance_utils.py
------------------
Shared handling for Yahoo Finance rate limiting. Every direct yfinance call
in this codebase (yf.download, Ticker.info, Ticker.history) can raise
YFRateLimitError, and Render's IP gets hit harder than local dev — so every
call site needs the same retry-then-clean-error treatment. Centralised here
so new call sites (there are three today: fetch_closing_prices,
/stock-detail/{ticker}, /api/fundamentals) reuse it instead of reimplementing
their own retry loop.
"""

import time

from yfinance.exceptions import YFRateLimitError

RATE_LIMIT_RETRIES = 1
RATE_LIMIT_BACKOFF_SECONDS = 2


class YFinanceRateLimitError(Exception):
    """
    Raised when a yfinance call still fails after retrying — either it kept
    raising YFRateLimitError, or (for multi-ticker calls) the result came
    back missing data for one or more of the requested tickers. Both usually
    mean Yahoo Finance is rate-limiting this server's IP.
    """
    def __init__(self, failed_tickers: list[str]):
        self.failed_tickers = failed_tickers
        super().__init__(
            f"Unable to fetch data for: {', '.join(failed_tickers)}. "
            "Yahoo Finance may be rate-limiting this server."
        )


def with_yfinance_retry(fetch_fn, tickers: list[str]):
    """
    Call fetch_fn() — a zero-arg callable wrapping one yfinance call —
    retrying once after a short backoff if it raises YFRateLimitError.
    Yahoo's rate limits are transient, so a single retry often succeeds.

    Raises YFinanceRateLimitError(tickers) if it still fails after retrying.
    `tickers` is just the ticker(s) involved in this call, used to build a
    clean error message — pass a single-element list for a one-ticker call.
    """
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            return fetch_fn()
        except YFRateLimitError:
            if attempt < RATE_LIMIT_RETRIES:
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                continue
            raise YFinanceRateLimitError(tickers)
