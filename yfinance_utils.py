"""
yfinance_utils.py
------------------
Shared handling for Yahoo Finance rate limiting. Every direct yfinance call
in this codebase (yf.download, Ticker.info, Ticker.history) can raise
YFRateLimitError, and Render's IP gets hit harder than local dev — so every
call site needs the same retry-then-clean-error treatment. Centralised here
so new call sites reuse it instead of reimplementing their own retry loop.

Only fetch_closing_prices (data_fetcher.py) actually uses this today — it
backs /api/analyse, /api/stress-test, and everything else that needs
historical price series. /stock-detail/{ticker} and /api/fundamentals are
Finnhub-backed (stock_detail_route.py, api.py's _fetch_ticker_fundamentals)
and already do their own independent not-found/rate-limit distinction —
they never touch yfinance or this module.

Two distinct failure modes, two distinct exceptions — don't conflate them:
  - YFinanceRateLimitError: yfinance itself raised YFRateLimitError (still,
    after a retry). Yahoo is genuinely rate-limiting this server's IP.
  - TickerNotFoundError: the fetch call completed with no YFRateLimitError
    at all, but came back with no data for one or more tickers. That's a bad
    symbol (e.g. "SPX" instead of "^GSPC"), not rate limiting — Yahoo
    doesn't error on an unknown ticker, it just returns nothing for it.
"""

import time

from yfinance.exceptions import YFRateLimitError

RATE_LIMIT_RETRIES = 1
RATE_LIMIT_BACKOFF_SECONDS = 2


class YFinanceRateLimitError(Exception):
    """
    Raised when a yfinance call still raises YFRateLimitError after
    retrying. This means Yahoo Finance is rate-limiting this server's IP —
    see TickerNotFoundError for the other, unrelated way a fetch can come
    back empty.
    """
    def __init__(self, failed_tickers: list[str]):
        self.failed_tickers = failed_tickers
        super().__init__(
            f"Unable to fetch data for: {', '.join(failed_tickers)}. "
            "Yahoo Finance may be rate-limiting this server."
        )


class TickerNotFoundError(Exception):
    """
    Raised when a yfinance call completes normally — no YFRateLimitError was
    ever thrown — but the result has no data for one or more of the
    requested tickers. yf.download() doesn't error on an invalid/unknown
    symbol, it just silently omits it from the result, so this is how that
    shows up: a bad ticker, not a rate limit.
    """
    def __init__(self, missing_tickers: list[str]):
        self.missing_tickers = missing_tickers
        if len(missing_tickers) == 1:
            message = (
                f"The ticker '{missing_tickers[0]}' could not be found. "
                "Please check the symbol and try again."
            )
        else:
            tickers_str = ", ".join(f"'{t}'" for t in missing_tickers)
            message = (
                f"These tickers could not be found: {tickers_str}. "
                "Please check the symbols and try again."
            )
        super().__init__(message)


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
