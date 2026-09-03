"""
twelvedata_utils.py
--------------------
Shared HTTP / error / retry handling for Twelve Data — the historical
price data vendor behind fetch_closing_prices and fetch_with_benchmark
(data_fetcher.py), which back /api/analyse, /api/stress-test, and
everything else needing historical price series.

This replaced yfinance (Sept 2026): Yahoo Finance's rate-limiting on
Render's datacenter IP caused repeated production incidents — a fetch
would silently come back empty for a subset of tickers, indistinguishable
from those tickers genuinely not existing, misreporting real, valid
tickers (GOOGL, AAPL, ...) as "not found".

Mirrors the role yfinance_utils.py used to play: this module only knows
about the transport layer (auth, rate limits, retries, daily-quota
visibility) — it has no idea what a "closing price" or a DataFrame is,
data_fetcher.py owns that. /stock-detail/{ticker} and /api/fundamentals
are unrelated — they're Finnhub-backed (stock_detail_route.py) and never
touch this module.

Twelve Data's free tier: 800 calls/day, 8 calls/minute — both hard caps,
shared across every user of this app (unlike yfinance, which had no
meaningful daily ceiling). One call = one symbol in a /time_series
request, regardless of how many rows come back, so a 5-ticker portfolio
+ benchmark costs 6 calls per analysis. See _track_call for the
visibility this module adds because of that.

Two distinct failure modes, two distinct exceptions — same split
yfinance_utils.py used to draw, just against Twelve Data's own signals
instead of yfinance's silence:
  - TwelveDataRateLimitError: the per-minute or per-day credit cap was
    hit (HTTP 429 — a whole request or one symbol within a batch), or
    Twelve Data returned something that isn't a definitive "bad symbol"
    signal. Default to this whenever there's genuine ambiguity: better
    to tell the user to retry a fine ticker than to wrongly blame their
    input.
  - TickerNotFoundError: Twelve Data explicitly said so (HTTP 404 with a
    "not found" message). Unlike yfinance, which never told us *why* a
    symbol came back empty, Twelve Data's error responses are specific
    enough to trust directly — no second-vendor (Finnhub) cross-check
    needed anymore, which is why that workaround is gone from
    data_fetcher.py as part of this migration.
  - TwelveDataAuthError: HTTP 401, or no API key configured at all — the
    key itself is being rejected, not a per-symbol problem. Every symbol
    in the request fails identically, so this is a service-level
    misconfiguration (TWELVEDATA_API_KEY missing/invalid), not a data
    issue — must never surface as "ticker not found" or "rate limited".
"""

import logging
import os
import time
from datetime import datetime, timezone

import requests

from cache import cache_incr

logger = logging.getLogger(__name__)

TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
TWELVEDATA_BASE = "https://api.twelvedata.com"

RATE_LIMIT_RETRIES = 1
RATE_LIMIT_BACKOFF_SECONDS = 8  # the per-minute cap resets on a rolling minute; a short wait often clears it

DAILY_CALL_LIMIT = 800
_CALL_COUNTER_TTL = 25 * 60 * 60  # a bit over a day, so a slow request near midnight UTC never reads a just-expired key


class TwelveDataRateLimitError(Exception):
    """
    Raised when Twelve Data's per-minute or per-day credit cap is hit, or a
    symbol failed for a reason that isn't a definitive bad-symbol signal.
    See module docstring — this is the "don't blame the user" default
    whenever Twelve Data's own response doesn't clearly say the ticker
    itself is invalid.
    """
    def __init__(self, failed_tickers: list[str]):
        self.failed_tickers = failed_tickers
        super().__init__(
            f"Unable to fetch data for: {', '.join(failed_tickers)}. "
            "Twelve Data's API quota may be temporarily exhausted."
        )


class TickerNotFoundError(Exception):
    """
    Raised only when Twelve Data explicitly confirmed a symbol doesn't
    exist (HTTP 404, "not found" message) — see module docstring.
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


class TwelveDataAuthError(Exception):
    """Raised on HTTP 401, or when TWELVEDATA_API_KEY isn't set at all."""
    pass


def _track_call(n_symbols: int) -> None:
    """
    Log-visible tracking of daily Twelve Data call volume — worth having
    here in a way it never was for yfinance, since the free tier's 800/day
    cap is a real, shared ceiling, not an "effectively unlimited but
    unreliable" one. Redis-backed so the count survives across requests;
    no-ops if Redis is unavailable (same fail-open contract as cache.py),
    since this is visibility, not something a request should ever fail on.
    """
    key = f"varense:twelvedata:calls:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    count = cache_incr(key, ttl=_CALL_COUNTER_TTL, by=n_symbols)
    if count is None:
        return
    if count >= DAILY_CALL_LIMIT:
        logger.error(
            "Twelve Data daily call volume: %d/%d — quota is likely exhausted for the rest of today (UTC)",
            count, DAILY_CALL_LIMIT,
        )
    elif count >= DAILY_CALL_LIMIT * 0.8:
        logger.warning("Twelve Data daily call volume: %d/%d — approaching the free-tier cap", count, DAILY_CALL_LIMIT)
    elif count % 50 < n_symbols:
        # Fires roughly every 50 calls without needing an exact multiple —
        # n_symbols can be >1 per call (a batched request), so an exact
        # `count % 50 == 0` check could skip past 50 entirely.
        logger.info("Twelve Data daily call volume: %d/%d", count, DAILY_CALL_LIMIT)


def with_twelvedata_retry(fetch_fn):
    """
    Call fetch_fn() — a zero-arg callable wrapping one Twelve Data request —
    retrying once after a short backoff if it raises TwelveDataRateLimitError.
    Mirrors yfinance_utils.with_yfinance_retry's contract.
    """
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            return fetch_fn()
        except TwelveDataRateLimitError:
            if attempt < RATE_LIMIT_RETRIES:
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                continue
            raise


def time_series_batch(symbols: list[str], start_date: str, end_date: str, interval: str = "1day") -> dict:
    """
    Fetch daily time series for one or more symbols in a single Twelve Data
    call — 1 credit per symbol regardless of row count, so batching
    multiple symbols into one call (rather than one call per symbol) costs
    the same credits but far fewer HTTP round trips and far less exposure
    to the 8-calls/minute cap.

    Returns a dict keyed by symbol, always in this per-symbol shape:
      {"AAPL": {"status": "ok", "values": [...]}, ...}
      {"FAKE": {"status": "error", "code": 404, "message": "..."}, ...}

    Twelve Data actually returns two different shapes depending on symbol
    count (a bare {meta,values,status} object for exactly one symbol, a
    symbol-keyed dict for 2+) — normalised here so every caller only ever
    deals with the one shape above.

    Raises TwelveDataAuthError / TwelveDataRateLimitError directly for a
    whole-request failure (bad API key, or every symbol rejected before
    any per-symbol processing happened) rather than returning it — those
    aren't per-symbol data problems, so they don't belong in the dict.
    """
    if not TWELVEDATA_API_KEY:
        raise TwelveDataAuthError()

    params = {
        "symbol": ",".join(symbols),
        "interval": interval,
        "start_date": start_date,
        "end_date": end_date,
        "outputsize": 5000,
        "adjust": "all",   # split + dividend adjusted — matches yfinance's auto_adjust=True
        "apikey": TWELVEDATA_API_KEY,
    }

    def _call():
        resp = requests.get(f"{TWELVEDATA_BASE}/time_series", params=params, timeout=20)
        _track_call(len(symbols))
        if resp.status_code == 401:
            raise TwelveDataAuthError()
        if resp.status_code == 429:
            raise TwelveDataRateLimitError(symbols)
        # Deliberately no raise_for_status() here. For a 2+ symbol batch,
        # Twelve Data returns HTTP 200 even when some symbols failed —
        # per-symbol errors ({"status":"error", "code":404, ...}) are
        # embedded in the body instead. For exactly one symbol, though, a
        # per-symbol failure like "not found" comes back as the *actual*
        # HTTP status (404) with that same error object as the *whole*
        # body — not embedded under a symbol key, since there's no batch
        # wrapper for a single symbol. raise_for_status() would turn that
        # into a raw requests.HTTPError (whose message embeds the full
        # request URL, API key included) before the per-symbol logic below
        # ever gets to classify it. So: always parse the JSON body and let
        # that classification handle both shapes uniformly — the body's
        # own "status"/"code"/"message" fields are trustworthy regardless
        # of which HTTP status carried them.
        return resp.json()

    body = with_twelvedata_retry(_call)

    if len(symbols) == 1:
        return {symbols[0]: body}

    # A batch response is symbol-keyed on success or partial success. The
    # one exception is a whole-request failure that happens before
    # per-symbol processing starts (e.g. a malformed shared parameter) —
    # that comes back as one flat {status, code, message} object with none
    # of the requested symbols as keys. Treat that as every symbol failing
    # identically rather than mis-indexing into a shape that isn't there.
    if "status" in body and not any(s in body for s in symbols):
        return {s: body for s in symbols}
    return body
