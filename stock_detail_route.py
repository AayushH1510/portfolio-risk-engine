# Add this to your api.py (or as a separate router file)
# Requires: requests (already in requirements.txt), fastapi
#
# Data source: Finnhub REST API (https://finnhub.io/docs/api).
# Requires FINNHUB_API_KEY set as an environment variable — see .env.example.
#
# Field mapping vs. the yfinance version this replaced:
#   price, change, change_pct  <- /quote                (c, d, dp)
#   company_name, sector       <- /stock/profile2        (name, finnhubIndustry —
#                                                          Finnhub only exposes one
#                                                          combined industry field,
#                                                          no separate sector/industry)
#   market_cap                 <- /stock/profile2        (marketCapitalization, in
#                                                          millions — scaled to raw USD)
#   pe_ratio, eps, beta,
#   week_52_high/low,
#   dividend_yield, avg_volume <- /stock/metric?metric=all
#   sparkline                  <- NOT AVAILABLE. /stock/candle (historical daily
#                                  bars) is gated behind a paid Finnhub plan — this
#                                  key gets {"error": "You don't have access to this
#                                  resource."} on every request. Returns [] instead;
#                                  StockDrawer.jsx already renders no chart when
#                                  sparkline is empty.

import logging
import os
import threading
import time
from collections import deque

import requests
from fastapi import APIRouter, HTTPException, Request

from cache import cached
from rate_limit import limiter

router = APIRouter()  # or use your existing `app` directly

logger = logging.getLogger(__name__)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB_BASE = "https://finnhub.io/api/v1"

MINUTE = 60
HOUR = 60 * MINUTE
DAY = 24 * HOUR

# Shown to the client for any unhandled error — the real exception is logged
# server-side via logger.exception() right before this gets raised, so
# nothing about the failure (message, internal state, stack trace) reaches
# the response body. Kept in sync with api.py's GENERIC_ERROR_DETAIL — not
# imported from there to avoid a circular import (api.py imports this module).
GENERIC_ERROR_DETAIL = "Something went wrong processing your request. Please try again."
RATE_LIMIT_DETAIL = "Finnhub rate limit reached (60 calls/minute). Please try again in a minute."


class FinnhubRateLimitError(Exception):
    """Raised when Finnhub returns 429 (60 calls/minute on the free tier)."""
    pass


class FinnhubAuthError(Exception):
    """
    Raised when Finnhub returns 401 — the API key itself is being rejected
    (missing, invalid, revoked, or a plan/quota issue Finnhub reports as an
    auth failure rather than a 429), not a problem with any one ticker.
    Every ticker in a batch fails identically when this happens, so callers
    should treat it as one service-level problem, not N unrelated failures —
    and it must never surface as "ticker not found", which is what letting
    this fall through to a bare requests.HTTPError used to look like.
    """
    pass


# Shared across every Finnhub call in the app (api.py's /api/fundamentals
# reuses this same _finnhub_get, not a second copy) — one API key, one
# 60-calls/minute budget, no matter how many concurrent users are behind it.
# A sliding window: calls made in the last FINNHUB_RATE_WINDOW seconds are
# tracked, and a call that would push the count past the limit blocks the
# calling thread until the oldest call in the window ages out, so a burst
# queues briefly and smooths itself out instead of firing 61 requests at
# once and letting Finnhub itself hand back the 429. This runs after each
# route became a plain `def` (see stock_detail below) — FastAPI's thread
# pool means multiple of these can genuinely run concurrently now, so the
# lock isn't decorative.
FINNHUB_RATE_LIMIT = 60
FINNHUB_RATE_WINDOW = 60.0
_finnhub_call_times = deque()
_finnhub_lock = threading.Lock()


def _throttle_finnhub():
    with _finnhub_lock:
        now = time.monotonic()
        while _finnhub_call_times and now - _finnhub_call_times[0] >= FINNHUB_RATE_WINDOW:
            _finnhub_call_times.popleft()
        if len(_finnhub_call_times) >= FINNHUB_RATE_LIMIT:
            wait = FINNHUB_RATE_WINDOW - (now - _finnhub_call_times[0])
        else:
            wait = 0.0
        # Reserve this call's slot at the time it will actually fire (not
        # `now`) *before* releasing the lock, so a burst of concurrent
        # callers queues in order — each one sees every already-reserved
        # future slot when it computes its own wait, rather than every
        # thread racing to compute the same wait against the same oldest
        # timestamp.
        _finnhub_call_times.append(now + wait)
    if wait > 0:
        logger.info("Finnhub throttle: queuing call for %.2fs (rate limit)", wait)
        time.sleep(wait)


def _finnhub_get(path: str, params: dict) -> dict:
    _throttle_finnhub()
    resp = requests.get(
        f"{FINNHUB_BASE}/{path}",
        params={**params, "token": FINNHUB_API_KEY},
        # 5s, not the original 10s — stock_detail below makes up to three
        # of these sequentially, so 10s meant up to 30s before the whole
        # request failed. Finnhub responding at all in under 5s is the
        # common case; when it isn't, failing faster matters more than
        # squeezing out a slow success, especially now that a timeout on
        # one call no longer takes the other two down with it (see
        # stock_detail's partial-data handling).
        timeout=5,
    )
    if resp.status_code == 429:
        raise FinnhubRateLimitError()
    if resp.status_code == 401:
        raise FinnhubAuthError()
    resp.raise_for_status()
    return resp.json()


# Each Finnhub endpoint cached separately, at its own TTL, rather than one
# TTL for the whole drawer — quote (price/change) is only meaningful fresh,
# so 60s; profile (name/sector/market cap) and metric (P/E, EPS, beta, 52w
# range, etc.) barely move within a day, so 24h. Only successful fetches are
# cached (the `cached` decorator only calls cache_set after `func` returns
# without raising) — a timeout or rate-limit here just isn't cached, so a
# request a few seconds later tries Finnhub again rather than being stuck
# behind a bad result. That's an accepted gap here (unlike
# _fetch_ticker_fundamentals in api.py, which does distinguish a transient
# failure from a genuine one) — the request that prompted this only asked
# for that distinction on the fundamentals path.
@cached(ttl=MINUTE, prefix="finnhub_quote")
def _get_quote(ticker: str) -> dict:
    return _finnhub_get("quote", {"symbol": ticker})


@cached(ttl=DAY, prefix="finnhub_profile")
def _get_profile(ticker: str) -> dict:
    return _finnhub_get("stock/profile2", {"symbol": ticker})


@cached(ttl=DAY, prefix="finnhub_metric")
def _get_metric(ticker: str) -> dict:
    return _finnhub_get("stock/metric", {"symbol": ticker, "metric": "all"}).get("metric", {})


def _try_fetch(fn, ticker: str):
    """Run one Finnhub fetch, returning (result, error) instead of raising —
    so a failure in one of the three independent calls below doesn't take
    the other two down with it."""
    try:
        return fn(ticker), None
    except Exception as e:
        return None, e


def _raise_for_total_failure(errors: list, ticker: str):
    """
    Reached only when quote, profile, AND metric all failed — there's
    nothing left to return partial data from. Maps whichever error is most
    specific/actionable to an HTTP response, same precedence and same
    messages the original single try/except chain used, just applied across
    three parallel failures instead of one linear one.
    """
    for e in errors:
        if isinstance(e, FinnhubAuthError):
            logger.error("Finnhub rejected the API key (401) fetching /stock-detail/%s — "
                          "check FINNHUB_API_KEY in the environment", ticker)
            raise HTTPException(status_code=503, detail=GENERIC_ERROR_DETAIL)
    for e in errors:
        if isinstance(e, FinnhubRateLimitError):
            raise HTTPException(status_code=503, detail=RATE_LIMIT_DETAIL)
    for e in errors:
        if isinstance(e, requests.RequestException):
            logger.error("Finnhub request failed in /stock-detail/%s: %s", ticker, e, exc_info=e)
            raise HTTPException(status_code=500, detail=GENERIC_ERROR_DETAIL)
    logger.error("Unhandled error(s) in /stock-detail/%s: %s", ticker, errors, exc_info=errors[0])
    raise HTTPException(status_code=500, detail=GENERIC_ERROR_DETAIL)


@router.get("/stock-detail/{ticker}")
@limiter.limit("30/minute")
def stock_detail(request: Request, ticker: str):
    """
    Returns current price, daily change, and key fundamentals for a single
    ticker. Sourced from Finnhub — see module docstring for the field
    mapping and the one gap (sparkline) vs. the previous yfinance-backed
    version.

    Plain `def`, not `async def` — every Finnhub call here is a blocking
    `requests.get` (_finnhub_get); FastAPI runs a sync route in its own
    thread pool automatically, where the previous `async def` version ran
    those same blocking calls directly on the event loop, stalling every
    other in-flight request on this worker for as long as Finnhub took to
    respond. Confirmed as the actual production mechanism, not a
    theoretical one: Render's logs showed Finnhub responding slowly (10s+
    ReadTimeouts), not erroring — exactly the case a blocking call inside
    `async def` turns into "every other request queues behind this one."

    The three Finnhub calls (quote, profile, metric) are fetched
    independently and merged — one timing out no longer fails the whole
    request. Only a ticker Finnhub positively confirms doesn't exist (a
    real quote response with a zero price) is a 404; a quote that merely
    failed to respond falls through to the same partial-data path as a
    failed profile/metric would.
    """
    if not FINNHUB_API_KEY:
        raise HTTPException(status_code=500, detail="FINNHUB_API_KEY is not configured")

    t = ticker.upper().strip()

    quote, quote_err     = _try_fetch(_get_quote, t)
    profile, profile_err = _try_fetch(_get_profile, t)
    metric, metric_err   = _try_fetch(_get_metric, t)
    metric = metric or {}

    # Finnhub returns an all-zero quote (c=0, d/dp=null) for an unknown
    # symbol rather than an error — a *successful* quote response with no
    # price means the ticker genuinely doesn't exist. That's distinct from
    # quote simply failing to respond at all (timeout/rate-limit/network),
    # which instead falls through to the total-failure check below,
    # alongside profile/metric.
    if quote is not None and not quote.get("c"):
        raise HTTPException(status_code=404, detail=f"Ticker '{t}' not found")

    if quote is None and not profile and not metric:
        _raise_for_total_failure([quote_err, profile_err, metric_err], ticker)

    price = quote.get("c") if quote else None
    change = quote.get("d") if quote else None
    # Finnhub's dp is already a percentage (-0.14 means -0.14%); the frontend
    # multiplies by 100 to display it, so convert to a decimal fraction here.
    change_pct = round(quote["dp"] / 100, 6) if quote and quote.get("dp") is not None else None

    market_cap = profile.get("marketCapitalization") if profile else None
    market_cap = round(market_cap * 1_000_000) if market_cap else None

    dividend_yield = metric.get("dividendYieldIndicatedAnnual")
    dividend_yield = round(dividend_yield / 100, 6) if dividend_yield is not None else None

    avg_volume = metric.get("10DayAverageTradingVolume")
    avg_volume = round(avg_volume * 1_000_000) if avg_volume else None

    return {
        "ticker":         t,
        "company_name":   profile.get("name") if profile else None,
        "sector":         profile.get("finnhubIndustry") if profile else None,
        "price":          round(price, 2) if price else None,
        "change":         round(change, 4) if change is not None else None,
        "change_pct":     change_pct,
        "sparkline":      [],
        "market_cap":     market_cap,
        "pe_ratio":       metric.get("peTTM") or metric.get("peBasicExclExtraTTM") or metric.get("peAnnual"),
        "eps":            metric.get("epsTTM") or metric.get("epsInclExtraItemsTTM"),
        "week_52_high":   metric.get("52WeekHigh"),
        "week_52_low":    metric.get("52WeekLow"),
        "beta":           metric.get("beta"),
        "dividend_yield": dividend_yield,
        "avg_volume":     avg_volume,
    }


# ─────────────────────────────────────────────
# HOW TO WIRE THIS UP IN YOUR api.py:
#
# Option A — if you're using a router:
#   from stock_detail_route import router as stock_router
#   app.include_router(stock_router)
#
# Option B — paste the @router.get(...) function
#   directly into api.py, replacing `router` with `app`
# ─────────────────────────────────────────────
