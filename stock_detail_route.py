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

import requests
from fastapi import APIRouter, HTTPException, Request

from rate_limit import limiter

router = APIRouter()  # or use your existing `app` directly

logger = logging.getLogger(__name__)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB_BASE = "https://finnhub.io/api/v1"

# Shown to the client for any unhandled error — the real exception is logged
# server-side via logger.exception() right before this gets raised, so
# nothing about the failure (message, internal state, stack trace) reaches
# the response body. Kept in sync with api.py's GENERIC_ERROR_DETAIL — not
# imported from there to avoid a circular import (api.py imports this module).
GENERIC_ERROR_DETAIL = "Something went wrong processing your request. Please try again."


class FinnhubRateLimitError(Exception):
    """Raised when Finnhub returns 429 (60 calls/minute on the free tier)."""
    pass


def _finnhub_get(path: str, params: dict) -> dict:
    resp = requests.get(
        f"{FINNHUB_BASE}/{path}",
        params={**params, "token": FINNHUB_API_KEY},
        timeout=10,
    )
    if resp.status_code == 429:
        raise FinnhubRateLimitError()
    resp.raise_for_status()
    return resp.json()


@router.get("/stock-detail/{ticker}")
@limiter.limit("30/minute")
async def stock_detail(request: Request, ticker: str):
    """
    Returns current price, daily change, and key fundamentals for a single
    ticker. Sourced from Finnhub — see module docstring for the field mapping
    and the one gap (sparkline) vs. the previous yfinance-backed version.
    """
    if not FINNHUB_API_KEY:
        raise HTTPException(status_code=500, detail="FINNHUB_API_KEY is not configured")

    try:
        t = ticker.upper().strip()

        quote = _finnhub_get("quote", {"symbol": t})
        price = quote.get("c")

        # Finnhub returns an all-zero quote (c=0, d/dp=null) for an unknown symbol
        # rather than an error, so that's how we detect an invalid ticker here.
        if not price:
            raise HTTPException(status_code=404, detail=f"Ticker '{t}' not found")

        profile = _finnhub_get("stock/profile2", {"symbol": t})
        metric = _finnhub_get("stock/metric", {"symbol": t, "metric": "all"}).get("metric", {})

        change = quote.get("d")
        # Finnhub's dp is already a percentage (-0.14 means -0.14%); the frontend
        # multiplies by 100 to display it, so convert to a decimal fraction here.
        change_pct = round(quote["dp"] / 100, 6) if quote.get("dp") is not None else None

        market_cap = profile.get("marketCapitalization")
        market_cap = round(market_cap * 1_000_000) if market_cap else None

        dividend_yield = metric.get("dividendYieldIndicatedAnnual")
        dividend_yield = round(dividend_yield / 100, 6) if dividend_yield is not None else None

        avg_volume = metric.get("10DayAverageTradingVolume")
        avg_volume = round(avg_volume * 1_000_000) if avg_volume else None

        return {
            "ticker":         t,
            "company_name":   profile.get("name"),
            "sector":         profile.get("finnhubIndustry"),
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

    except HTTPException:
        raise
    except FinnhubRateLimitError:
        raise HTTPException(
            status_code=503,
            detail="Finnhub rate limit reached (60 calls/minute). Please try again in a minute.",
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Finnhub request failed: {e}")
    except Exception as e:
        logger.exception("Unhandled error in /stock-detail/%s: %s", ticker, e)
        raise HTTPException(status_code=500, detail=GENERIC_ERROR_DETAIL)


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
