# Add this to your api.py (or as a separate router file)
# Requires: yfinance (already in your requirements), fastapi

from datetime import datetime, timedelta

import yfinance as yf
from fastapi import APIRouter, HTTPException
from typing import Optional

from yfinance_utils import with_yfinance_retry, YFinanceRateLimitError

router = APIRouter()  # or use your existing `app` directly


@router.get("/stock-detail/{ticker}")
async def stock_detail(ticker: str):
    """
    Returns current price, daily change, 30-day sparkline,
    and key fundamentals for a single ticker.
    """
    try:
        t = ticker.upper().strip()
        stock = yf.Ticker(t)
        info  = with_yfinance_retry(lambda: stock.info, [t])

        # info can come back near-empty (e.g. just {"trailingPegRatio": None})
        # both for a genuinely invalid ticker AND for a rate-limited/degraded
        # response that never raised YFRateLimitError — yfinance makes the two
        # indistinguishable at this point. Cross-check with a short yf.download(),
        # a different endpoint than .info: if it also has no data, the ticker is
        # genuinely invalid (404); if it succeeds with data, .info's failure was
        # a rate-limit/degradation specific to that endpoint, not a bad ticker.
        if not info or info.get("regularMarketPrice") is None:
            end = datetime.today().strftime("%Y-%m-%d")
            start = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
            probe = with_yfinance_retry(
                lambda: yf.download(t, start=start, end=end, auto_adjust=True, progress=False),
                [t],
            )
            if probe.empty:
                raise HTTPException(status_code=404, detail=f"Ticker '{t}' not found")
            raise YFinanceRateLimitError([t])

        # 30-day price history for sparkline
        hist = with_yfinance_retry(lambda: stock.history(period="1mo"), [t])
        sparkline = []
        if not hist.empty:
            sparkline = [round(float(v), 4) for v in hist["Close"].tolist()]

        price      = info.get("regularMarketPrice") or info.get("currentPrice")
        prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose")
        change     = round(price - prev_close, 4) if price and prev_close else None
        change_pct = round(change / prev_close, 6) if change and prev_close else None

        return {
            "ticker":         t,
            "company_name":   info.get("longName") or info.get("shortName"),
            "sector":         info.get("sector"),
            "price":          round(price, 2) if price else None,
            "change":         change,
            "change_pct":     change_pct,
            "sparkline":      sparkline,
            "market_cap":     info.get("marketCap"),
            "pe_ratio":       info.get("trailingPE") or info.get("forwardPE"),
            "eps":            info.get("trailingEps"),
            "week_52_high":   info.get("fiftyTwoWeekHigh"),
            "week_52_low":    info.get("fiftyTwoWeekLow"),
            "beta":           info.get("beta"),
            "dividend_yield": info.get("dividendYield"),
            "avg_volume":     info.get("averageVolume"),
        }

    except HTTPException:
        raise
    except YFinanceRateLimitError as e:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Unable to fetch data for: {', '.join(e.failed_tickers)}. "
                "Yahoo Finance may be rate-limiting this server. Please try again in a minute."
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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