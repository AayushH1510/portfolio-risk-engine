"""
api.py
------
FastAPI backend that wraps stats_engine.py and data_fetcher.py.
Run with: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import date, timedelta
import pandas as pd
import numpy as np

from data_fetcher import fetch_with_benchmark, fetch_closing_prices, validate_tickers
from stats_engine import compute_all_metrics, compute_stress_scenario, compute_monte_carlo
from stock_detail_route import router as stock_router
from cache import cached

HOUR = 60 * 60

STRESS_SCENARIOS = [
    {"name": "2008 Financial Crisis", "start": "2008-09-01", "end": "2009-03-31"},
    {"name": "COVID Crash",           "start": "2020-02-01", "end": "2020-03-31"},
    {"name": "2022 Rate Shock",       "start": "2022-01-01", "end": "2022-12-31"},
]

app = FastAPI(title="Portfolio Risk Engine API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stock_router)


class AnalyseRequest(BaseModel):
    tickers: List[str]
    weights: List[float]
    start_date: str
    end_date: str
    portfolio_value: float = 10_000
    confidence: float = 0.95
    rolling_window: int = 30
    show_benchmark: bool = True


class ValidateRequest(BaseModel):
    tickers: List[str]


class StressTestRequest(BaseModel):
    tickers: List[str]
    weights: List[float]
    portfolio_value: float = 10_000


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/validate")
async def validate(req: ValidateRequest):
    valid, invalid = validate_tickers(req.tickers)
    return {"valid": valid, "invalid": invalid}


@cached(ttl=HOUR, prefix="fundamentals")
def _fetch_ticker_fundamentals(ticker: str) -> dict:
    """Fetch + derive one ticker's fundamentals. Raises on failure — callers
    handle the error so a bad fetch never gets cached."""
    info = __import__('yfinance').Ticker(ticker).info

    def g(key, fallback=None):
        val = info.get(key)
        return val if val not in (None, 'N/A', float('inf'), float('-inf')) else fallback

    ps_ratio      = g('priceToSalesTrailing12Months')
    ev_ebitda     = g('enterpriseToEbitda')
    gross_margin  = g('grossMargins')
    rev_growth    = g('revenueGrowth')
    profit_margin = g('profitMargins')
    debt_equity   = g('debtToEquity')
    current_ratio = g('currentRatio')
    roe           = g('returnOnEquity')
    beta          = g('beta')
    market_cap    = g('marketCap')
    sector        = g('sector', 'Unknown')
    industry      = g('industry', 'Unknown')
    name          = g('longName', ticker)
    pct_insiders  = g('heldPercentInsiders')
    short_pct     = g('shortPercentOfFloat')
    revenue       = g('totalRevenue')
    free_cashflow = g('freeCashflow')

    vg_score = None
    if ps_ratio and rev_growth and rev_growth > 0:
        vg_score = round(ps_ratio / (rev_growth * 100), 3)

    flags = []

    if profit_margin is not None and profit_margin < -0.1:
        flags.append({ "category": "accounting", "severity": "high",
            "title": "Significant losses",
            "detail": f"Net margin is {profit_margin*100:.1f}%. Company is burning cash — watch cash runway." })
    elif profit_margin is not None and profit_margin < 0:
        flags.append({ "category": "accounting", "severity": "medium",
            "title": "Unprofitable",
            "detail": f"Net margin is {profit_margin*100:.1f}%. Not yet profitable — common in growth stage but adds risk." })

    if free_cashflow is not None and revenue is not None and revenue > 0:
        fcf_margin = free_cashflow / revenue
        if fcf_margin < -0.15:
            flags.append({ "category": "accounting", "severity": "high",
                "title": "Heavy cash burn",
                "detail": f"Free cash flow margin is {fcf_margin*100:.1f}%. Company is consuming cash significantly." })

    if debt_equity is not None:
        if debt_equity > 200:
            flags.append({ "category": "concentration", "severity": "high",
                "title": "Very high leverage",
                "detail": f"Debt/Equity ratio is {debt_equity:.0f}%. Heavily indebted — vulnerable in rising rate environments." })
        elif debt_equity > 100:
            flags.append({ "category": "concentration", "severity": "medium",
                "title": "Elevated debt",
                "detail": f"Debt/Equity ratio is {debt_equity:.0f}%. Above average leverage — monitor debt servicing capacity." })

    if current_ratio is not None and current_ratio < 1.0:
        flags.append({ "category": "accounting", "severity": "high",
            "title": "Liquidity concern",
            "detail": f"Current ratio is {current_ratio:.2f}. Short-term liabilities exceed short-term assets." })

    if short_pct is not None and short_pct > 0.20:
        flags.append({ "category": "competitive", "severity": "high",
            "title": "High short interest",
            "detail": f"{short_pct*100:.1f}% of float is sold short. Significant bearish sentiment from sophisticated investors." })
    elif short_pct is not None and short_pct > 0.10:
        flags.append({ "category": "competitive", "severity": "medium",
            "title": "Elevated short interest",
            "detail": f"{short_pct*100:.1f}% of float sold short. Worth monitoring — indicates some investor scepticism." })

    if pct_insiders is not None and pct_insiders < 0.03:
        flags.append({ "category": "concentration", "severity": "medium",
            "title": "Low insider ownership",
            "detail": f"Insiders own only {pct_insiders*100:.1f}% of shares. Low skin in the game from management." })

    if beta is not None and beta > 2.0:
        flags.append({ "category": "competitive", "severity": "medium",
            "title": "Very high market sensitivity",
            "detail": f"Beta of {beta:.2f} means this stock moves more than 2x the market. Amplified losses in downturns." })

    if rev_growth is not None and rev_growth < -0.05:
        flags.append({ "category": "competitive", "severity": "high",
            "title": "Revenue declining",
            "detail": f"Revenue shrank {abs(rev_growth)*100:.1f}% YoY. Declining top line is a serious competitive concern." })

    positives = []
    if gross_margin is not None and gross_margin > 0.60:
        positives.append(f"High gross margin ({gross_margin*100:.1f}%) — strong pricing power")
    if roe is not None and roe > 0.20:
        positives.append(f"Strong ROE ({roe*100:.1f}%) — efficient use of shareholder capital")
    if rev_growth is not None and rev_growth > 0.20:
        positives.append(f"Strong revenue growth ({rev_growth*100:.1f}% YoY)")
    if current_ratio is not None and current_ratio > 2.0:
        positives.append(f"Strong liquidity (current ratio {current_ratio:.1f}x)")
    if pct_insiders is not None and pct_insiders > 0.10:
        positives.append(f"High insider ownership ({pct_insiders*100:.1f}%) — management aligned with shareholders")

    return {
        "ticker":       ticker,
        "name":         name,
        "sector":       sector,
        "industry":     industry,
        "ps_ratio":     round(ps_ratio, 2)      if ps_ratio      else None,
        "ev_ebitda":    round(ev_ebitda, 2)     if ev_ebitda     else None,
        "gross_margin": round(gross_margin, 4)  if gross_margin  else None,
        "rev_growth":   round(rev_growth, 4)    if rev_growth    else None,
        "profit_margin":round(profit_margin, 4) if profit_margin else None,
        "debt_equity":  round(debt_equity, 1)   if debt_equity   else None,
        "current_ratio":round(current_ratio, 2) if current_ratio else None,
        "roe":          round(roe, 4)           if roe           else None,
        "beta":         round(beta, 2)          if beta          else None,
        "market_cap":   market_cap,
        "vg_score":     vg_score,
        "flags":        flags,
        "positives":    positives,
    }


@app.get("/api/fundamentals")
async def fundamentals(tickers: str):
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        results = []

        for ticker in ticker_list:
            try:
                results.append(_fetch_ticker_fundamentals(ticker))
            except Exception as e:
                results.append({ "ticker": ticker, "error": str(e) })

        results.sort(key=lambda r: (r.get("vg_score") is None, r.get("vg_score") or 999))
        return { "tickers": results }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stress-test")
async def stress_test(req: StressTestRequest):
    try:
        tickers = [t.strip().upper() for t in req.tickers]
        scenarios = []

        for scenario in STRESS_SCENARIOS:
            start = scenario["start"]
            crash_end = scenario["end"]
            # Search up to 5 years past the crash for recovery — comfortably
            # covers all three scenarios, capped at today for recent ones.
            extended_end = min(
                date.today(),
                date.fromisoformat(start) + timedelta(days=5 * 365),
            ).isoformat()

            prices = fetch_closing_prices(tickers, start_date=start, end_date=extended_end)
            result = compute_stress_scenario(
                prices=prices,
                crash_start=start,
                crash_end=crash_end,
                tickers=tickers,
                weights=req.weights,
            )

            if result is None:
                scenarios.append({
                    "name":             scenario["name"],
                    "period":           f"{start} to {crash_end}",
                    "portfolio_return": None,
                    "worst_day":        None,
                    "recovery_days":    None,
                    "excluded_tickers": tickers,
                })
                continue

            scenarios.append({
                "name":             scenario["name"],
                "period":           f"{start} to {crash_end}",
                "portfolio_return": round(result["portfolio_return"], 4),
                "worst_day":        round(result["worst_day"], 4),
                "recovery_days":    result["recovery_days"],
                "excluded_tickers": result["excluded_tickers"],
            })

        return { "scenarios": scenarios, "portfolio_value": req.portfolio_value }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _serialize_var_cvar(vc: dict) -> dict:
    return {
        "var_pct":     round(vc["var_pct"], 6),
        "var_dollar":  round(vc["var_dollar"], 2),
        "cvar_pct":    round(vc["cvar_pct"], 6),
        "cvar_dollar": round(vc["cvar_dollar"], 2),
        "confidence":  vc["confidence"],
        "n_tail_days": vc["n_tail_days"],
    }


def _serialize_monte_carlo(mc: dict) -> dict:
    return {
        "all_paths":       [[round(v, 2) for v in mc["all_paths"][:, i].tolist()] for i in range(min(300, mc["all_paths"].shape[1]))],
        "percentile_5":    [round(v, 2) for v in mc["percentile_5"].tolist()],
        "percentile_50":   [round(v, 2) for v in mc["percentile_50"].tolist()],
        "percentile_95":   [round(v, 2) for v in mc["percentile_95"].tolist()],
        "p5_final":        round(mc["p5_final"], 2),
        "p50_final":       round(mc["p50_final"], 2),
        "p95_final":       round(mc["p95_final"], 2),
        "prob_profit":     round(mc["prob_profit"], 4),
        "prob_loss_10pct": round(mc["prob_loss_10pct"], 4),
        "portfolio_value": mc["portfolio_value"],
        "n_simulations":   mc["n_simulations"],
        "scenario":        mc["scenario"],
    }


@app.post("/api/analyse")
async def analyse(req: AnalyseRequest):
    try:
        portfolio_prices, benchmark_prices = fetch_with_benchmark(
            tickers=req.tickers,
            start_date=req.start_date,
            end_date=req.end_date,
        )

        if portfolio_prices.empty:
            raise HTTPException(status_code=400, detail="No data returned. Check tickers and date range.")

        m = compute_all_metrics(
            prices=portfolio_prices,
            weights=req.weights,
            portfolio_value=req.portfolio_value,
            benchmark_prices=benchmark_prices if req.show_benchmark else None,
            rolling_window=req.rolling_window,
        )

        port_rets  = m["portfolio_returns"]
        cum_rets   = (1 + port_rets).cumprod() - 1
        dd_series  = m["max_drawdown"]["drawdown_series"]
        roll_vol   = m["rolling"]["rolling_volatility"].dropna()
        roll_sh    = m["rolling"]["rolling_sharpe"].dropna()
        ef         = m["efficient_frontier"]
        mc         = m["monte_carlo"]
        ds         = m["diversification_score"]

        result = {
            "period":                m["period"],
            "annualised_return":     round(m["annualised_return"], 6),
            "annualised_volatility": round(m["annualised_volatility"], 6),
            "sharpe_ratio":          round(m["sharpe_ratio"], 4),
            "sortino_ratio":         round(m["sortino_ratio"], 4),
            "var_cvar":              _serialize_var_cvar(m["var_cvar"]),
            "var_cvar_99":           _serialize_var_cvar(m["var_cvar_99"]),
            "max_drawdown":          round(m["max_drawdown"]["max_drawdown"], 6),
            "beta_alpha":            m["beta_alpha"],
            "treynor_ratio":         round(m["treynor_ratio"], 4)     if m["treynor_ratio"]     is not None else None,
            "information_ratio":     round(m["information_ratio"], 4) if m["information_ratio"] is not None else None,
            "diversification_score": {
                "score":             ds["score"],
                "avg_pairwise_corr": ds["avg_pairwise_corr"],
                "label":             ds["label"],
            },

            "portfolio_returns": [round(v, 6) for v in port_rets.values.tolist()],
            "cumulative_returns": {
                "dates":  cum_rets.index.strftime("%Y-%m-%d").tolist(),
                "values": [round(v, 6) for v in cum_rets.values.tolist()],
            },
            "drawdown_series": {
                "dates":  dd_series.index.strftime("%Y-%m-%d").tolist(),
                "values": [round(v, 6) for v in dd_series.values.tolist()],
            },
            "rolling_volatility": {
                "dates":  roll_vol.index.strftime("%Y-%m-%d").tolist(),
                "values": [round(v, 6) for v in roll_vol.values.tolist()],
            },
            "rolling_sharpe": {
                "dates":  roll_sh.index.strftime("%Y-%m-%d").tolist(),
                "values": [round(v, 6) for v in roll_sh.values.tolist()],
            },
            "correlation_matrix": {
                "tickers": list(m["correlation_matrix"].columns),
                "values":  m["correlation_matrix"].round(4).values.tolist(),
            },
            "efficient_frontier": {
                "vols":               [round(v, 6) for v in ef["vols"].tolist()],
                "returns":            [round(v, 6) for v in ef["returns"].tolist()],
                "sharpes":            [round(v, 4) for v in ef["sharpes"].tolist()],
                "max_sharpe_vol":     round(ef["max_sharpe_vol"], 6),
                "max_sharpe_return":  round(ef["max_sharpe_return"], 6),
                "max_sharpe_sharpe":  round(ef["max_sharpe_sharpe"], 4),
                "max_sharpe_weights": ef["max_sharpe_weights"],
                "min_vol_vol":        round(ef["min_vol_vol"], 6),
                "min_vol_return":     round(ef["min_vol_return"], 6),
                "min_vol_weights":    ef["min_vol_weights"],
            },
            "monte_carlo":      _serialize_monte_carlo(mc),
            "monte_carlo_base": _serialize_monte_carlo(mc),
            "monte_carlo_bear": _serialize_monte_carlo(compute_monte_carlo(
                portfolio_returns=port_rets,
                asset_returns=m["returns"],
                weights=req.weights,
                portfolio_value=req.portfolio_value,
                n_simulations=mc["n_simulations"],
                scenario="bear",
            )),
            "monte_carlo_bull": _serialize_monte_carlo(compute_monte_carlo(
                portfolio_returns=port_rets,
                asset_returns=m["returns"],
                weights=req.weights,
                portfolio_value=req.portfolio_value,
                n_simulations=mc["n_simulations"],
                scenario="bull",
            )),
        }

        if req.show_benchmark and benchmark_prices is not None:
            bench_rets   = benchmark_prices.pct_change().dropna().iloc[:, 0]
            bench_cumret = (1 + bench_rets).cumprod() - 1
            bench_cumret = bench_cumret.reindex(cum_rets.index)
            result["benchmark_cumulative"] = {
                "dates":  bench_cumret.index.strftime("%Y-%m-%d").tolist(),
                "values": [round(v, 6) if not np.isnan(v) else None
                           for v in bench_cumret.values.tolist()],
            }

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))