# Varense

**Portfolio risk analytics for individual investors.**

Varense runs correlated Monte Carlo simulation, historical crisis replay, and full risk decomposition across a stock portfolio — the kind of analysis that normally sits behind an institutional terminal.

**Live:** [varense.vercel.app](https://varense.vercel.app)

> Educational tool only. Not financial advice. Past performance does not guarantee future results.

---

## What it does

Enter up to five tickers, set your weights, pick a time period, and Varense computes:

**Risk metrics**
- Value at Risk (VaR) and Conditional VaR / Expected Shortfall at both 95% and 99% confidence
- Maximum drawdown with a full drawdown time series
- Annualised volatility, rolling volatility over a configurable window
- Diversification score derived from the portfolio's average pairwise correlation

**Risk-adjusted return**
- Sharpe, Sortino, Treynor, and Information ratios
- Beta and Jensen's Alpha against the S&P 500
- Rolling Sharpe over time

**Simulation**
- Monte Carlo with Cholesky decomposition, so simulated paths honour the real covariance structure between holdings rather than treating each asset independently
- Bear / base / bull scenarios with adjustable drift assumptions
- 1,000 paths with percentile bands and probability-of-profit

**Historical stress testing**
- Replays the portfolio through the 2008 financial crisis, the COVID crash, and the 2022 rate shock using actual historical returns
- Reports portfolio loss, worst single day, and days to recovery per scenario
- Handles holdings with no data for a given window by excluding them and reweighting the remainder

**Optimisation**
- Efficient frontier across 5,000 simulated weight combinations
- Maximum-Sharpe and minimum-volatility portfolios identified, with suggested weights
- CAGR-consistent return calculation so the user's actual portfolio always plots inside the simulation cloud

**Backtesting**
- Runs the user's weights, an equal-weight allocation, and the S&P 500 through the same historical period
- Cumulative return chart, year-by-year breakdown, and comparative Sharpe / drawdown

**Fundamentals**
- Per-ticker valuation ratios, margins, leverage, and a composite value/growth score
- Rule-based risk flags (leverage, liquidity, cash burn, declining revenue) and strengths
- Sector exposure aggregated across the portfolio

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | React, Vite, Recharts, HTML Canvas |
| Backend | FastAPI, NumPy, Pandas |
| Auth & persistence | Supabase (email + Google OAuth, saved portfolios) |
| Caching | Redis via Upstash, fail-open |
| Market data | Finnhub (quotes, fundamentals), Twelve Data (historical price series) |
| Hosting | Vercel (frontend), Render (backend) |

---

## Architecture

```
varense/
├── api.py                  # FastAPI — all HTTP endpoints
├── stats_engine.py         # Pure maths layer, no HTTP, fully vectorised
├── data_fetcher.py         # Historical price fetching, Redis-cached
├── stock_detail_route.py   # Single-ticker quotes and fundamentals (Finnhub)
├── twelvedata_utils.py     # Shared retry + rate-limit handling
├── cache.py                # Redis wrapper, degrades silently if unavailable
└── portfolio-risk-ui/      # React frontend
    └── src/
        ├── pages/          # Dashboard, Risk Analysis, Monte Carlo, Frontier,
        │                   # Valuation, Backtest, Compare, Learn, Landing
        ├── components/     # Shared UI, charts, modals, drawer
        ├── hooks/          # useAnalysis, useAuth, usePortfolios, useComparison
        └── content/        # Metric explanations, Learn tiers, landing copy
```

The maths layer is deliberately isolated from the API layer — `stats_engine.py` takes DataFrames and returns dictionaries, with no knowledge of HTTP, so it can be tested and reasoned about independently.

---

## Notes on the maths

A few implementation decisions worth calling out:

**Returns are annualised geometrically, not arithmetically.** `compute_annualised_return` uses CAGR — `cumulative^(252/n) − 1` — rather than `mean × 252`. The efficient frontier simulation uses the identical formula, which is what keeps the user's real portfolio plotting inside the simulated cloud rather than floating outside it.

**Monte Carlo models correlation.** Rather than drawing portfolio returns from a single univariate normal distribution, the simulation Cholesky-decomposes the asset covariance matrix and draws correlated returns per asset, then applies portfolio weights. Ignoring correlation systematically understates tail risk, because assets that fall together in reality are treated as independent. Falls back to the univariate approach if the covariance matrix is non-positive-definite.

**CVaR is reported alongside VaR at two confidence levels.** VaR marks a quantile boundary but says nothing about severity beyond it. Expected Shortfall describes the tail itself, which is why the Basel Committee's Fundamental Review of the Trading Book replaced 99% VaR with 97.5% Expected Shortfall as the required market-risk capital measure.

**Diversification is scored from realised correlation, not holdings count.** A five-stock portfolio of large-cap US tech is less diversified than a three-asset portfolio spanning uncorrelated sectors. The score is `(1 − average pairwise correlation) × 100`, which makes that visible. It is also window-dependent by design — the same holdings can score very differently across periods, which is surfaced in the UI rather than hidden.

---

## Running locally

**Prerequisites:** Python 3.11+, Node 18+

**Backend**

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

**Frontend**

```bash
cd portfolio-risk-ui
npm install
npm run dev
```

Open `http://localhost:5173`.

**Environment variables**

Copy `.env.example` to `.env` and fill in:

```
FINNHUB_API_KEY=      # free tier at finnhub.io
REDIS_URL=            # optional — app runs fine without it
ALLOWED_ORIGINS=http://localhost:5173
```

Frontend, in `portfolio-risk-ui/.env`:

```
VITE_API_URL=http://localhost:8000
```

Redis is genuinely optional. `cache.py` fails open, so every cache operation becomes a no-op if Redis is unreachable and the app continues working, just without caching.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/validate` | Validate ticker symbols |
| `POST` | `/api/analyse` | Full portfolio analysis — all metrics, simulation, frontier, backtest |
| `POST` | `/api/stress-test` | Historical crisis scenarios |
| `GET` | `/api/fundamentals` | Per-ticker fundamentals and risk flags |
| `GET` | `/stock-detail/{ticker}` | Single-ticker quote and fundamentals |

---

## Roadmap

- Mobile and tablet responsive layouts
- GARCH / EWMA volatility forecasting feeding into Monte Carlo drift assumptions
- Volatility regime detection
- Watchlists
- Caching for the stress-test endpoint