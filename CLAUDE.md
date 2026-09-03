# Varense — Portfolio Risk Analysis Engine
> Context file for Claude Code / Codex. Read this before touching any file.

---

## What this is
A professional portfolio risk analytics platform. Not a toy project — the math engine
uses institutional-grade techniques (Cholesky-corrected Monte Carlo, CAGR-based frontier
simulation, dual confidence VaR/CVaR). Target users: serious retail investors, finance
students, early-stage fintech teams. Goal: something worth purchasing or investing in.

---

## How to run

**Backend (Terminal 1):**
```bash
cd "C:\Users\aayus\Desktop\Career\Portfolio Risk Analysis Engine"
venv\Scripts\activate
uvicorn api:app --reload --port 8000
```

**Frontend (Terminal 2):**
```bash
cd "C:\Users\aayus\Desktop\Career\Portfolio Risk Analysis Engine\portfolio-risk-ui"
npm run dev
```

Open: `http://localhost:5173`

---

## Project structure

```
Portfolio Risk Analysis Engine/
├── api.py                  # FastAPI — all HTTP endpoints
├── stats_engine.py         # All maths — pure Python/NumPy/Pandas, no HTTP
├── data_fetcher.py         # yFinance data fetching
├── stock_detail_route.py   # /stock-detail/{ticker} endpoint (separate router)
├── requirements.txt
├── venv/
└── portfolio-risk-ui/      # React + Vite frontend
    └── src/
        ├── App.jsx         # Root — tab routing, auth, drawer state, global fetch
        ├── pages/
        │   ├── Dashboard.jsx
        │   ├── RiskAnalysis.jsx
        │   ├── MonteCarlo.jsx
        │   ├── Frontier.jsx
        │   ├── Valuation.jsx
        │   ├── Comparison.jsx
        │   └── Learn.jsx
        ├── components/
        │   ├── Sidebar.jsx         # All portfolio controls (tickers, weights, period)
        │   ├── StockDrawer.jsx     # Slide-in panel — price, fundamentals, sparkline
        │   ├── MetricCard.jsx
        │   ├── RiskGauge.jsx
        │   ├── ReturnHistogram.jsx
        │   ├── InsightBox.jsx
        │   ├── ExportPDF.jsx
        │   ├── AuthModal.jsx
        │   └── SavedPortfolios.jsx
        └── hooks/
            ├── useAnalysis.js
            ├── useAuth.js
            ├── usePortfolios.js
            └── useComparison.js
```

---

## Backend — API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/validate` | Validate ticker symbols |
| POST | `/api/analyse` | Main analysis — returns all metrics |
| GET | `/api/fundamentals?tickers=AAPL,MSFT` | Per-ticker fundamentals + risk flags |
| GET | `/stock-detail/{ticker}` | Price, sparkline, fundamentals for drawer |

**`/api/analyse` request shape:**
```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "weights": [0.34, 0.33, 0.33],
  "start_date": "2024-01-01",
  "end_date": "2025-01-01",
  "portfolio_value": 10000,
  "rolling_window": 30,
  "show_benchmark": true
}
```

**`/api/analyse` response includes:**
- `period`, `annualised_return`, `annualised_volatility`
- `sharpe_ratio`, `sortino_ratio`
- `var_cvar` (95%), `var_cvar_99` (99%)
- `max_drawdown`, `beta_alpha`
- `diversification_score` — `{ score: 0-100, label, avg_pairwise_corr }`
- `cumulative_returns`, `drawdown_series`
- `rolling_volatility`, `rolling_sharpe`
- `correlation_matrix`
- `efficient_frontier` — 5,000 simulated portfolios (CAGR-based)
- `monte_carlo` — 1,000 paths with Cholesky correlation
- `benchmark_cumulative` — S&P 500 comparison
- `portfolio_returns` — daily return series

---

## Stats engine — key implementation details

**`stats_engine.py` — DO NOT break these:**

- `compute_annualised_return` uses **CAGR** (geometric): `cumulative^(252/n) - 1`
  — NOT arithmetic mean × 252. This is critical — the frontier simulation must match.
- `compute_efficient_frontier` uses the same CAGR formula for consistency — this is
  what keeps "Your portfolio" dot inside the simulation cloud.
- `compute_monte_carlo` uses **Cholesky decomposition** to model asset correlations.
  Falls back to univariate normal if matrix is non-positive-definite (rare).
- `compute_diversification_score` — avg off-diagonal correlation → 0-100 scale.
  Score = (1 − avg_corr) × 100, clamped. ≥70 = Well diversified, 40-69 = Moderate, <40 = Concentrated.
- `compute_cvar` is called twice — once at 0.95, once at 0.99. Both go into the response.
- `TRADING_DAYS = 252`, `RISK_FREE_RATE = 0.045`

---

## Frontend — critical conventions

### Styling
- **All inline styles** — no Tailwind utility classes in component JSX.
- CSS variables defined globally: `--accent`, `--accent-dark`, `--card`, `--border`,
  `--border-light`, `--text-primary`, `--text-secondary`, `--text-muted`,
  `--positive`, `--negative`, `--warning`, `--bg`
- Primary green: `#52b788` (accent)
- Danger red: `#e05c5c`
- Warning amber: `#e09a30`
- Dark bg: `#0d1a10` (used for canvas areas like the frontier chart)
- Card bg: `var(--card)` — do not hardcode

### Component patterns
- `MetricCard` — takes `label`, `value`, `tone` ('good'|'bad'|'warning'|'neutral'), `small`, `sub`
- `InsightBox` — takes `label`, `text` (HTML string), `tone`
- Charts use **Recharts** (`AreaChart`, `LineChart`, `ResponsiveContainer`)
- Frontier + Histogram use **raw Canvas** (not Recharts) — `useRef(canvasRef)`
- Stock drawer is a fixed-position overlay triggered globally from `App.jsx` via `openDrawer(ticker, weight)`

### State / data flow
- `useAnalysis` hook owns all analysis state (tickers, weights, period, data, loading)
- `App.jsx` calls `openDrawer(ticker, weight)` which sets `drawerTicker` + `drawerWeight` state
- `onTickerClick` prop is passed down to `Sidebar`, `Dashboard`, `RiskAnalysis`, `Valuation`
- All pages receive `data` (the full API response) as a prop — they do NOT fetch themselves
- Exception: `StockDrawer` fetches `/stock-detail/{ticker}` itself on mount

### Sidebar
- Supports 3–5 tickers (max 5, cap enforced)
- Two weight modes: `% Split` (sliders) and `$ Amount`
- Last ticker always auto-fills to make weights sum to 100%
- Tickers are clickable (dotted underline + arrow) — triggers the stock drawer

---

## What's been built ✅

- All 7 tabs working: Dashboard, Risk Analysis, Monte Carlo, Efficient Frontier, Valuation, Compare, Learn
- Stock detail drawer (slide-in from right, 30-day sparkline, fundamentals)
- Supabase auth — Google sign-in + email, saved portfolios
- PDF export
- Risk gauge (canvas-based semicircle)
- Returns histogram (canvas — VaR/CVaR lines, normal curve overlay)
- Efficient frontier (canvas — 5,000 dots, animated optimal marker, hover tooltips)
- Monte Carlo (Recharts — fan of 1,000 paths, percentile bands)
- Correlation matrix (auto-scales for 3–5 tickers)
- Diversification score on Dashboard (dynamic second metrics row)
- VaR/CVaR at 95% and 99% on Risk Analysis tab
- 5-ticker support across all views
- Redis caching (Upstash) — `cache.py` wraps `fetch_closing_prices` and
  `fetch_with_benchmark` (`data_fetcher.py`) at a 24hr TTL, and
  `_fetch_ticker_fundamentals` (`api.py`, backs `/api/fundamentals`) at a 1hr
  TTL. Fail-open by design: if `REDIS_URL` is unset or Redis is unreachable,
  every cache op silently no-ops and the app runs exactly as it would with no
  cache. Known gap: `/api/stress-test` doesn't use this layer yet — it still
  fetches each crisis-window's price data fresh on every call. Worth adding
  the same `@cached` wrapper there, not a missing feature so much as an
  unfinished rollout.

---

## What's next — in order

### Before rebrand:
1. **Sector exposure chart** — frontend only. `/api/fundamentals` already returns `sector`
   per ticker. Fetch after analysis runs, aggregate by sector weighted by portfolio weights,
   display as horizontal bar chart on Dashboard. Preferred approach: call fundamentals
   inside `App.jsx` after analysis completes, pass `sectorData` as prop to Dashboard.

### After rebrand:
- Rebrand to **Varense** — name, logo (`/public/favicon.ico`), `index.html` title,
  sidebar logo, footer copy
- Stress testing (2008/2020/2022 scenario shocks) — new tab or section in Risk Analysis
- Portfolio backtesting engine — new tab
- Bear/base/bull Monte Carlo scenarios — extend Monte Carlo tab
- Treynor Ratio + Information Ratio — 2 lines in stats_engine, display in RiskAnalysis
- CSV export
- Watchlists (Supabase table)
- Volatility regime detection

### Deploy:
- Frontend → **Vercel**
- Backend → **Render**
- Redis → **Upstash**, wired via `REDIS_URL` (see `cache.py`)
- Add rate limiting to `/api/analyse` before going public

---

## Known decisions / things NOT to change

- Do NOT switch to Tailwind — all styling is inline on purpose for this codebase
- Do NOT replace Recharts — it's already imported and working
- Do NOT add Celery — asyncio is sufficient, Celery adds unnecessary complexity
- Do NOT roll custom auth — Supabase handles it
- Do NOT use arithmetic mean × 252 for annualised returns anywhere — always use CAGR
- The frontier simulation MUST use the same return calculation as `compute_annualised_return`
  or "Your portfolio" dot will appear outside the simulation cloud

---

## Auth / external services

- **Supabase URL:** `https://wzjtyosijxacgytfmjtu.supabase.co`
- **Data:** two sources, split by purpose —
  - **Finnhub** (`stock_detail_route.py`, `FINNHUB_API_KEY`): real-time quotes and company fundamentals, powering the stock detail drawer and the Valuation tab.
  - **Twelve Data** (`data_fetcher.py` via `twelvedata_utils.py`, `TWELVEDATA_API_KEY`): historical price series, powering portfolio analysis, backtesting, and Monte Carlo simulation. Replaced yfinance in Sept 2026 — Yahoo's rate-limiting on Render's IP was silently misreporting valid tickers as "not found." Free tier: 800 calls/day, 8/minute, shared across every user — `twelvedata_utils.py` logs call volume as it approaches the daily cap. No free-tier access to the raw S&P 500 index (`^GSPC`/`SPX` are paywalled there), so the benchmark is `SPY` (S&P 500 ETF) instead of the literal index — see `fetch_with_benchmark`'s docstring.
- **Streamlit prototype:** `https://portfolio-risk-engine.streamlit.app` (legacy — ignore)

In stats_engine.py compute_diversification_score — verified correct Aug 2026. AAPL/MSFT/GOOGL scoring ~86/100 over trailing 1yr is accurate — the three stocks genuinely decorrelated in this window due to AI capex divergence. Not a bug. Score is window-dependent by design.