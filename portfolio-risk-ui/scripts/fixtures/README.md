# Fixtures

Used by `scripts/shots.mjs` to mock `/api/analyse-summary`, `/api/analyse-full`,
`/api/analyse` (Compare's Portfolio B), and `/api/fundamentals` so in-app views
can be screenshotted without a live backend or Twelve Data/Finnhub calls.

Portfolio: **AAPL / MSFT / GOOGL / AMZN**, weights 30/25/25/20, benchmark **QQQ**
(Nasdaq 100), 1-year window ending 2026-09-12.

- `analyse-summary.json`, `analyse-full.json` — **genuine**, captured from a
  real `POST /api/analyse-summary` / `POST /api/analyse-full` against the
  local backend with a live `TWELVEDATA_API_KEY`. Real Twelve Data prices,
  real computed Sharpe/VaR/Monte Carlo/efficient frontier/backtest numbers.

- `fundamentals.json` — **not genuine**. A real `GET /api/fundamentals` call
  was made for the same 4 tickers, but this environment's `.env` has no
  `FINNHUB_API_KEY` configured (pre-existing, unrelated to this harness), so
  the real response is all four tickers returning
  `{"error": "Unable to load data for this ticker right now."}` — accurate to
  what the endpoint actually does without a key, but useless for screenshotting
  the Valuation tab's real layout. This file is hand-built instead, matching
  the real endpoint's schema exactly (same fields as a genuine response,
  reusing 3 entries already validated against the real shape earlier this
  project, plus one added for AMZN). Recapture for real once
  `FINNHUB_API_KEY` is set: `curl "http://localhost:8000/api/fundamentals?tickers=AAPL,MSFT,GOOGL,AMZN"`.

To recapture the two genuine fixtures after a stats_engine/data_fetcher change
(same tickers/benchmark/window so the harness's ticker-input fill still lines
up with what's displayed):

```bash
curl -X POST http://localhost:8000/api/analyse-summary -H 'Content-Type: application/json' \
  -d '{"tickers":["AAPL","MSFT","GOOGL","AMZN"],"weights":[0.30,0.25,0.25,0.20],"start_date":"<1y ago>","end_date":"<today>","portfolio_value":10000,"show_benchmark":true,"benchmark":"QQQ","rolling_window":30}' \
  -o scripts/fixtures/analyse-summary.json

curl -X POST http://localhost:8000/api/analyse-full -H 'Content-Type: application/json' \
  -d '{...same body...}' -o scripts/fixtures/analyse-full.json
```
