# Fixtures

Used by `scripts/shots.mjs` to mock `/api/analyse-summary`, `/api/analyse-full`,
`/api/analyse` (Compare's Portfolio B), `/api/fundamentals`, and
`/api/stress-test` so in-app views can be screenshotted without a live backend
or Twelve Data/Finnhub calls.

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
  the real endpoint's schema exactly (same fields as a genuine response).
  Recapture for real once `FINNHUB_API_KEY` is set:
  `curl "http://localhost:8000/api/fundamentals?tickers=AAPL,MSFT,GOOGL,AMZN"`.

  **`sector`/`industry` were rewritten to hold the same value per ticker,
  not left as the two different GICS-style strings an earlier version of
  this file used** (e.g. GOOGL previously read `"sector": "Communication
  Services"` / `"industry": "Internet Content & Information"`). That was
  never a shape the real endpoint can produce: Finnhub's `/stock/profile2`
  only exposes one combined `finnhubIndustry` field, and `api.py`/
  `stock_detail_route.py` both read that single field into *both* the
  `sector` and `industry` response keys — the two are never actually
  independent live data, only ever the same string under two labels. Every
  screenshot this project had taken of the Valuation tab, before this fix,
  showed a more differentiated (and more polished-looking) Sector/Industry
  pairing than production ever renders. Fixed at the display layer too —
  Valuation.jsx no longer has a separate Industry row at all — but the
  fixture needed to stop lying about the shape independently of that, since
  anyone screenshotting or reasoning from this file should see what the
  real endpoint actually returns.

  **The specific values (`"Technology"` for AAPL/MSFT, `"Media"` for
  GOOGL, `"Retail"` for AMZN) are illustrative, not captured** — still no
  `FINNHUB_API_KEY` available in this environment to confirm Finnhub's
  actual current classification for any of these four tickers. What *is*
  faithful is the shape (`sector === industry`, always, one value not two)
  and the general style (Finnhub's own `finnhubIndustry` taxonomy runs
  short, single-tier labels like these, not GICS's two-tier sector+industry
  split) — not a verified live value. Recapture for real the same way as
  above the moment a key is available, and replace these four values with
  whatever actually comes back.

- `stress-test.json` — **not genuine, same reason as `fundamentals.json` above,
  different key.** A real `POST /api/stress-test` call for this portfolio was
  attempted while adding this fixture and got a 401 from Twelve Data (see the
  backend log: `Twelve Data rejected the API key (401) in /api/stress-test`) —
  this environment's `TWELVEDATA_API_KEY` doesn't currently authenticate,
  unrelated to this harness. Hand-built instead, matching `/api/stress-test`'s
  real response schema exactly (`api.py`'s three `STRESS_SCENARIOS`: 2008
  Financial Crisis, COVID Crash, 2022 Rate Shock — same names/date ranges the
  endpoint hardcodes) with plausible, not-computed loss/recovery figures.
  **Before this fixture existed, `/api/stress-test` was the one call in the
  Risk Analysis tab this harness left hitting a live backend** — the
  Historical Scenarios section stayed on its loading spinner (sometimes
  several seconds, per its own "can take up to a minute" copy) well past the
  fixed `waitForTimeout` `captureAppViews` gives each tab before screenshotting,
  so its 3-card grid never actually existed in the DOM yet when `capture()`
  measured overflow — a real, previously-shipped `repeat(3,1fr)` phone-width
  overflow in that grid went undetected for exactly this reason. Recapture for
  real once `TWELVEDATA_API_KEY` authenticates:
  `curl -X POST http://localhost:8000/api/stress-test -H 'Content-Type: application/json' -d '{"tickers":["AAPL","MSFT","GOOGL","AMZN"],"weights":[0.30,0.25,0.25,0.20],"portfolio_value":10000}'`.

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
