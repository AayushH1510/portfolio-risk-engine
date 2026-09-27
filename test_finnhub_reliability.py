"""
test_finnhub_reliability.py
----------------------------
Regression tests for the Finnhub reliability fixes: event-loop blocking,
stock-detail/fundamentals caching, transient-vs-permanent failure handling,
the shared throttle, and /api/health's Redis reporting.

Everything here runs against mocked Finnhub responses (requests.get is
patched in every test — nothing here ever makes a real HTTP call) and a
fake in-memory Redis stand-in, per the request that prompted these fixes:
"Test against mocked Finnhub responses ... rather than live production."

Uses the stdlib `unittest` rather than pytest — no test framework was
already a dependency of this project, and stdlib is enough for what's
being verified here.

Run: venv/Scripts/python.exe -m unittest test_finnhub_reliability -v
"""

import inspect
import logging
import time
import unittest
from unittest.mock import patch, Mock

import requests
from fastapi.testclient import TestClient

import api
import cache
import stock_detail_route as sdr

# This environment's own .env only defines TWELVEDATA_API_KEY (confirmed —
# FINNHUB_API_KEY is unset here), and stock_detail's route hard-fails with
# a 500 before ever attempting a Finnhub call if it's falsy. Patched at
# import time, for the whole test run: every test in this file mocks
# requests.get anyway, so the key's actual value is never used for a real
# call, only checked for truthiness.
sdr.FINNHUB_API_KEY = "test-key"


# ─────────────────────────────────────────────────────────────────────────
# Test doubles
# ─────────────────────────────────────────────────────────────────────────

class FakeRedis:
    """
    Minimal in-memory stand-in for the real redis-py client — just enough
    of .get/.setex/.ping for cache.py's own cache_get/cache_set/is_available
    to work unmodified against it. Patched onto cache._client (not onto
    cache_get/cache_set themselves) so both api.py's `from cache import
    cache_get` binding and stock_detail_route.py's `@cached` decorator (which
    resolves cache_get/cache_set from cache.py's own module globals at call
    time) go through the exact same fake store, rather than needing two
    separate patches that could silently drift out of sync.
    """
    def __init__(self):
        self.store = {}  # key -> (raw_bytes, expire_at_monotonic | None)

    def get(self, key):
        entry = self.store.get(key)
        if entry is None:
            return None
        value, expire_at = entry
        if expire_at is not None and time.monotonic() > expire_at:
            del self.store[key]
            return None
        return value

    def setex(self, key, ttl, value):
        self.store[key] = (value, time.monotonic() + ttl)

    def ping(self):
        return True

    def ttl_of(self, key):
        """Test helper, not part of the real redis-py API surface — lets a
        test assert *which* TTL bucket a value landed in (the ~60s failure
        bucket vs. the 24h success bucket) without sleeping for real."""
        entry = self.store.get(key)
        if entry is None:
            return None
        return entry[1] - time.monotonic()


def make_response(status_code=200, json_data=None):
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    if status_code >= 400 and status_code not in (401, 429):
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status_code} error")
    else:
        resp.raise_for_status.return_value = None
    return resp


QUOTE_OK    = {"c": 150.25, "d": 1.5, "dp": 1.01}
QUOTE_ZERO  = {"c": 0, "d": None, "dp": None}
PROFILE_OK  = {"name": "Apple Inc", "finnhubIndustry": "Technology", "marketCapitalization": 3_000_000}
METRIC_OK   = {"metric": {"peTTM": 30.5, "beta": 1.2, "52WeekHigh": 200, "52WeekLow": 100}}


def route_by_path(app, path):
    for r in app.routes:
        if getattr(r, "path", None) == path:
            return r.endpoint
    raise AssertionError(f"route {path} not found")


# ─────────────────────────────────────────────────────────────────────────
# 1. Every route touching blocking I/O or heavy compute is a plain `def`
# ─────────────────────────────────────────────────────────────────────────

class RouteIsSyncTests(unittest.TestCase):
    """Item 1: confirms the actual mechanism of the fix, not just its
    symptom — a route that's still `async def` would still block the event
    loop on a slow Finnhub/Twelve Data call no matter how everything else
    below behaves."""

    def test_stock_detail_is_sync(self):
        self.assertFalse(inspect.iscoroutinefunction(sdr.stock_detail))

    def test_api_routes_are_sync(self):
        for path in ["/api/health", "/api/validate", "/api/fundamentals",
                     "/api/stress-test", "/api/analyse",
                     "/api/analyse-summary", "/api/analyse-full"]:
            with self.subTest(path=path):
                self.assertFalse(inspect.iscoroutinefunction(route_by_path(api.app, path)),
                                  f"{path} is still async def")


# ─────────────────────────────────────────────────────────────────────────
# 2. /stock-detail/{ticker} — partial data, timeouts, 429s, not-found
# ─────────────────────────────────────────────────────────────────────────

class StockDetailPartialDataTests(unittest.TestCase):
    def setUp(self):
        self.fake_redis = FakeRedis()
        self._client_patch = patch.object(cache, "_client", self.fake_redis)
        self._client_patch.start()
        self.client = TestClient(api.app)

    def tearDown(self):
        self._client_patch.stop()

    @patch("requests.get")
    def test_all_three_succeed(self, mock_get):
        mock_get.side_effect = [
            make_response(200, QUOTE_OK),
            make_response(200, PROFILE_OK),
            make_response(200, METRIC_OK),
        ]
        r = self.client.get("/stock-detail/AAPL")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["price"], 150.25)
        self.assertEqual(body["company_name"], "Apple Inc")
        self.assertEqual(body["pe_ratio"], 30.5)

    @patch("requests.get")
    def test_profile_and_metric_timeout_still_returns_price(self, mock_get):
        """Item 6: 'a drawer with price and profile but no metrics beats an
        error' — here it's price with NO profile and NO metrics, an even
        more degraded case, which should still succeed with 200."""
        def side_effect(url, params, timeout):
            if "quote" in url:
                return make_response(200, QUOTE_OK)
            raise requests.exceptions.ReadTimeout("Read timed out.")
        mock_get.side_effect = side_effect

        r = self.client.get("/stock-detail/AAPL")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["price"], 150.25)
        self.assertIsNone(body["company_name"])
        self.assertIsNone(body["pe_ratio"])

    @patch("requests.get")
    def test_quote_times_out_but_profile_succeeds(self, mock_get):
        """The less obvious partial case: the headline number (price) is
        the one that's missing, not an enhancement — still 200, not a hard
        error, per the same 'return whatever arrived' instruction."""
        def side_effect(url, params, timeout):
            if "quote" in url:
                raise requests.exceptions.ReadTimeout("Read timed out.")
            if "profile2" in url:
                return make_response(200, PROFILE_OK)
            return make_response(200, METRIC_OK)
        mock_get.side_effect = side_effect

        r = self.client.get("/stock-detail/AAPL")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIsNone(body["price"])
        self.assertEqual(body["company_name"], "Apple Inc")

    @patch("requests.get")
    def test_all_three_timeout_returns_error_not_hang(self, mock_get):
        mock_get.side_effect = requests.exceptions.ReadTimeout("Read timed out.")
        start = time.monotonic()
        r = self.client.get("/stock-detail/AAPL")
        elapsed = time.monotonic() - start
        self.assertEqual(r.status_code, 500)
        # Bounded, not "eventually" — three independent calls, each mocked
        # to raise instantly, so this should return in well under a second,
        # not the 15-30s a real timeout chain could have taken pre-fix.
        self.assertLess(elapsed, 2.0)

    @patch("requests.get")
    def test_all_three_rate_limited_returns_503(self, mock_get):
        mock_get.return_value = make_response(429)
        r = self.client.get("/stock-detail/AAPL")
        self.assertEqual(r.status_code, 503)
        self.assertIn("rate limit", r.json()["detail"].lower())

    @patch("requests.get")
    def test_genuinely_invalid_ticker_is_404(self, mock_get):
        mock_get.return_value = make_response(200, QUOTE_ZERO)
        r = self.client.get("/stock-detail/NOTATICKER")
        self.assertEqual(r.status_code, 404)

    @patch("requests.get")
    def test_finnhub_timeout_reduced_to_5s(self, mock_get):
        """Item 5 — confirms the actual timeout value passed to requests,
        not just that a timeout error is handled somewhere."""
        mock_get.return_value = make_response(200, QUOTE_OK)
        sdr._get_quote.__wrapped__("AAPL") if hasattr(sdr._get_quote, "__wrapped__") else sdr._finnhub_get("quote", {"symbol": "AAPL"})
        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs["timeout"], 5)


# ─────────────────────────────────────────────────────────────────────────
# 3. stock-detail caching (item 2) — quote 60s, profile/metric 24h
# ─────────────────────────────────────────────────────────────────────────

class StockDetailCachingTests(unittest.TestCase):
    def setUp(self):
        self.fake_redis = FakeRedis()
        self._client_patch = patch.object(cache, "_client", self.fake_redis)
        self._client_patch.start()

    def tearDown(self):
        self._client_patch.stop()

    @patch("requests.get")
    def test_quote_cached_across_calls(self, mock_get):
        mock_get.return_value = make_response(200, QUOTE_OK)
        sdr._get_quote("AAPL")
        sdr._get_quote("AAPL")
        self.assertEqual(mock_get.call_count, 1, "second call should be served from cache")

    @patch("requests.get")
    def test_profile_and_metric_cached_across_calls(self, mock_get):
        mock_get.side_effect = [make_response(200, PROFILE_OK), make_response(200, METRIC_OK)]
        sdr._get_profile("AAPL")
        sdr._get_metric("AAPL")
        sdr._get_profile("AAPL")
        sdr._get_metric("AAPL")
        self.assertEqual(mock_get.call_count, 2, "profile+metric should each be fetched once, then cached")

    @patch("requests.get")
    def test_quote_ttl_is_60s_profile_ttl_is_24h(self, mock_get):
        mock_get.side_effect = [make_response(200, QUOTE_OK), make_response(200, PROFILE_OK)]
        sdr._get_quote("AAPL")
        sdr._get_profile("AAPL")
        quote_key  = [k for k in self.fake_redis.store if "finnhub_quote" in k][0]
        profile_key = [k for k in self.fake_redis.store if "finnhub_profile" in k][0]
        self.assertLessEqual(self.fake_redis.ttl_of(quote_key), 60)
        self.assertGreater(self.fake_redis.ttl_of(profile_key), 3600)  # much more than an hour

    @patch("requests.get")
    def test_quote_failure_is_not_cached(self, mock_get):
        """Only successes are cached for stock-detail (unlike fundamentals,
        which gets the two-tier treatment) — a failed quote should retry
        on the very next call, not get stuck."""
        mock_get.side_effect = requests.exceptions.ReadTimeout("Read timed out.")
        with self.assertRaises(requests.exceptions.ReadTimeout):
            sdr._get_quote("AAPL")
        mock_get.side_effect = [make_response(200, QUOTE_OK)]
        result = sdr._get_quote("AAPL")
        self.assertEqual(result["c"], 150.25)


# ─────────────────────────────────────────────────────────────────────────
# 4. /api/fundamentals two-tier caching (item 3 + item 4)
# ─────────────────────────────────────────────────────────────────────────

class FundamentalsCachingTests(unittest.TestCase):
    def setUp(self):
        self.fake_redis = FakeRedis()
        self._client_patch = patch.object(cache, "_client", self.fake_redis)
        self._client_patch.start()

    def tearDown(self):
        self._client_patch.stop()

    @patch("requests.get")
    def test_success_cached_24h_not_1h(self, mock_get):
        mock_get.side_effect = [make_response(200, QUOTE_OK), make_response(200, PROFILE_OK), make_response(200, METRIC_OK)]
        api._fetch_ticker_fundamentals("AAPL")
        key = "varense:fundamentals:AAPL"
        ttl = self.fake_redis.ttl_of(key)
        self.assertGreater(ttl, 23 * 3600, "should be cached for ~24h (DAY), not the old 1h")
        # Second call must not touch Finnhub again.
        call_count_before = mock_get.call_count
        api._fetch_ticker_fundamentals("AAPL")
        self.assertEqual(mock_get.call_count, call_count_before)

    @patch("requests.get")
    def test_timeout_cached_briefly_and_retried_on_every_call_otherwise(self, mock_get):
        """The exact bug this fixes: before this change, a ticker that
        timed out was NEVER cached and hit Finnhub fresh on every single
        request. Confirms both halves — it's cached at all (call count
        doesn't grow within the window) and the TTL is the short bucket,
        not the 24h one (so it recovers quickly, doesn't get stuck for a
        day behind one bad timeout)."""
        mock_get.side_effect = requests.exceptions.ReadTimeout("Read timed out.")
        with self.assertRaises(requests.exceptions.ReadTimeout):
            api._fetch_ticker_fundamentals("MSFT")
        # _fetch_ticker_fundamentals_raw is unchanged, linear code (fetches
        # quote, then profile, then metric) — it aborts on quote's own
        # timeout without ever attempting profile/metric. Unlike
        # stock-detail (item 6), fundamentals was never asked to fetch the
        # three independently; only its caching behaviour changed here.
        calls_after_first_failure = mock_get.call_count
        self.assertEqual(calls_after_first_failure, 1, "aborts at quote, never reaches profile/metric")

        # Immediately retrying should NOT hit Finnhub again — served from
        # the brief failure cache instead.
        with self.assertRaises(Exception):
            api._fetch_ticker_fundamentals("MSFT")
        self.assertEqual(mock_get.call_count, calls_after_first_failure,
                          "a second request within the failure window must not re-hit Finnhub")

        key = "varense:fundamentals:MSFT"
        ttl = self.fake_redis.ttl_of(key)
        self.assertLessEqual(ttl, 60, "transient failures must use the short ~60s bucket")
        self.assertLess(ttl, 3600, "must not accidentally land in the 24h success bucket")

    @patch("requests.get")
    def test_rate_limit_cached_briefly(self, mock_get):
        mock_get.return_value = make_response(429)
        with self.assertRaises(sdr.FinnhubRateLimitError):
            api._fetch_ticker_fundamentals("GOOGL")
        calls_after_first = mock_get.call_count
        with self.assertRaises(sdr.FinnhubRateLimitError):
            api._fetch_ticker_fundamentals("GOOGL")
        self.assertEqual(mock_get.call_count, calls_after_first, "429 must be cached briefly too")

    @patch("requests.get")
    def test_not_found_cached_24h(self, mock_get):
        """A genuinely bad ticker (Finnhub responds, price is zero) is
        cached at the SAME long TTL as a success — it's not going to start
        existing on retry, so there's no reason to keep re-asking Finnhub
        about it every 60 seconds the way a real transient failure should."""
        mock_get.return_value = make_response(200, QUOTE_ZERO)
        with self.assertRaises(ValueError):
            api._fetch_ticker_fundamentals("NOTATICKER")
        key = "varense:fundamentals:NOTATICKER"
        ttl = self.fake_redis.ttl_of(key)
        self.assertGreater(ttl, 23 * 3600)

        calls_before = mock_get.call_count
        with self.assertRaises(ValueError):
            api._fetch_ticker_fundamentals("NOTATICKER")
        self.assertEqual(mock_get.call_count, calls_before)

    @patch("requests.get")
    def test_fundamentals_route_still_returns_error_shape_for_failed_ticker(self, mock_get):
        """End-to-end through the actual route, not just the cached helper
        function directly — confirms the route's existing per-ticker error
        response shape is unaffected by the caching rewrite."""
        def side_effect(url, params, timeout):
            if "AAPL" in params.get("symbol", ""):
                if "quote" in url:
                    return make_response(200, QUOTE_OK)
                if "profile2" in url:
                    return make_response(200, PROFILE_OK)
                return make_response(200, METRIC_OK)
            raise requests.exceptions.ReadTimeout("Read timed out.")
        mock_get.side_effect = side_effect

        client = TestClient(api.app)
        r = client.get("/api/fundamentals", params={"tickers": "AAPL,MSFT"})
        self.assertEqual(r.status_code, 200)
        results = {row["ticker"]: row for row in r.json()["tickers"]}
        self.assertNotIn("error", results["AAPL"])
        self.assertIn("error", results["MSFT"])


# ─────────────────────────────────────────────────────────────────────────
# 5. Shared Finnhub throttle (item 7)
# ─────────────────────────────────────────────────────────────────────────

class ThrottleTests(unittest.TestCase):
    def setUp(self):
        # Swap in small, fast-to-test values rather than the real 60/60s —
        # the mechanism being tested (does a call past the limit queue
        # instead of firing immediately) doesn't depend on the specific
        # numbers, and this test shouldn't take a real minute to run.
        self._orig_limit = sdr.FINNHUB_RATE_LIMIT
        self._orig_window = sdr.FINNHUB_RATE_WINDOW
        sdr.FINNHUB_RATE_LIMIT = 3
        sdr.FINNHUB_RATE_WINDOW = 1.0
        sdr._finnhub_call_times.clear()

    def tearDown(self):
        sdr.FINNHUB_RATE_LIMIT = self._orig_limit
        sdr.FINNHUB_RATE_WINDOW = self._orig_window
        sdr._finnhub_call_times.clear()

    @patch("time.sleep")
    def test_burst_past_limit_queues_instead_of_firing_immediately(self, mock_sleep):
        for _ in range(3):
            sdr._throttle_finnhub()
        mock_sleep.assert_not_called()

        sdr._throttle_finnhub()
        mock_sleep.assert_called_once()
        waited = mock_sleep.call_args[0][0]
        self.assertGreater(waited, 0)
        self.assertLessEqual(waited, sdr.FINNHUB_RATE_WINDOW)

    def test_throttle_is_shared_across_fundamentals_and_stock_detail(self):
        """Item 7 says "one shared limiter" — confirms it's literally the
        same module-level state both call paths go through, not two
        independent copies that could each individually stay under the
        limit while the combined real call rate exceeds it."""
        self.assertIs(sdr._finnhub_get, api._finnhub_get)


# ─────────────────────────────────────────────────────────────────────────
# 6. /api/health reports Redis reachability (item 8)
# ─────────────────────────────────────────────────────────────────────────

class HealthEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    @patch("api.redis_is_available", return_value=True)
    def test_health_reports_redis_up(self, _mock):
        r = self.client.get("/api/health")
        self.assertEqual(r.json(), {"status": "ok", "redis": True})

    @patch("api.redis_is_available", return_value=False)
    def test_health_reports_redis_down_but_status_still_ok(self, _mock):
        """Redis is a soft dependency by design (cache.py fails open) — the
        app is genuinely fine with it down, just uncached, so overall
        `status` must stay "ok" even when `redis` is False."""
        r = self.client.get("/api/health")
        self.assertEqual(r.json(), {"status": "ok", "redis": False})


# ─────────────────────────────────────────────────────────────────────────
# 7. cache.py logs hits, misses, and connection failures (item 8)
# ─────────────────────────────────────────────────────────────────────────

class CacheLoggingTests(unittest.TestCase):
    def test_miss_then_hit_are_both_logged(self):
        fake = FakeRedis()
        with patch.object(cache, "_client", fake):
            with self.assertLogs("cache", level="INFO") as log:
                cache.cache_get("some:key")
            self.assertTrue(any("miss" in m for m in log.output))

            cache.cache_set("some:key", {"x": 1}, 60)
            with self.assertLogs("cache", level="INFO") as log:
                cache.cache_get("some:key")
            self.assertTrue(any("hit" in m for m in log.output))

    def test_connection_failure_is_logged_and_url_not_leaked(self):
        with patch.object(cache, "redis") as mock_redis_module:
            mock_redis_module.from_url.side_effect = Exception("connection refused")
            with patch.object(cache, "REDIS_URL", "redis://user:supersecret@myhost:6379/0"):
                with self.assertLogs("cache", level="WARNING") as log:
                    cache._connect()
                joined = " ".join(log.output)
                self.assertIn("myhost", joined)
                self.assertNotIn("supersecret", joined, "password must never reach the logs")

    def test_is_available_reflects_live_ping_not_stale_state(self):
        fake = FakeRedis()
        with patch.object(cache, "_client", fake):
            self.assertTrue(cache.is_available())

        broken = Mock()
        broken.ping.side_effect = Exception("connection reset")
        with patch.object(cache, "_client", broken):
            self.assertFalse(cache.is_available())


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main(verbosity=2)
