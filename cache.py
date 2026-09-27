"""
cache.py
--------
Redis-backed caching for external data fetches (Twelve Data, Finnhub).

Fails open: if Redis isn't installed, isn't running, or drops mid-session,
every cache operation silently becomes a no-op and the app behaves exactly
as it would with no cache at all. Redis is a soft dependency — never a hard
one.

Key pattern: varense:{prefix}:{args_hash}
"""

import functools
import hashlib
import logging
import os
import pickle
from urllib.parse import urlparse

try:
    import redis
except ImportError:
    redis = None

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

_client = None


def _redact(url: str) -> str:
    """Host:port only, for logging — REDIS_URL embeds a password (Upstash
    always does), so the full URL must never reach the logs."""
    try:
        parsed = urlparse(url)
        return f"{parsed.hostname}:{parsed.port}" if parsed.hostname else "unknown-host"
    except Exception:
        return "unknown-host"


def _connect():
    """Try once to reach Redis. Any failure just leaves the client as None.

    Logs the outcome either way — previously this was silent, so a dead
    Redis at startup looked identical to a healthy one right up until every
    cache_get() started quietly missing forever (see cache_get's own
    comment for why misses are otherwise indistinguishable from heavy load
    on nothing but request latency).
    """
    global _client
    if redis is None:
        logger.warning("redis package not installed — caching disabled")
        _client = None
        return
    try:
        client = redis.from_url(REDIS_URL, socket_connect_timeout=1, socket_timeout=1)
        client.ping()
        _client = client
        logger.info("Redis connected (%s)", _redact(REDIS_URL))
    except Exception as e:
        logger.warning("Redis connection failed (%s): %s", _redact(REDIS_URL), e)
        _client = None


_connect()


def is_available() -> bool:
    """
    Live reachability check for /api/health — pings Redis right now rather
    than trusting the result of the startup connection attempt above, so a
    mid-session Redis outage (or recovery) is visible immediately instead
    of only after a restart. Self-healing: if the client is currently None
    (never connected, or a previous ping killed it below), this retries the
    connection rather than reporting "down" forever.
    """
    global _client
    if _client is None:
        _connect()
        if _client is None:
            return False
    try:
        _client.ping()
        return True
    except Exception as e:
        logger.warning("Redis ping failed, marking connection dead: %s", e)
        _client = None
        return False


def _make_key(prefix: str, args: tuple, kwargs: dict) -> str:
    raw = repr(args) + repr(sorted(kwargs.items()))
    args_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"varense:{prefix}:{args_hash}"


def cache_get(key: str):
    # INFO, not DEBUG — the whole point of this logging (see cache.py's
    # module docstring / the request this addresses) is that a dead Redis
    # previously looked *identical* to heavy upstream load: same symptom
    # (everything slow), nothing in the logs to tell them apart. That only
    # works as a diagnostic if it shows up at whatever level Render's
    # logs actually display by default, not behind a DEBUG flag nobody
    # will think to flip mid-incident.
    if _client is None:
        logger.info("cache miss (no connection): %s", key)
        return None
    try:
        raw = _client.get(key)
        if raw is None:
            logger.info("cache miss: %s", key)
            return None
        logger.info("cache hit: %s", key)
        return pickle.loads(raw)
    except Exception as e:
        logger.warning("Redis get failed for %s: %s", key, e)
        return None


def cache_set(key: str, value, ttl: int) -> None:
    if _client is None:
        return
    try:
        _client.setex(key, ttl, pickle.dumps(value))
    except Exception as e:
        logger.warning("Redis set failed for %s: %s", key, e)


def cache_incr(key: str, ttl: int, by: int = 1) -> int | None:
    """
    Atomically increment a counter (creating it at 0 first if absent) and
    return the new value. TTL is set only on the increment that actually
    creates the key (detected as "the new value equals what we just added",
    i.e. it started from 0) so repeated calls don't keep pushing the
    expiry back — this is a fixed-window daily counter (see
    twelvedata_utils.py's call-volume tracking), not a rolling one.

    Separate from `cached`/`cache_get`/`cache_set` above: those cache a
    function's pickled return value, this just counts. No-ops (returns
    None) whenever Redis is unavailable, same fail-open contract as the
    rest of this module.
    """
    if _client is None:
        return None
    try:
        count = _client.incrby(key, by)
        if count == by:
            _client.expire(key, ttl)
        return count
    except Exception:
        return None


def cached(ttl: int, prefix: str | None = None, should_cache=None):
    """
    Decorator that caches a function's return value in Redis (pickled),
    keyed on its arguments. No-ops transparently whenever Redis is
    unavailable, so the decorated function just runs normally.

    `should_cache`, if given, is called with the function's result — the
    result is only cached if it returns True. Lets a caller skip caching a
    bad or partial result (e.g. too few rows from a rate-limited fetch) so
    a transient failure can't poison the cache for the full TTL.
    """
    def decorator(func):
        key_prefix = prefix or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = _make_key(key_prefix, args, kwargs)
            cached_value = cache_get(key)
            if cached_value is not None:
                return cached_value
            result = func(*args, **kwargs)
            if should_cache is None or should_cache(result):
                cache_set(key, result, ttl)
            return result

        return wrapper
    return decorator
