"""
cache.py
--------
Redis-backed caching for external data fetches (yfinance).

Fails open: if Redis isn't installed, isn't running, or drops mid-session,
every cache operation silently becomes a no-op and the app behaves exactly
as it would with no cache at all. Redis is a soft dependency — never a hard
one.

Key pattern: varense:{prefix}:{args_hash}
"""

import functools
import hashlib
import os
import pickle

try:
    import redis
except ImportError:
    redis = None

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

_client = None


def _connect():
    """Try once to reach Redis. Any failure just leaves the client as None."""
    global _client
    if redis is None:
        return
    try:
        client = redis.from_url(REDIS_URL, socket_connect_timeout=1, socket_timeout=1)
        client.ping()
        _client = client
    except Exception:
        _client = None


_connect()


def _make_key(prefix: str, args: tuple, kwargs: dict) -> str:
    raw = repr(args) + repr(sorted(kwargs.items()))
    args_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"varense:{prefix}:{args_hash}"


def cache_get(key: str):
    if _client is None:
        return None
    try:
        raw = _client.get(key)
        return pickle.loads(raw) if raw is not None else None
    except Exception:
        return None


def cache_set(key: str, value, ttl: int) -> None:
    if _client is None:
        return
    try:
        _client.setex(key, ttl, pickle.dumps(value))
    except Exception:
        pass


def cached(ttl: int, prefix: str | None = None):
    """
    Decorator that caches a function's return value in Redis (pickled),
    keyed on its arguments. No-ops transparently whenever Redis is
    unavailable, so the decorated function just runs normally.
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
            cache_set(key, result, ttl)
            return result

        return wrapper
    return decorator
