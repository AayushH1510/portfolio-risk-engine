"""
rate_limit.py
-------------
Shared slowapi Limiter, keyed by client IP (via X-Forwarded-For on Render,
falling back to the direct connecting IP). Applied to the endpoints in
api.py and stock_detail_route.py that hit an external data provider
(yfinance, Finnhub) and cost real compute per request.

Lives in its own module (rather than api.py) so stock_detail_route.py can
import the same `limiter` instance without a circular import — api.py
already imports from stock_detail_route.py.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
