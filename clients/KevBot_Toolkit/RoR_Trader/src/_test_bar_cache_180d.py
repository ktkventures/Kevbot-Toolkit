"""180-day cache vs direct-Polygon timing test. No strategy state touched.

Run from src/:
    ../.venv/bin/python _test_bar_cache_180d.py [SYMBOL] [DAYS]
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import importlib
import os
import sys
import time

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "SPY"
DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 180
TF = "1Min"

print(f"180-day test: {SYMBOL} {TF} × {DAYS} days RTH")
print("=" * 70)

# ---------- Direct Polygon (cache disabled) ----------
os.environ["BAR_CACHE_ENABLED"] = "false"
import bar_cache, data_loader  # noqa
importlib.reload(bar_cache)
importlib.reload(data_loader)

t0 = time.time()
direct = data_loader.load_market_data(
    symbol=SYMBOL, days=DAYS, timeframe=TF, session="RTH")
t_direct = time.time() - t0
n_direct = len(direct) if direct is not None else 0
print(f"Direct Polygon:  {t_direct:6.2f}s   rows={n_direct}")

# ---------- Cache cold (will full-backfill if symbol/range not cached) ----------
os.environ["BAR_CACHE_ENABLED"] = "true"
importlib.reload(bar_cache)
importlib.reload(data_loader)

t0 = time.time()
cold = data_loader.load_market_data(
    symbol=SYMBOL, days=DAYS, timeframe=TF, session="RTH")
t_cold = time.time() - t0
n_cold = len(cold) if cold is not None else 0
print(f"Cache cold:      {t_cold:6.2f}s   rows={n_cold}")

# ---------- Cache warm ----------
t0 = time.time()
warm = data_loader.load_market_data(
    symbol=SYMBOL, days=DAYS, timeframe=TF, session="RTH")
t_warm = time.time() - t0
n_warm = len(warm) if warm is not None else 0
print(f"Cache warm:      {t_warm:6.2f}s   rows={n_warm}")

print()
# Coverage delta
if n_direct > 0:
    cold_pct = 100.0 * n_cold / n_direct
    warm_pct = 100.0 * n_warm / n_direct
    print(f"Coverage  cold: {cold_pct:5.1f}%   warm: {warm_pct:5.1f}%")
if t_warm > 0:
    print(f"Speedup direct→warm: {t_direct / t_warm:.1f}x")

# Cache stats
from bar_cache import cache_stats
print(f"Cache stats: {cache_stats()}")
