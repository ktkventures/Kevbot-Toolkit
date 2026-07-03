"""Offline test of prime_1s_cache_from_rest_bars chunking + breaker.

Fakes the bar_cache module — no DB, no network.
"""
import sys, types, os
from datetime import datetime, timedelta, timezone
import pandas as pd

SRC = str(__import__("pathlib").Path(__file__).resolve().parent)
sys.path.insert(0, SRC)

calls = []
fail_mode = {"on": False}


# Fixed global synthetic dataset — read_bars slices it, so chunked and
# whole-span reads MUST return identical rows (like the real store).
_GLOBAL_IDX = pd.date_range("2026-05-30", "2026-06-12", freq="3600s", tz="UTC")
_GLOBAL_DF = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                           "close": 1.0, "volume": 1}, index=_GLOBAL_IDX)


def _fake_read_bars(ticker, tf, start, end):
    calls.append((start, end))
    if fail_mode["on"]:
        raise RuntimeError("canceling statement due to statement timeout")
    return _GLOBAL_DF[(_GLOBAL_DF.index >= pd.Timestamp(start)) &
                      (_GLOBAL_DF.index < pd.Timestamp(end))]


fake_bc = types.ModuleType("bar_cache")
fake_bc.is_enabled = lambda: True
fake_bc.direct_pg_available = lambda: True
fake_bc.read_bars = _fake_read_bars
sys.modules["bar_cache"] = fake_bc

import data_loader as dl

fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


S = datetime(2026, 6, 1, tzinfo=timezone.utc)
E = datetime(2026, 6, 10, 23, 0, tzinfo=timezone.utc)

# ── A: chunked read, 10-day span, chunk=7 → 2 windows ──
os.environ["RORT_PRIME1S_CHUNK_DAYS"] = "7"
dl.clear_1s_cache()
calls.clear()
n = dl.prime_1s_cache_from_rest_bars("SPY", S, E)
check("A1 two chunk queries (not one giant)", len(calls) == 2)
check("A2 primed 10 day frames (padded May 31 has no rows on hourly grid)", n == 10)
check("A3 chunk bounds clamp to padded span",
      calls[0][0] == S - timedelta(seconds=60)
      and calls[-1][1] == E + timedelta(seconds=60))
chunked_cache = {k: v.copy() for k, v in dl._1s_cache.items()}

# ── B: warm-skip — second call issues ZERO queries ──
calls.clear()
n2 = dl.prime_1s_cache_from_rest_bars("SPY", S, E)
check("B1 all-warm second call → 0 queries", len(calls) == 0)
check("B2 returns 0 primed", n2 == 0)

# ── C: failure trips breaker; next call short-circuits ──
dl.clear_1s_cache()
calls.clear()
fail_mode["on"] = True
n3 = dl.prime_1s_cache_from_rest_bars("SPY", S, E)
check("C1 failure → exactly 1 query then abandon", len(calls) == 1)
check("C2 returns 0", n3 == 0)
calls.clear()
n4 = dl.prime_1s_cache_from_rest_bars("SPY", S, E)
check("C3 breaker OPEN → 0 queries on next call", len(calls) == 0)
dl._prime_1s_breaker_until = 0.0
fail_mode["on"] = False

# ── D: kill-switch → legacy single whole-span read ──
os.environ["RORT_PRIME1S_CHUNKED"] = "0"
dl.clear_1s_cache()
calls.clear()
n5 = dl.prime_1s_cache_from_rest_bars("SPY", S, E)
check("D1 legacy single query", len(calls) == 1)
check("D2 primed 10 days", n5 == 10)
legacy_cache = {k: v.copy() for k, v in dl._1s_cache.items()}
os.environ["RORT_PRIME1S_CHUNKED"] = "1"

# ── E: chunked cache content identical to legacy ──
same_keys = set(chunked_cache) == set(legacy_cache)
same_frames = same_keys and all(
    chunked_cache[k].equals(legacy_cache[k]) for k in chunked_cache)
check("E1 chunked vs legacy: identical day keys", same_keys)
check("E2 chunked vs legacy: byte-identical frames", same_frames)

# ── F: partial warm — only cold run queried ──
dl.clear_1s_cache()
poly = "SPY"
# Warm everything through Jun 5 (including padded-in May 31) → cold run =
# Jun 6..Jun 10 → exactly one chunk window starting Jun 6.
for i in range(-1, 5):
    d = S.date() + timedelta(days=i)
    dl._1s_cache[f"{poly}_{d.isoformat()}"] = pd.DataFrame({"close": [1.0]})
calls.clear()
dl.prime_1s_cache_from_rest_bars("SPY", S, E)
check("F1 only cold tail queried (1 window, days 6-10)",
      len(calls) == 1 and calls[0][0].date() == S.date() + timedelta(days=5))

print("\n%d failures" % len(fails))
sys.exit(1 if fails else 0)
