"""Offline test of the RORT_PRIME1S_CACHE_MAX_ROWS row budget (board #68).

Bounds the KPI-HiFi whole-history load-once: LRU-bounded _1s_cache +
budget-stopped prime + full-day on-demand reads + coverage-aware warm
checks (a window-primed partial day-frame must never look warm to a later
window). Fakes the bar_cache module — no DB, no network. Companion to
test_prime1s_chunked.py (which must still pass with the flag unset).
"""
import sys, types, os
from datetime import datetime, timedelta, timezone
import pandas as pd

SRC = str(__import__("pathlib").Path(__file__).resolve().parent)
sys.path.insert(0, SRC)

calls = []

# Fixed global synthetic dataset — hourly grid, 24 rows/day.
_GLOBAL_IDX = pd.date_range("2026-05-30", "2026-06-12", freq="3600s", tz="UTC")
_GLOBAL_DF = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                           "close": 1.0, "volume": 1}, index=_GLOBAL_IDX)


def _fake_read_bars(ticker, tf, start, end):
    calls.append((pd.Timestamp(start), pd.Timestamp(end)))
    return _GLOBAL_DF[(_GLOBAL_DF.index >= pd.Timestamp(start)) &
                      (_GLOBAL_DF.index < pd.Timestamp(end))]


fake_bc = types.ModuleType("bar_cache")
fake_bc.is_enabled = lambda: True
fake_bc.direct_pg_available = lambda: True
fake_bc.read_bars = _fake_read_bars
sys.modules["bar_cache"] = fake_bc

import data_loader as dl

polygon_calls = []
_real_polygon_fetch = dl._polygon_fetch_bars


def _fake_polygon_fetch(ticker, multiplier, timespan, from_date, to_date):
    polygon_calls.append((from_date, to_date))
    day = pd.Timestamp(from_date, tz="UTC")
    sel = _GLOBAL_DF[(_GLOBAL_DF.index >= day) &
                     (_GLOBAL_DF.index < day + pd.Timedelta(days=1))]
    return [{"t": int(ts.value // 10**6), "o": r.open, "h": r.high,
             "l": r.low, "c": r.close, "v": r.volume}
            for ts, r in sel.iterrows()]


dl._polygon_fetch_bars = _fake_polygon_fetch

fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def cache_rows():
    return sum(len(v) for v in dl._1s_cache.values())


S = datetime(2026, 6, 1, tzinfo=timezone.utc)
E = datetime(2026, 6, 10, 23, 0, tzinfo=timezone.utc)
os.environ["RORT_PRIME1S_CHUNK_DAYS"] = "7"
os.environ.pop("RORT_PRIME1S_CACHE_MAX_ROWS", None)

# ── Baseline (flag OFF): whole-span prime + per-day windows ──
dl.clear_1s_cache()
calls.clear()
n = dl.prime_1s_cache_from_rest_bars("SPY", S, E)
check("O1 flag OFF: primes all 10 days (legacy behavior)", n == 10)
check("O2 flag OFF: nothing evicted", len(dl._1s_cache) == 10)
baseline = {}
for i in range(10):
    d = S + timedelta(days=i)
    baseline[i] = dl.fetch_1s_bars_for_window(
        "SPY", d + timedelta(hours=10), d + timedelta(hours=11),
        padding_seconds=30)
check("O3 flag OFF: baseline windows non-empty",
      all(len(v) > 0 for v in baseline.values()))
check("O4 flag OFF: no Polygon fallback", len(polygon_calls) == 0)

# ── T: flag ON, cap=100 rows (~4 of 24-row days) ──
os.environ["RORT_PRIME1S_CACHE_MAX_ROWS"] = "100"
dl.clear_1s_cache()
calls.clear()
n = dl.prime_1s_cache_from_rest_bars("SPY", S, E)
check("T1 budget-stop: 1 chunk read (168 rows ≥ cap), not 2",
      len(calls) == 1)
check("T2 resident rows ≤ cap after prime", cache_rows() <= 100)
check("T3 some days deferred/evicted", len(dl._1s_cache) < 10)

# Chronological sweep — the KPI-HiFi access pattern.
per_window = {}
for i in range(10):
    d = S + timedelta(days=i)
    per_window[i] = dl.fetch_1s_bars_for_window(
        "SPY", d + timedelta(hours=10), d + timedelta(hours=11),
        padding_seconds=30)
    if cache_rows() > 100:
        check("T4 cap held during sweep (day %d)" % i, False)
check("T4 cap held during sweep", cache_rows() <= 100)
check("T5 every window byte-identical to flag-OFF baseline",
      all(per_window[i].equals(baseline[i]) for i in range(10)))
check("T6 no Polygon fallback under budget (bar_cache served all)",
      len(polygon_calls) == 0)
# No-thrash: forward sweep must never re-read the same span twice.
check("T7 no repeated identical reads (no thrash)",
      len(calls) == len(set(calls)))
check("T8 total reads bounded (initial + ≤1/day on demand)",
      len(calls) <= 11)
check("T9 LRU: earliest day evicted, latest resident",
      "SPY_2026-06-01" not in dl._1s_cache
      and "SPY_2026-06-10" in dl._1s_cache)

# Second, LATER window on a still-resident day: full-day on-demand frames
# must serve it with zero new reads (the partial-poisoning fix).
calls.clear()
d9 = S + timedelta(days=8)  # Jun 9
w2 = dl.fetch_1s_bars_for_window(
    "SPY", d9 + timedelta(hours=15), d9 + timedelta(hours=16),
    padding_seconds=30)
check("U1 later same-day window served warm (0 reads)", len(calls) == 0)
check("U2 later same-day window correct rows", len(w2) == 2)

# ── V: prime-side coverage guard — a poisoned PARTIAL frame must be
#      evicted + re-read, not served ──
partial = _GLOBAL_DF[(_GLOBAL_DF.index >= "2026-06-05 10:00:00+00:00") &
                     (_GLOBAL_DF.index <= "2026-06-05 11:00:00+00:00")]
dl._1s_cache_put("SPY_2026-06-05", partial.copy(),
                 cov_start=pd.Timestamp("2026-06-05 10:00:00+00:00"),
                 cov_end=pd.Timestamp("2026-06-05 11:00:00+00:00"))
calls.clear()
d5 = S + timedelta(days=4)  # Jun 5
w3 = dl.fetch_1s_bars_for_window(
    "SPY", d5 + timedelta(hours=13), d5 + timedelta(hours=14),
    padding_seconds=30)
check("V1 partial frame evicted + full day re-read (1 read)",
      len(calls) == 1)
check("V2 out-of-coverage window served correctly (not silently empty)",
      len(w3) == 2 and w3.equals(
          _GLOBAL_DF[(_GLOBAL_DF.index >= "2026-06-05 13:00:00+00:00") &
                     (_GLOBAL_DF.index <= "2026-06-05 14:00:00+00:00")]))

# ── W: fetch-side coverage guard (inline prime disabled) → full-day
#      Polygon fallback replaces the partial frame ──
os.environ["RORT_FETCH1S_FROM_CACHE"] = "0"
dl._1s_cache_put("SPY_2026-06-05", partial.copy(),
                 cov_start=pd.Timestamp("2026-06-05 10:00:00+00:00"),
                 cov_end=pd.Timestamp("2026-06-05 11:00:00+00:00"))
calls.clear()
polygon_calls.clear()
w4 = dl.fetch_1s_bars_for_window(
    "SPY", d5 + timedelta(hours=13), d5 + timedelta(hours=14),
    padding_seconds=30)
check("W1 prime skipped: 0 bar_cache reads", len(calls) == 0)
check("W2 partial frame evicted → Polygon full-day fetch",
      len(polygon_calls) == 1)
check("W3 rows correct via fallback", len(w4) == 2)
os.environ["RORT_FETCH1S_FROM_CACHE"] = "1"

# ── X: a single frame larger than the cap survives (never evict the
#      just-inserted frame), then gets evicted by the next insert ──
os.environ["RORT_PRIME1S_CACHE_MAX_ROWS"] = "10"
dl.clear_1s_cache()
big = _GLOBAL_DF.head(24).copy()
dl._1s_cache_put("SPY_2026-05-30", big)
check("X1 oversized lone frame stays resident", len(dl._1s_cache) == 1)
dl._1s_cache_put("SPY_2026-05-31", _GLOBAL_DF.head(5).copy())
check("X2 next insert evicts the oversized LRU frame",
      "SPY_2026-05-30" not in dl._1s_cache
      and "SPY_2026-05-31" in dl._1s_cache)

# ── Y: flag OFF again → exact legacy behavior (no budget, no eviction) ──
os.environ.pop("RORT_PRIME1S_CACHE_MAX_ROWS", None)
dl.clear_1s_cache()
calls.clear()
n = dl.prime_1s_cache_from_rest_bars("SPY", S, E)
check("Y1 flag OFF: 2 chunk reads again", len(calls) == 2)
check("Y2 flag OFF: all 10 days resident", len(dl._1s_cache) == 10
      and n == 10)
off_sweep_ok = True
for i in range(10):
    d = S + timedelta(days=i)
    got = dl.fetch_1s_bars_for_window(
        "SPY", d + timedelta(hours=10), d + timedelta(hours=11),
        padding_seconds=30)
    off_sweep_ok = off_sweep_ok and got.equals(baseline[i])
check("Y3 flag OFF sweep byte-identical to baseline", off_sweep_ok)

print("\n%d failures" % len(fails))
sys.exit(1 if fails else 0)
