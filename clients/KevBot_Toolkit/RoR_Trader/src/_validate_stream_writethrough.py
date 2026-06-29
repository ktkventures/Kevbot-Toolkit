"""Step 1 of the bar_cache write-through unification (Design_BarCache_WriteThrough_Unification.md):
prove the data-worker's REST stream is byte-identical to a fresh Polygon pull on SETTLED bars,
BEFORE we wire any write-through. Offline — settled historical window, no live market needed.

Compares, for one (symbol, settled window):
  1. store 1Sec   vs Polygon native 1Sec   — ingest fidelity (should be trivially identical)
  2. store 1Min (RESAMPLED from 1s) vs Polygon native 1Min — THE write-through question
  3. bar_cache 1Sec/1Min vs Polygon native — current cache staleness (what write-through fixes)

Usage: PYTHONPATH=. ../.venv/bin/python _validate_stream_writethrough.py [SYMBOL] [YYYY-MM-DD] [HH:MM] [MINUTES]
"""
import os, sys, logging
from datetime import datetime, timedelta, timezone
import pandas as pd

os.environ["USE_DB"] = "true"
from dotenv import load_dotenv
load_dotenv(".env", override=True)
logging.basicConfig(level=logging.ERROR)

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "TSLA"
DAY = sys.argv[2] if len(sys.argv) > 2 else "2026-06-26"
HHMM = sys.argv[3] if len(sys.argv) > 3 else "14:30"
MINUTES = int(sys.argv[4]) if len(sys.argv) > 4 else 60

h, m = map(int, HHMM.split(":"))
start = datetime.fromisoformat(f"{DAY}T{h:02d}:{m:02d}:00+00:00")
end = start + timedelta(minutes=MINUTES)
print(f"=== write-through stream validation: {SYMBOL}  [{start.isoformat()} → {end.isoformat()}] (settled) ===", flush=True)

OHLCV = ["open", "high", "low", "close", "volume"]


def norm(df):
    """UTC index, numeric OHLCV, sorted, deduped — for byte comparison."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=OHLCV)
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    for c in OHLCV:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[[c for c in OHLCV if c in df.columns]]
    return df[~df.index.duplicated(keep="last")].sort_index()


def compare(name, a, b, tol=1e-6):
    """Compare two normalized bar frames on SHARED timestamps, broken out per column."""
    a, b = norm(a), norm(b)
    # restrict both to the requested window for a clean shared comparison
    a = a[(a.index >= pd.Timestamp(start)) & (a.index < pd.Timestamp(end))]
    b = b[(b.index >= pd.Timestamp(start)) & (b.index < pd.Timestamp(end))]
    only_a = a.index.difference(b.index)
    only_b = b.index.difference(a.index)
    both = a.index.intersection(b.index)
    per_col = {c: 0 for c in OHLCV}
    worst = {c: 0.0 for c in OHLCV}
    for ts in both:
        for c in OHLCV:
            if c in a.columns and c in b.columns:
                d = abs(float(a.at[ts, c]) - float(b.at[ts, c]))
                if d > tol:
                    per_col[c] += 1
                    worst[c] = max(worst[c], d)
    total = sum(per_col.values())
    ohlc_diffs = sum(per_col[c] for c in ("open", "high", "low", "close"))
    ok = (len(only_a) == 0 and len(only_b) == 0 and total == 0)
    flag = "✅ IDENTICAL" if ok else "❌ DIVERGE"
    print(f"  {flag}  {name}: A={len(a)} B={len(b)} shared={len(both)} "
          f"A_only={len(only_a)} B_only={len(only_b)}", flush=True)
    if total or len(only_a) or len(only_b):
        bycol = ", ".join(f"{c}={per_col[c]}(max {worst[c]:.4f})" for c in OHLCV if per_col[c])
        print(f"        diffs by col: {bycol or 'none'}  | OHLC diffs={ohlc_diffs} volume diffs={per_col['volume']}", flush=True)
    return ok


# ---- sources ----
from data_worker_ingest import _polygon_1s
import bar_cache
from bar_store import SymbolBarStore


def _polygon_native(symbol, mult, timespan, start_dt, end_dt):
    """Windowed native Polygon fetch via the same ms-epoch primitive the ingest uses."""
    from data_loader import _to_polygon_ticker, _polygon_fetch_bars, _polygon_bars_to_df
    ticker = _to_polygon_ticker(symbol)
    from_ms = int(start_dt.timestamp() * 1000)
    to_ms = int(end_dt.timestamp() * 1000)
    res = _polygon_fetch_bars(ticker, mult, timespan, str(from_ms), str(to_ms))
    return _polygon_bars_to_df(res)


print("\n-- fetching sources --", flush=True)
poly_1s = _polygon_1s(SYMBOL, start, end)
print(f"  Polygon native 1Sec: {0 if poly_1s is None else len(poly_1s)} bars", flush=True)
poly_1m = _polygon_native(SYMBOL, 1, "minute", start, end)
print(f"  Polygon native 1Min: {0 if poly_1m is None else len(poly_1m)} bars", flush=True)

# feed the store exactly as the data-worker ingest does
store = SymbolBarStore(SYMBOL, window_minutes=MINUTES + 120)
store.upsert_bars(poly_1s)
store_1s = store.get_1s(start, end)
store_1m = store.get_timeframe("1Min")
store_1m = store_1m[(store_1m.index >= pd.Timestamp(start)) & (store_1m.index < pd.Timestamp(end))]
print(f"  store 1Sec={len(store_1s)}  store 1Min(resampled)={len(store_1m)}", flush=True)

# bar_cache
bc_1s = bar_cache.read_bars(SYMBOL, "1Sec", start, end)
bc_1m = bar_cache.read_bars(SYMBOL, "1Min", start, end)
print(f"  bar_cache 1Sec={0 if bc_1s is None else len(bc_1s)}  1Min={0 if bc_1m is None else len(bc_1m)}", flush=True)

print("\n-- comparisons --", flush=True)
r = {}
r["store1s_vs_poly1s"] = compare("store 1Sec  vs Polygon 1Sec (ingest fidelity)", store_1s, poly_1s)
r["resample1m_vs_native1m"] = compare("store 1Min RESAMPLED vs Polygon native 1Min (THE question)", store_1m, poly_1m)
r["cache1s_vs_poly1s"] = compare("bar_cache 1Sec vs Polygon 1Sec (current staleness)", bc_1s, poly_1s)
r["cache1m_vs_poly1m"] = compare("bar_cache 1Min vs Polygon 1Min (current staleness)", bc_1m, poly_1m)

print("\n=== VERDICT ===", flush=True)
print(f"  Ingest fidelity (store==Polygon 1s): {'PASS' if r['store1s_vs_poly1s'] else 'FAIL'}")
print(f"  Resample==native 1Min:               {'PASS' if r['resample1m_vs_native1m'] else 'FAIL — write-through must persist NATIVE 1Min, not resampled'}")
print(f"  bar_cache currently fresh:           {'yes' if (r['cache1s_vs_poly1s'] and r['cache1m_vs_poly1m']) else 'NO — stale; write-through would fix'}")
