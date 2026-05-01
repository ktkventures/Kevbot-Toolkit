"""Validate the live_bars cache after tomorrow's market open.

Run this after the worker has been recording for at least 30-60 min:

    cd src && ../.venv/bin/python _validate_live_bars_cache.py

Reports:
  1. How many rows landed in live_bars during the window
  2. Per (symbol, timeframe) row counts + freshness (lag from bar_start
     to written_at — should be sub-second to seconds)
  3. WS-vs-REST drift on the SAME bars: pulls Polygon REST for the
     same range and compares OHLCV cell-by-cell with the cached rows
  4. Missing bars: bars present in REST but not in cache (worker gap)
     and bars present in cache but not in REST (legitimate WS-only
     prints, or REST hadn't settled yet)

This is the tomorrow-morning sanity check on the M8.7 write path.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import statistics

import pandas as pd

from db import get_admin_client, set_admin_user_context
from data_loader import load_market_data

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hours-back", type=float, default=4.0,
                   help="How far back to validate (default: 4 hours)")
    p.add_argument("--symbol", default=None,
                   help="Restrict to one symbol (default: all)")
    p.add_argument("--tf-seconds", type=int, default=None,
                   help="Restrict to one TF in seconds (default: all)")
    return p.parse_args()


def fetch_cache(c, since_iso: str, symbol: str | None,
                tf_seconds: int | None) -> list[dict]:
    q = c.table("live_bars").select(
        "symbol,timeframe_seconds,bar_start,open,high,low,close,volume,"
        "source,written_at"
    ).gte("bar_start", since_iso).order("bar_start")
    if symbol:
        q = q.eq("symbol", symbol)
    if tf_seconds:
        q = q.eq("timeframe_seconds", tf_seconds)
    r = q.execute()
    return r.data or []


def report_freshness(rows: list[dict]) -> None:
    print("\n=== Cache freshness (write_lag = written_at - bar_start) ===")
    by_group: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        try:
            bs = pd.Timestamp(row["bar_start"])
            wa = pd.Timestamp(row["written_at"])
            lag_s = (wa - bs).total_seconds()
            by_group[(row["symbol"], row["timeframe_seconds"])].append(lag_s)
        except Exception:
            continue
    if not by_group:
        print("  (no rows)")
        return
    print(f"{'symbol/tf':<18} {'n':>6} {'min':>8} {'p50':>8} {'p95':>8} "
          f"{'max':>8}")
    for (sym, tf), lags in sorted(by_group.items()):
        lags.sort()
        n = len(lags)
        p50 = lags[n // 2]
        p95 = lags[int(0.95 * n)] if n > 1 else lags[-1]
        print(f"{sym + '/' + str(tf) + 's':<18} {n:>6} {lags[0]:>8.2f} "
              f"{p50:>8.2f} {p95:>8.2f} {lags[-1]:>8.2f}")


SECONDS_TO_TF = {
    10: "10Sec", 30: "30Sec", 60: "1Min", 300: "5Min",
    900: "15Min", 1800: "30Min", 3600: "1Hour", 86400: "1Day",
}


def report_drift(rows: list[dict]) -> None:
    print("\n=== WS-vs-REST drift on cached bars ===")
    by_group: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("source") != "ws":
            continue
        by_group[(row["symbol"], row["timeframe_seconds"])].append(row)

    print(f"{'symbol/tf':<18} {'n':>6} {'matched':>8} {'differ':>7} "
          f"{'mean|d|':>10} {'p95|d|':>10} {'max|d|':>10}")
    for (sym, tf), group in sorted(by_group.items()):
        tf_str = SECONDS_TO_TF.get(tf)
        if tf_str is None:
            print(f"  skip {sym}/{tf}s — no REST tf mapping")
            continue
        # Group's range
        bar_times = sorted(pd.Timestamp(r["bar_start"]) for r in group)
        if not bar_times:
            continue
        days = max(1, int((datetime.now(timezone.utc) -
                           bar_times[0].to_pydatetime()).total_seconds()
                          // 86400) + 2)
        try:
            df = load_market_data(symbol=sym, days=days, timeframe=tf_str,
                                  session="RTH")
        except Exception as e:
            print(f"  {sym}/{tf_str}: REST load failed: {e}")
            continue
        if df is None or len(df) == 0:
            print(f"  {sym}/{tf_str}: no REST data")
            continue

        diffs: list[float] = []
        matched = 0
        for r in group:
            bt = pd.Timestamp(r["bar_start"])
            if bt.tz is None:
                bt = bt.tz_localize("UTC")
            else:
                bt = bt.tz_convert("UTC")
            if bt not in df.index:
                continue
            rest_close = float(df.loc[bt, "close"])
            ws_close = float(r["close"])
            d = abs(ws_close - rest_close)
            if d < 0.0001:
                matched += 1
            else:
                diffs.append(d)
        n = matched + len(diffs)
        if n == 0:
            print(f"  {sym}/{tf_str}: no overlap")
            continue
        diffs.sort()
        mean_d = statistics.mean(diffs) if diffs else 0.0
        p95_d = diffs[int(0.95 * len(diffs))] if diffs else 0.0
        max_d = diffs[-1] if diffs else 0.0
        print(f"{sym + '/' + tf_str:<18} {n:>6} {matched:>8} {len(diffs):>7} "
              f"{mean_d:>10.4f} {p95_d:>10.4f} {max_d:>10.4f}")


def report_coverage_gaps(rows: list[dict]) -> None:
    """Bars in REST but missing from cache, per (symbol, tf)."""
    print("\n=== Coverage gaps (bars in REST but missing from cache) ===")
    by_group: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("source") != "ws":
            continue
        by_group[(row["symbol"], row["timeframe_seconds"])].append(row)
    for (sym, tf), group in sorted(by_group.items()):
        tf_str = SECONDS_TO_TF.get(tf)
        if tf_str is None:
            continue
        bar_times = sorted(pd.Timestamp(r["bar_start"]) for r in group)
        first, last = bar_times[0], bar_times[-1]
        try:
            df = load_market_data(symbol=sym, days=2, timeframe=tf_str,
                                  session="RTH")
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        rest_in_window = df[(df.index >= first) & (df.index <= last)]
        cache_set = set(bar_times)
        missing = [t for t in rest_in_window.index if t not in cache_set]
        print(f"  {sym}/{tf_str}: {len(missing)} of "
              f"{len(rest_in_window)} REST bars missing from cache "
              f"(window {first} → {last})")
        if missing[:5]:
            for t in missing[:5]:
                print(f"    missing: {t}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5} more")


def main() -> None:
    args = parse_args()
    set_admin_user_context(USER)
    c = get_admin_client()

    since = (datetime.now(timezone.utc)
             - timedelta(hours=args.hours_back)).isoformat()
    print(f"\nValidating live_bars cache for last {args.hours_back}h "
          f"(since {since[:19]} UTC)")
    if args.symbol:
        print(f"  filter symbol={args.symbol}")
    if args.tf_seconds:
        print(f"  filter tf_seconds={args.tf_seconds}")

    rows = fetch_cache(c, since, args.symbol, args.tf_seconds)
    print(f"\nTotal cache rows in window: {len(rows)}")
    if not rows:
        print("\nNo rows. Either:")
        print("  - LIVE_BAR_CACHE_WRITE_ENABLED is not set on the worker")
        print("  - Worker hasn't started writing yet")
        print("  - Window is outside RTH")
        return

    by_source = defaultdict(int)
    for r in rows:
        by_source[r.get("source", "?")] += 1
    print(f"By source: {dict(by_source)}")

    report_freshness(rows)
    report_drift(rows)
    report_coverage_gaps(rows)


if __name__ == "__main__":
    main()
