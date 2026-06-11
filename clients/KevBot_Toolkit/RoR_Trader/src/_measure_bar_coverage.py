"""Measure live_bars cache coverage vs Polygon REST at matched timestamps.

Gap-healer verification tool (2026-06-11, iter 0611a). For a (symbol, tf,
window): REST is ground truth for which bar windows actually traded; the
cache (engine-consumed sources: ws, ws_agg, rest_correction, rest_insert)
is what the live engine consumed. Coverage = cache∩REST / REST. Also
reports cache-only bars (should be ~0 post Bug A) and the source mix —
`rest_insert` count shows the healer working.

Auth-free (admin client). Run from src/:
  ../.venv/bin/python _measure_bar_coverage.py --symbol SPY --tf 10 --minutes 60
  ../.venv/bin/python _measure_bar_coverage.py --symbol DIA --tf 10 \
      --start-utc 2026-06-11T14:40 --end-utc 2026-06-11T15:40
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

sys.path.insert(0, '.')

try:
    from dotenv import load_dotenv
    load_dotenv('.env', override=True)
except Exception:
    pass
os.environ.setdefault('USE_DB', 'true')
os.environ['USE_DB'] = 'true'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', default='SPY')
    p.add_argument('--tf', type=int, default=10)
    p.add_argument('--minutes', type=int, default=60,
                   help='lookback window ending now-2min (ignored if --start-utc)')
    p.add_argument('--start-utc', default=None)
    p.add_argument('--end-utc', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    from db import get_admin_client
    from data_loader import (
        _polygon_fetch_bars, _polygon_bars_to_df, _to_polygon_ticker,
        ENGINE_CONSUMED_SOURCES)
    import pandas as pd

    now = datetime.now(timezone.utc)
    if args.start_utc:
        start = datetime.fromisoformat(args.start_utc)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        end = (datetime.fromisoformat(args.end_utc)
               if args.end_utc else now - timedelta(minutes=2))
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
    else:
        end = now - timedelta(minutes=2)
        start = end - timedelta(minutes=args.minutes)

    tf = args.tf
    sym = args.symbol

    # REST ground truth: native tf-second aggregates (timestamps are
    # bucket starts aligned to tf).
    res = _polygon_fetch_bars(
        _to_polygon_ticker(sym), tf, 'second',
        str(int(start.timestamp() * 1000)), str(int(end.timestamp() * 1000)))
    rest_df = _polygon_bars_to_df(res) if res else None
    rest_ts = set()
    if rest_df is not None and len(rest_df):
        for ts in rest_df.index:
            t = pd.Timestamp(ts)
            if t.tzinfo is None:
                t = t.tz_localize('UTC')
            # align down to tf boundary (defensive)
            epoch = int(t.timestamp())
            rest_ts.add(epoch - epoch % tf)

    # Cache (engine-consumed sources)
    c = get_admin_client()
    rows, page, off = [], 1000, 0
    while True:
        r = (c.table('live_bars')
             .select('bar_start,source')
             .eq('symbol', sym).eq('timeframe_seconds', tf)
             .gte('bar_start', start.isoformat())
             .lt('bar_start', end.isoformat())
             .in_('source', ENGINE_CONSUMED_SOURCES)
             .order('bar_start').range(off, off + page - 1)
             .execute()).data or []
        rows.extend(r)
        if len(r) < page:
            break
        off += page
    cache_ts = {}
    for r in rows:
        t = datetime.fromisoformat(r['bar_start'].replace('Z', '+00:00'))
        cache_ts[int(t.timestamp())] = r['source']

    both = rest_ts & set(cache_ts)
    rest_only = rest_ts - set(cache_ts)
    cache_only = set(cache_ts) - rest_ts
    cov = 100.0 * len(both) / max(len(rest_ts), 1)
    src_mix = Counter(cache_ts.values())

    print(f"{sym} {tf}s  window {start:%Y-%m-%d %H:%M}Z .. {end:%H:%M}Z")
    print(f"  REST traded windows : {len(rest_ts)}")
    print(f"  cache (engine srcs) : {len(cache_ts)}  mix={dict(src_mix)}")
    print(f"  COVERAGE cache∩REST : {len(both)}  = {cov:.1f}% of REST")
    print(f"  REST-only (missing) : {len(rest_only)}")
    print(f"  cache-only          : {len(cache_only)}")
    if rest_only:
        sample = sorted(rest_only)[-5:]
        print("  sample missing:",
              [datetime.fromtimestamp(t, tz=timezone.utc).strftime('%H:%M:%S')
               for t in sample])


if __name__ == '__main__':
    main()
