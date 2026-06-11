"""Measure flat zero-volume gap-fill bars in the live_bars cache, per UTC day,
split RTH vs after-hours — to confirm the 2026-06-10 gap-fill fix
(ralph_engine BAR_GAP_FILL_ENABLED=False) stops the live engine from carrying
forward flat bars through empty periods.

A "gap-fill flat" = open==high==low==close AND volume==0 (the synthetic
carry-forward the backtest never produces). Pre-fix these dominated the
after-hours (~30% of SPY 1Min). Post-fix, after-hours days should read ~0.

Run from src/:
    ../.venv/bin/python _measure_gapfill_flats.py            # SPY 60s, last 4 days
    ../.venv/bin/python _measure_gapfill_flats.py --symbol SPY --tf 60 --days 5
    ../.venv/bin/python _measure_gapfill_flats.py --symbol TSLA --tf 10

Auth-free: reads live_bars directly via db.get_admin_client (service role).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

from db import get_admin_client

# RTH window in UTC (09:30–16:00 ET during EDT ≈ 13:30–20:00 UTC).
RTH_START_UTC = (13, 30)
RTH_END_UTC = (20, 0)


def _is_rth(ts: datetime) -> bool:
    mins = ts.hour * 60 + ts.minute
    return (RTH_START_UTC[0] * 60 + RTH_START_UTC[1]) <= mins < (RTH_END_UTC[0] * 60 + RTH_END_UTC[1])


def _flat_zero_vol(row: dict) -> bool:
    try:
        o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
        v = float(row.get('volume', 0) or 0)
        return o == h == l == c and v == 0
    except Exception:
        return False


def fetch_rows(symbol: str, tf_seconds: int, start_dt: datetime, end_dt: datetime):
    c = get_admin_client()
    cols = "bar_start,open,high,low,close,volume,source"
    rows, offset, page = [], 0, 1000
    while True:
        r = (c.table('live_bars').select(cols)
             .eq('symbol', symbol).eq('timeframe_seconds', tf_seconds)
             .gte('bar_start', start_dt.isoformat())
             .lte('bar_start', end_dt.isoformat())
             .order('bar_start').range(offset, offset + page - 1).execute())
        batch = r.data or []
        rows.extend(batch)
        if len(batch) < page:
            break
        offset += page
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='SPY')
    ap.add_argument('--tf', type=int, default=60, help='timeframe_seconds')
    ap.add_argument('--days', type=int, default=4)
    ap.add_argument('--start-utc', default=None,
                    help='ISO start (overrides --days), e.g. 2026-06-10T20:00')
    ap.add_argument('--end-utc', default=None, help='ISO end (default now)')
    args = ap.parse_args()

    end_dt = (datetime.fromisoformat(args.end_utc).replace(tzinfo=timezone.utc)
              if args.end_utc else datetime.now(timezone.utc))
    if args.start_utc:
        start_dt = datetime.fromisoformat(args.start_utc).replace(tzinfo=timezone.utc)
    else:
        start_dt = end_dt - timedelta(days=args.days)
    rows = fetch_rows(args.symbol, args.tf, start_dt, end_dt)
    print(f"{args.symbol} {args.tf}s · {len(rows)} bars · "
          f"{start_dt.isoformat()[:16]} → {end_dt.isoformat()[:16]} UTC\n")

    # bucket by (utc_date, session)
    agg = defaultdict(lambda: {'total': 0, 'flat': 0})
    for row in rows:
        ts = datetime.fromisoformat(row['bar_start'].replace('Z', '+00:00'))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        sess = 'RTH' if _is_rth(ts) else 'AH '
        key = (ts.date().isoformat(), sess)
        agg[key]['total'] += 1
        if _flat_zero_vol(row):
            agg[key]['flat'] += 1

    print(f"{'date':<12}{'sess':<5}{'bars':>7}{'flatZV':>8}{'pct':>8}")
    print('-' * 40)
    for key in sorted(agg.keys()):
        d = agg[key]
        pct = 100 * d['flat'] / max(d['total'], 1)
        flag = '  <-- still gap-filling' if (key[1] == 'AH ' and pct > 2) else ''
        print(f"{key[0]:<12}{key[1]:<5}{d['total']:>7}{d['flat']:>8}{pct:>7.1f}%{flag}")
    print("\nExpect AH flatZV% to drop to ~0 on days AFTER the worker redeployed "
          "with BAR_GAP_FILL_ENABLED=False.")


if __name__ == '__main__':
    main()
