"""One-time cleanup of fabricated flat gap-fill bars in the live_bars cache.

The pre-2026-06-10 live engine carried forward flat zero-volume bars through
empty/no-trade periods (after-hours). A "flat gap-fill" = open==high==low==close
AND volume==0 — a bar REST never produces (no trade → no bar). These pollute the
Alert Lens indicator recompute and the cross-TF (2m) bars resampled from them.

Deleting them makes the cache match the backtest bar set (no bar for empty
periods). Safe: the live engine warms from REST (not the cache), so its
decisions are unaffected; only the cache display / cache_locked recompute change.

DEFAULT IS COUNT-ONLY (read-only, nothing deleted). To actually delete you must
pass BOTH --delete AND --yes.

Run from src/:
    # dry-run / count (read-only):
    ../.venv/bin/python _cleanup_gapfill_flats.py --symbol SPY --tf 10
    ../.venv/bin/python _cleanup_gapfill_flats.py --symbol SPY --tf 120
    # actually delete (guarded):
    ../.venv/bin/python _cleanup_gapfill_flats.py --symbol SPY --tf 10 --delete --yes
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

RTH_START = (13, 30)
RTH_END = (20, 0)


def _is_rth(ts: datetime) -> bool:
    m = ts.hour * 60 + ts.minute
    return (RTH_START[0] * 60 + RTH_START[1]) <= m < (RTH_END[0] * 60 + RTH_END[1])


def _flat(row: dict) -> bool:
    try:
        o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
        v = float(row.get('volume', 0) or 0)
        return o == h == l == c and v == 0
    except Exception:
        return False


def fetch_all(c, symbol, tf, start_dt, end_dt):
    cols = "bar_start,open,high,low,close,volume,source"
    rows, offset, page = [], 0, 1000
    while True:
        r = (c.table('live_bars').select(cols)
             .eq('symbol', symbol).eq('timeframe_seconds', tf)
             .gte('bar_start', start_dt.isoformat()).lte('bar_start', end_dt.isoformat())
             .order('bar_start').range(offset, offset + page - 1).execute())
        batch = r.data or []
        rows.extend(batch)
        if len(batch) < page:
            break
        offset += page
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--tf', type=int, required=True, help='timeframe_seconds (e.g. 10 primary, 120 the 2m gate)')
    ap.add_argument('--days', type=int, default=45)
    ap.add_argument('--delete', action='store_true', help='actually delete (needs --yes)')
    ap.add_argument('--yes', action='store_true', help='confirm deletion')
    args = ap.parse_args()

    c = get_admin_client()
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=args.days)
    rows = fetch_all(c, args.symbol, args.tf, start_dt, end_dt)

    flats = [r for r in rows if _flat(r)]
    by_day = defaultdict(lambda: {'total': 0, 'flat': 0})
    src = defaultdict(int)
    for r in rows:
        ts = datetime.fromisoformat(r['bar_start'].replace('Z', '+00:00'))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        key = (ts.date().isoformat(), 'RTH' if _is_rth(ts) else 'AH ')
        by_day[key]['total'] += 1
        if _flat(r):
            by_day[key]['flat'] += 1
            src[r.get('source', '?')] += 1

    print(f"\n{args.symbol} {args.tf}s · {len(rows)} bars in window · "
          f"FLAT gap-fill rows: {len(flats)} ({100*len(flats)/max(len(rows),1):.1f}%)")
    print(f"window: {start_dt.isoformat()[:16]} → {end_dt.isoformat()[:16]} UTC")
    print(f"flat-row sources: {dict(src)}\n")
    print(f"{'date':<12}{'sess':<5}{'bars':>7}{'flat':>7}{'pct':>8}")
    print('-' * 40)
    for key in sorted(by_day):
        d = by_day[key]
        if d['flat'] == 0:
            continue
        print(f"{key[0]:<12}{key[1]:<5}{d['total']:>7}{d['flat']:>7}{100*d['flat']/max(d['total'],1):>7.1f}%")

    if not args.delete:
        print(f"\n[DRY-RUN] Would delete {len(flats)} flat rows. "
              f"Re-run with --delete --yes to apply.")
        return

    if not args.yes:
        print("\nRefusing to delete without --yes.")
        return

    # Delete by bar_start in batches (PostgREST can't compare columns, so we
    # delete the specific flat timestamps we identified client-side).
    bts = [r['bar_start'] for r in flats]
    deleted = 0
    for i in range(0, len(bts), 100):
        chunk = bts[i:i + 100]
        (c.table('live_bars').delete()
         .eq('symbol', args.symbol).eq('timeframe_seconds', args.tf)
         .in_('bar_start', chunk).execute())
        deleted += len(chunk)
        print(f"  deleted {deleted}/{len(bts)}...")
    print(f"\nDONE. Deleted {deleted} flat gap-fill rows for {args.symbol} {args.tf}s.")


if __name__ == '__main__':
    main()
