"""Scrub clobber-contaminated >=120s live_bars cache rows.

Every >=120s ws/ws_agg row written BEFORE the dc3f3bb merge fix came from
the rebroadcast-replace branch — i.e. single-minute OHLCV masquerading as
the full period (volume ~0.50x REST, collapsed range). The chart-data lens
falls back to a clean 10s/60s resample when rows are absent, so deletion
is safe and self-healing.

Usage:
    cd src && ../.venv/bin/python _scrub_clobbered_cache.py            # dry run
    cd src && ../.venv/bin/python _scrub_clobbered_cache.py --execute  # delete

Scope: timeframe_seconds >= 120, source in (ws, ws_agg),
bar_start < CUTOFF (dc3f3bb deploy, 2026-06-12T21:15:00Z).
"""
import argparse
import os
import sys

sys.path.insert(0, '.')
from dotenv import load_dotenv

load_dotenv('.env', override=True)
os.environ['USE_DB'] = 'true'

from db import get_admin_client  # noqa: E402

CUTOFF = '2026-06-12T21:15:00+00:00'  # dc3f3bb deploy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true',
                    help='actually delete (default: dry-run count)')
    args = ap.parse_args()
    c = get_admin_client()

    # Inventory by (symbol, tf, source)
    print(f"Scope: tf>=120, source in (ws, ws_agg), bar_start < {CUTOFF}\n")
    inv = {}
    page = 0
    while True:
        r = (c.table('live_bars')
             .select('symbol,timeframe_seconds,source')
             .gte('timeframe_seconds', 120)
             .in_('source', ['ws', 'ws_agg'])
             .lt('bar_start', CUTOFF)
             .range(page * 1000, page * 1000 + 999).execute().data)
        for row in r or []:
            k = (row['symbol'], row['timeframe_seconds'], row['source'])
            inv[k] = inv.get(k, 0) + 1
        if not r or len(r) < 1000:
            break
        page += 1
    total = sum(inv.values())
    for k in sorted(inv):
        print(f"  {k[0]:<8} tf={k[1]:<5} src={k[2]:<7} rows={inv[k]}")
    print(f"\nTOTAL contaminated rows: {total}")

    if not args.execute:
        print("\nDRY RUN — rerun with --execute to delete.")
        return

    print("\nDeleting...")
    deleted = 0
    # Delete per (symbol, tf) to keep each statement bounded
    for (sym, tf, src) in sorted(inv):
        c.table('live_bars').delete() \
            .eq('symbol', sym).eq('timeframe_seconds', tf) \
            .eq('source', src).lt('bar_start', CUTOFF).execute()
        deleted += inv[(sym, tf, src)]
        print(f"  scrubbed {sym} tf={tf} src={src} (~{inv[(sym, tf, src)]})")
    print(f"\nDone. ~{deleted} rows removed. Lens now resamples from "
          "sub-minute cache for pre-cutoff windows.")


if __name__ == '__main__':
    main()
