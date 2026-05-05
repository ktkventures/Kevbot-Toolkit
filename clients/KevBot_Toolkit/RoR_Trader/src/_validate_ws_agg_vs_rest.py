"""Validate ws_agg (client-side A-aggregation) against Polygon's REST API
canonical 1Min bars.

Why: AM events have been so sparse on AAPL/AMD/SPY/TSLA that we can't
get enough paired (ws, ws_agg) rows in cache to do an OHLCV diff
directly. Polygon's REST endpoint
`/v2/aggs/ticker/<SYM>/range/1/minute/<FROM>/<TO>` is the canonical,
fully-late-print-corrected view that AM (or AM+rebroadcasts) would
eventually settle to. So we compare ws_agg against REST instead.

If ws_agg matches REST closely (e.g. close diff < $0.01 typical, open
identical) → ship Mode 2 (always A-agg). Acceptable trade-off: bars
locked at minute close miss any subsequent FINRA late-print
corrections, but they capture all trades that arrived during the
minute itself.

If ws_agg diverges materially → consider Mode 3 (AM-preferred with
A-agg fallback) so we get late-print accuracy when AM arrives + A-agg
reliability when it doesn't.

Run from src/:
  ../.venv/bin/python _validate_ws_agg_vs_rest.py [--hours-back N] [--symbol SYM]
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from db import get_admin_client
from data_loader import _polygon_fetch_bars


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--hours-back', type=float, default=2.0)
    p.add_argument('--symbol', default=None,
                   help='Restrict to one symbol (default: all symbols with ws_agg rows)')
    return p.parse_args()


def main():
    args = parse_args()
    c = get_admin_client()
    since = datetime.now(timezone.utc) - timedelta(hours=args.hours_back)

    print(f"Window: {since.isoformat()} → now")

    # Pull all ws_agg 1Min rows from cache
    q = c.table('live_bars').select(
        'symbol,bar_start,open,high,low,close,volume'
    ).gte('bar_start', since.isoformat()) \
     .eq('timeframe_seconds', 60) \
     .eq('source', 'ws_agg')
    if args.symbol:
        q = q.eq('symbol', args.symbol)

    cache_rows = []
    offset = 0
    batch = 1000
    while True:
        r = q.range(offset, offset + batch - 1).execute()
        page = r.data or []
        cache_rows.extend(page)
        if len(page) < batch:
            break
        offset += batch
    print(f"Loaded {len(cache_rows)} ws_agg rows from cache")

    if not cache_rows:
        print("No ws_agg rows. Set WS_AGG_SHADOW_ENABLED=true on Worker first.")
        return

    # Group by symbol so we can fetch one REST range per symbol instead of
    # hitting the API per bar
    by_sym: dict[str, list[dict]] = defaultdict(list)
    for row in cache_rows:
        by_sym[row['symbol']].append(row)

    print()
    overall_pairs = 0
    overall_identical = 0
    sym_summary: list[dict] = []

    for sym in sorted(by_sym.keys()):
        rows = sorted(by_sym[sym], key=lambda r: r['bar_start'])
        # Fetch REST 1Min for the date range covering these bars
        first_dt = datetime.fromisoformat(rows[0]['bar_start'].replace('Z', '+00:00'))
        last_dt = datetime.fromisoformat(rows[-1]['bar_start'].replace('Z', '+00:00'))
        # Polygon accepts unix-ms ranges directly
        from_ms = int(first_dt.timestamp() * 1000)
        to_ms = int((last_dt + timedelta(minutes=1)).timestamp() * 1000) - 1

        rest_bars = _polygon_fetch_bars(
            sym, multiplier=1, timespan='minute',
            from_date=str(from_ms), to_date=str(to_ms)
        )
        rest_by_ts: dict[int, dict] = {}
        for bar in rest_bars:
            t = bar.get('t')  # unix ms (start of bar)
            if t is None:
                continue
            rest_by_ts[int(t // 1000)] = bar  # index by unix sec

        print(f"{sym}: {len(rows)} ws_agg rows · {len(rest_bars)} REST bars in range")

        # Compare each cache row to its REST counterpart
        diffs = {
            'open': [], 'high': [], 'low': [], 'close': [],
            'volume': [], 'volume_pct': [],
        }
        identical = 0
        compared = 0
        unmatched = 0
        for cr in rows:
            cr_ts = int(datetime.fromisoformat(
                cr['bar_start'].replace('Z', '+00:00')).timestamp())
            rb = rest_by_ts.get(cr_ts)
            if rb is None:
                unmatched += 1
                continue
            compared += 1
            do = float(cr['open']) - float(rb.get('o', 0))
            dh = float(cr['high']) - float(rb.get('h', 0))
            dl = float(cr['low']) - float(rb.get('l', 0))
            dc = float(cr['close']) - float(rb.get('c', 0))
            dv = float(cr['volume']) - float(rb.get('v', 0))
            v_rest = float(rb.get('v', 0)) or 1.0
            dvp = 100 * dv / v_rest
            diffs['open'].append(do)
            diffs['high'].append(dh)
            diffs['low'].append(dl)
            diffs['close'].append(dc)
            diffs['volume'].append(dv)
            diffs['volume_pct'].append(dvp)
            if do == 0 and dh == 0 and dl == 0 and dc == 0 and dv == 0:
                identical += 1

        overall_pairs += compared
        overall_identical += identical

        def _stat(vals):
            if not vals:
                return (0.0, 0.0, 0.0)
            absvals = [abs(v) for v in vals]
            return (statistics.median(absvals), max(absvals), sum(absvals) / len(absvals))

        sym_summary.append({
            'symbol': sym,
            'compared': compared,
            'identical': identical,
            'unmatched': unmatched,
            'open': _stat(diffs['open']),
            'high': _stat(diffs['high']),
            'low': _stat(diffs['low']),
            'close': _stat(diffs['close']),
            'volume_pct': _stat(diffs['volume_pct']),
        })

    print()
    print("=" * 110)
    print("DIFF SUMMARY (ws_agg vs Polygon REST canonical 1Min)")
    print("=" * 110)
    hdr = (f"{'symbol':<7} {'compared':>9} {'identical':>10} "
           f"{'open|d|.max':>12} {'high|d|.max':>12} {'low|d|.max':>12} "
           f"{'close|d| p50/max':>20} {'vol% |d| p50/max':>20}")
    print(hdr)
    for s in sym_summary:
        o = s['open']; h = s['high']; lo = s['low']; cl = s['close']; vp = s['volume_pct']
        print(f"{s['symbol']:<7} {s['compared']:>9} {s['identical']:>10} "
              f"{o[1]:>11.4f}  {h[1]:>11.4f}  {lo[1]:>11.4f}  "
              f"{cl[0]:>9.4f}/{cl[1]:>8.4f}  "
              f"{vp[0]:>8.2f}%/{vp[1]:>8.2f}%")
    print()
    print(f"Overall: {overall_pairs} bars compared, {overall_identical} bit-identical "
          f"({100 * overall_identical / max(1, overall_pairs):.1f}%)")
    print()
    print("Decision rule:")
    print("  - open|d|.max ≤ $0.01 AND close|d|.max ≤ $0.05 AND volume% p50 ≤ 5%")
    print("    → ship Mode 2 (A-agg only, lock at minute close)")
    print("  - close|d|.max routinely > $0.05 OR volume% p50 > 10%")
    print("    → consider Mode 3 (AM-preferred with A-agg fallback)")
    print("  - Anything wild (e.g. open|d|.max > $0.10) → halt + investigate aggregator")


if __name__ == "__main__":
    main()
