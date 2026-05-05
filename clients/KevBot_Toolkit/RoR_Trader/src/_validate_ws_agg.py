"""Compare AM-channel rows (source='ws') against client-side aggregated
rows (source='ws_agg') for the same (symbol, tf=60, bar_start).

Used to pick between live_bar_source modes:
  - Bit-identical / trivial diff → ship Mode 2 (A-agg only)
  - Material diff → build Mode 3 (AM-preferred with A-agg fallback)
  - Wild divergence → halt + investigate aggregator

Run from src/:
  ../.venv/bin/python _validate_ws_agg.py [--hours-back N] [--symbol SYM]
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import statistics

from db import get_admin_client


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--hours-back', type=float, default=2.0)
    p.add_argument('--symbol', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    c = get_admin_client()
    since = (datetime.now(timezone.utc) - timedelta(hours=args.hours_back))

    # Pull all 1Min cache rows since `since`, both sources
    print(f"Window: {since.isoformat()} → now")
    q = c.table('live_bars').select(
        'symbol,timeframe_seconds,bar_start,source,open,high,low,close,volume'
    ).gte('bar_start', since.isoformat()).eq('timeframe_seconds', 60)
    if args.symbol:
        q = q.eq('symbol', args.symbol)

    all_rows = []
    offset = 0
    batch = 1000
    while True:
        r = q.range(offset, offset + batch - 1).execute()
        page = r.data or []
        all_rows.extend(page)
        if len(page) < batch:
            break
        offset += batch
    print(f"Loaded {len(all_rows)} 1Min rows")

    # Index: (symbol, bar_start) → {source → row}
    by_bar: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in all_rows:
        by_bar[(row['symbol'], row['bar_start'])][row['source']] = row

    # For each bar with both sources, diff
    pair_count = 0
    sym_stats: dict[str, dict] = defaultdict(lambda: {
        'pairs': 0,
        'identical': 0,
        'open_diffs': [],
        'high_diffs': [],
        'low_diffs': [],
        'close_diffs': [],
        'volume_diffs': [],
        'volume_pct_diffs': [],
    })
    sym_only_ws: dict[str, int] = defaultdict(int)
    sym_only_agg: dict[str, int] = defaultdict(int)

    for (sym, bar_start), bysrc in by_bar.items():
        ws = bysrc.get('ws')
        ws_agg = bysrc.get('ws_agg')
        if ws and ws_agg:
            pair_count += 1
            s = sym_stats[sym]
            s['pairs'] += 1
            do = float(ws_agg['open']) - float(ws['open'])
            dh = float(ws_agg['high']) - float(ws['high'])
            dl = float(ws_agg['low']) - float(ws['low'])
            dc = float(ws_agg['close']) - float(ws['close'])
            dv = float(ws_agg['volume']) - float(ws['volume'])
            v_ws = float(ws['volume']) or 1.0
            dvp = 100 * dv / v_ws
            if do == 0 and dh == 0 and dl == 0 and dc == 0 and dv == 0:
                s['identical'] += 1
            s['open_diffs'].append(do)
            s['high_diffs'].append(dh)
            s['low_diffs'].append(dl)
            s['close_diffs'].append(dc)
            s['volume_diffs'].append(dv)
            s['volume_pct_diffs'].append(dvp)
        elif ws and not ws_agg:
            sym_only_ws[sym] += 1
        elif ws_agg and not ws:
            sym_only_agg[sym] += 1

    print()
    print("=" * 100)
    print(f"PAIR DIFF SUMMARY  ({pair_count} bars where both sources exist)")
    print("=" * 100)
    if pair_count == 0:
        print("No paired bars yet.  Either WS_AGG_SHADOW_ENABLED is off, the")
        print("worker hasn't restarted with the new code, or the active symbol")
        print("doesn't have an A.* subscription (only L-type / sub-minute strats).")
    else:
        hdr = (f"{'symbol':<8} {'pairs':>5} {'ident':>5} "
               f"{'open|d| p50/max':>18} {'high|d| p50/max':>18} "
               f"{'low|d| p50/max':>18} {'close|d| p50/max':>18} "
               f"{'vol% |d| p50/max':>18}")
        print(hdr)

        def _stat(vals: list[float]) -> tuple[float, float]:
            abs_vals = [abs(v) for v in vals]
            if not abs_vals:
                return (0.0, 0.0)
            return (statistics.median(abs_vals), max(abs_vals))

        for sym in sorted(sym_stats.keys()):
            s = sym_stats[sym]
            o_p50, o_max = _stat(s['open_diffs'])
            h_p50, h_max = _stat(s['high_diffs'])
            l_p50, l_max = _stat(s['low_diffs'])
            c_p50, c_max = _stat(s['close_diffs'])
            vp_p50, vp_max = _stat(s['volume_pct_diffs'])
            print(f"{sym:<8} {s['pairs']:>5} {s['identical']:>5} "
                  f"{o_p50:>9.4f}/{o_max:>7.4f} "
                  f"{h_p50:>9.4f}/{h_max:>7.4f} "
                  f"{l_p50:>9.4f}/{l_max:>7.4f} "
                  f"{c_p50:>9.4f}/{c_max:>7.4f} "
                  f"{vp_p50:>8.2f}%/{vp_max:>7.2f}%")

    print()
    print("=" * 100)
    print("ASYMMETRY (bars with only one source)")
    print("=" * 100)
    print(f"{'symbol':<8} {'ws-only':>10} {'ws_agg-only':>14}")
    syms = sorted(set(sym_only_ws) | set(sym_only_agg))
    for sym in syms:
        print(f"{sym:<8} {sym_only_ws.get(sym, 0):>10} {sym_only_agg.get(sym, 0):>14}")

    print()
    print("Interpretation:")
    print("  - 'ws-only' rows are AM bars that arrived but A-agg didn't")
    print("    cover (likely no A.* subscription for that symbol).")
    print("  - 'ws_agg-only' rows are A-aggregated bars that AM dropped —")
    print("    these are the bars Mode 2 / Mode 3 would recover.")
    print("  - Per-symbol diff: if open|d|.max == 0 and close|d|.max < 0.01,")
    print("    A-agg matches AM closely enough → ship Mode 2.")
    print("  - Otherwise → consider Mode 3 (AM-preferred, A-agg fallback).")


if __name__ == "__main__":
    main()
