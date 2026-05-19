"""Grace-window final sweep — each grace variant vs REST 1Sec-rolled-10s.

For the last MINUTES minutes, compares each grace_Ns 10Sec shadow series
in live_bars_trades against Polygon's settled REST 1-second aggregates
rolled into 10s buckets. Reports, per (symbol, variant):
  - close-match %  (|grace.close - rest.close| <= TOL)
  - high/low-match %
  - flat bars       (v=0 carry-forward — the Bug B empty-bar symptom)

Usage:  MINUTES=35 .venv/bin/python src/_grace_final_sweep.py
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

from datetime import datetime, timezone, timedelta
from collections import defaultdict
import os

from db import get_admin_client
from data_loader import _polygon_fetch_bars, _to_polygon_ticker

SYMBOLS = ['SPY', 'TSLA']
VARIANTS = ['grace_2s', 'grace_2.5s', 'grace_3s', 'grace_4s', 'grace_5s']
TF = 10
TOL = 0.01
MINUTES = int(os.environ.get('MINUTES', '35'))


def _epoch(ts: str) -> int:
    return int(datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp())


def fetch_grace(sb, symbol: str, source: str, since_iso: str) -> dict:
    rows, off = [], 0
    while True:
        page = (sb.table('live_bars_trades')
                .select('bar_start, open, high, low, close, volume')
                .eq('symbol', symbol).eq('timeframe_seconds', TF)
                .eq('source', source).gte('bar_start', since_iso)
                .order('bar_start').range(off, off + 999)
                .execute().data or [])
        rows.extend(page)
        if len(page) < 1000:
            break
        off += 1000
    return {_epoch(r['bar_start']): r for r in rows}


def fetch_rest_rolled(symbol: str, start_dt: datetime,
                      end_dt: datetime) -> dict:
    """REST 1Sec aggs for just the window (ms-epoch range — keeps the
    response small; a full-day 1Sec pull truncates the JSON)."""
    from_ms = str(int(start_dt.timestamp() * 1000))
    to_ms = str(int(end_dt.timestamp() * 1000))
    raw = _polygon_fetch_bars(_to_polygon_ticker(symbol), 1, 'second',
                              from_ms, to_ms)
    out: dict = {}
    for r in raw:
        b = (int(r.get('t', 0) / 1000) // TF) * TF
        cur = out.get(b)
        if cur is None:
            out[b] = {'high': r['h'], 'low': r['l'], 'close': r['c']}
        else:
            cur['high'] = max(cur['high'], r['h'])
            cur['low'] = min(cur['low'], r['l'])
            cur['close'] = r['c']  # raw sorted asc → last wins
    return out


def main() -> None:
    sb = get_admin_client()
    now = datetime.now(timezone.utc)
    since = now - timedelta(minutes=MINUTES)
    since_iso = since.isoformat()
    day = now.strftime('%Y-%m-%d')
    print(f"Grace-window sweep — last {MINUTES} min "
          f"({since.strftime('%H:%M')}–{now.strftime('%H:%M')} UTC), "
          f"10Sec, tol ${TOL}\n")

    rest_by_sym = {s: fetch_rest_rolled(s, since, now) for s in SYMBOLS}

    hdr = (f"{'symbol':<7}{'variant':<12}{'bars':>6}{'close=':>9}"
           f"{'high=':>8}{'low=':>8}{'flat':>6}")
    print(hdr)
    print('-' * len(hdr))
    best: dict = {}
    for symbol in SYMBOLS:
        rest = rest_by_sym[symbol]
        for v in VARIANTS:
            g = fetch_grace(sb, symbol, v, since_iso)
            common = sorted(set(g) & set(rest))
            if not common:
                print(f"{symbol:<7}{v:<12}{'—':>6}  (no overlap)")
                continue
            cm = hm = lm = flat = 0
            for b in common:
                gb, rb = g[b], rest[b]
                if abs(float(gb['close']) - rb['close']) <= TOL:
                    cm += 1
                if abs(float(gb['high']) - rb['high']) <= TOL:
                    hm += 1
                if abs(float(gb['low']) - rb['low']) <= TOL:
                    lm += 1
                if float(gb.get('volume') or 0) == 0:
                    flat += 1
            n = len(common)
            pct = 100.0 * cm / n
            best.setdefault(symbol, []).append((v, pct, n))
            print(f"{symbol:<7}{v:<12}{n:>6}{pct:>8.1f}%"
                  f"{100.0*hm/n:>7.1f}%{100.0*lm/n:>7.1f}%{flat:>6}")
        print()

    print("Smallest grace reaching >=99% close-match:")
    for symbol in SYMBOLS:
        winner = next((v for v, p, _ in best.get(symbol, []) if p >= 99.0),
                      None)
        rows = best.get(symbol, [])
        topv, topp, _ = max(rows, key=lambda x: x[1]) if rows else ('—', 0, 0)
        print(f"  {symbol}: {winner or f'none (best: {topv} {topp:.1f}%)'}")


if __name__ == '__main__':
    main()
