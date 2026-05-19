"""Cache vs REST-1Sec parity — for 1Min+ timeframes.

Verifies the claim that the live_bars cache (decision-time AND
cache+backfill) matches REST 1Sec-rolled well at >=1Min — i.e. the
grace paradigm is only needed sub-minute.

For each (symbol, tf): compares two cache views against REST 1Sec aggs
rolled into tf buckets, close-match within TOL:
  - cache (decision)  = live_bars.first_close (what the engine saw live)
  - cache + backfill  = live_bars.close, all sources (ws_agg settled +
                        rest_backfill gap rows)

Usage:  HOURS=2 .venv/bin/python src/_cache_vs_rest1s.py
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

from datetime import datetime, timezone, timedelta
import os

from db import get_admin_client
from data_loader import _polygon_fetch_bars, _to_polygon_ticker

SYMBOLS = ['SPY', 'AAPL', 'TSLA']
TFS = [60, 300]          # 1Min, 5Min
TOL = 0.01
HOURS = float(os.environ.get('HOURS', '2'))


def _epoch(ts: str) -> int:
    return int(datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp())


def fetch_cache(sb, symbol: str, tf: int, since_iso: str) -> dict:
    rows, off = [], 0
    while True:
        page = (sb.table('live_bars')
                .select('bar_start, first_close, close, source')
                .eq('symbol', symbol).eq('timeframe_seconds', tf)
                .gte('bar_start', since_iso)
                .order('bar_start').range(off, off + 999)
                .execute().data or [])
        rows.extend(page)
        if len(page) < 1000:
            break
        off += 1000
    return {_epoch(r['bar_start']): r for r in rows}


def fetch_rest_rolled(symbol: str, tf: int, start_dt: datetime,
                      end_dt: datetime) -> dict:
    """REST 1Sec aggs for the window (ms-epoch range), rolled to tf."""
    from_ms = str(int(start_dt.timestamp() * 1000))
    to_ms = str(int(end_dt.timestamp() * 1000))
    raw = _polygon_fetch_bars(_to_polygon_ticker(symbol), 1, 'second',
                              from_ms, to_ms)
    out: dict = {}
    for r in raw:
        b = (int(r.get('t', 0) / 1000) // tf) * tf
        cur = out.get(b)
        if cur is None:
            out[b] = {'close': r['c']}
        else:
            cur['close'] = r['c']  # raw sorted asc → last wins
    return out


def main() -> None:
    sb = get_admin_client()
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=HOURS)
    since_iso = since.isoformat()
    print(f"Cache vs REST-1Sec parity — last {HOURS}h "
          f"({since.strftime('%H:%M')}–{now.strftime('%H:%M')} UTC), "
          f"tol ${TOL}\n")
    hdr = (f"{'symbol':<7}{'tf':>5}{'bars':>7}"
           f"{'cache(decision)=REST':>23}{'cache+backfill=REST':>22}")
    print(hdr)
    print('-' * len(hdr))
    for symbol in SYMBOLS:
        for tf in TFS:
            cache = fetch_cache(sb, symbol, tf, since_iso)
            rest = fetch_rest_rolled(symbol, tf, since, now)
            common = sorted(set(cache) & set(rest))
            if not common:
                print(f"{symbol:<7}{tf:>5}{'—':>7}  (no overlap)")
                continue
            dm = sum(1 for b in common
                     if cache[b].get('first_close') is not None
                     and abs(float(cache[b]['first_close'])
                             - rest[b]['close']) <= TOL)
            dn = sum(1 for b in common
                     if cache[b].get('first_close') is not None)
            bm = sum(1 for b in common
                     if cache[b].get('close') is not None
                     and abs(float(cache[b]['close'])
                             - rest[b]['close']) <= TOL)
            bn = len(common)
            dpct = 100.0 * dm / dn if dn else 0.0
            bpct = 100.0 * bm / bn if bn else 0.0
            print(f"{symbol:<7}{tf:>5}{bn:>7}"
                  f"{f'{dm}/{dn}={dpct:.1f}%':>23}"
                  f"{f'{bm}/{bn}={bpct:.1f}%':>22}")
        print()


if __name__ == '__main__':
    main()
