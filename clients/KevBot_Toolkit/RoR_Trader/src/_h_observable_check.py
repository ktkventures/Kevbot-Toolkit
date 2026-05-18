"""Phase H parity check: trade-channel bars vs REST 1Min (and observable).

Aligns `live_bars_trades` (three wait variants) against Polygon REST 1Min
aggregates on bar_start, reports close-price match %. REST 1Min is the
standard CTA/UTP pipeline (per Polygon's aggregates KB) — the right
baseline now that per-second REST is known to be a separate pipeline.

If observable flat-file data overlaps the window it is also compared;
ground truth when present.

Usage:  .venv/bin/python src/_h_observable_check.py [SYMBOL ...]
        WINDOW_START / WINDOW_END env vars override the default window.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

from datetime import datetime, timezone
import os
import sys

from db import get_admin_client

SOURCES = ['t_wait0', 't_wait200', 't_wait500']
DEFAULT_SYMBOLS = ['SPY', 'TSLA']
WINDOW_START = os.environ.get('WINDOW_START', '2026-05-18T13:30:00+00:00')
WINDOW_END = os.environ.get('WINDOW_END',
                            datetime.now(timezone.utc).isoformat())
TF = 60  # compare at 1Min — standard REST pipeline
TOL = 0.01  # |Δclose| <= $0.01 counts as a match


def _epoch(ts: str) -> int:
    return int(datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp())


def _paginate(query_fn) -> list[dict]:
    PAGE, out, offset = 1000, [], 0
    while True:
        page = query_fn(offset, offset + PAGE - 1)
        out.extend(page)
        if len(page) < PAGE:
            return out
        offset += PAGE


def fetch_trade_bars(sb, symbol: str, source: str) -> dict[int, dict]:
    rows = _paginate(lambda lo, hi: (
        sb.table("live_bars_trades")
        .select("bar_start, open, high, low, close, volume, trade_count")
        .eq("symbol", symbol).eq("source", source).eq("timeframe_seconds", TF)
        .gte("bar_start", WINDOW_START).lt("bar_start", WINDOW_END)
        .order("bar_start").range(lo, hi).execute().data or []))
    return {_epoch(r['bar_start']): r for r in rows}


def fetch_rest_1min(symbol: str) -> dict[int, dict]:
    from data_loader import _polygon_fetch_bars, _to_polygon_ticker
    raw = _polygon_fetch_bars(_to_polygon_ticker(symbol), 1, 'minute',
                              WINDOW_START[:10], WINDOW_END[:10])
    s, e = _epoch(WINDOW_START), _epoch(WINDOW_END)
    out: dict[int, dict] = {}
    for r in raw:
        sec = int(r.get('t', 0) / 1000)
        if s <= sec < e:
            out[sec] = {'open': r.get('o'), 'high': r.get('h'),
                        'low': r.get('l'), 'close': r.get('c'),
                        'volume': int(r.get('v', 0) or 0)}
    return out


def fetch_observable(sb, symbol: str) -> dict[int, dict]:
    rows = _paginate(lambda lo, hi: (
        sb.table("polygon_observable_bars")
        .select("sip_second_ts, open, high, low, close, volume")
        .eq("ticker", symbol)
        .gte("sip_second_ts", WINDOW_START).lt("sip_second_ts", WINDOW_END)
        .order("sip_second_ts").range(lo, hi).execute().data or []))
    out: dict[int, dict] = {}
    for r in rows:
        b = (_epoch(r['sip_second_ts']) // TF) * TF
        cur = out.get(b)
        if cur is None:
            out[b] = {'close': r['close'], 'high': r['high'], 'low': r['low']}
        else:
            cur['high'] = max(cur['high'], r['high'])
            cur['low'] = min(cur['low'], r['low'])
            cur['close'] = r['close']
    return out


def _compare(label: str, tb: dict[int, dict], ref: dict[int, dict]) -> None:
    common = sorted(set(tb) & set(ref))
    if not common:
        print(f"    vs {label:11s}: overlap=0")
        return
    diffs = [tb[b]['close'] - ref[b]['close'] for b in common]
    matches = sum(1 for d in diffs if abs(d) <= TOL)
    worst = sorted(diffs, key=abs, reverse=True)[:5]
    pct = 100.0 * matches / len(common)
    print(f"    vs {label:11s}: overlap={len(common):4d}  "
          f"close-match={matches}/{len(common)} = {pct:.1f}%  "
          f"worst Δ={[round(x, 3) for x in worst]}")


def main() -> None:
    symbols = [s.upper() for s in sys.argv[1:]] or DEFAULT_SYMBOLS
    sb = get_admin_client()
    print(f"Window: {WINDOW_START} .. {WINDOW_END}")
    print(f"TF: {TF}s   tol: ${TOL}\n")
    for symbol in symbols:
        rest = fetch_rest_1min(symbol)
        obs = fetch_observable(sb, symbol)
        print(f"=== {symbol} — REST 1Min bars: {len(rest)}  "
              f"observable bars: {len(obs)} ===")
        for source in SOURCES:
            tb = fetch_trade_bars(sb, symbol, source)
            print(f"  {source}  (trade bars={len(tb)})")
            _compare("REST 1Min", tb, rest)
            if obs:
                _compare("observable", tb, obs)
        print()


if __name__ == "__main__":
    main()
