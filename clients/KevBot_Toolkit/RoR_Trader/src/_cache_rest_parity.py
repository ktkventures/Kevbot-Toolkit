"""Stress-test: decision-time cache vs settled REST vs Polygon correction.

For each (symbol, tf, day) it aligns three bar series on bar_start and
reports close-price match %:

  first  vs REST   = decision-time cache vs settled REST
                     — did the live engine see the right close?
  settled vs REST  = our post-correction cache vs settled REST
                     — sanity check on our own settled write
  first  vs settled= how often Polygon corrected the bar after first
                     WS emission (the Phase G "~16%" number)

`first_*` = what the live alert engine gated on. REST = Polygon's
settled aggregates (what a backtest runs on). High first-vs-REST match
means backtest and live are looking at effectively the same data.

Usage:  .venv/bin/python src/_cache_rest_parity.py
        DAYS env var = comma-separated YYYY-MM-DD (default: recent 3).
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

from datetime import datetime, timezone
import os

from db import get_admin_client
from data_loader import _polygon_fetch_bars, _to_polygon_ticker

COMBOS = [('SPY', 60), ('TSLA', 60), ('AAPL', 60), ('AMD', 60),
          ('META', 60), ('SPY', 10), ('TSLA', 10)]
DAYS = os.environ.get('DAYS', '2026-05-18,2026-05-15,2026-05-14').split(',')
TOL = 0.01
# Restrict to RTH (13:30-20:00 UTC) — ETH has Form-T / REST 1-sec gaps.
RTH_START_H, RTH_END_H = 13.5, 20.0


def _epoch(ts: str) -> int:
    return int(datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp())


def _in_rth(epoch: int) -> bool:
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    h = dt.hour + dt.minute / 60.0
    return RTH_START_H <= h < RTH_END_H


def fetch_cache(sb, symbol: str, tf: int, day: str) -> dict[int, dict]:
    rows, offset = [], 0
    while True:
        page = (sb.table("live_bars")
                .select("bar_start, first_close, close")
                .eq("symbol", symbol).eq("timeframe_seconds", tf)
                .gte("bar_start", f"{day}T00:00:00+00:00")
                .lt("bar_start", f"{day}T23:59:59+00:00")
                .order("bar_start").range(offset, offset + 999)
                .execute().data or [])
        rows.extend(page)
        if len(page) < 1000:
            break
        offset += 1000
    return {_epoch(r['bar_start']): r for r in rows
            if r.get('first_close') is not None}


def fetch_rest(symbol: str, tf: int, day: str) -> dict[int, dict]:
    """REST bars rolled to `tf`. minute aggs for tf>=60, second for tf<60."""
    timespan = 'minute' if tf >= 60 else 'second'
    native = 60 if tf >= 60 else 1
    raw = _polygon_fetch_bars(_to_polygon_ticker(symbol), 1, timespan, day, day)
    out: dict[int, dict] = {}
    for r in raw:
        sec = int(r.get('t', 0) / 1000)
        b = (sec // tf) * tf
        cur = out.get(b)
        if cur is None:
            out[b] = {'high': r['h'], 'low': r['l'], 'close': r['c']}
        else:
            cur['high'] = max(cur['high'], r['h'])
            cur['low'] = min(cur['low'], r['l'])
            cur['close'] = r['c']  # raw is sorted asc → last wins
    return out


def _match(pairs: list[tuple[float, float]]) -> tuple[int, int, list[float]]:
    diffs = [a - b for a, b in pairs]
    matches = sum(1 for d in diffs if abs(d) <= TOL)
    worst = sorted(diffs, key=abs, reverse=True)[:3]
    return matches, len(diffs), worst


def main() -> None:
    sb = get_admin_client()
    print(f"Decision-time cache vs settled REST  (RTH only, tol ${TOL})\n")
    hdr = (f"{'day':<11}{'sym':<6}{'tf':>4}  {'bars':>5}  "
           f"{'first=REST':>22}  {'settled=REST':>22}  {'first=settled':>22}")
    for day in DAYS:
        print(hdr)
        print("-" * len(hdr))
        for symbol, tf in COMBOS:
            cache = fetch_cache(sb, symbol, tf, day)
            rest = fetch_rest(symbol, tf, day)
            common = sorted(b for b in (set(cache) & set(rest)) if _in_rth(b))
            if not common:
                print(f"{day:<11}{symbol:<6}{tf:>4}  {'—':>5}  (no overlap)")
                continue
            fr = _match([(cache[b]['first_close'], rest[b]['close']) for b in common])
            sr = _match([(cache[b]['close'], rest[b]['close']) for b in common])
            fs = _match([(cache[b]['first_close'], cache[b]['close']) for b in common])

            def fmt(m):
                pct = 100.0 * m[0] / m[1] if m[1] else 0.0
                return f"{m[0]:>4}/{m[1]:<4}={pct:5.1f}% {str([round(x,2) for x in m[2]]):>9}"
            print(f"{day:<11}{symbol:<6}{tf:>4}  {len(common):>5}  "
                  f"{fmt(fr)}  {fmt(sr)}  {fmt(fs)}")
        print()


if __name__ == "__main__":
    main()
