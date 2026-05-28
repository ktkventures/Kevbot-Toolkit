"""Hybrid WS + REST simulation.

Question: if a new live model used WS-tip values at bar close (so it
could fire at +0s instead of waiting +2-3s for REST), how often would
the WS-tip value match the eventually-settled REST value?

Approach: pull historical `live_bars` (Ralph's actual WS captures)
and compare to REST 1-sec aggregates resampled to the same TF.

Two angles measured:
1. Value agreement: when both sides have the bar, how often does
   close (and first_close, which preserves Ralph's first WS read)
   match REST?
2. Coverage: how often does WS have a bar but REST doesn't (or
   vice versa)?

Run: cd src && ../.venv/bin/python _hybrid_ws_rest_sim.py
"""
from __future__ import annotations
import os
import sys
from datetime import datetime, timezone, timedelta
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dotenv import load_dotenv
load_dotenv(override=True)

import pandas as pd

from db import get_admin_client
from data_loader import fetch_1s_bars_for_window


def _pct(arr, p):
    if not arr:
        return float('nan')
    return arr[min(len(arr) - 1, int(len(arr) * p / 100))]


def compare_window(symbol: str, tf_sec: int, hours_back: float):
    c = get_admin_client()
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=hours_back)

    print(f"\n=== {symbol} {tf_sec}s, last {hours_back}h "
          f"({start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')} UTC) ===")

    # Pull WS-captured bars from live_bars
    r = c.table('live_bars').select(
        'bar_start,open,high,low,close,first_open,first_high,first_low,'
        'first_close,first_volume,volume,written_at,source'
    ).eq('symbol', symbol).eq('timeframe_seconds', tf_sec).gte(
        'bar_start', start_dt.isoformat()).lte(
        'bar_start', end_dt.isoformat()).order('bar_start').execute()
    ws_bars = r.data or []
    print(f"  WS bars in live_bars:                     {len(ws_bars)}")

    # Pull REST 1s and resample
    rest_1s = fetch_1s_bars_for_window(
        symbol, start_dt - timedelta(seconds=60), end_dt + timedelta(seconds=60),
        padding_seconds=30)
    if len(rest_1s) == 0:
        print(f"  REST returned no data — skipping")
        return
    freq = f"{tf_sec}s"
    rest_tf = rest_1s.resample(freq, label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    # Trim to actual window
    rest_tf = rest_tf[(rest_tf.index >= start_dt) & (rest_tf.index <= end_dt)]
    print(f"  REST {tf_sec}s bars in window:               {len(rest_tf)}")

    # Coverage analysis
    ws_ts = {pd.Timestamp(b['bar_start']).tz_convert('UTC') for b in ws_bars}
    rest_ts = set(rest_tf.index)
    in_both = ws_ts & rest_ts
    ws_only = ws_ts - rest_ts
    rest_only = rest_ts - ws_ts
    print(f"  Coverage:")
    print(f"    bars in BOTH:                            {len(in_both)}")
    print(f"    bars in WS only (REST missing):          {len(ws_only)}")
    print(f"    bars in REST only (WS missing — gap):    {len(rest_only)}")

    # Value comparison on the intersection
    close_match_first, close_match_settled = 0, 0
    close_drift_first, close_drift_settled = [], []
    has_rebroadcast = 0
    bars_by_first_close_key = {pd.Timestamp(b['bar_start']).tz_convert('UTC'): b
                                for b in ws_bars}
    for ts in sorted(in_both):
        ws_bar = bars_by_first_close_key[ts]
        rest_row = rest_tf.loc[ts]
        fc = ws_bar.get('first_close')
        c_ = ws_bar.get('close')
        rc = float(rest_row['close'])
        if fc is not None:
            d = abs(fc - rc)
            close_drift_first.append(d)
            if d < 0.0001:
                close_match_first += 1
        if c_ is not None:
            d = abs(c_ - rc)
            close_drift_settled.append(d)
            if d < 0.0001:
                close_match_settled += 1
        if fc is not None and c_ is not None and abs(fc - c_) > 0.0001:
            has_rebroadcast += 1

    if not in_both:
        print(f"  (no overlapping bars to compare values)")
        return

    n = len(in_both)
    print(f"  Value match (close, |Δ| < 0.0001):")
    print(f"    WS first_close vs REST settled:          "
          f"{close_match_first}/{n} = {close_match_first*100/n:.1f}%")
    print(f"    WS settled close vs REST settled:        "
          f"{close_match_settled}/{n} = {close_match_settled*100/n:.1f}%")
    print(f"  WS bars that received rebroadcast correction (close != first_close): "
          f"{has_rebroadcast}/{n} = {has_rebroadcast*100/n:.1f}%")
    if close_drift_first:
        s = sorted(close_drift_first)
        print(f"  Drift distribution (WS first_close vs REST):")
        print(f"    max={s[-1]:.4f}  p99={_pct(s,99):.4f}  p95={_pct(s,95):.4f}  "
              f"p90={_pct(s,90):.4f}  p75={_pct(s,75):.4f}  median={_pct(s,50):.4f}")


def main():
    # Last 4 hours covers post-RTH window where we saw the problem
    for symbol in ('SPY',):
        for tf_sec in (10, 60):
            compare_window(symbol, tf_sec, hours_back=4)


if __name__ == '__main__':
    main()
