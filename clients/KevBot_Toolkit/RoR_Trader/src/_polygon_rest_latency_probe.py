"""Polygon REST latency probe.

Goals (2026-05-27):
1. Measure how fast Polygon REST per-second aggregates settle after each
   second closes. This is the latency floor for a `rest_polled` live model.
2. Measure how fast 1Min aggregates settle. If 1Min is faster than
   summing 60 second-aggs, the new model could use the 1Min endpoint
   directly for 1Min strategies.
3. Compare to `live_bars` (Ralph's WS pipeline) write timing for the
   same bars.

Run while market is open OR during extended hours when there's some
activity. Polls every 250ms for ~10 minutes.

Run: cd src && ../.venv/bin/python _polygon_rest_latency_probe.py
"""
from __future__ import annotations
import os
import sys
import time
import statistics
from collections import defaultdict
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dotenv import load_dotenv
load_dotenv(override=True)

import requests

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
SYMBOL = "SPY"
POLL_INTERVAL_SEC = 0.25   # 4 polls/sec
RUN_DURATION_SEC = 600     # 10 min
LOOKBACK_SEC = 30          # look back 30 sec on each poll

# Track each (bar_start_ts -> [(poll_time, close, volume)]) so we can
# compute: first-appearance latency, settle latency, value drift.
sec_observations: dict = defaultdict(list)
min_observations: dict = defaultdict(list)


def _fetch_polygon(timespan: str, from_iso: str, to_iso: str):
    """Hit Polygon /v2/aggs for the given range. Returns list of bar dicts."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{SYMBOL}/range/1/{timespan}/{from_iso}/{to_iso}"
    params = {'apiKey': POLYGON_API_KEY, 'limit': 50000, 'sort': 'asc', 'adjusted': 'true'}
    try:
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code != 200:
            return []
        return resp.json().get('results', [])
    except Exception:
        return []


def main():
    if not POLYGON_API_KEY:
        print("POLYGON_API_KEY not set"); return
    start_wall = datetime.now(timezone.utc)
    deadline = start_wall + timedelta(seconds=RUN_DURATION_SEC)
    print(f"Starting probe at {start_wall.isoformat()}")
    print(f"Will run for {RUN_DURATION_SEC}s ({RUN_DURATION_SEC//60}min), poll every {POLL_INTERVAL_SEC*1000:.0f}ms")
    print(f"Tracking {LOOKBACK_SEC}s lookback per poll")
    print()

    poll_count = 0
    while datetime.now(timezone.utc) < deadline:
        poll_start = datetime.now(timezone.utc)
        # Polygon expects date-only or millis-since-epoch. Use millis for second-precision.
        from_ms = int((poll_start - timedelta(seconds=LOOKBACK_SEC)).timestamp() * 1000)
        to_ms = int(poll_start.timestamp() * 1000)
        # Per-second aggregates
        sec_bars = _fetch_polygon('second', str(from_ms), str(to_ms))
        for bar in sec_bars:
            bar_t = bar['t']  # ms since epoch, bar START
            sec_observations[bar_t].append({
                'poll_ms': int(poll_start.timestamp() * 1000),
                'open': bar.get('o'), 'high': bar.get('h'), 'low': bar.get('l'),
                'close': bar.get('c'), 'volume': bar.get('v'),
            })
        poll_count += 1

        # Every 4 polls (1 sec), also pull last 5 min of 1Min aggregates
        if poll_count % 4 == 0:
            min_from = int((poll_start - timedelta(minutes=5)).timestamp() * 1000)
            min_bars = _fetch_polygon('minute', str(min_from), str(to_ms))
            for bar in min_bars:
                bar_t = bar['t']
                min_observations[bar_t].append({
                    'poll_ms': int(poll_start.timestamp() * 1000),
                    'open': bar.get('o'), 'high': bar.get('h'), 'low': bar.get('l'),
                    'close': bar.get('c'), 'volume': bar.get('v'),
                })

        # Sleep to maintain poll cadence
        elapsed = (datetime.now(timezone.utc) - poll_start).total_seconds()
        sleep_for = max(0, POLL_INTERVAL_SEC - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)
        if poll_count % 60 == 0:
            print(f"  ...poll #{poll_count} at {poll_start.strftime('%H:%M:%S')}, "
                  f"sec_bars_tracked={len(sec_observations)}, "
                  f"min_bars_tracked={len(min_observations)}")

    end_wall = datetime.now(timezone.utc)
    print()
    print(f"Probe complete. Polled {poll_count} times in {(end_wall - start_wall).total_seconds():.0f}s")
    print(f"Tracked {len(sec_observations)} distinct second-bars and {len(min_observations)} minute-bars")
    print()

    # Analysis: per-second latency
    def _analyze(observations: dict, label: str, tf_sec: int):
        first_lats = []   # bar_close → first appearance
        settle_lats = []  # bar_close → last value-change
        for bar_t_ms, obs_list in observations.items():
            bar_start_ms = bar_t_ms
            bar_close_ms = bar_start_ms + tf_sec * 1000
            # Sort observations by poll_ms
            obs_sorted = sorted(obs_list, key=lambda x: x['poll_ms'])
            if not obs_sorted:
                continue
            first_poll = obs_sorted[0]['poll_ms']
            first_lat = (first_poll - bar_close_ms) / 1000.0
            # Settle: find last poll that had a DIFFERENT close from the eventual final close
            final_close = obs_sorted[-1]['close']
            settle_poll = first_poll
            for o in obs_sorted:
                if o['close'] != final_close:
                    settle_poll = o['poll_ms']
            settle_lat = (settle_poll - bar_close_ms) / 1000.0
            # Sanity: only count bars where first_lat > 0 (bar was closed when we first saw it)
            # and first_lat < 60 (don't count weird outliers)
            if 0 <= first_lat <= 60:
                first_lats.append(first_lat)
            if 0 <= settle_lat <= 60:
                settle_lats.append(settle_lat)

        if not first_lats:
            print(f"{label}: no usable observations"); return
        first_lats.sort(); settle_lats.sort()
        def pct(arr, p):
            if not arr: return float('nan')
            return arr[min(len(arr) - 1, int(len(arr) * p / 100))]
        print(f"{label} latency (n={len(first_lats)}):")
        print(f"  first-appearance lat:  min={first_lats[0]:.2f}s  med={pct(first_lats,50):.2f}s  p75={pct(first_lats,75):.2f}s  p90={pct(first_lats,90):.2f}s  p95={pct(first_lats,95):.2f}s  max={first_lats[-1]:.2f}s")
        print(f"  value-settle lat:      min={settle_lats[0]:.2f}s  med={pct(settle_lats,50):.2f}s  p75={pct(settle_lats,75):.2f}s  p90={pct(settle_lats,90):.2f}s  p95={pct(settle_lats,95):.2f}s  max={settle_lats[-1]:.2f}s")

    _analyze(sec_observations, "SPY per-second", 1)
    print()
    _analyze(min_observations, "SPY per-minute", 60)


if __name__ == '__main__':
    main()
