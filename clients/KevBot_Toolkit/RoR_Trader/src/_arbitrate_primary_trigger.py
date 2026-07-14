"""Three-lane arbitration for a PRIMARY-side divergence (gates open on both
lanes, but one lane fires an entry the other doesn't).

Computes the ALGO lane offline (decision-time / cache-locked model) for one
strategy + window, and lines it up against:
  - backtest lane (trades.data_source='backtest_rest_hifi')  = settled REST
  - live lane      (alerts, side=entry/exit, fill_ts)        = WS dispatch

Arbitration (bug-hunt skill):
  algo ~= live, both != backtest  -> WS-vs-REST decision-time floor (NOTE it)
  algo ~= backtest, live != both  -> live real-time path bug (usually)
  live ~= backtest, algo != both  -> algo/model config mismatch (check flags)

ALSO prints the primary-bar diff at the disputed timestamps (WS live_bars
first_* vs settled bar_cache) so a bar-level cause is visible immediately.

Load discipline (memory feedback_local_analysis_starves_live_worker): ONE
strategy, ONE window, single script, run OUTSIDE RTH. Aborts if the window
is wider than a day.

Usage: cd src && ../.venv/bin/python _arbitrate_primary_trigger.py \
           --sid 333 --since 2026-07-14T17:52:00Z --until 2026-07-14T19:15:00Z
"""
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv('.env')
os.environ['USE_DB'] = 'true'
os.environ['BAR_CACHE_ENABLED'] = '1'
# Mirror prod OFFLINE engine flags (api/batch-worker) — local-update hard rule 0.
for _k, _v in {
    'RORT_COARSE_SECONDARY_FROM_1MIN': '1', 'RORT_ENFORCE_1MIN_GATE': '1',
    'RORT_RESAMPLED_STORE_READ': '1', 'RORT_RESAMPLED_STORE_SERVE': '1',
    'RORT_RESUME_INHERIT_POSITION': '1', 'RORT_RIGHTSIZE_WARMUP': '1',
    'RORT_SCOPE_CONFLUENCE_GROUPS': '1', 'RORT_SECONDARY_TF_CACHE': '1',
    'RORT_SECONDARY_TF_SNAPSHOT': '1', 'RORT_USE_DOUGH_CACHE': '1',
}.items():
    os.environ.setdefault(_k, _v)

from db import set_admin_user_context, get_admin_client  # noqa: E402
ADMIN = "19d47e46-f718-49a6-af32-5f5407f5b170"
set_admin_user_context(ADMIN)
import pack_registry  # noqa: E402
pack_registry.scan_and_load_all()

ap = argparse.ArgumentParser()
ap.add_argument('--sid', type=int, required=True)
ap.add_argument('--since', required=True)
ap.add_argument('--until', required=True)
args = ap.parse_args()

since = datetime.fromisoformat(args.since.replace('Z', '+00:00'))
until = datetime.fromisoformat(args.until.replace('Z', '+00:00'))
assert until - since <= timedelta(days=1), "window too wide (load discipline)"

sb = get_admin_client()
from db import get_strategy_by_id_admin  # noqa: E402
strat = get_strategy_by_id_admin(args.sid, ADMIN)
sym = strat['symbol']
tf = strat['timeframe']
print(f"sid={args.sid} {sym}/{tf}  window {since:%H:%M}–{until:%H:%M}Z")
print(f"  entry={strat.get('entry_trigger')} exit={strat.get('exit_trigger')} "
      f"conf={strat.get('confluence')}")

# ---- LIVE lane (alerts) ----
al = (sb.table('alerts').select('side,fill_ts,timestamp')
      .eq('strategy_id', args.sid)
      .gte('timestamp', (since - timedelta(minutes=30)).isoformat())
      .lte('timestamp', (until + timedelta(minutes=30)).isoformat())
      .order('timestamp').execute()).data
live = [(a['side'], a['fill_ts']) for a in al
        if a['fill_ts'] and since.isoformat() <= a['fill_ts'] <= until.isoformat()]
lag = [(a['fill_ts'][11:19],
        (datetime.fromisoformat(a['timestamp'].replace('Z', '+00:00'))
         - datetime.fromisoformat(a['fill_ts'].replace('Z', '+00:00'))).total_seconds())
       for a in al if a['fill_ts']]
print(f"\nLIVE  : {[(s[:3], f[11:19]) for s, f in live]}")
stalled = [(t, l) for t, l in lag if l > 60]
if stalled:
    print(f"  ⚠️  dispatch lag >60s on {stalled} — window partly CONTAMINATED "
          f"(see memory feedback_local_analysis_starves_live_worker)")

# ---- BACKTEST lane ----
bt = (sb.table('trades').select('entry_fill_ts,exit_fill_ts')
      .eq('strategy_id', args.sid).eq('data_source', 'backtest_rest_hifi')
      .gte('entry_fill_ts', since.isoformat())
      .lte('entry_fill_ts', until.isoformat())
      .order('entry_fill_ts').execute()).data
print(f"BT    : {[(t['entry_fill_ts'][11:19], (t['exit_fill_ts'] or '')[11:19]) for t in bt]}")

# ---- ALGO lane (computed offline, decision-time / cache-locked model) ----
import strategy_data  # noqa: E402
import services  # noqa: E402
algo_trades = None
for model in ('cache_cache_locked', 'cache_locked', 'algo'):
    try:
        s2 = dict(strat)
        s2['backtest_model'] = model
        s2['data_source'] = model
        res = services.get_strategy_trades(s2)
        if res is not None:
            algo_trades = res
            print(f"\nALGO  : computed with model={model}")
            break
    except Exception as e:  # noqa: BLE001
        print(f"  (model={model} failed: {str(e)[:90]})")
if algo_trades is not None:
    df = algo_trades if isinstance(algo_trades, pd.DataFrame) else pd.DataFrame(algo_trades)
    col = 'entry_fill_ts' if 'entry_fill_ts' in df.columns else 'entry_time'
    if len(df):
        df[col] = pd.to_datetime(df[col], utc=True)
        w = df[(df[col] >= since) & (df[col] <= until)]
        print(f"ALGO  : {[t.strftime('%H:%M:%S') for t in w[col]]}")
    else:
        print("ALGO  : (no trades)")

# ---- Primary-bar diff at the disputed times (WS decision-time vs settled) ----
tf_s = {'10Sec': 10, '15Sec': 15, '30Sec': 30, '1Min': 60}.get(tf, 60)
lb = pd.DataFrame((sb.table('live_bars')
    .select('bar_start,open,high,low,close,first_open,first_high,first_low,first_close')
    .eq('symbol', sym).eq('timeframe_seconds', tf_s)
    .gte('bar_start', since.isoformat()).lte('bar_start', until.isoformat())
    .order('bar_start').execute()).data)
print(f"\nprimary {tf} live_bars rows in window: {len(lb)}")
if len(lb):
    lb = lb.set_index('bar_start')
    n_rev = sum(1 for _, r in lb.iterrows()
                if r.get('first_close') is not None
                and abs(float(r['close']) - float(r['first_close'])) > 0.005)
    print(f"  bars whose WS first-write CLOSE != healed close (>0.5c): {n_rev}/{len(lb)}")
