"""Retro-replay: what WOULD the fine-TF gate keys have held, had
RORT_SHADOW_RETRUE_FORCE_FULL been armed — replayed over recorded data.

Simulates the 120s refresher tick-by-tick across a window: at each tick T,
builds the df the refresher would have had (bars whose period is fully
closed at T) and derives records via a FRESH shadow (fresh shadow ==
force-full semantics by construction). Compares against (a) the actual
live_gate telemetry (what the key really held) and (b) the settled
construction (the backtest's view), then re-scores the divergent events.

Recipe nuances (preserve — feed into the replay harness):
- LIVE-side values: prefer live_bars.first_* (decision-time WS view);
  today they are byte-identical to settled, so bar_cache/settled 1Min is
  faithful here. Always CHECK equality first; if they differ, run both.
- Fresh shadow per tick == force-full re-true. Never reuse a warmed
  shadow across ticks without force_full (the snapshot fast-path bug).
- Slice rule: at tick T a tf-bar is usable iff bar_start + tf <= T
  (equivalently index < floor(T, tf)) — mirrors _closed_bars_only.
- Anchor precondition: verify state anchor-insensitivity (2d vs 10d) once
  per interpreter before trusting a short replay window.
- Standalone-engine env SOP applies (admin ctx + pack registry + USE_DB).

Usage: cd src && ../.venv/bin/python _retro_replay_gate_states.py
"""
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv('.env')
os.environ['USE_DB'] = 'true'
os.environ['BAR_CACHE_ENABLED'] = '1'
from db import set_admin_user_context, get_admin_client
set_admin_user_context("19d47e46-f718-49a6-af32-5f5407f5b170")
import pack_registry
pack_registry.scan_and_load_all()

from data_loader import load_market_data, resample_to_timeframe
from ralph_engine import _ShadowIndicatorEngine

SYMBOL = 'TSLA'
SESSION = 'RTH'
DAY = '2026-07-14'
WINDOW = (f'{DAY} 14:30:00+00:00', f'{DAY} 15:08:00+00:00')
TICK_S = 120           # refresher cadence
PARAMS = {'ema_periods': [9, 21], 'macd_fast_period': 12,
          'macd_slow_period': 26, 'macd_signal_period': 9, 'atr_period': 14}
KEYS = {                # (tf_seconds, needed record at the 14:39 events)
    300: ('5M', {'atr', '_user_pack_swing_123'}, {'SWING_123'},
          '5M-SWING_123-BULL_C2'),
    120: ('2M', {'atr', '_user_pack_swing_123'}, {'SWING_123'},
          '2M-SWING_123-NEUTRAL'),
}

print(f"loading 1Min {SYMBOL} {SESSION} (3d window for warmup depth)...")
df1 = load_market_data(SYMBOL, days=3, timeframe='1Min', feed='sip',
                       session=SESSION)
df1 = df1[['open', 'high', 'low', 'close', 'volume']]

sb = get_admin_client()


def telemetry_series(tf):
    q = (sb.table('bar_diagnostics').select('values,created_at')
         .eq('source', 'live_gate').eq('strategy_id', 328 if tf == 300 else 327)
         .gte('created_at', f'{DAY}T13:30:00Z')
         .lte('created_at', f'{DAY}T15:30:00Z')
         .order('created_at').execute())
    out = []
    for row in q.data:
        v = row['values']
        if v.get('tf') != tf:
            continue
        sw = [r for r in v.get('records', []) if 'SWING' in r]
        out.append((row['created_at'][:19], sw[0] if sw else '(none)'))
    return out


def actual_at(series, ts_iso):
    cur = '(pre-window)'
    for ts, sw in series:
        if ts <= ts_iso:
            cur = sw
        else:
            break
    return cur


ticks = pd.date_range(WINDOW[0], WINDOW[1], freq=f'{TICK_S}s', tz='UTC')
print(f"\n{'tick':<22}{'tf':>5}  {'replayed (force-full)':<28}"
      f"{'actual key held':<28}")
results = {}
for tf, (lbl, req_ind, req_interp, needed) in KEYS.items():
    sec = resample_to_timeframe(df1.copy(), {300: '5Min', 120: '2Min'}[tf])
    tele = telemetry_series(tf)
    results[tf] = []
    for T in ticks:
        cutoff = T.floor(f'{tf}s')
        df_T = sec[sec.index < cutoff]
        sh = _ShadowIndicatorEngine(tf, set(req_ind), set(req_interp),
                                    dict(PARAMS))
        recs = sh.recompute_confluence(df_T)
        sw = sorted(r for r in recs if 'SWING' in r)
        replayed = sw[0] if sw else '(none)'
        actual = actual_at(tele, T.isoformat()[:19])
        results[tf].append((T, replayed, actual))
        print(f"{T.isoformat()[:19]:<22}{lbl:>5}  {replayed:<28}{actual:<28}")

print("\n=== VERDICT for the 14:39:00 bt-only entries (327/328) ===")
for tf, (lbl, _, _, needed) in KEYS.items():
    last = [r for r in results[tf] if r[0] <= pd.Timestamp(f'{DAY} 14:39:00+00:00')]
    replayed = last[-1][1] if last else '?'
    actual = last[-1][2] if last else '?'
    ok = needed.endswith(replayed.split('-')[-1]) and replayed == needed
    print(f"  {lbl}: needed {needed:<26} replayed-key {replayed:<26} "
          f"actual-key {actual:<26} -> {'GATE OPEN (would pair)' if replayed == needed else 'still blocked'}")
