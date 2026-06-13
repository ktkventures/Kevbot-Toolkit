"""WS-vs-REST trigger divergence attributor (parity-sim V2, offline).

The existing parity tools measure divergence at the PRICE/bar level
(_cache_rest_parity.py: close-match %). This one measures it at the
TRIGGER level and ATTRIBUTES every divergence to a root cause — turning
"thin-tape divergence, known class" into hard per-strategy numbers.

For a saved strategy + window, it:
  1. Loads the WS-cache view (live_bars first_*, ENGINE_CONSUMED_SOURCES) —
     what the live engine actually gated on.
  2. Loads the REST view (Polygon 1-second rolled to the strategy's TF) —
     what the backtest runs on.
  3. Runs the IDENTICAL trigger pipeline on each.
  4. Aligns on bar_start and, for each bar where the entry/exit trigger
     boolean differs, attributes:
       BAR_ONLY_IN_WS   — WS had a bar REST doesn't (thin-tape phantom);
                          a live fire here can never exist in backtest.
       BAR_ONLY_IN_REST — REST had a bar WS doesn't (gap; should be ~0
                          after the 2026-06-11 gap healer).
       VALUE_FLIP       — both have the bar, but OHLC/indicator state
                          differs enough to flip the trigger boolean.

This is purely offline (reads cached live_bars + Polygon REST). It does
NOT touch ralph_engine.py or worker.py and cannot affect the live engine.

Usage:
    cd src && ../.venv/bin/python _ws_rest_trigger_divergence.py --sid 303 \
        --ws 2026-06-12T13:30 --we 2026-06-13T00:00
    # defaults: window = Friday RTH+ext if omitted; session Extended Hours
"""
from __future__ import annotations

import argparse
import warnings
from datetime import datetime, timezone

warnings.filterwarnings('ignore')

from dotenv import load_dotenv

load_dotenv('.env', override=True)
import os

os.environ.setdefault('USE_DB', 'true')

import pandas as pd

import pack_registry
import services
from data_loader import (ENGINE_CONSUMED_SOURCES, TF_TO_SECONDS,
                         fetch_1s_bars_for_window, fetch_cache_as_df,
                         resample_to_timeframe)
from db import get_admin_client, set_admin_user_context

CLOSE_FLIP_EPS = 0.001  # $ delta that can flip an EMA-cross boolean


def _parse(s: str) -> datetime:
    d = datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _enrich(strat, df, tf, s, e, session, model):
    return services.prepare_data_with_indicators(
        strat['symbol'], primary_df=df.copy(), strat=strat, timeframe=tf,
        start_date=s, end_date=e, session=session, model_override=model)


def analyze(sid: int, ws_start: datetime, ws_end: datetime,
            session: str = 'Extended Hours') -> dict:
    pack_registry.scan_and_load_all()
    c = get_admin_client()
    strat = c.table('strategies').select('*').eq('id', sid).execute().data[0]
    set_admin_user_context(strat['user_id'])
    cfg = strat['config']
    sym = strat['symbol']
    tf = cfg.get('timeframe') or strat.get('timeframe') or '10Sec'
    tf_sec = TF_TO_SECONDS.get(tf)
    entry_t = cfg.get('entry_trigger')
    exit_t = cfg.get('exit_trigger')
    model = cfg.get('backtest_model') or 'cache_locked'

    # --- load the two data views ---
    ws = fetch_cache_as_df(sym, tf_sec, ws_start, ws_end,
                           value_type='first', sources=ENGINE_CONSUMED_SOURCES)
    rest1s = fetch_1s_bars_for_window(sym, ws_start, ws_end)
    rest = resample_to_timeframe(rest1s, tf)
    rest = rest[(rest.index >= ws_start) & (rest.index <= ws_end)]

    # --- run the identical trigger pipeline on each ---
    enr_ws = _enrich(strat, ws, tf, ws_start, ws_end, session, model)
    enr_rest = _enrich(strat, rest, tf, ws_start, ws_end, session, model)

    ws_idx = set(enr_ws.index)
    rest_idx = set(enr_rest.index)
    only_ws = ws_idx - rest_idx
    only_rest = rest_idx - ws_idx
    both = ws_idx & rest_idx

    out = {
        'sid': sid, 'symbol': sym, 'tf': tf, 'session': session,
        'window': f'{ws_start.isoformat()} -> {ws_end.isoformat()}',
        'ws_bars': len(enr_ws), 'rest_bars': len(enr_rest),
        'bars_only_in_ws': len(only_ws), 'bars_only_in_rest': len(only_rest),
        'bars_in_both': len(both),
        'triggers': {},
    }

    for label, trig in (('entry', entry_t), ('exit', exit_t)):
        if not trig:
            continue
        col = f'trig_{trig}'
        if col not in enr_ws.columns or col not in enr_rest.columns:
            out['triggers'][label] = {'trigger': trig, 'error': 'col missing'}
            continue
        ws_fires = set(enr_ws.index[enr_ws[col].fillna(False).astype(bool)])
        rest_fires = set(
            enr_rest.index[enr_rest[col].fillna(False).astype(bool)])
        ws_only = ws_fires - rest_fires   # live phantom candidates
        rest_only = rest_fires - ws_fires  # missed candidates

        def _attrib(ts_set):
            # VALUE_FLIP is split: CLOSE_DIFF = this bar's OHLC differs
            # enough to flip the boolean; INDICATOR_DRIFT = this bar's
            # close matches (Δ<eps) but the trigger still flips, i.e. the
            # divergence is accumulated indicator state from earlier
            # differing bars, not this bar's price.
            buckets = {'BAR_ONLY_IN_WS': [], 'BAR_ONLY_IN_REST': [],
                       'CLOSE_DIFF': [], 'INDICATOR_DRIFT': []}
            for ts in sorted(ts_set):
                if ts in only_ws:
                    buckets['BAR_ONLY_IN_WS'].append(ts)
                elif ts in only_rest:
                    buckets['BAR_ONLY_IN_REST'].append(ts)
                else:
                    dc = abs(float(enr_ws.loc[ts, 'close'])
                             - float(enr_rest.loc[ts, 'close']))
                    if dc >= CLOSE_FLIP_EPS:
                        buckets['CLOSE_DIFF'].append(ts)
                    else:
                        buckets['INDICATOR_DRIFT'].append(ts)
            return buckets

        # close-delta context on value-flip bars
        def _flip_detail(ts_list):
            rows = []
            for ts in ts_list[:8]:
                wc = float(enr_ws.loc[ts, 'close'])
                rc = float(enr_rest.loc[ts, 'close'])
                rows.append(f'{ts.strftime("%H:%M:%S")}(Δclose={wc-rc:+.4f})')
            return rows

        ph = _attrib(ws_only)
        mi = _attrib(rest_only)
        out['triggers'][label] = {
            'trigger': trig,
            'ws_fires': len(ws_fires), 'rest_fires': len(rest_fires),
            'agreed': len(ws_fires & rest_fires),
            'phantom_total': len(ws_only),
            'phantom_by_cause': {k: len(v) for k, v in ph.items()},
            'phantom_flip_samples': _flip_detail(ph['CLOSE_DIFF']),
            'missed_total': len(rest_only),
            'missed_by_cause': {k: len(v) for k, v in mi.items()},
            'missed_flip_samples': _flip_detail(mi['CLOSE_DIFF']),
        }
    return out


def _print(r: dict):
    print(f"\n=== WS-vs-REST trigger divergence — sid {r['sid']} "
          f"({r['symbol']} {r['tf']}, {r['session']}) ===")
    print(f"window: {r['window']}")
    print(f"bars: WS={r['ws_bars']} REST={r['rest_bars']} "
          f"| only-WS={r['bars_only_in_ws']} only-REST={r['bars_only_in_rest']} "
          f"both={r['bars_in_both']}")
    for label, t in r['triggers'].items():
        if 'error' in t:
            print(f"  [{label}] {t['trigger']}: {t['error']}")
            continue
        print(f"\n  [{label}] {t['trigger']}")
        print(f"    fires: WS={t['ws_fires']} REST={t['rest_fires']} "
              f"agreed={t['agreed']}")
        print(f"    PHANTOM (WS-only fires): {t['phantom_total']}  "
              f"by-cause={t['phantom_by_cause']}")
        if t['phantom_flip_samples']:
            print(f"      flip samples: {t['phantom_flip_samples']}")
        print(f"    MISSED (REST-only fires): {t['missed_total']}  "
              f"by-cause={t['missed_by_cause']}")
        if t['missed_flip_samples']:
            print(f"      flip samples: {t['missed_flip_samples']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sid', type=int, required=True)
    ap.add_argument('--ws', default='2026-06-12T13:30:00+00:00')
    ap.add_argument('--we', default='2026-06-13T00:00:00+00:00')
    ap.add_argument('--session', default='Extended Hours')
    a = ap.parse_args()
    _print(analyze(a.sid, _parse(a.ws), _parse(a.we), a.session))
