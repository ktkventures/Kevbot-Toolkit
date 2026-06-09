"""Gate Parity Harness — diagnose live↔backtest cross-TF gate divergence.

Phase 2 / H1. For a gated strategy, classify each LIVE entry by whether its
cross-TF (e.g. 2m) confluence gate is satisfied under:

    PB (Previous Bar)  — the previous *closed* secondary bar's state.
                         This is what BACKTEST uses (services.py shifts the
                         secondary TF forward one period + ffill).
    CB (Current Bar)   — the secondary bar the entry falls inside, at its
                         close. PB and CB differ by exactly one secondary bar.

Then pairs each live entry against the BACKTEST lane (`trades` table,
data_source LIKE 'backtest%') at ±tolerance and flags phantoms (live entry
with no backtest match). The decisive signal: do phantoms cluster on
"CB-pass / PB-fail"? That is the off-by-one fingerprint — live behaving as
if CB while backtest gates on PB.

This is the data layer behind the "Gate Parity" detail-page tab. Read-only.

Usage:
    python gate_parity_harness.py --sid 303 --window-hours 8
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
load_dotenv(override=True)

import pack_registry
pack_registry.scan_and_load_all()

from db import get_admin_client, _row_to_strategy
from ralph_engine import StrategyMonitor, SymbolHub, _LABEL_TO_TF_SECONDS
from data_loader import load_market_data, resample_to_timeframe

TF_LABEL = {60: '1Min', 120: '2Min', 300: '5Min', 900: '15Min'}


def _parse_iso(s) -> Optional[datetime]:
    if not s:
        return None
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(s).replace('Z', '+00:00'))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def load_context(sid: int) -> dict:
    """Resolve the strategy's gate condition, entry trigger, and the
    secondary-TF shadow that computes the gate's interpreter."""
    c = get_admin_client()
    row = c.table('strategies').select('*').eq('id', sid).execute().data[0]
    strat = _row_to_strategy(row)
    conf = list(strat.get('confluence') or [])
    if not conf:
        raise SystemExit(f"sid {sid} has no confluence gate — nothing to diagnose")
    gate = conf[0]                       # e.g. '2m-UT_BOT_V4-BULL_TREND'
    tf_label = gate.split('-', 1)[0]     # '2m'
    sec_tf = _LABEL_TO_TF_SECONDS.get(tf_label, 0)
    interp = gate.split('-', 2)[1]       # 'UT_BOT_V4'
    state = gate.split('-', 2)[2]        # 'BULL_TREND'

    mon = StrategyMonitor(strat, None, general_packs=[])
    hub = SymbolHub(strat['symbol']); hub.add_monitor(mon); hub.finalize_shadow_engines()
    shadow = hub._shadow_engines.get(sec_tf)
    if shadow is None:
        raise SystemExit(f"sid {sid}: no shadow engine for {sec_tf}s — gate TF not resolved")
    return {
        'sid': sid, 'symbol': strat['symbol'], 'user_id': row['user_id'],
        'gate': gate, 'sec_tf': sec_tf, 'interp': interp, 'state': state,
        'entry_trigger': strat.get('entry_trigger'),
        'live_model': strat.get('live_model'), 'backtest_model': strat.get('backtest_model'),
        'shadow': shadow,
    }


def gate_state_series(ctx: dict, days: int = 1) -> pd.Series:
    """Closed secondary-bar state of the gate interpreter, indexed by 2m
    bar-start. Built from 10Sec base → resampled to the secondary TF, mirroring
    how the live shadow aggregates and how backtest resamples."""
    sec_tf = ctx['sec_tf']; shadow = ctx['shadow']; interp = ctx['interp']
    df1 = load_market_data(ctx['symbol'], days=days, timeframe='10Sec')
    sec = resample_to_timeframe(df1, TF_LABEL[sec_tf]).dropna(subset=['open'])
    shadow.warmup(sec.iloc[:20])
    out = {}
    for ts, r in sec.iloc[20:].iterrows():
        recs = shadow.on_bar_close({
            'open': float(r.open), 'high': float(r.high), 'low': float(r.low),
            'close': float(r.close), 'volume': float(r.volume),
            'timestamp': ts.isoformat()})
        st = next((x.split('-', 2)[2] for x in recs if f'-{interp}-' in x), None)
        out[pd.Timestamp(ts)] = st
    return pd.Series(out).sort_index()


def classify(ctx: dict, window_hours: float, tol_sec: float = 5.0) -> dict:
    c = get_admin_client()
    now = datetime.now(timezone.utc)
    lo = now - timedelta(hours=window_hours)
    lo_iso = lo.isoformat()

    states = gate_state_series(ctx)
    buckets = states.index
    want = ctx['state']

    def pb_cb(ts: datetime):
        """Return (PB_state, CB_state) for an entry at ts."""
        bs = pd.Timestamp(ts).floor(f'{ctx["sec_tf"]}s')
        # CB = bucket containing ts; PB = the bucket before it.
        cb = states.get(bs, None)
        prev = buckets[buckets < bs]
        pb = states.get(prev[-1], None) if len(prev) else None
        return pb, cb

    # live entries = alerts on the entry trigger
    al = c.table('alerts').select('timestamp,fill_ts,trigger_id').eq('strategy_id', ctx['sid'])\
        .eq('trigger_id', ctx['entry_trigger']).gte('timestamp', lo_iso).limit(5000).execute().data
    live = sorted(_parse_iso(a.get('fill_ts') or a.get('timestamp')) for a in al if _parse_iso(a.get('fill_ts') or a.get('timestamp')))

    # backtest entries (trades table, backtest lane)
    tr = c.table('trades').select('entry_fill_ts,entry_trigger_ts,data_source').eq('strategy_id', ctx['sid'])\
        .like('data_source', 'backtest%').gte('entry_fill_ts', lo_iso).limit(5000).execute().data
    bt = sorted(_parse_iso(t.get('entry_fill_ts') or t.get('entry_trigger_ts')) for t in tr if _parse_iso(t.get('entry_fill_ts') or t.get('entry_trigger_ts')))

    # two-pointer pairing live↔bt within ±tol
    bt_ts = [b.timestamp() for b in bt]
    rows = []
    for e in live:
        et = e.timestamp()
        paired = any(abs(b - et) <= tol_sec for b in bt_ts)
        pb, cb = pb_cb(e)
        rows.append({'ts': e, 'pb': pb, 'cb': cb,
                     'pb_pass': pb == want, 'cb_pass': cb == want, 'paired': paired})
    return {'rows': rows, 'live_n': len(live), 'bt_n': len(bt), 'want': want}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sid', type=int, required=True)
    ap.add_argument('--window-hours', type=float, default=8.0)
    ap.add_argument('--tolerance', type=float, default=5.0)
    args = ap.parse_args()

    ctx = load_context(args.sid)
    print(f"sid {ctx['sid']} {ctx['symbol']} | gate={ctx['gate']} | "
          f"entry={ctx['entry_trigger']} | live={ctx['live_model']} bt={ctx['backtest_model']}")
    res = classify(ctx, args.window_hours, args.tolerance)
    rows = res['rows']
    print(f"live entries={res['live_n']}  backtest entries={res['bt_n']}  "
          f"(want gate state={res['want']})\n")

    phantom = [r for r in rows if not r['paired']]
    paired = [r for r in rows if r['paired']]
    def frac(rs, key): return f"{sum(1 for r in rs if r[key])}/{len(rs)}" if rs else "0/0"

    print(f"{'entries':<10}{'n':>5}{'PB-pass':>10}{'CB-pass':>10}{'CBpass&PBfail':>15}")
    for label, rs in [('paired', paired), ('phantom', phantom), ('ALL', rows)]:
        cbpb = sum(1 for r in rs if r['cb_pass'] and not r['pb_pass'])
        print(f"{label:<10}{len(rs):>5}{frac(rs,'pb_pass'):>10}{frac(rs,'cb_pass'):>10}{cbpb:>15}")

    print("\nOff-by-one signal: of phantom entries, "
          f"{sum(1 for r in phantom if r['cb_pass'] and not r['pb_pass'])}/{len(phantom)} "
          "pass under CB but fail under PB (=> live behaving like CB while BT gates on PB).")
    print("\nsample phantom rows (ts, PB, CB):")
    for r in phantom[:10]:
        print(f"   {r['ts']:%H:%M:%S}  PB={str(r['pb']):<12} CB={str(r['cb']):<12} "
              f"pb_pass={int(r['pb_pass'])} cb_pass={int(r['cb_pass'])}")


if __name__ == '__main__':
    main()
