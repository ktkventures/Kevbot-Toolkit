"""Fidelity Gate — Tier-2 live before/after (the [FC]-change protocol).

Before an [FC] change:  python -m fidelity.fidelity_gate --snapshot
  → freezes current backtest+algo trades for the live strategies from the DB
    lanes into fidelity/baseline_live.json (instant; the trusted "before").

After the change (code prepared locally):  python -m fidelity.fidelity_gate --check
  → regenerates trades via the REAL engine (get_strategy_trades) and diffs vs the
    baseline; reports any strategy whose entries/exits/prices/timestamps shifted.

Catches data-path drift that the hermetic Tier-1 fixtures can't (live data,
full window). See docs/Fidelity_Gate.md. Run deliberately around fidelity-critical
changes — not every push.
"""
import os, sys, json
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('.env', override=True)
os.environ['USE_DB'] = 'true'

import pandas as pd
from db import get_admin_client, set_admin_user_context, get_strategy_by_id_admin
import pack_registry
import services

HERE = os.path.dirname(__file__)
BASELINE = os.path.join(HERE, 'baseline_live.json')
# Live real-money set + per-pack canaries (mirror the Tier-1 coverage set).
DEFAULT_SIDS = [308, 309, 310, 311, 312, 313, 314,
                174, 288, 290, 294, 296, 298, 304, 280, 282, 284, 286, 276, 278]
_KEY_COLS = ['entry_fill_ts', 'exit_fill_ts', 'entry_price', 'exit_price', 'exit_reason']


def _lane_signature(sid):
    """Durable signature of a strategy's current backtest trades from the DB."""
    c = get_admin_client()
    rows = c.table('trades').select(','.join(_KEY_COLS)) \
        .eq('strategy_id', sid).like('data_source', 'backtest_%') \
        .order('entry_fill_ts').execute().data
    return [[str(r.get(k)) for k in _KEY_COLS] for r in (rows or [])]


def snapshot(sids):
    base = {}
    for sid in sids:
        try:
            base[str(sid)] = _lane_signature(sid)
        except Exception as e:
            base[str(sid)] = {'error': str(e)[:120]}
    with open(BASELINE, 'w') as f:
        json.dump(base, f)
    n = sum(len(v) for v in base.values() if isinstance(v, list))
    print(f"snapshot: {len(sids)} strategies, {n} backtest trades → {BASELINE}")


def check(sids):
    if not os.path.exists(BASELINE):
        print("no baseline — run --snapshot first"); return 1
    with open(BASELINE) as f:
        base = json.load(f)
    pack_registry.scan_and_load_all()
    drift = 0
    for sid in sids:
        before = base.get(str(sid))
        if not isinstance(before, list):
            print(f"  [SKIP] sid {sid}: no clean baseline"); continue
        c = get_admin_client()
        row = c.table('strategies').select('user_id').eq('id', sid).single().execute().data
        set_admin_user_context(row['user_id'])
        strat = get_strategy_by_id_admin(sid, row['user_id'])
        trades = services.get_strategy_trades(strat)
        trades = trades if trades is not None else pd.DataFrame()
        after = [[str(t.get(k)) for k in _KEY_COLS] for _, t in trades.iterrows()] \
            if len(trades) else []
        if before == after:
            print(f"  [OK]   sid {sid}: {len(after)} trades, no drift")
        else:
            drift += 1
            print(f"  [DRIFT] sid {sid}: before={len(before)} after={len(after)} trades — "
                  f"engine output CHANGED")
    print(f"\nFidelity Gate (Tier-2 live): {'CLEAN' if not drift else str(drift)+' DRIFTED'}")
    return 1 if drift else 0


if __name__ == '__main__':
    args = sys.argv[1:]
    sids = [int(a) for a in args if a.isdigit()] or DEFAULT_SIDS
    if '--snapshot' in args:
        snapshot(sids)
    elif '--check' in args:
        sys.exit(check(sids))
    else:
        print("usage: python -m fidelity.fidelity_gate --snapshot|--check [sids...]")
