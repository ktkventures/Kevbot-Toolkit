"""Fidelity Gate — fixture capture (one-time, non-hermetic).

Freezes a strategy's RAW primary bars + golden enriched gate/trigger columns +
golden trades into fidelity/fixtures/, so the hermetic Tier-1 golden test can
replay the REAL prepare_data_with_indicators pipeline offline and assert no
drift. See docs/Fidelity_Gate.md.

Usage (from src/):  python -m fidelity.capture <sid> [window_days]
"""
import os, sys, json, time
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('.env', override=True)
os.environ['USE_DB'] = 'true'

import pandas as pd
from db import get_admin_client, set_admin_user_context, get_strategy_by_id_admin
import pack_registry
import services
from data_loader import get_required_tfs_from_confluence, get_tf_from_label

FIX_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


def capture(sid: int, window_days: int = 30) -> dict:
    pack_registry.scan_and_load_all()
    c = get_admin_client()
    row = c.table('strategies').select('user_id').eq('id', sid).single().execute().data
    uid = row['user_id']
    set_admin_user_context(uid)
    strat = get_strategy_by_id_admin(sid, uid)
    cfg = strat.get('config') or {}
    conf = cfg.get('confluence', strat.get('confluence', []))
    sec = tuple(sorted(get_tf_from_label(l) for l in get_required_tfs_from_confluence(conf)))
    sym = strat.get('symbol'); tf = strat.get('timeframe')

    # RECORD the raw primary load
    _orig = services.load_market_data
    rec = {}
    def _rec(symbol, **kw):
        df = _orig(symbol, **kw)
        rec['primary'] = df.copy()
        return df
    services.load_market_data = _rec
    t0 = time.time()
    try:
        enriched = services.prepare_data_with_indicators(
            symbol=sym, timeframe=tf, days=window_days,
            session=strat.get('trading_session', 'RTH'),
            secondary_tfs=sec, strat=strat)
        golden = services.unified_trades(enriched, strat)
    finally:
        services.load_market_data = _orig
    elapsed = time.time() - t0

    if 'primary' not in rec or rec['primary'] is None or len(rec['primary']) == 0:
        raise RuntimeError(f"sid {sid}: loader captured no bars")
    golden = golden if golden is not None else pd.DataFrame()

    base = os.path.join(FIX_DIR, f"sid_{sid}")
    rec['primary'].to_parquet(base + "_bars.parquet")
    # gate/trigger/interpreter columns are the fidelity surface
    gate_cols = [col for col in enriched.columns
                 if col.startswith('trig_') or '__' in col or col == 'atr']
    enriched[gate_cols].to_parquet(base + "_gatecols.parquet")
    golden.to_parquet(base + "_trades.parquet")
    manifest = {
        'sid': sid, 'symbol': sym, 'timeframe': tf, 'secondary_tfs': list(sec),
        'window_days': window_days, 'session': strat.get('trading_session', 'RTH'),
        'strat': strat, 'n_bars': int(len(rec['primary'])),
        'n_trades': int(len(golden)), 'n_gate_cols': len(gate_cols),
        'captured_elapsed_s': round(elapsed, 1),
    }
    with open(base + "_manifest.json", 'w') as f:
        json.dump(manifest, f, default=str, indent=0)
    return manifest


if __name__ == '__main__':
    sid = int(sys.argv[1]); win = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    m = capture(sid, win)
    print(f"sid {m['sid']} {m['symbol']}/{m['timeframe']} sec={m['secondary_tfs']} "
          f"bars={m['n_bars']} trades={m['n_trades']} cols={m['n_gate_cols']} "
          f"{m['captured_elapsed_s']}s")
