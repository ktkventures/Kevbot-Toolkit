"""Fidelity Gate — Tier-1 golden replay (real-bar, local/nightly).

For each captured fixture: patch the data loader to the frozen raw bars, re-run
the REAL prepare_data_with_indicators + unified_trades, and assert the gate/
trigger columns AND trades are byte-identical to the golden snapshot. Any drift
in the prep path (#21 group scoping, #29 warmup/resample/trim) or the engine
fails here. See docs/Fidelity_Gate.md.

Runs where Supabase is reachable (local/nightly) — confluence groups load live;
the strategy config + bars + golden outputs are frozen. (Per-push CI runs the
fast synthetic suite instead.)

Usage (from src/):  python -m fidelity.golden [sid ...]   (no args = all fixtures)
"""
import os, sys, json, glob
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('.env', override=True)
os.environ['USE_DB'] = 'true'

import pandas as pd
from pandas.testing import assert_frame_equal
from db import set_admin_user_context
import pack_registry
import services

FIX_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
_TRADE_COLS = ['entry_fill_ts', 'exit_fill_ts', 'entry_price', 'exit_price',
               'exit_reason', 'entry_trigger', 'exit_trigger']


def _load_manifest(sid):
    with open(os.path.join(FIX_DIR, f"sid_{sid}_manifest.json")) as f:
        return json.load(f)


def replay(sid) -> dict:
    """Return {sid, ok, detail} after asserting fixture parity."""
    m = _load_manifest(sid)
    strat = m['strat']
    uid = strat.get('user_id')
    if uid:
        set_admin_user_context(uid)
    pack_registry.scan_and_load_all()

    bars = pd.read_parquet(os.path.join(FIX_DIR, f"sid_{sid}_bars.parquet"))
    gold_cols = pd.read_parquet(os.path.join(FIX_DIR, f"sid_{sid}_gatecols.parquet"))
    gold_trades = pd.read_parquet(os.path.join(FIX_DIR, f"sid_{sid}_trades.parquet"))

    _orig = services.load_market_data
    services.load_market_data = lambda symbol, **kw: bars.copy()
    try:
        enriched = services.prepare_data_with_indicators(
            symbol=m['symbol'], timeframe=m['timeframe'], days=m['window_days'],
            session=m['session'], secondary_tfs=tuple(m['secondary_tfs']), strat=strat)
        trades = services.unified_trades(enriched, strat)
    finally:
        services.load_market_data = _orig
    trades = trades if trades is not None else pd.DataFrame()

    # 1) gate/trigger/interpreter columns byte-identical
    missing = [c for c in gold_cols.columns if c not in enriched.columns]
    if missing:
        return {'sid': sid, 'ok': False,
                'detail': f"{len(missing)} gate cols MISSING (e.g. {missing[:4]})"}
    try:
        assert_frame_equal(enriched[gold_cols.columns].reset_index(drop=True),
                           gold_cols.reset_index(drop=True),
                           check_exact=True, check_dtype=False)
    except AssertionError as e:
        return {'sid': sid, 'ok': False, 'detail': f"GATE COLS DRIFT: {str(e)[:200]}"}

    # 2) trades byte-identical (on the durable columns)
    if len(trades) != len(gold_trades):
        return {'sid': sid, 'ok': False,
                'detail': f"TRADE COUNT {len(trades)} != golden {len(gold_trades)}"}
    if len(gold_trades) > 0:
        cols = [c for c in _TRADE_COLS if c in gold_trades.columns and c in trades.columns]
        try:
            assert_frame_equal(
                trades[cols].reset_index(drop=True).astype(str),
                gold_trades[cols].reset_index(drop=True).astype(str),
                check_exact=True)
        except AssertionError as e:
            return {'sid': sid, 'ok': False, 'detail': f"TRADE DRIFT: {str(e)[:200]}"}

    return {'sid': sid, 'ok': True,
            'detail': f"{len(gold_cols.columns)} cols, {len(gold_trades)} trades OK"}


def main(argv):
    if argv:
        sids = [int(x) for x in argv]
    else:
        sids = sorted(int(os.path.basename(p).split('_')[1])
                      for p in glob.glob(os.path.join(FIX_DIR, "sid_*_manifest.json")))
    if not sids:
        print("no fixtures found"); return 1
    fails = 0
    for sid in sids:
        try:
            r = replay(sid)
        except Exception as e:
            r = {'sid': sid, 'ok': False, 'detail': f"EXC {type(e).__name__}: {str(e)[:160]}"}
        mark = "PASS" if r['ok'] else "FAIL"
        if not r['ok']:
            fails += 1
        print(f"  [{mark}] sid {sid}: {r['detail']}")
    print(f"\nFidelity Gate (Tier-1 golden): {len(sids)-fails}/{len(sids)} passed")
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
