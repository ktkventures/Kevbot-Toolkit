"""Tier B + Tier C migration: explicit per-strategy swap plan.

Tier B (ema_price_position_v2 → ema_pp_v3 OR ema_pp_v4):
  Per-strategy decision based on trigger semantic:
    `_ib` suffix → v3 (L0 intra-bar matches the original)
    bare        → v4 (L1 1-bar lag closest to original C-type behavior)

Tier C (utbot_v2 → ut_bot_v4):
  utbot_v2_default_buy   → utv4_default_bull_flip
  utbot_v2_default_sell  → utv4_default_bear_flip
  utbot_v2_default_buy_ib → utv4_default_bull_flip (no _ib variant in v4 —
                            behavior-change risk; flagged below)
  Confluence record: '{tf}-UTBOT-BULL'  → '{tf}-UT_BOT_V4-BULL_TREND'
                     '{tf}-UTBOT-BEAR'  → '{tf}-UT_BOT_V4-BEAR_TREND'

Plus incidental Tier A swaps (macd_line, macd_histogram, vwap, rvol,
ema_stack) wherever they appear alongside Tier B/C references — the
mirror should be ALL_MIRRORED, not partial.

sid 66 has BOTH Tier B (entry) and Tier C (exit) references — produces
one combined mirror.

Usage:
  python _migrate_tier_bc.py                  # dry run
  python _migrate_tier_bc.py --execute        # create mirrors (no backtest)
  python _migrate_tier_bc.py --execute --backtest  # also run backtests

Originals are never modified. Per Kevin's plan, originals firing alerts
stay live as side-by-side reference baselines.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
import copy
import json
import sys
from typing import Optional

import pack_registry
pack_registry.scan_and_load_all()

from db import (
    get_admin_client, set_admin_user_context, get_strategy_by_id_db,
    save_strategy_db,
)

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"


# ============================================================================
# Per-strategy swap plan
# ============================================================================
#
# Each entry maps a strategy sid → list of swap operations applied to its
# config before saving as a new mirror. Operations are applied in order.
#
# Operations:
#   ('entry', new_trigger_id)          — replace entry_trigger_confluence_id
#   ('exit', old_trigger_id, new_trigger_id) — replace one exit trigger
#   ('confluence', old_record, new_record)   — replace one confluence record

SWAP_PLAN: dict[int, list[tuple]] = {
    # ─── Tier B (ema_price_position_v2 → ema_pp_v3 OR ema_pp_v4) ───
    66: [
        # Tier B entry: bare cross_short_up → v4 (L1, closest to bar-close)
        ('entry', 'eppv4_default_cross_short_up'),
        # Tier C exit: utbot_v2_default_sell → utv4_default_bear_flip
        ('exit', 'utbot_v2_default_sell', 'utv4_default_bear_flip'),
        # Confluences: 1d MACD_LINE → MACD_LINE_V2; 5m UTBOT BEAR → UT_BOT_V4 BEAR_TREND
        ('confluence', '1d-MACD_LINE-M>S-', '1d-MACD_LINE_V2-M>S-'),
        ('confluence', '5m-UTBOT-BEAR', '5m-UT_BOT_V4-BEAR_TREND'),
    ],
    88: [
        # Entry is already user-pack (stochastic_oscillator) — no change.
        # Exit: ema_price_position_v2_default_cross_short_down_ib → v4
        # (v3 retired 2026-04-29 per Kevin; consolidated to v4. Trade-off:
        #  v4's L1 1-bar lag is more conservative than the original _ib
        #  intra-bar semantic; loses some responsiveness in exchange for
        #  fewer false fills.)
        ('exit', 'ema_price_position_v2_default_cross_short_down_ib',
                 'eppv4_default_cross_short_down'),
    ],
    124: [
        # Entry: _ib suffix → v4 (was v3 pre-consolidation)
        ('entry', 'eppv4_default_cross_short_up'),
        # Exit: _ib suffix → v4
        ('exit', 'ema_price_position_v2_default_cross_mid_down_ib',
                 'eppv4_default_cross_mid_down'),
    ],
    134: [
        # Entry: bare → v4
        ('entry', 'eppv4_default_cross_short_up'),
        # Exit: bare → v4
        ('exit', 'ema_price_position_v2_default_cross_mid_down',
                 'eppv4_default_cross_mid_down'),
        # Confluence already references UT_BOT_V4 (no change)
    ],

    # ─── Tier C (utbot_v2 → ut_bot_v4) ───
    59: [
        ('entry', 'utv4_default_bull_flip'),
        ('exit', 'utbot_v2_default_sell', 'utv4_default_bear_flip'),
        ('confluence', '1d-MACD_LINE-M>S-', '1d-MACD_LINE_V2-M>S-'),
        ('confluence', '5m-UTBOT-BEAR', '5m-UT_BOT_V4-BEAR_TREND'),
    ],
    64: [
        ('entry', 'utv4_default_bull_flip'),
        ('exit', 'macd_line_default_cross_bear', 'macd_line_v2_default_cross_bear'),
        ('confluence', '1d-MACD_LINE-M>S-', '1d-MACD_LINE_V2-M>S-'),
        ('confluence', '1M-MACD_LINE-M<S-', '1M-MACD_LINE_V2-M<S-'),
    ],
    # sid 66 is in Tier B above
    71: [
        ('entry', 'utv4_default_bull_flip'),
    ],
    114: [
        # ⚠ behaviour-change risk: utbot_v2 _ib has no v4 equivalent. v4 L1
        # semantic via _prev columns is the closest match but not identical.
        ('entry', 'utv4_default_bull_flip'),
        # Exit (swing_123_default_bear_c3) is already user-pack — no change.
    ],
    117: [
        ('entry', 'utv4_default_bull_flip'),
    ],
    131: [
        ('entry', 'utv4_default_bull_flip'),
        # Exit (swing_123_default_bear_c2) is already user-pack — no change.
    ],
}


# ============================================================================
# Apply swaps
# ============================================================================

def apply_swaps(cfg: dict, swaps: list[tuple]) -> tuple[dict, list[str]]:
    """Apply the swap operations to a config dict. Returns (new_cfg, log)."""
    out = copy.deepcopy(cfg)
    log: list[str] = []

    for op in swaps:
        kind = op[0]

        if kind == 'entry':
            new_trig = op[1]
            old = out.get('entry_trigger_confluence_id')
            if old is None:
                log.append(f"  ⚠ entry_swap requested but no entry_trigger_confluence_id on strategy")
                continue
            out['entry_trigger_confluence_id'] = new_trig
            log.append(f"  entry: {old!r} → {new_trig!r}")

        elif kind == 'exit':
            old_trig, new_trig = op[1], op[2]
            exits = out.get('exit_trigger_confluence_ids') or []
            if isinstance(exits, str):
                try:
                    exits = json.loads(exits)
                except Exception:
                    exits = []
            new_exits = []
            swapped = False
            for e in exits:
                if e == old_trig:
                    new_exits.append(new_trig)
                    swapped = True
                else:
                    new_exits.append(e)
            if swapped:
                out['exit_trigger_confluence_ids'] = new_exits
                log.append(f"  exit:  {old_trig!r} → {new_trig!r}")
            else:
                log.append(f"  ⚠ exit_swap target {old_trig!r} not found in exits {exits}")

        elif kind == 'confluence':
            old_rec, new_rec = op[1], op[2]
            confs = out.get('confluence') or []
            if isinstance(confs, str):
                try:
                    confs = json.loads(confs)
                except Exception:
                    confs = []
            new_confs = []
            swapped = False
            for r in confs:
                if r == old_rec:
                    new_confs.append(new_rec)
                    swapped = True
                else:
                    new_confs.append(r)
            if swapped:
                out['confluence'] = new_confs
                log.append(f"  confl: {old_rec!r} → {new_rec!r}")
            else:
                log.append(f"  ⚠ confluence_swap target {old_rec!r} not found in {confs}")

        else:
            log.append(f"  ⚠ unknown op kind: {op!r}")

    return out, log


# ============================================================================
# Migrate one strategy
# ============================================================================

def migrate_one(sid: int, dry_run: bool, run_backtest: bool) -> dict:
    original = get_strategy_by_id_db(sid)
    if original is None:
        return {'sid': sid, 'status': 'NOT_FOUND'}

    swaps = SWAP_PLAN.get(sid)
    if not swaps:
        return {'sid': sid, 'status': 'NO_PLAN'}

    new_cfg, log = apply_swaps(original, swaps)
    info = {
        'sid': sid,
        'name': original.get('name'),
        'symbol': original.get('symbol'),
        'timeframe': original.get('timeframe'),
        'log': log,
    }

    if dry_run:
        info['status'] = 'DRY_RUN'
        return info

    duplicate = copy.deepcopy(new_cfg)
    for k in ('id', 'user_id', 'created_at', 'updated_at',
              'data_refreshed_at', 'kpis_computed_at', 'kpis_stale_since',
              'discrepancies', 'discrepancies_dismissed_at',
              'live_executions', 'equity_curve_data',
              'alert_tracking_reset_at', 'parity_status'):
        duplicate.pop(k, None)
    duplicate['stored_trades'] = []
    duplicate['kpis'] = {}
    duplicate['data_source'] = 'rapid'
    duplicate['name'] = (original.get('name') or '') + f' [mirror {sid}]'
    duplicate['strategy_origin'] = 'migration'

    saved = save_strategy_db(duplicate)
    new_sid = saved.get('id')
    info['new_sid'] = new_sid

    if not run_backtest:
        info['status'] = 'CREATED_NO_BACKTEST'
        return info

    # Run backtest on the new mirror to populate stored_trades
    from api.services.forward_test_service import recompute_and_persist_stored_trades
    try:
        result = recompute_and_persist_stored_trades(new_sid, USER, compute_parity=False)
        info['backtest_status'] = result.get('status')
        info['backtest_trades'] = result.get('trades')
        info['status'] = 'CREATED_AND_BACKTESTED'
    except Exception as e:
        info['backtest_status'] = 'ERROR'
        info['backtest_error'] = f"{type(e).__name__}: {e}"
        info['status'] = 'BACKTEST_FAILED'

    return info


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true',
                    help='Actually create the mirror strategies (default: dry run)')
    ap.add_argument('--backtest', action='store_true',
                    help='Also run backtest on each new mirror to populate stored_trades')
    ap.add_argument('--sids', type=str, default='',
                    help='Comma-separated subset of sids (default: all in SWAP_PLAN)')
    args = ap.parse_args()

    set_admin_user_context(USER)

    if args.sids:
        sids = sorted({int(s.strip()) for s in args.sids.split(',') if s.strip()})
    else:
        sids = sorted(SWAP_PLAN.keys())

    print(f"\n=== Tier B+C migration ({'EXECUTE' if args.execute else 'DRY RUN'},"
          f" backtest={'ON' if args.backtest else 'OFF'}) ===\n")
    print(f"Strategies to migrate: {sids}\n")

    results = []
    for sid in sids:
        print(f"sid {sid}:")
        info = migrate_one(sid, dry_run=not args.execute,
                           run_backtest=args.backtest)
        for line in info.get('log', []):
            print(line)
        if info.get('status') == 'CREATED_NO_BACKTEST':
            print(f"  ✓ created mirror sid {info['new_sid']}")
        elif info.get('status') == 'CREATED_AND_BACKTESTED':
            print(f"  ✓ created mirror sid {info['new_sid']}, "
                  f"backtest={info.get('backtest_status')} "
                  f"trades={info.get('backtest_trades', 0)}")
        elif info.get('status') == 'BACKTEST_FAILED':
            print(f"  ✓ created mirror sid {info['new_sid']} "
                  f"but backtest failed: {info.get('backtest_error')}")
        elif info.get('status') == 'NO_PLAN':
            print(f"  (no swap plan for this sid)")
        elif info.get('status') == 'NOT_FOUND':
            print(f"  ✗ not found")
        elif info.get('status') == 'DRY_RUN':
            print(f"  (dry run)")
        results.append(info)
        print()

    # Summary
    n_created = sum(1 for r in results if r.get('new_sid'))
    n_backtested = sum(1 for r in results
                      if r.get('status') == 'CREATED_AND_BACKTESTED')
    n_failed = sum(1 for r in results if r.get('status') == 'BACKTEST_FAILED')
    print(f"\n=== Summary ===")
    print(f"  Mirrors {'planned' if not args.execute else 'created'}: "
          f"{len(sids) if not args.execute else n_created}")
    if args.execute and args.backtest:
        print(f"  Backtested: {n_backtested}")
        print(f"  Backtest failures: {n_failed}")

    if args.execute:
        print(f"\nNew mirror sids:")
        for r in results:
            if r.get('new_sid'):
                trades = r.get('backtest_trades', '?')
                print(f"  sid {r['sid']:>4} → sid {r['new_sid']:<4}  "
                      f"({r.get('symbol')}/{r.get('timeframe')})  "
                      f"trades={trades}")


if __name__ == '__main__':
    main()
