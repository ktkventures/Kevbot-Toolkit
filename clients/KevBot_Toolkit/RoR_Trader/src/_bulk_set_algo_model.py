"""One-time bulk migration: set algo_model='cache_locked' on all
existing strategies; set backtest_model='rest_hifi' (preserve existing
rest_hifi values, flip everyone else).

Run once on Supabase via dev environment. Safe to re-run (idempotent).

2026-05-07 — algo_model split. See:
- Plan file: ~/.claude/plans/synchronous-tickling-yeti.md
- strategy_models.py for the new defaults

Usage:
    python -m _bulk_set_algo_model           # dry-run, prints what would change
    python -m _bulk_set_algo_model --apply   # actually writes
"""
import sys
import json
from dotenv import load_dotenv
load_dotenv(override=True)

from db import get_admin_client


def main():
    apply = '--apply' in sys.argv
    client = get_admin_client()

    rows = client.table('strategies').select('id,name,config').execute()
    strategies = rows.data or []

    print(f"Loaded {len(strategies)} strategies")
    print()

    plan = []
    for s in strategies:
        sid = s['id']
        name = s.get('name', '')[:40]
        cfg = s.get('config') or {}

        old_bt = cfg.get('backtest_model')
        old_algo = cfg.get('algo_model')

        # Backtest target: preserve rest_hifi if already set; flip
        # everyone else (including cache_locked, rest_only) to rest_hifi.
        new_bt = 'rest_hifi' if old_bt != 'rest_hifi' else 'rest_hifi'

        # Algo target: cache_locked for everyone
        new_algo = 'cache_locked'

        bt_change = old_bt != new_bt
        algo_change = old_algo != new_algo

        if bt_change or algo_change:
            plan.append({
                'sid': sid, 'name': name,
                'old_bt': old_bt, 'new_bt': new_bt, 'bt_change': bt_change,
                'old_algo': old_algo, 'new_algo': new_algo, 'algo_change': algo_change,
            })

    if not plan:
        print("Nothing to migrate — all strategies already aligned.")
        return

    print(f"{'sid':>4}  {'name':<40}  {'old_bt':<14} -> {'new_bt':<14}  {'old_algo':<14} -> {'new_algo':<14}")
    print("-" * 130)
    for p in plan:
        print(f"{p['sid']:>4}  {p['name']:<40}  {str(p['old_bt'] or 'None'):<14} -> {p['new_bt']:<14}  {str(p['old_algo'] or 'None'):<14} -> {p['new_algo']:<14}")
    print()
    print(f"Total to update: {len(plan)}")

    if not apply:
        print()
        print("DRY RUN — no changes written. Re-run with --apply to commit.")
        return

    print()
    print("Applying changes...")
    success = 0
    for p in plan:
        sid = p['sid']
        try:
            # Re-read the current config to merge cleanly (don't clobber
            # other fields). Then write back the full config.
            row = client.table('strategies').select('config').eq('id', sid).limit(1).execute()
            current_cfg = (row.data[0].get('config') if row.data else {}) or {}
            current_cfg['backtest_model'] = p['new_bt']
            current_cfg['algo_model'] = p['new_algo']
            client.table('strategies').update({'config': current_cfg}).eq('id', sid).execute()
            print(f"  sid={sid}: OK")
            success += 1
        except Exception as e:
            print(f"  sid={sid}: FAILED — {e}")

    print()
    print(f"Done — {success}/{len(plan)} strategies migrated")


if __name__ == '__main__':
    main()
