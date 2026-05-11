"""One-time backfill: migrate stored_trades JSONB → trades table
with data_source='backtest_<model>'.

Phase 41 — 2026-05-11.

Iterates every strategy. For each:
  1. Reads stored_trades JSONB
  2. Tags each trade with data_source = 'backtest_' + (config.backtest_model or 'rest_hifi')
  3. INSERTs into trades table with ON CONFLICT DO NOTHING for idempotency
  4. Reports per-strategy backfill count

After Phase 41 schema migration has been applied (unique constraint
includes data_source), these inserts won't collide with existing
algo data (NULL → 'cache_locked' via the same migration).

Usage:
    python -m _backfill_stored_trades_to_table           # dry-run
    python -m _backfill_stored_trades_to_table --apply   # actually writes
"""
import sys
from dotenv import load_dotenv
load_dotenv(override=True)

from db import get_admin_client, insert_trade_admin, set_admin_user_context


def main():
    apply = '--apply' in sys.argv
    client = get_admin_client()

    rows = client.table('strategies').select(
        'id,name,user_id,config,stored_trades').execute()
    strategies = rows.data or []

    print(f"Loaded {len(strategies)} strategies")
    print()

    plan = []
    for s in strategies:
        sid = s['id']
        name = s.get('name', '')[:36]
        uid = s['user_id']
        cfg = s.get('config') or {}
        backtest_model = cfg.get('backtest_model') or 'rest_hifi'
        stored = s.get('stored_trades') or []
        if not stored:
            continue
        plan.append({
            'sid': sid, 'name': name, 'user_id': uid,
            'backtest_model': backtest_model,
            'stored_count': len(stored),
            'data_source_tag': f'backtest_{backtest_model}',
            'stored': stored,
        })

    print(f'{"sid":>4}  {"name":<36}  {"trades":>6}  {"data_source":<22}')
    print('-' * 80)
    total_to_insert = 0
    for p in plan:
        print(f'{p["sid"]:>4}  {p["name"]:<36}  {p["stored_count"]:>6}  {p["data_source_tag"]:<22}')
        total_to_insert += p['stored_count']
    print()
    print(f'Total trades to backfill: {total_to_insert}')

    if not apply:
        print()
        print('DRY RUN — no changes. Re-run with --apply to commit.')
        return

    print()
    print('Applying backfill...')
    total_inserted = 0
    total_skipped = 0  # ON CONFLICT
    total_errors = 0
    for p in plan:
        sid = p['sid']
        uid = p['user_id']
        ds_tag = p['data_source_tag']
        set_admin_user_context(uid)
        inserted = 0
        skipped = 0
        errors = 0
        for t in p['stored']:
            # Skip rows missing entry_fill_ts — can't dedupe them
            if not t.get('entry_fill_ts'):
                skipped += 1
                continue
            try:
                trade_record = dict(t)  # copy
                trade_record['data_source'] = ds_tag
                row = insert_trade_admin(sid, uid, trade_record)
                if row is not None:
                    inserted += 1
                else:
                    skipped += 1  # ON CONFLICT
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f'  sid={sid} insert error: {str(e)[:100]}')
        total_inserted += inserted
        total_skipped += skipped
        total_errors += errors
        print(f'  sid={sid:>3}: inserted={inserted:>5} skipped={skipped:>5} errors={errors}')

    print()
    print(f'Done — total inserted: {total_inserted}, skipped: {total_skipped}, errors: {total_errors}')


if __name__ == '__main__':
    main()
