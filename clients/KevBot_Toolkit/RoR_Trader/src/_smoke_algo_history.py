"""Smoke-test the new append_new_trades_for_strategy helper.

WARNING: this script writes to PRODUCTION DB.  There is no dev/staging
Supabase — db.get_admin_client() returns the live service-role client.
A successful run will:
  - Insert any closed trades the engine produces that aren't already in
    the trades table (by (entry_fill_ts, exit_fill_ts) tuple — idempotent).
  - Update kpis / equity_curve_data / data_refreshed_at on the strategy.
  - Stamp config.last_recompute_until_ts.

Confirm with the user before running on a strategy you don't want to
mutate. Pre-flight gate below requires --i-understand-this-mutates-prod
to actually do work; without it the script does a dry inspection.
"""
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import sys
from db import get_admin_client


def _pre_flight_check(target_sids):
    c = get_admin_client()
    print(f"=== PRE-FLIGHT (read-only) ===")
    rows = c.table('strategies').select('id,name,config').in_('id', target_sids).execute().data or []
    if not rows:
        print(f"No strategies match {target_sids}")
        return False
    for s in rows:
        cfg = s.get('config') or {}
        print(f"  sid={s['id']:>4} {s['name'][:30]:<30} "
              f"config_keys={len(cfg)} live_model={cfg.get('live_model')}")
    print()
    return True


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    apply = '--i-understand-this-mutates-prod' in args
    target_sids = [int(x) for x in args if x.lstrip('-').isdigit()]
    if not target_sids:
        print('No strategy IDs supplied.')
        sys.exit(1)

    if not _pre_flight_check(target_sids):
        sys.exit(1)

    if not apply:
        print("Dry-run only. Re-run with "
              "'--i-understand-this-mutates-prod' to actually invoke "
              "append_new_trades_for_strategy().")
        sys.exit(0)

    c = get_admin_client()
    r = c.table('strategies').select('user_id').limit(1).execute()
    user_id = r.data[0]['user_id']

    from api.services.forward_test_service import append_new_trades_for_strategy

    for sid in target_sids:
        print(f'\n=== sid={sid} (PROD WRITE) ===')
        try:
            res = append_new_trades_for_strategy(sid, user_id, lag_minutes=15)
        except Exception as e:
            print(f'  EXCEPTION: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
            continue
        out = {k: v for k, v in res.items() if k != 'detail'}
        print(f'  {json.dumps(out, default=str)}')

    # Post-write inspection: did config preserve other fields?
    print()
    print(f"=== POST-WRITE INSPECTION ===")
    rows = c.table('strategies').select('id,name,config').in_('id', target_sids).execute().data or []
    for s in rows:
        cfg = s.get('config') or {}
        print(f"  sid={s['id']:>4} {s['name'][:30]:<30} "
              f"config_keys={len(cfg)} live_model={cfg.get('live_model')} "
              f"last_recompute={cfg.get('last_recompute_until_ts')}")


if __name__ == '__main__':
    main()
