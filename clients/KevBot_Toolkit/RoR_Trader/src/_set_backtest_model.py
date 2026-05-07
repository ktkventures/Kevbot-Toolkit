"""Bulk-set strategy.config.backtest_model on a list of strategies.

Per-strategy: read full config JSONB, mutate only the backtest_model
key, write back via .update({'config': new_cfg}). Preserves every
other key in config (per feedback_jsonb_partial_updates.md).

Usage:
    ../.venv/bin/python _set_backtest_model.py <model_id> <sid1> [sid2] ...

Examples (Phase E preview canary staging, 2026-05-06):
    ../.venv/bin/python _set_backtest_model.py cache_locked 152 88
    ../.venv/bin/python _set_backtest_model.py rest_only 152 88   # revert

Valid model_ids per src/strategy_models.py BACKTEST_MODELS registry.
"""
from dotenv import load_dotenv
load_dotenv(override=True)

import sys
from db import get_admin_client
from strategy_models import is_valid_backtest_model


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    target = sys.argv[1]
    sids = [int(x) for x in sys.argv[2:]]

    if target.upper() == 'NULL':
        target_value = None
    else:
        if not is_valid_backtest_model(target):
            sys.exit(f"Unknown backtest_model: {target}")
        target_value = target

    c = get_admin_client()
    target_label = target_value if target_value is not None else '<null>'

    for sid in sids:
        r = c.table('strategies').select('id,user_id,name,config').eq('id', sid).single().execute()
        s = r.data
        if not s:
            print(f"sid={sid}: not found, skipping")
            continue
        cfg = dict(s.get('config') or {})
        before = cfg.get('backtest_model')
        if before == target_value:
            print(f"sid={sid} {s['name'][:30]}: already at {target_label}, skipping")
            continue
        if target_value is None:
            cfg.pop('backtest_model', None)
        else:
            cfg['backtest_model'] = target_value
        c.table('strategies').update({'config': cfg}).eq('id', sid).eq('user_id', s['user_id']).execute()
        print(f"sid={sid} {s['name'][:30]}: backtest_model {before} -> {target_label}")

    print(f"\nDone. Set {len(sids)} strategies to backtest_model={target_label}")


if __name__ == "__main__":
    main()
