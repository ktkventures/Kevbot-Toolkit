"""Bulk-set strategy.config.live_model on every strategy.

Per-strategy: read full config JSONB, mutate only the live_model key,
write back via .update({'config': new_cfg}).  Preserves every other
key in config (per feedback_jsonb_partial_updates.md).

Usage:
    ../.venv/bin/python _bulk_set_live_model.py <model_id>
    e.g. ../.venv/bin/python _bulk_set_live_model.py ws_agg_locked

Reverts:
    ../.venv/bin/python _bulk_set_live_model.py ws_with_corrections
or to nullify (fall back to default):
    ../.venv/bin/python _bulk_set_live_model.py NULL
"""
from dotenv import load_dotenv
load_dotenv(override=True)

import sys
import time
from db import get_admin_client
from strategy_models import is_valid_live_model


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    target = sys.argv[1]

    if target.upper() == 'NULL':
        target_value = None
    else:
        if not is_valid_live_model(target):
            sys.exit(f"Unknown live_model: {target}")
        target_value = target

    c = get_admin_client()
    rows = None
    last_err = None
    for attempt in range(8):
        try:
            r = c.table('strategies').select('id,name,config').execute()
            rows = r.data or []
            break
        except Exception as e:
            last_err = e
            wait_s = min(60, 5 * (attempt + 1))
            print(f"[fetch attempt {attempt+1}] {str(e)[:100]} — retrying in {wait_s}s")
            time.sleep(wait_s)
    if rows is None:
        sys.exit(f"Fetch failed after retries: {last_err}")

    print(f"Loaded {len(rows)} strategies")
    target_label = target_value if target_value is not None else '<null>'
    confirmed = 0
    skipped = 0
    failed = []

    for row in rows:
        sid = row['id']
        cfg = dict(row.get('config') or {})
        before = cfg.get('live_model')
        if before == target_value:
            skipped += 1
            continue
        if target_value is None:
            cfg.pop('live_model', None)
        else:
            cfg['live_model'] = target_value
        try:
            for attempt in range(5):
                try:
                    c.table('strategies').update({'config': cfg}).eq('id', sid).execute()
                    confirmed += 1
                    break
                except Exception as e:
                    last_err = e
                    wait_s = min(30, 3 * (attempt + 1))
                    print(f"  strat {sid} [attempt {attempt+1}] {str(e)[:80]} — retrying in {wait_s}s")
                    time.sleep(wait_s)
            else:
                failed.append((sid, str(last_err)[:120]))
        except Exception as e:
            failed.append((sid, str(e)[:120]))

    print()
    print(f"Updated:   {confirmed}")
    print(f"Skipped:   {skipped} (already at target)")
    print(f"Failed:    {len(failed)}")
    if failed:
        for sid, err in failed[:10]:
            print(f"  {sid}: {err}")
        sys.exit(1)
    print(f"All strategies now have config.live_model = {target_label}")


if __name__ == "__main__":
    main()
