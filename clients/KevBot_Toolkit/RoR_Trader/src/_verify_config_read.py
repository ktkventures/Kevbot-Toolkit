"""READ-ONLY verification that the new config-read path actually fetches
the existing config dict from JSONB (not the flattened strat shape).

Compares:
  - get_strategy_by_id_admin(...).get('config') → flattened, returns None
  - Direct admin client SELECT config → returns the actual JSONB dict

If raw read returns a non-empty dict, the fix is correct: merging into
THIS dict and writing back preserves the existing fields.
"""
from dotenv import load_dotenv
load_dotenv(override=True)

import sys
from db import get_admin_client, get_strategy_by_id_admin


def main():
    sids = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [144, 152, 154]
    c = get_admin_client()
    r = c.table('strategies').select('user_id').limit(1).execute()
    user_id = r.data[0]['user_id']

    for sid in sids:
        print(f'\n=== sid={sid} ===')
        # Method A: admin getter (flattened) — what the old code did
        strat = get_strategy_by_id_admin(sid, user_id)
        flat_cfg = strat.get('config') if strat else None
        print(f'  admin getter strat.get(config) = {flat_cfg!r}')

        # Method B: raw SELECT — what the new code does
        raw = c.table('strategies').select('config') \
            .eq('id', sid).eq('user_id', user_id).single().execute()
        raw_cfg = raw.data.get('config') if raw.data else None
        if raw_cfg:
            print(f'  raw SELECT config keys ({len(raw_cfg)}): '
                  f'{sorted(raw_cfg.keys())[:8]}...')
            print(f'  live_model = {raw_cfg.get("live_model")}')
        else:
            print(f'  raw SELECT config = {raw_cfg!r}')


if __name__ == '__main__':
    main()
