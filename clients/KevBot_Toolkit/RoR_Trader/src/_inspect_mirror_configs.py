"""Quick inspector: print entry trigger, exit triggers, and confluence
records for the 10 Tier B/C mirrors so we can correlate verdicts with
which packs are actually in play."""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import json
from db import get_admin_client, set_admin_user_context

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
SIDS = [145, 146, 147, 148, 149, 150, 151, 152, 153, 154]

set_admin_user_context(USER)
c = get_admin_client()
r = c.table('strategies').select('id,name,symbol,timeframe,config,parity_status').in_('id', SIDS).execute()

for row in sorted((r.data or []), key=lambda x: x['id']):
    cfg = row.get('config') or {}
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    ps = row.get('parity_status') or {}
    if isinstance(ps, str):
        ps = json.loads(ps)
    verdict = ps.get('verdict', '?')
    score = ps.get('score')
    score_str = f'{score:.2f}' if isinstance(score, (int, float)) else '-'

    entry = cfg.get('entry_trigger_confluence_id') or row.get('entry_trigger_confluence_id') or '?'
    exits = cfg.get('exit_trigger_confluence_ids') or []
    confs = cfg.get('confluence') or []

    print(f"\nsid {row['id']} ({row['symbol']}/{row['timeframe']}): {row['name']}")
    print(f"  verdict: {verdict:<22}  score={score_str}")
    print(f"  entry:   {entry}")
    print(f"  exits:   {exits}")
    print(f"  conf:    {confs}")
