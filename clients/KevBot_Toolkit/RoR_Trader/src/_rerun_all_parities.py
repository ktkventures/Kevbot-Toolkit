"""Force-rerun parity for all 10 Tier-A mirrors after the live-dispatch
fix (`f68b410`). Runs locally so Railway redeploys don't kill threads.

Run from src/:
    ../.venv/bin/python _rerun_all_parities.py
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import threading
import time as _time

import pack_registry
pack_registry.scan_and_load_all()

from db import set_admin_user_context
from api.services.forward_test_service import queue_parity_for_strategy

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"

# Optional --sids override (e.g. --sids 139,140,141)
import sys as _sys
SIDS = list(range(135, 145))  # 135-144 default
for _i, _a in enumerate(_sys.argv):
    if _a == '--sids' and _i + 1 < len(_sys.argv):
        SIDS = [int(s.strip()) for s in _sys.argv[_i + 1].split(',') if s.strip()]
        break

set_admin_user_context(USER)

print(f"\nRe-queueing parity for {len(SIDS)} mirrors...\n")
for sid in SIDS:
    r = queue_parity_for_strategy(sid, USER)
    print(f"  sid {sid}: {r.get('status')}")

# Wait for all bg threads to finish
print("\nWaiting for bg parity threads to complete...")
deadline = _time.monotonic() + 7200  # 2 hour ceiling — sub-minute strats with 180-day windows are slow
while _time.monotonic() < deadline:
    active = [t for t in threading.enumerate() if t.name.startswith('parity-bg-')]
    if not active:
        print("  All threads finished.")
        break
    print(f"  {len(active)} active: {[t.name for t in active]}")
    _time.sleep(15)

# Pull final verdicts
from db import get_admin_client
import json
client = get_admin_client()
r = client.table('strategies').select('id,name,parity_status').in_('id', SIDS).execute()

print(f"\n{'sid':>4}  {'name':<48}  {'verdict':<22}  {'score':>6}  matched/stored  replay_only")
print('-' * 130)
for row in sorted((r.data or []), key=lambda x: x['id']):
    ps = row.get('parity_status') or {}
    if isinstance(ps, str):
        ps = json.loads(ps)
    verdict = ps.get('verdict', '?')
    score = ps.get('score')
    score_str = f'{score:.2f}' if isinstance(score, (int, float)) else '-'
    m = ps.get('matched_count', '?')
    s = ps.get('stored_count', '?')
    ro = ps.get('replay_only_count', '?')
    print(f"{row['id']:>4}  {row['name'][:48]:<48}  {verdict:<22}  {score_str:>6}  {m:>4}/{s:<4}      {ro}")
