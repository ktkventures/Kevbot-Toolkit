"""Re-queue parity replays for strategies stuck at parity_status.verdict='PENDING'.

When a Railway redeploy happens mid-bulk-parity, the daemon bg threads
get killed before they reach the semaphore. The DB row stays PENDING
forever. This script identifies those orphan rows and re-queues them.

Run from src/:
    ../.venv/bin/python _kick_stuck_parities.py [--apply] [--sids 135,136]
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import sys
import time

import pack_registry
pack_registry.scan_and_load_all()

from db import set_admin_user_context, get_admin_client
from api.services.forward_test_service import queue_parity_for_strategy

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
APPLY = '--apply' in sys.argv

# Optional --sids override
SIDS_ARG = None
for i, a in enumerate(sys.argv):
    if a == '--sids' and i + 1 < len(sys.argv):
        SIDS_ARG = [int(s.strip()) for s in sys.argv[i + 1].split(',') if s.strip()]
        break

set_admin_user_context(USER)
client = get_admin_client()

print(f"\n=== Kick stuck parities ({'APPLY' if APPLY else 'DRY RUN'}) ===\n")

# Find PENDING rows
if SIDS_ARG:
    r = client.table('strategies').select('id,name,parity_status').in_('id', SIDS_ARG).execute()
else:
    r = client.table('strategies').select('id,name,parity_status').eq('user_id', USER).execute()

stuck = []
for row in (r.data or []):
    ps = row.get('parity_status') or {}
    if isinstance(ps, str):
        try:
            ps = json.loads(ps)
        except Exception:
            ps = {}
    if ps.get('verdict') == 'PENDING':
        stuck.append(row)

print(f"Found {len(stuck)} strategies stuck at PENDING:")
for row in sorted(stuck, key=lambda x: x['id']):
    ps = row.get('parity_status') or {}
    if isinstance(ps, str):
        ps = json.loads(ps)
    when = (ps.get('computed_at') or '')[:19].replace('T', ' ')
    print(f"  sid {row['id']:>4}  {row['name'][:50]:<50}  PENDING since {when}")

if not stuck:
    print("\nNothing to kick.")
    sys.exit(0)

if not APPLY:
    print(f"\n(Dry run — pass --apply to re-queue {len(stuck)} parity threads.)")
    sys.exit(0)

print(f"\nRe-queueing {len(stuck)} strategies via queue_parity_for_strategy...")
for row in sorted(stuck, key=lambda x: x['id']):
    sid = row['id']
    try:
        result = queue_parity_for_strategy(sid, USER)
        print(f"  sid {sid}: {result.get('status', '?')}")
    except Exception as e:
        print(f"  sid {sid}: ERROR — {type(e).__name__}: {e}")

print("\nQueued. Threads run in this Python process — wait for them to finish "
      "(don't Ctrl-C until verdicts settle in the DB).")
print("\nWaiting up to 20 minutes for verdicts to settle...")

# Give the daemon threads time to do their work in this process.
# We're in a long-lived Python process here, not the API. The threads
# acquire the LOCAL Semaphore(1), so they run serial. ~30-200s each.
import threading
deadline = time.monotonic() + 1200  # 20 min cap
while time.monotonic() < deadline:
    # Count active non-main threads
    active = [t for t in threading.enumerate() if t.name.startswith('parity-bg-')]
    if not active:
        print("  All bg parity threads finished.")
        break
    print(f"  {len(active)} threads still running: "
          f"{[t.name for t in active]}")
    time.sleep(15)
else:
    print("  TIMEOUT — some threads still running after 20 min.")

# Final verification
print("\nFinal verdicts:")
r = client.table('strategies').select('id,parity_status').in_(
    'id', [row['id'] for row in stuck]).execute()
for row in sorted((r.data or []), key=lambda x: x['id']):
    ps = row.get('parity_status') or {}
    if isinstance(ps, str):
        ps = json.loads(ps)
    print(f"  sid {row['id']}: verdict={ps.get('verdict', '?')} "
          f"score={ps.get('score', '-')} "
          f"matched={ps.get('matched_count', '-')}/{ps.get('stored_count', '-')}")

print("\nDone.")
