"""Drill: invoke run_strategy_parity directly and dump the FULL
stored_only list with reasons + failing_gates. The persisted
parity_status only keeps summary counts, so we need an in-memory call
to see per-bar diagnostics.

Usage:
    ../.venv/bin/python _drill_parity_full.py SID

Picks one of: 145 (cross-TF MACD daily gate), 153 (utv4 flip drift no
cross-TF), 154 (cross-TF UT_BOT_V4 15m gate).
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import sys
from collections import Counter

import pack_registry
pack_registry.scan_and_load_all()

from db import get_admin_client, set_admin_user_context

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
SID = int(sys.argv[1]) if len(sys.argv) > 1 else 145

set_admin_user_context(USER)
c = get_admin_client()

# Pull strategy + stored_trades
r = c.table('strategies').select('id,name,symbol,timeframe').eq('id', SID).single().execute()
strategy_meta = r.data
print(f"\n=== sid {SID}: {strategy_meta.get('name')} ({strategy_meta.get('symbol')}/{strategy_meta.get('timeframe')}) ===")

from api.services.parity_service import run_strategy_parity
report = run_strategy_parity(SID, last_n=200, forward_test_only=False)

print(f"\nverdict: {report['verdict']}  score: {report.get('parity_score')}")
print(f"stored: {report['stored_count']}  matched: {report['matched_count']}  "
      f"stored_only: {report['stored_only_count']}  replay_only: {report['replay_only_count']}")
print(f"\nmeta: {json.dumps(report.get('meta'), indent=2, default=str)}")

# Reason histogram
reasons = Counter(u.get('reason') for u in report.get('stored_only') or [])
print(f"\nreason breakdown:")
for k, v in reasons.most_common():
    print(f"  {k:<30} {v}")

# Failing gate histogram (only entries with reason=GATE_FAILED)
gates = Counter()
for u in (report.get('stored_only') or []):
    for fg in (u.get('failing_gates') or []):
        gates[(fg.get('required'), fg.get('replay_actual'))] += 1
if gates:
    print(f"\nfailing gates (required → replay_actual, count):")
    for (req, actual), n in gates.most_common(10):
        print(f"  {req!r:<55} → actual={actual!r:<25}  ×{n}")

# Sample of unique TRIGGER_NOT_FIRED entries
not_fired = [u for u in (report.get('stored_only') or []) if u.get('reason') == 'TRIGGER_NOT_FIRED']
if not_fired:
    print(f"\nTRIGGER_NOT_FIRED samples (first 5):")
    for u in not_fired[:5]:
        print(f"  ts={u.get('stored_ts')}  trigger={u.get('trigger')!r}")
        print(f"     triggers actually fired in replay: {u.get('triggers_fired_in_replay') or 'NONE'}")

# Sample of POSITION_BLOCKED entries
pos_blocked = [u for u in (report.get('stored_only') or []) if u.get('reason') == 'POSITION_BLOCKED']
if pos_blocked:
    print(f"\nPOSITION_BLOCKED samples (first 5):")
    for u in pos_blocked[:5]:
        print(f"  ts={u.get('stored_ts')}  trigger={u.get('trigger')!r}  position={u.get('position_state')}")

# Sample of GATE_FAILED entries
gate_failed = [u for u in (report.get('stored_only') or []) if u.get('reason') == 'GATE_FAILED']
if gate_failed:
    print(f"\nGATE_FAILED samples (first 3):")
    for u in gate_failed[:3]:
        print(f"  ts={u.get('stored_ts')}  trigger={u.get('trigger')!r}")
        for fg in (u.get('failing_gates') or [])[:5]:
            print(f"    required: {fg.get('required')!r}  →  replay_actual: {fg.get('replay_actual')!r}")

# replay_only summary
ro = report.get('replay_only') or []
if ro:
    print(f"\nreplay_only samples (first 5) — live fired but no stored trade matched:")
    for f in ro[:5]:
        print(f"  ts={f.get('ts')}  trigger={f.get('trigger')!r}")
