"""Drill sid 145 — pull everything parity_service recorded so we can
see what gates failed, on which bars, and how the live shadow engine's
MACD_LINE_V2 1Day classification compares to the batch path.

Sid 145 = AAPL/1Min, all entries gated by `1d-MACD_LINE_V2-M>S-` and
`5m-UT_BOT_V4-BEAR_TREND`. Verdict: FAIL_LIVE_BLOCKED 0/132.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

import json
from db import get_admin_client, set_admin_user_context

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
SID = 145

set_admin_user_context(USER)
c = get_admin_client()

# 1. Strategy + parity_status
r = c.table('strategies').select('*').eq('id', SID).single().execute()
s = r.data or {}
ps = s.get('parity_status') or {}
if isinstance(ps, str):
    ps = json.loads(ps)

print(f"\n{'='*80}\nSTRATEGY: sid {SID} — {s.get('name')}\n{'='*80}")
print(f"  symbol/tf:   {s.get('symbol')}/{s.get('timeframe')}")
cfg = s.get('config') or {}
if isinstance(cfg, str):
    cfg = json.loads(cfg)
print(f"  entry:       {cfg.get('entry_trigger_confluence_id')}")
print(f"  exits:       {cfg.get('exit_trigger_confluence_ids')}")
print(f"  confluence:  {cfg.get('confluence')}")

print(f"\n{'='*80}\nPARITY_STATUS\n{'='*80}")
print(json.dumps({k: v for k, v in ps.items() if k not in ('stored_only', 'replay_only', 'matched')}, indent=2, default=str))

# 2. stored_only entries with failing_gates
print(f"\n{'='*80}\nSTORED_ONLY (first 5) — backtest fired, live did not\n{'='*80}")
stored_only = ps.get('stored_only') or []
for i, e in enumerate(stored_only[:5]):
    print(f"\n  [{i}] {e.get('ts')}  bar_time={e.get('bar_time')}")
    print(f"      entry_fill_ts: {e.get('entry_fill_ts')}")
    print(f"      failing_gates: {e.get('failing_gates')}")
    print(f"      most_common_failing_gate: {ps.get('most_common_failing_gate')}")

# 3. stored_trades sample (the backtest ground truth)
print(f"\n{'='*80}\nSTORED_TRADES (first 3 entries) — backtest's ground truth\n{'='*80}")
t = c.table('trades').select('*').eq('strategy_id', SID).limit(3).execute()
for tr in (t.data or []):
    print(f"  trade {tr.get('id')}: entry_fill_ts={tr.get('entry_fill_ts')}  entry_price={tr.get('entry_price')}")
    print(f"             exit_fill_ts={tr.get('exit_fill_ts')}    direction={tr.get('direction')}  exec_type={tr.get('exec_type')}")
