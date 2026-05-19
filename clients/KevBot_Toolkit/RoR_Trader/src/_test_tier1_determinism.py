"""Tier 1 determinism test — parallel vs sequential bulk refresh.

Runs `bulk_refresh_all_strategies(max_workers=1)` and
`bulk_refresh_all_strategies(max_workers=6)` against the SAME starting
strategies.json (USE_DB=false / JSON mode), and asserts the per-strategy
output (stored_trades / kpis / equity_curve_data) is byte-identical.

Usage:
    USE_DB=false PYTHONPATH=src .venv/bin/python src/_test_tier1_determinism.py

Set N_STRATEGIES env to limit how many strategies to test (default 5).
The script pulls real strategies (with stored_trades) from the prod DB
in read-only mode, dumps them to a temp file, runs refresh against the
temp file — production data is NEVER mutated.
"""
from __future__ import annotations
import os, sys, json, copy, shutil
from pathlib import Path
from dotenv import load_dotenv

# Pull source strategies via admin client BEFORE USE_DB is flipped.
# USE_DB env is read at db.py import; we want it true for the pull,
# false for the refresh. So: import db with USE_DB=true, pull, then
# force-flip db.USE_DB to false for the test run.
os.environ['USE_DB'] = 'true'  # for the read-only pull
load_dotenv(Path(__file__).parent / '.env', override=True)
# (`.env` may set USE_DB; the pull needs it true)
os.environ['USE_DB'] = 'true'  # re-assert in case .env overrode

import db
from db import get_admin_client
import pack_registry
pack_registry.scan_and_load_all()

N = int(os.environ.get('N_STRATEGIES', '5'))
print(f"[1] Pulling {N} strategies (with stored_trades) from prod DB...")
sb = get_admin_client()
rows = (sb.table('strategies')
        .select('*')
        .order('id', desc=True).limit(50)
        .execute().data or [])
# pick the first N that have stored_trades + entry_trigger_confluence_id
candidates = []
for r in rows:
    cfg = r.get('config') or {}
    stored = (r.get('stored_trades') or cfg.get('stored_trades') or [])
    has_etc = (
        'entry_trigger_confluence_id' in cfg or
        'entry_trigger_confluence_id' in r or
        cfg.get('strategy_origin') == 'webhook_inbound' or
        r.get('strategy_origin') == 'webhook_inbound'
    )
    if stored and has_etc:
        candidates.append(r)
    if len(candidates) >= N:
        break
print(f"    found {len(candidates)} eligible strategies")
if not candidates:
    sys.exit("no eligible strategies — abort")

# Flatten DB row → strategy dict format the JSON-mode code expects.
# DB row has top-level fields + a `config` JSONB. The JSON-mode code
# expects a flat dict with everything at top level. Merge config in.
def flatten(row):
    d = dict(row)
    cfg = d.pop('config', None) or {}
    for k, v in cfg.items():
        if k not in d:
            d[k] = v
    return d

source_strats = [flatten(r) for r in candidates]
print(f"    pulled: {[s.get('id') for s in source_strats]}")

# Now flip to JSON mode for the rest. We override db.USE_DB directly
# (the env-var-at-import pattern has already cached it).
db.USE_DB = False

# Stash the original repo strategies.json (if any), point app.py at a
# temp file containing our pulled strategies.
import app
ORIG_FILE = app.STRATEGIES_FILE
TMP_FILE = '/tmp/_tier1_determinism_strategies.json'
START_FILE = '/tmp/_tier1_determinism_start.json'

# Save a "before refresh" snapshot for later restore between runs.
with open(START_FILE, 'w') as f:
    json.dump(source_strats, f, indent=2)
print(f"[2] Snapshot saved to {START_FILE}")

app.STRATEGIES_FILE = TMP_FILE

def run_with_workers(n):
    shutil.copy(START_FILE, TMP_FILE)
    res = app.bulk_refresh_all_strategies(max_workers=n)
    with open(TMP_FILE) as f:
        return res, json.load(f)

print(f"[3] Sequential run (max_workers=1)...")
res_seq, out_seq = run_with_workers(1)
print(f"    {res_seq}")

print(f"[4] Parallel run (max_workers=6)...")
res_par, out_par = run_with_workers(6)
print(f"    {res_par}")

# Diff per-strategy, excluding wall-clock timestamps.
# `data_refreshed_at` and `detected_at` (inside discrepancies/
# live_executions) are wall-clock and naturally differ between runs.
print(f"[5] Diffing per-strategy output "
      f"(excluding data_refreshed_at, detected_at)...")
keys_check = ['stored_trades', 'kpis', 'equity_curve_data',
              'live_executions', 'discrepancies']

def _scrub_wall_clock(obj):
    """Recursively drop wall-clock keys so the diff is content-only."""
    DROP = {'detected_at', 'data_refreshed_at'}
    if isinstance(obj, dict):
        return {k: _scrub_wall_clock(v) for k, v in obj.items()
                if k not in DROP}
    if isinstance(obj, list):
        return [_scrub_wall_clock(v) for v in obj]
    return obj

mismatches = []
for s_seq in out_seq:
    sid = s_seq.get('id')
    s_par = next((s for s in out_par if s.get('id') == sid), None)
    if s_par is None:
        mismatches.append((sid, 'missing in parallel output'))
        continue
    for k in keys_check:
        v_seq = _scrub_wall_clock(s_seq.get(k))
        v_par = _scrub_wall_clock(s_par.get(k))
        if json.dumps(v_seq, sort_keys=True, default=str) != \
           json.dumps(v_par, sort_keys=True, default=str):
            mismatches.append((sid, f'field {k!r} differs'))

# Compare failed_ids as SETS — parallel completes in non-deterministic
# order via `as_completed`, so list order can legitimately differ.
ok = (res_seq['success_count'] == res_par['success_count']
      and set(res_seq['failed_ids']) == set(res_par['failed_ids'])
      and not mismatches)

if ok:
    print(f"\nPASS — sequential and parallel produced byte-identical "
          f"stored_trades/kpis/equity_curve_data/live_executions/"
          f"discrepancies for all {len(out_seq)} strategies.")
else:
    print(f"\nFAIL — mismatches:")
    for sid, why in mismatches[:20]:
        print(f"  sid={sid}: {why}")
    print(f"  seq result: {res_seq}")
    print(f"  par result: {res_par}")

# Cleanup
for p in (TMP_FILE, START_FILE):
    try: os.remove(p)
    except FileNotFoundError: pass
app.STRATEGIES_FILE = ORIG_FILE

sys.exit(0 if ok else 1)
