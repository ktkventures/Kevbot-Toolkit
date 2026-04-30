"""Real-world fidelity check.

For each strategy that fired alerts overnight, compare the alert
timestamps to what the current backtest engine would have produced on
the same bars. If parity says a strategy is PASS but its overnight
alerts don't match a fresh backtest, that's a real-world gap we
missed.

Step 1: query alerts for the last ~14 hours grouped by strategy_id.
Step 2: pull each strategy's stored_trades and current configuration.
Step 3: compare alert timestamps to stored_trades entry timestamps
        within the alert window. Compute a per-strategy match rate.
Step 4: report.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

from datetime import datetime, timedelta, timezone
import json

from db import get_admin_client, set_admin_user_context

USER = "19d47e46-f718-49a6-af32-5f5407f5b170"
HOURS_BACK = 18  # cover yesterday's late session through this morning's open

set_admin_user_context(USER)
c = get_admin_client()

since = (datetime.now(timezone.utc) - timedelta(hours=HOURS_BACK)).isoformat()
print(f"\nLooking at alerts from the last {HOURS_BACK} hours (since {since[:19]} UTC)\n")

# Pull overnight alerts. The alerts table has both 'entry' and 'exit'
# event types — we focus on entries for the fidelity check (exits are
# downstream of entries; matching entries gets us 80% of the signal).
r = c.table('alerts').select(
    'id,strategy_id,strategy_name,symbol,timeframe,trigger_id,'
    'event_type,side,timestamp,fill_ts,bar_time,price,exec_type'
).gte('timestamp', since).execute()
alerts = r.data or []

# Group by strategy_id; only side='entry' alerts are entry signals
# (side='exit' alerts are downstream exits that we don't fidelity-check
# directly — they follow from entries). type='exit_signal' often
# co-exists with side='exit' but the 'side' field is the canonical
# entry/exit discriminator post Trade Timestamps Spec.
from collections import defaultdict
by_strat = defaultdict(list)
for a in alerts:
    sid = a.get('strategy_id')
    if (a.get('side') or '').lower() == 'entry':
        by_strat[sid].append(a)

print(f"Total entry alerts: {sum(len(v) for v in by_strat.values())} across {len(by_strat)} strategies\n")

# Pull strategy configs + stored_trades for each
print(f"{'sid':>4}  {'name':<40}  {'sym/tf':<11}  {'alerts':>7}  {'matched':>8}  {'rate':>6}  notes")
print('-' * 130)

results = []
for sid, alerts_for in sorted(by_strat.items()):
    if sid is None:
        continue
    s_resp = c.table('strategies').select(
        'id,name,symbol,timeframe,parity_status'
    ).eq('id', sid).maybe_single().execute()
    if not s_resp or not s_resp.data:
        continue
    s = s_resp.data
    name = (s.get('name') or '?')[:40]
    sym = s.get('symbol', '?')
    tf = s.get('timeframe', '?')

    # Fetch the strategy's current stored_trades (post-fix backtest)
    full_s = c.table('strategies').select('stored_trades').eq('id', sid).single().execute()
    st = (full_s.data or {}).get('stored_trades') or []
    if isinstance(st, str):
        st = json.loads(st)

    # Build a set of stored entry minute keys
    stored_minutes = set()
    for t in st:
        ts_field = t.get('entry_fill_ts') or t.get('entry_time')
        if not ts_field:
            continue
        try:
            dt = datetime.fromisoformat(str(ts_field).replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            stored_minutes.add(dt.astimezone(timezone.utc).isoformat()[:16])
        except Exception:
            pass

    # For each alert, check if its minute is in stored_minutes
    alert_minutes = []
    matched = 0
    alert_window_start = None
    alert_window_end = None
    for a in alerts_for:
        # Prefer fill_ts (post Trade Timestamps Spec); fallback to timestamp
        ts_field = a.get('fill_ts') or a.get('timestamp') or a.get('bar_time')
        if not ts_field:
            continue
        try:
            dt = datetime.fromisoformat(str(ts_field).replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            normalized = dt.astimezone(timezone.utc).isoformat()[:16]
            alert_minutes.append(normalized)
            if normalized in stored_minutes:
                matched += 1
            if alert_window_start is None or dt < alert_window_start:
                alert_window_start = dt
            if alert_window_end is None or dt > alert_window_end:
                alert_window_end = dt
        except Exception:
            pass

    n_alerts = len(alert_minutes)
    rate = matched / n_alerts if n_alerts > 0 else 0.0

    # Parity status for cross-reference
    ps = s.get('parity_status') or {}
    if isinstance(ps, str):
        ps = json.loads(ps)
    parity_score = ps.get('score')
    parity_str = (f'parity={parity_score:.2f}'
                  if isinstance(parity_score, (int, float)) else 'parity=?')

    notes = f"{parity_str}  alerts {alert_window_start} → {alert_window_end}" if n_alerts > 0 else parity_str

    print(f"{sid:>4}  {name:<40}  {sym}/{tf:<7}  "
          f"{n_alerts:>7}  {matched:>8}  {rate:>6.2f}  {notes}")
    results.append({
        'sid': sid, 'name': name, 'symbol': sym, 'timeframe': tf,
        'n_alerts': n_alerts, 'matched': matched, 'rate': rate,
        'parity_score': parity_score,
        'alert_window': (alert_window_start, alert_window_end),
    })

# Summary
print()
total_alerts = sum(r['n_alerts'] for r in results)
total_matched = sum(r['matched'] for r in results)
overall_rate = total_matched / total_alerts if total_alerts > 0 else 0
print(f"OVERALL: {total_matched}/{total_alerts} alerts matched "
      f"a stored backtest entry  →  rate {overall_rate:.2f}")
