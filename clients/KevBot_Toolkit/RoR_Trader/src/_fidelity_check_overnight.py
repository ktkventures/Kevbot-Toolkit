"""Real-world fidelity check (with near-bar tolerance).

For each strategy that fired alerts overnight, compare the alert
timestamps to what the current backtest engine would have produced on
the same bars. Counts:
  - exact: alert minute exactly equals a stored entry minute
  - near:  alert minute within ±1 bar period of a stored entry
  - miss:  no stored entry within ±1 bar

The ±1 bar tolerance handles legitimate timing edges (close→next-bar
fill, bar_close vs bar_start anchoring, sub-second worker jitter). A
near match is a "trade-level match" per Kevin's framing — same trigger
fire, same general bar, just slightly different timestamp anchor.

Step 1: query alerts for the last ~18 hours grouped by strategy_id.
Step 2: pull each strategy's stored_trades.
Step 3: per alert, search stored entries within ±1 bar period. Tag
        as exact/near/miss.
Step 4: compute exact-rate, total-rate (exact+near), and dump per-
        strategy summary.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)

from datetime import datetime, timedelta, timezone
import json

from db import get_admin_client, set_admin_user_context

# How many bar periods of tolerance count as a "near" match
_NEAR_BAR_TOLERANCE = 1

_TF_TO_SECONDS = {
    '1Sec': 1, '5Sec': 5, '10Sec': 10, '15Sec': 15, '30Sec': 30,
    '1Min': 60, '2Min': 120, '3Min': 180, '5Min': 300,
    '10Min': 600, '15Min': 900, '30Min': 1800,
    '1Hour': 3600, '2Hour': 7200, '4Hour': 14400, '1Day': 86400,
}

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
print(f"{'sid':>4}  {'name':<40}  {'sym/tf':<11}  {'alerts':>7}  "
      f"{'exact':>6}  {'near':>5}  {'miss':>5}  {'fire-rate':>10}  notes")
print('-' * 145)

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
    tf_seconds = _TF_TO_SECONDS.get(tf, 60)
    tolerance_sec = tf_seconds * _NEAR_BAR_TOLERANCE

    # Fetch the strategy's current stored_trades (post-fix backtest)
    full_s = c.table('strategies').select('stored_trades').eq('id', sid).single().execute()
    st = (full_s.data or {}).get('stored_trades') or []
    if isinstance(st, str):
        st = json.loads(st)

    # Build a sorted list of stored entry datetimes for fast bisect lookup
    stored_dts: list[datetime] = []
    for t in st:
        ts_field = t.get('entry_fill_ts') or t.get('entry_time')
        if not ts_field:
            continue
        try:
            dt = datetime.fromisoformat(str(ts_field).replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            stored_dts.append(dt)
        except Exception:
            pass
    stored_dts.sort()

    import bisect

    def classify(alert_dt: datetime) -> str:
        """Return 'exact', 'near', or 'miss'."""
        if not stored_dts:
            return 'miss'
        # Find nearest stored entry
        idx = bisect.bisect_left(stored_dts, alert_dt)
        candidates = []
        if idx < len(stored_dts):
            candidates.append(stored_dts[idx])
        if idx > 0:
            candidates.append(stored_dts[idx - 1])
        nearest = min(candidates, key=lambda d: abs((d - alert_dt).total_seconds()))
        delta = abs((nearest - alert_dt).total_seconds())
        if delta == 0:
            return 'exact'
        if delta <= tolerance_sec:
            return 'near'
        return 'miss'

    exact = near = miss = 0
    n_alerts = 0
    alert_window_start = alert_window_end = None
    for a in alerts_for:
        ts_field = a.get('fill_ts') or a.get('timestamp') or a.get('bar_time')
        if not ts_field:
            continue
        try:
            dt = datetime.fromisoformat(str(ts_field).replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            n_alerts += 1
            cls = classify(dt)
            if cls == 'exact':
                exact += 1
            elif cls == 'near':
                near += 1
            else:
                miss += 1
            if alert_window_start is None or dt < alert_window_start:
                alert_window_start = dt
            if alert_window_end is None or dt > alert_window_end:
                alert_window_end = dt
        except Exception:
            pass

    fire_rate = (exact + near) / n_alerts if n_alerts > 0 else 0.0

    ps = s.get('parity_status') or {}
    if isinstance(ps, str):
        ps = json.loads(ps)
    parity_score = ps.get('score')
    parity_str = (f'parity={parity_score:.2f}'
                  if isinstance(parity_score, (int, float)) else 'parity=?')
    notes = parity_str + (f"  tol=±{tolerance_sec}s" if n_alerts > 0 else '')

    print(f"{sid:>4}  {name:<40}  {sym}/{tf:<7}  "
          f"{n_alerts:>7}  {exact:>6}  {near:>5}  {miss:>5}  "
          f"{fire_rate:>10.2f}  {notes}")
    results.append({
        'sid': sid, 'n_alerts': n_alerts,
        'exact': exact, 'near': near, 'miss': miss,
        'fire_rate': fire_rate, 'parity_score': parity_score,
    })

print()
total_alerts = sum(r['n_alerts'] for r in results)
total_exact = sum(r['exact'] for r in results)
total_near = sum(r['near'] for r in results)
total_miss = sum(r['miss'] for r in results)
overall_rate = (total_exact + total_near) / total_alerts if total_alerts > 0 else 0
print(f"OVERALL: {total_alerts} alerts → {total_exact} exact + "
      f"{total_near} near + {total_miss} miss  →  fire-rate {overall_rate:.2f}")
print(f"  (fire-rate = (exact + near) / total — Q1 'will it fire' metric, "
      f"±{_NEAR_BAR_TOLERANCE} bar tolerance)")
