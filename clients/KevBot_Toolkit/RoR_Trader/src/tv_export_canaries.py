"""TV-export parity canaries — the single source of truth for which Pine emitters
have been VALIDATED against TradingView trade-for-trade, not merely "emit without
error". See docs/_active/Design_TV_Export_Readiness.md §5.3.

Why this exists: tv_export_readiness() answers "can this strategy emit Pine?".
That is necessary but NOT sufficient — the EOD re-entry churn emitted fine and was
still a divergence. A "✅ TV-ready" badge should mean "validated", and this registry
is what distinguishes validated from merely-emitting. Update it whenever a new
pack / stop method / time-exit method passes a trade-for-trade parity check
(the sid-308 method: TV List-of-Trades vs the RoR backtest via _tv_parity_compare).

Component keys are "<kind>:<name>":
  kind ∈ {trigger, gate, stop, time_exit}
  name  = emitter slug (trigger/gate) or method name (stop/time_exit)

status ∈ {validated, pending}
  validated — proven trade-for-trade against TV on the named canary strategy.
  pending   — emits + passes the offline audit, but no TV parity run yet.
"""

# component key -> {status, sid, tf, date, result, note}
CANARIES = {
    # ── pack triggers ──
    'trigger:ut_bot_v4': {
        'status': 'validated', 'sid': 269, 'tf': '1Min', 'date': '2026-06-26',
        'result': 'trade-for-trade match where data overlaps',
        'note': 'SPY UT Bot primary entry/exit, no gates'},
    'trigger:swing_123': {
        'status': 'validated', 'sid': 308, 'tf': '15Sec', 'date': '2026-06-26',
        'result': '178/178 entries; bull_c2 entry vs RoR 15Sec to the cent',
        'note': 'TSLA, cross-pack (swing entry / ut_bot exit)'},
    'trigger:ema_pp_v3': {
        'status': 'pending', 'sid': 306, 'tf': '10Sec', 'date': None,
        'result': None, 'note': 'emitter shipped (Fix C); TV parity not run yet'},
    'trigger:macd_line_v2': {
        'status': 'pending', 'sid': 136, 'tf': '1Min', 'date': None,
        'result': None, 'note': 'emitter shipped (Fix C); TV parity not run yet'},

    # ── stop methods ──
    'stop:swing': {
        'status': 'validated', 'sid': 308, 'tf': '15Sec', 'date': '2026-06-26',
        'result': 'stop level + intrabar L-type fill byte-exact (3/3 earlier; '
                  '7/8 fills == stored stop, 1 legit gap-through)',
        'note': 'lowest-low lookback − padding, swLow[1]'},
    'stop:atr': {
        'status': 'validated', 'sid': 303, 'tf': '10Sec', 'date': '2026-06',
        'result': 'byte-identical port; historical RTH parity 98.6%',
        'note': 'rorEma(rorTrueRange)'},

    # ── time exits ──
    'time_exit:eod_exit': {
        'status': 'validated', 'sid': 308, 'tf': '15Sec', 'date': '2026-06-26',
        'result': 'EOD exits matched RoR to the cent (incl. re-entry churn, '
                  'which is faithful — engine does the same)',
        'note': '15:55 ET flat-by; fills at bar close'},
    'time_exit:max_hold_bars': {
        'status': 'pending', 'sid': 136, 'tf': '1Min', 'date': None,
        'result': None,
        'note': 'emitter shipped (Fix C); barsHeld off-by-one unconfirmed on TV'},
}


def is_validated(kind: str, name: str) -> bool:
    """True iff this component has passed a trade-for-trade TV parity check."""
    rec = CANARIES.get(f'{kind}:{name}')
    return bool(rec and rec.get('status') == 'validated')


def canary_for(kind: str, name: str):
    """Return the canary record for a component, or None if untracked."""
    return CANARIES.get(f'{kind}:{name}')


def coverage_summary() -> dict:
    """Counts by status — for a dashboard / at-a-glance health line."""
    validated = sum(1 for r in CANARIES.values() if r.get('status') == 'validated')
    pending = sum(1 for r in CANARIES.values() if r.get('status') == 'pending')
    return {'validated': validated, 'pending': pending, 'total': len(CANARIES)}
