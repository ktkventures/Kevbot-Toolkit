"""Strategy-Detail parity service.

Replays a strategy's stored backtest trades through the live engine
(StrategyMonitor + SymbolHub + ShadowIndicatorEngine, the same code the
worker runs) against historical bars, and diffs the resulting fires
against the strategy's stored_trades. Surfaces parity gaps as concrete
"this stored trade would NOT fire in live, and here's the failing gate"
rows for the Strategy Detail UI.

Why this exists
---------------
The recurring debug pattern is: backtest produces a trade, live worker
doesn't fire the matching alert, and we lose hours figuring out which
gate disagreed. parity_simulator.py covers single-pack V1 (one indicator,
one timeframe). This is the deferred V2: full strategy mode with
cross-TF gate evaluation via real shadow engines.

Output is a ParityReport dict shaped for the frontend Parity tab.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# Default warmup bars per timeframe. Longer warmup = better path-dependent
# indicator convergence (UT_BOT trail, ATR, EMA). These match the worker's
# load_market_data days arg ranges; if a strategy's data_days is smaller
# than the default warmup, we cap it.
_DEFAULT_PRIMARY_WARMUP_BARS = 200
_DEFAULT_SHADOW_WARMUP_BARS = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_ts(ts: Any) -> Optional[str]:
    """Convert any timestamp form to a normalized UTC ISO string."""
    if ts is None:
        return None
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except Exception:
            return ts
    elif isinstance(ts, datetime):
        dt = ts
    elif hasattr(ts, 'isoformat'):
        try:
            dt = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts
        except Exception:
            return str(ts)
    else:
        return str(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _stored_entry_records(stored_trades: list[dict]) -> list[dict]:
    """Extract the entry-side fields we'll diff against."""
    out = []
    for t in stored_trades or []:
        ts_raw = t.get('entry_fill_ts') or t.get('entry_time')
        ts = _normalize_ts(ts_raw)
        if not ts:
            continue
        out.append({
            'ts': ts,
            'trigger': t.get('entry_trigger') or t.get('trigger') or '',
            'price': t.get('entry_price'),
            'trade_id': t.get('id'),
        })
    return out


# ---------------------------------------------------------------------------
# Replay infrastructure
# ---------------------------------------------------------------------------

def _build_monitor_and_hub(strategy: dict, user_id: Optional[str]):
    """Construct StrategyMonitor + SymbolHub the way the worker does.

    Returns (hub, monitor). Shadow engines are NOT yet finalized — caller
    must call hub.finalize_shadow_engines() after add_monitor.
    """
    from ralph_engine import StrategyMonitor, SymbolHub

    # Load this user's enabled general packs so GEN- confluence records
    # evaluate identically to the worker.
    general_packs: list = []
    try:
        from db import load_general_packs_admin
        from general_packs import _parse_pack_list
        if user_id:
            raw = load_general_packs_admin(user_id) or []
            all_packs = _parse_pack_list(raw)
            general_packs = [
                p for p in all_packs if getattr(p, 'enabled', True)]
    except Exception as e:
        logger.warning("[parity] general pack load failed: %s", e)

    sym = strategy.get('symbol', 'SPY')
    hub = SymbolHub(sym, publisher=None)
    monitor = StrategyMonitor(strategy, None, general_packs=general_packs)
    hub.add_monitor(monitor)
    return hub, monitor


def _load_primary_and_secondary_bars(
    strategy: dict, secondary_tfs: set[int], days: int,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Load OHLCV for the primary TF and every required secondary TF.

    Returns (primary_df, {sec_tf_seconds: secondary_df}).
    """
    from data_loader import load_market_data
    from ralph_engine import SECONDS_TO_TIMEFRAME

    sym = strategy.get('symbol', 'SPY')
    tf_label = strategy.get('timeframe', '1Min')
    session = strategy.get('trading_session', 'RTH')

    primary_df = load_market_data(
        symbol=sym, days=days, timeframe=tf_label, session=session)

    sec_dfs: dict[int, pd.DataFrame] = {}
    for sec_tf in secondary_tfs:
        sec_label = SECONDS_TO_TIMEFRAME.get(sec_tf)
        if not sec_label:
            logger.warning("[parity] no label for tf=%ss; skipping", sec_tf)
            continue
        try:
            sec_dfs[sec_tf] = load_market_data(
                symbol=sym, days=days, timeframe=sec_label, session=session)
        except Exception as e:
            logger.warning(
                "[parity] secondary bar load failed for %s/%s: %s",
                sym, sec_label, e)
    return primary_df, sec_dfs


def _replay_strategy(
    strategy: dict, user_id: Optional[str],
) -> tuple[list[dict], list[dict], dict]:
    """Run the strategy through StrategyMonitor + shadow engines on
    historical bars. Returns (replay_fires, replay_bar_states, meta).

    `replay_fires` is the list of entry signals StrategyMonitor produced.
    `replay_bar_states` is per-bar diagnostics (current_confluence,
    triggers fired, position state). Used to attribute "why didn't this
    bar fire" answers in the diff phase.
    """
    from ralph_engine import StrategyMonitor

    hub, monitor = _build_monitor_and_hub(strategy, user_id)
    hub.finalize_shadow_engines()

    # Window the replay to span every stored trade plus a warmup buffer.
    # Strategy.data_days alone often undercounts because stored_trades
    # accumulate across multiple historical backtest runs of varying
    # lengths. We start from the earliest stored trade (or now - data_days,
    # whichever is older) and add ~30 days of warmup buffer.
    cfg_days = int(strategy.get('data_days', 30) or 30)
    earliest_iso = strategy.get('_earliest_stored_ts')
    if earliest_iso:
        try:
            earliest_dt = datetime.fromisoformat(
                earliest_iso.replace('Z', '+00:00'))
            if earliest_dt.tzinfo is None:
                earliest_dt = earliest_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            span_days = (now - earliest_dt).days + 30
            days = max(cfg_days, span_days, 7)
        except Exception:
            days = max(cfg_days, 7)
    else:
        days = max(cfg_days, 7)
    # Cap at 365 to avoid pathological loads
    days = min(days, 365)

    primary_df, sec_dfs = _load_primary_and_secondary_bars(
        strategy, monitor._required_secondary_tf, days)

    if primary_df is None or len(primary_df) < 20:
        return [], [], {
            'error': f'Not enough primary-TF bars loaded: {len(primary_df) if primary_df is not None else 0}',
            'days': days,
        }

    # Warmup: each engine consumes its first N bars to seed indicator state.
    warmup = min(_DEFAULT_PRIMARY_WARMUP_BARS, max(20, len(primary_df) // 4))
    monitor.warmup(primary_df.iloc[:warmup])

    for sec_tf, sec_df in sec_dfs.items():
        shadow = hub._shadow_engines.get(sec_tf)
        if shadow is None or sec_df is None or len(sec_df) < 5:
            continue
        sec_warmup = min(_DEFAULT_SHADOW_WARMUP_BARS,
                         max(5, len(sec_df) // 4))
        try:
            shadow.warmup(sec_df.iloc[:sec_warmup])
        except Exception as e:
            logger.warning(
                "[parity] shadow warmup failed for %ss: %s", sec_tf, e)

    # Track secondary iter positions so we can advance them in lockstep
    # with the primary timestamp. Each shadow.on_bar_close() updates the
    # hub's _mtf_confluence buffer, mirroring on_polygon_bar's fan-out.
    sec_state: dict[int, dict] = {}
    for sec_tf, sec_df in sec_dfs.items():
        if sec_df is None or len(sec_df) < 5:
            continue
        sec_warmup = min(_DEFAULT_SHADOW_WARMUP_BARS,
                         max(5, len(sec_df) // 4))
        sec_state[sec_tf] = {
            'df': sec_df,
            'idx': sec_warmup,  # next bar to play after warmup
        }

    replay_fires: list[dict] = []
    replay_bar_states: list[dict] = []
    bar_count = warmup

    for ts, row in primary_df.iloc[warmup:].iterrows():
        # Advance any secondary TF whose next bar timestamp <= primary ts.
        # Mirrors the production fan-out (on_polygon_bar / on_second_bar).
        for sec_tf, st in sec_state.items():
            shadow = hub._shadow_engines.get(sec_tf)
            if shadow is None:
                continue
            sec_df = st['df']
            while st['idx'] < len(sec_df):
                sec_ts = sec_df.index[st['idx']]
                # Compare as pandas Timestamps (both come from DataFrame index)
                if sec_ts > ts:
                    break
                sec_row = sec_df.iloc[st['idx']]
                sec_bar = {
                    'open': float(sec_row['open']),
                    'high': float(sec_row['high']),
                    'low': float(sec_row['low']),
                    'close': float(sec_row['close']),
                    'volume': float(sec_row.get('volume', 0)),
                    'timestamp': sec_ts,
                }
                try:
                    records = shadow.on_bar_close(sec_bar)
                    hub._mtf_confluence[sec_tf] = records
                except Exception as e:
                    logger.warning(
                        "[parity] shadow bar_close failed (%ss @ %s): %s",
                        sec_tf, sec_ts, e)
                st['idx'] += 1

        # Run primary monitor on this bar with current cross-TF buffer.
        bar = {
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row.get('volume', 0)),
            'timestamp': ts,
        }
        try:
            signals, audit = monitor.on_bar_close(
                bar, bar_count,
                mtf_confluence=dict(hub._mtf_confluence))
        except Exception as e:
            logger.exception(
                "[parity] monitor.on_bar_close failed at %s: %s", ts, e)
            bar_count += 1
            continue

        # Capture diagnostic for this bar (compact)
        triggers_fired = [
            k for k, v in (audit.get('trigger_booleans') or {}).items() if v]
        replay_bar_states.append({
            'ts': _normalize_ts(ts),
            'bar_count': bar_count,
            'position': audit.get('position_state'),
            'triggers_fired': triggers_fired,
            'confluence_records': sorted(monitor._current_confluence),
        })

        for sig in signals:
            stype = sig.get('signal_type') or sig.get('type') or ''
            if stype.lower().startswith('entry'):
                replay_fires.append({
                    'ts': _normalize_ts(sig.get('time') or ts),
                    'bar_count': bar_count,
                    'trigger': sig.get('trigger', ''),
                    'price': sig.get('price') or sig.get('fill_price'),
                })

        bar_count += 1

    meta = {
        'primary_bars_total': int(len(primary_df)),
        'primary_bars_replayed': int(len(primary_df) - warmup),
        'warmup_bars': warmup,
        'secondary_tfs': sorted(sec_dfs.keys()),
        'data_days': days,
    }
    return replay_fires, replay_bar_states, meta


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def _find_bar_state(states: list[dict], ts_iso: str) -> Optional[dict]:
    """Find the bar state at the given timestamp (exact match)."""
    for s in states:
        if s.get('ts') == ts_iso:
            return s
    return None


def _explain_unmatched(
    stored: dict, bar_state: Optional[dict],
    confluence_set: set[str],
) -> dict:
    """For a stored trade with no matching live fire, return a structured
    reason: which gate failed, or did the trigger itself not fire?"""
    if bar_state is None:
        return {
            'reason': 'NO_REPLAY_BAR',
            'detail': (
                'No replay bar at this timestamp. The strategy may have '
                'been created before the data window we replayed, or the '
                'timestamp falls outside RTH.'
            ),
        }

    expected_trigger = stored.get('trigger') or ''
    triggers_fired = bar_state.get('triggers_fired') or []
    actual_records = set(bar_state.get('confluence_records') or [])

    # Did the entry trigger fire at all?
    trigger_fired_in_replay = (
        expected_trigger in triggers_fired
        if expected_trigger else bool(triggers_fired))

    if not trigger_fired_in_replay:
        return {
            'reason': 'TRIGGER_NOT_FIRED',
            'detail': (
                f'Live replay did not fire {expected_trigger!r} at this bar. '
                f'Triggers that did fire: {triggers_fired or "none"}.'
            ),
            'triggers_fired_in_replay': triggers_fired,
        }

    # Trigger fired in replay — diagnose the gate
    if confluence_set:
        missing = sorted(confluence_set - actual_records)
        if missing:
            # Group by interpreter name so the UI can show "live state was X"
            # for each missing gate.
            missing_with_actual: list[dict] = []
            for req in missing:
                # req format: "{tf}-{INTERP}-{state}"
                parts = req.split('-', 2)
                actual_state = None
                if len(parts) == 3:
                    tf, interp, _state = parts
                    prefix = f'{tf}-{interp}-'
                    for r in actual_records:
                        if r.startswith(prefix):
                            actual_state = r[len(prefix):]
                            break
                missing_with_actual.append({
                    'required': req,
                    'replay_actual': actual_state,
                })
            return {
                'reason': 'GATE_FAILED',
                'detail': (
                    f'Trigger {expected_trigger!r} fired but the cross-TF / '
                    f'interpreter gate was not satisfied. Missing: '
                    f'{[m["required"] for m in missing_with_actual]}'
                ),
                'failing_gates': missing_with_actual,
                'triggers_fired_in_replay': triggers_fired,
            }

    # Trigger fired, gates matched, no fire — likely position state /
    # cooldown / direction filter. Surface as POSITION_BLOCKED.
    return {
        'reason': 'POSITION_BLOCKED',
        'detail': (
            f'Trigger {expected_trigger!r} fired and confluence was '
            f'satisfied in replay, but no entry signal emitted. The '
            f'position state machine likely held the strategy IN_POSITION '
            f'from an earlier replay entry, blocking re-entry.'
        ),
        'triggers_fired_in_replay': triggers_fired,
        'position_state': bar_state.get('position'),
    }


def _build_report(
    strategy: dict,
    stored_entries: list[dict],
    replay_fires: list[dict],
    replay_bar_states: list[dict],
    meta: dict,
) -> dict:
    """Diff stored vs. replay and emit a structured report."""
    confluence_set = (set(strategy.get('confluence', []) or [])
                      | set(strategy.get('general_confluences', []) or []))

    # Index replay fires by minute-truncated ts so timestamps that drift by
    # a few seconds (sub-second alert fill_ts vs bar_start) still match.
    fire_by_minute: dict[str, dict] = {}
    for f in replay_fires:
        if not f.get('ts'):
            continue
        key = f['ts'][:16]  # YYYY-MM-DDTHH:MM
        fire_by_minute.setdefault(key, f)

    # Index bar states by minute as well for the unmatched-explanation lookup.
    state_by_minute: dict[str, dict] = {}
    for s in replay_bar_states:
        if not s.get('ts'):
            continue
        state_by_minute.setdefault(s['ts'][:16], s)

    matched: list[dict] = []
    stored_only: list[dict] = []  # backtest fired, live wouldn't — the bug

    for s in stored_entries:
        if not s.get('ts'):
            continue
        key = s['ts'][:16]
        if key in fire_by_minute:
            matched.append({
                'stored_ts': s['ts'],
                'replay_ts': fire_by_minute[key].get('ts'),
                'trigger': s.get('trigger'),
                'trade_id': s.get('trade_id'),
            })
        else:
            bar_state = state_by_minute.get(key)
            stored_only.append({
                'stored_ts': s['ts'],
                'trigger': s.get('trigger'),
                'trade_id': s.get('trade_id'),
                **_explain_unmatched(s, bar_state, confluence_set),
            })

    # replay_only = live fires with no corresponding stored trade. Less
    # critical than stored_only (it means live would over-fire), but still
    # useful for catching backtest-side gaps.
    stored_minutes = {s['ts'][:16] for s in stored_entries if s.get('ts')}
    replay_only: list[dict] = []
    for f in replay_fires:
        if not f.get('ts'):
            continue
        if f['ts'][:16] not in stored_minutes:
            replay_only.append(f)

    total = len(matched) + len(stored_only) + len(replay_only)
    parity_score = (len(matched) / total) if total > 0 else None

    if total == 0:
        verdict = 'NO_DATA'
    elif not stored_only and not replay_only:
        verdict = 'PASS'
    elif stored_only and not replay_only:
        verdict = 'FAIL_LIVE_BLOCKED'
    elif replay_only and not stored_only:
        verdict = 'FAIL_OVER_FIRES'
    else:
        verdict = 'PARTIAL'

    # Aggregate the most common failing gate so the UI can show a one-line
    # summary at the top of the panel.
    gate_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for u in stored_only:
        reason_counts[u.get('reason', 'UNKNOWN')] = (
            reason_counts.get(u.get('reason', 'UNKNOWN'), 0) + 1)
        for fg in u.get('failing_gates', []) or []:
            req = fg.get('required', '')
            gate_counts[req] = gate_counts.get(req, 0) + 1
    most_common_gate = max(gate_counts.items(),
                            key=lambda kv: kv[1])[0] if gate_counts else None

    return {
        'strategy_id': strategy.get('id'),
        'symbol': strategy.get('symbol'),
        'timeframe': strategy.get('timeframe'),
        'verdict': verdict,
        'parity_score': parity_score,
        'stored_count': len(stored_entries),
        'matched_count': len(matched),
        'stored_only_count': len(stored_only),
        'replay_only_count': len(replay_only),
        'most_common_failing_gate': most_common_gate,
        'reason_breakdown': reason_counts,
        'matched': matched,
        'stored_only': stored_only,
        'replay_only': replay_only,
        'meta': meta,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_strategy_parity(
    strategy_id: int, user_id: Optional[str] = None,
) -> dict:
    """Run a parity check on the given strategy.

    Loads the strategy + its stored trades, replays bars through the live
    engine (StrategyMonitor + shadow engines), and diffs the resulting
    fires against the stored trades. Returns a structured report.

    Caller must have admin DB context set (set_admin_user_context) or pass
    a user_id; the underlying load_general_packs_admin / load_trades_admin
    use the admin client.
    """
    from db import get_strategy_by_id_db, load_trades_admin

    strat = get_strategy_by_id_db(strategy_id)
    if strat is None:
        return {'error': 'Strategy not found', 'strategy_id': strategy_id}

    # If user_id wasn't provided, derive it from the strategy row.
    user_id = user_id or strat.get('user_id')

    stored_trades = load_trades_admin(strategy_id, user_id) or []
    stored_entries = _stored_entry_records(stored_trades)

    if not stored_entries:
        return {
            'strategy_id': strategy_id,
            'symbol': strat.get('symbol'),
            'timeframe': strat.get('timeframe'),
            'verdict': 'NO_TRADES',
            'parity_score': None,
            'stored_count': 0,
            'matched_count': 0,
            'stored_only_count': 0,
            'replay_only_count': 0,
            'most_common_failing_gate': None,
            'reason_breakdown': {},
            'matched': [],
            'stored_only': [],
            'replay_only': [],
            'meta': {'note': 'Strategy has no stored trades to diff against'},
        }

    # Pass earliest stored ts so the replay can size its window correctly.
    earliest = min((s['ts'] for s in stored_entries if s.get('ts')),
                   default=None)
    if earliest:
        strat = {**strat, '_earliest_stored_ts': earliest}

    try:
        replay_fires, replay_bar_states, meta = _replay_strategy(
            strat, user_id)
    except Exception as e:
        logger.exception(
            "[parity] replay crashed for strategy %s: %s", strategy_id, e)
        return {
            'strategy_id': strategy_id,
            'verdict': 'ERROR',
            'error': f'{type(e).__name__}: {e}',
        }

    return _build_report(
        strat, stored_entries, replay_fires, replay_bar_states, meta)
