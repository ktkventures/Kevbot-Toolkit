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
import time
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

    Secondary bars are RESAMPLED from primary bars when the secondary TF
    is coarser than the primary — NEVER loaded as native bars from
    Polygon. Native daily/hourly bars from Polygon have stock-split
    adjustment inconsistencies (per CLAUDE.md and verified empirically:
    AAPL native 1Day closes at $185, AAPL resampled-from-1Min at $271,
    a ~30% divergence that breaks every cross-TF gate's classification).
    Backtest (services.prepare_data_with_indicators) and production
    live (worker BarBuilders) both work from primary-or-finer bars and
    aggregate up — parity replay must do the same to match.

    Returns (primary_df, {sec_tf_seconds: secondary_df}).
    """
    from data_loader import load_market_data, resample_to_timeframe
    from ralph_engine import SECONDS_TO_TIMEFRAME, TIMEFRAME_SECONDS

    sym = strategy.get('symbol', 'SPY')
    tf_label = strategy.get('timeframe', '1Min')
    session = strategy.get('trading_session', 'RTH')

    primary_df = load_market_data(
        symbol=sym, days=days, timeframe=tf_label, session=session)

    primary_tf_seconds = TIMEFRAME_SECONDS.get(tf_label, 60)

    sec_dfs: dict[int, pd.DataFrame] = {}
    for sec_tf in secondary_tfs:
        sec_label = SECONDS_TO_TIMEFRAME.get(sec_tf)
        if not sec_label:
            logger.warning("[parity] no label for tf=%ss; skipping", sec_tf)
            continue
        try:
            if sec_tf > primary_tf_seconds and primary_df is not None and len(primary_df) > 0:
                # Resample from primary — matches backtest + production
                # live aggregation paths
                sec_dfs[sec_tf] = resample_to_timeframe(
                    primary_df[['open', 'high', 'low', 'close', 'volume']].copy(),
                    sec_label,
                )
            else:
                # Secondary is finer than primary OR primary failed —
                # fall back to direct load (rare; usually a config bug)
                logger.warning(
                    "[parity] secondary tf %s (%ds) <= primary tf (%ds), "
                    "falling back to native load — parity may diverge",
                    sec_label, sec_tf, primary_tf_seconds,
                )
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

    sid = strategy.get('id', '?')
    _t = time.time()
    primary_df, sec_dfs = _load_primary_and_secondary_bars(
        strategy, monitor._required_secondary_tf, days)
    t_load = time.time() - _t
    logger.warning(
        "[PARITY:%s] bar_load=%.2fs primary_bars=%d secondary_tfs=%s days=%d",
        sid, t_load, len(primary_df) if primary_df is not None else 0,
        list(sec_dfs.keys()), days)

    if primary_df is None or len(primary_df) < 20:
        return [], [], {
            'error': f'Not enough primary-TF bars loaded: {len(primary_df) if primary_df is not None else 0}',
            'days': days,
        }

    # Warmup: each engine consumes its first N bars to seed indicator state.
    warmup = min(_DEFAULT_PRIMARY_WARMUP_BARS, max(20, len(primary_df) // 4))
    _t = time.time()
    monitor.warmup(primary_df.iloc[:warmup])

    for sec_tf, sec_df in sec_dfs.items():
        # Bug Hunt Wave 1 #1: shadow keys are (tf, session) — scan by tf.
        shadow = next(
            (sh for (_tf, _s), sh in hub._shadow_engines.items()
             if _tf == sec_tf), None)
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
    # Pre-extract numpy arrays per secondary df — `sec_df.iloc[i]` per bar
    # was the second-largest source of pandas overhead in the inner loop.
    sec_state: dict[int, dict] = {}
    for sec_tf, sec_df in sec_dfs.items():
        if sec_df is None or len(sec_df) < 5:
            continue
        sec_warmup = min(_DEFAULT_SHADOW_WARMUP_BARS,
                         max(5, len(sec_df) // 4))
        sec_state[sec_tf] = {
            'idx': sec_warmup,
            'len': len(sec_df),
            'index': sec_df.index.values,  # numpy datetime64 array
            'open': sec_df['open'].to_numpy(dtype=float),
            'high': sec_df['high'].to_numpy(dtype=float),
            'low': sec_df['low'].to_numpy(dtype=float),
            'close': sec_df['close'].to_numpy(dtype=float),
            'volume': sec_df['volume'].to_numpy(dtype=float)
                       if 'volume' in sec_df.columns
                       else None,
            'sec_df_index': sec_df.index,  # keep for sec_ts pandas Timestamp
        }

    t_warmup = time.time() - _t
    logger.warning("[PARITY:%s] warmup=%.2fs warmup_bars=%d", sid, t_warmup, warmup)

    # Pre-extract primary df columns. iterrows() instantiates a Series per
    # row which dwarfs the dict-construction cost we were doing inside the
    # loop. With 67k+ bars on a sub-minute strategy this added up to
    # tens of seconds of pure pandas overhead before the engine even ran.
    primary_index = primary_df.index
    p_open  = primary_df['open'].to_numpy(dtype=float)
    p_high  = primary_df['high'].to_numpy(dtype=float)
    p_low   = primary_df['low'].to_numpy(dtype=float)
    p_close = primary_df['close'].to_numpy(dtype=float)
    p_volume = (primary_df['volume'].to_numpy(dtype=float)
                if 'volume' in primary_df.columns
                else None)

    replay_fires: list[dict] = []
    replay_bar_states: list[dict] = []
    bar_count = warmup
    _t_loop = time.time()
    n_to_replay = len(primary_df) - warmup
    progress_step = max(10_000, n_to_replay // 10) if n_to_replay > 0 else 0

    for i in range(warmup, len(primary_df)):
        ts = primary_index[i]
        # Advance any secondary TF whose bar has FULLY CLOSED before the
        # current primary ts. The check is `sec_ts + tf_period <= ts` —
        # only feed a secondary bar when its full period has elapsed
        # (so a 1Day bar with ts=2025-12-02 00:00 is fed at the FIRST
        # primary bar at-or-after 2025-12-03 00:00). This mirrors the
        # 1-period forward shift that `prepare_data_with_indicators`
        # applies to secondary indicators in the batch path (look-ahead
        # protection: primary bar X must use the previously-closed
        # secondary bar's state, never the secondary bar that's still
        # forming).
        #
        # Without this shift, parity replay fed the 12-02 daily bar to
        # the shadow at primary 14:30 on 12-02, while batch (per the
        # forward shift) used the 12-01 daily bar's state at that
        # moment — producing legitimate same-bar state divergence on
        # every cross-TF gate (sid 141 TSLL: VWAP_V2 78/200 opposite
        # state). Found via `_probe_vwap_v2_divergence.py` 2026-04-29.
        for sec_tf, st in sec_state.items():
            shadow = next(
                (sh for (_tf, _s), sh in hub._shadow_engines.items()
                 if _tf == sec_tf), None)
            if shadow is None:
                continue
            s_idx = st['idx']
            s_len = st['len']
            s_index = st['sec_df_index']
            tf_period = pd.Timedelta(seconds=sec_tf)
            while s_idx < s_len:
                sec_ts = s_index[s_idx]
                # Only feed when the bar has fully closed before primary ts.
                if sec_ts + tf_period > ts:
                    break
                sec_bar = {
                    'open':   st['open'][s_idx],
                    'high':   st['high'][s_idx],
                    'low':    st['low'][s_idx],
                    'close':  st['close'][s_idx],
                    'volume': float(st['volume'][s_idx])
                              if st['volume'] is not None else 0.0,
                    'timestamp': sec_ts,
                }
                try:
                    records = shadow.on_bar_close(sec_bar)
                    hub._mtf_confluence[(sec_tf, shadow.session)] = records
                except Exception as e:
                    logger.warning(
                        "[parity] shadow bar_close failed (%ss @ %s): %s",
                        sec_tf, sec_ts, e)
                s_idx += 1
            st['idx'] = s_idx

        # Run primary monitor on this bar with current cross-TF buffer.
        bar = {
            'open':   p_open[i],
            'high':   p_high[i],
            'low':    p_low[i],
            'close':  p_close[i],
            'volume': float(p_volume[i]) if p_volume is not None else 0.0,
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

        # Periodic progress log so a hung replay shows up in the log.
        if progress_step and (bar_count - warmup) % progress_step == 0 \
                and bar_count > warmup:
            elapsed = time.time() - _t_loop
            done = bar_count - warmup
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_to_replay - done) / rate if rate > 0 else -1
            logger.warning(
                "[PARITY:%s] progress %d/%d (%.0f bars/s, eta %.0fs)",
                sid, done, n_to_replay, rate, eta)

        # Capture diagnostic for this bar (compact)
        triggers_fired = [
            k for k, v in (audit.get('trigger_booleans') or {}).items() if v]
        # Record both bar_start and bar_close timestamps so the
        # diagnostic explain function can locate this bar regardless of
        # which anchor the stored trade was recorded with. C-type stored
        # trades use entry_fill_ts = bar_close (bar_start + tf_seconds);
        # L-type stored trades use entry_fill_ts = bar_start. Without
        # the bar_close key the C-type lookup misses by exactly one bar
        # (one minute on 1Min strats), causing every unmatched stored
        # entry to report TRIGGER_NOT_FIRED on the next bar's state when
        # the actual trigger fired on this bar — masking real GATE_FAILED
        # diagnoses for cross-TF shadow gate failures. Found via Phase B
        # drill on sid 145 (132/132 stored entries match incremental fires
        # at offset -1m).
        bar_close_ts = ts + pd.Timedelta(seconds=monitor.tf_seconds)
        replay_bar_states.append({
            'ts': _normalize_ts(ts),
            'ts_close': _normalize_ts(bar_close_ts),
            'bar_count': bar_count,
            'position': audit.get('position_state'),
            'triggers_fired': triggers_fired,
            'confluence_records': sorted(monitor._current_confluence),
        })

        for sig in signals:
            stype = sig.get('signal_type') or sig.get('type') or ''
            if stype.lower().startswith('entry'):
                # Anchor on the same field stored_trades use: entry_fill_ts.
                # The engine emits 'entry_fill_ts' on entry signals (Trade
                # Timestamps Spec; equals bar-close anchor). Falling back to
                # 'time' or 'bar_time' (bar-start) was the bug — bar_start
                # vs bar_close are different minutes after :16 truncation,
                # making matched_count==0 even when the live engine fired
                # the same trades the backtest did. Sid 137 probe (2026-04-29)
                # showed 108 replay fires + 141 stored fires with 0 match
                # before this fix because of this exact anchor mismatch.
                ts_field = (sig.get('entry_fill_ts')
                            or sig.get('bar_time')
                            or sig.get('time')
                            or ts)
                replay_fires.append({
                    'ts': _normalize_ts(ts_field),
                    'bar_count': bar_count,
                    'trigger': sig.get('trigger', ''),
                    'price': sig.get('price') or sig.get('fill_price'),
                })

        bar_count += 1

    t_loop = time.time() - _t_loop
    bars_per_sec = (n_to_replay / t_loop) if t_loop > 0 else 0
    logger.warning(
        "[PARITY:%s] replay_loop=%.2fs bars=%d (%.0f bars/s) fires=%d",
        sid, t_loop, n_to_replay, bars_per_sec, len(replay_fires))

    meta = {
        'primary_bars_total': int(len(primary_df)),
        'primary_bars_replayed': int(len(primary_df) - warmup),
        'warmup_bars': warmup,
        'secondary_tfs': sorted(sec_dfs.keys()),
        'data_days': days,
        'timing': {
            'bar_load_s': round(t_load, 2),
            'warmup_s': round(t_warmup, 2),
            'replay_loop_s': round(t_loop, 2),
            'bars_per_s': round(bars_per_sec, 0),
        },
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
    # Normalize confluence labels to match what the engine emits
    # (e.g., '1d-MACD_LINE_V2-M>S-' → '1D-MACD_LINE_V2-M>S-'). Live
    # engine uppercases the TF prefix in monitor._current_confluence
    # via _normalize_confluence_label; without applying the same
    # normalization here, every gate looks "missing" in the diagnostic
    # explain even when the engine correctly evaluates it. Fixed during
    # Phase B drill on sid 145 — was masking GATE_FAILED entries with
    # replay_actual=None when the actual emitted state existed but was
    # cased differently.
    from ralph_engine import _normalize_confluence_label
    raw_conf = (set(strategy.get('confluence', []) or [])
                | set(strategy.get('general_confluences', []) or []))
    confluence_set = {_normalize_confluence_label(r) for r in raw_conf}

    # Index replay fires by minute-truncated ts so timestamps that drift by
    # a few seconds (sub-second alert fill_ts vs bar_start) still match.
    fire_by_minute: dict[str, dict] = {}
    for f in replay_fires:
        if not f.get('ts'):
            continue
        key = f['ts'][:16]  # YYYY-MM-DDTHH:MM
        fire_by_minute.setdefault(key, f)

    # Index bar states by minute as well for the unmatched-explanation lookup.
    # Index by BOTH bar_start and bar_close minute keys so a stored trade
    # recorded with entry_fill_ts at either anchor lands on the correct bar.
    state_by_minute: dict[str, dict] = {}
    for s in replay_bar_states:
        if s.get('ts'):
            state_by_minute.setdefault(s['ts'][:16], s)
        if s.get('ts_close'):
            state_by_minute.setdefault(s['ts_close'][:16], s)

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
    # Constrain replay_only to the same time window covered by stored_entries
    # so it doesn't drown the table when last_n / forward_test_only filters
    # tighten the stored side.
    stored_minutes = {s['ts'][:16] for s in stored_entries if s.get('ts')}
    if stored_entries:
        sorted_ts = sorted(s['ts'] for s in stored_entries if s.get('ts'))
        window_start = sorted_ts[0] if sorted_ts else None
        window_end = sorted_ts[-1] if sorted_ts else None
    else:
        window_start = window_end = None

    replay_only: list[dict] = []
    for f in replay_fires:
        if not f.get('ts'):
            continue
        if window_start and f['ts'] < window_start:
            continue
        if window_end and f['ts'] > window_end:
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
    strategy_id: int,
    user_id: Optional[str] = None,
    last_n: Optional[int] = None,
    forward_test_only: bool = False,
) -> dict:
    """Run a parity (fire-test) check on the given strategy.

    Primary question this answers: **does the engine fire when conditions
    meet?** This is the Q1 "will it fire / is something fundamentally
    broken" check. PASS means a fresh live replay produced approximately
    the same fires the backtest stored; FAIL_LIVE_BLOCKED means live
    emits nothing; PARTIAL means live fires but on slightly different
    bars (often legitimate timing edges, not a broken engine).

    What this DOES NOT measure: live↔backtest agreement on TODAY's bars
    (Q3 fidelity — see _fidelity_check_overnight.py). The replay window
    is fixed (last N stored trades) and stored trades may be days old;
    a PASS here doesn't guarantee a fresh backtest on today's bars
    matches today's live alerts. Q3 is a separate measurement.

    Loads the strategy + its stored trades, replays bars through the live
    engine (StrategyMonitor + shadow engines), and diffs the resulting
    fires against the stored trades. Returns a structured report.

    Args:
        strategy_id: which strategy to check.
        user_id: admin context user (defaults to strategy.user_id).
        last_n: limit comparison to the most recent N stored trades.
            Older backtest data was often computed with different pack
            code / Polygon snapshots and naturally diverges; the recent
            tail tells us if the *current* engine matches the latest
            backtest. None = compare every stored trade.
        forward_test_only: if True, only compare stored trades whose
            entry_fill_ts >= strategy.forward_test_start. Pre-forward-test
            trades are pure-historical backtest output and aren't
            expected to match live engine behaviour.

    Caller must have admin DB context set (set_admin_user_context) or pass
    a user_id; the underlying load_general_packs_admin / load_trades_admin
    use the admin client.
    """
    from db import get_strategy_by_id_db, load_trades_admin

    _t_total = time.time()
    logger.warning(
        "[PARITY:%s] start last_n=%s forward_test_only=%s",
        strategy_id, last_n, forward_test_only)

    strat = get_strategy_by_id_db(strategy_id)
    if strat is None:
        return {'error': 'Strategy not found', 'strategy_id': strategy_id}

    # If user_id wasn't provided, derive it from the strategy row.
    user_id = user_id or strat.get('user_id')

    # Source: prefer trades table; fall back to JSONB on the strategies
    # row when trades table is empty (recompute_and_persist may have
    # written only to JSONB depending on the trades-table flag state).
    stored_trades = load_trades_admin(strategy_id, user_id) or []
    if not stored_trades:
        st = strat.get('stored_trades') or []
        if isinstance(st, list):
            stored_trades = st

    stored_entries = _stored_entry_records(stored_trades)

    # Apply forward-test-only filter BEFORE last_n so "last N forward-test
    # trades" works as expected.
    pre_filter_count = len(stored_entries)
    if forward_test_only:
        fwd_start = strat.get('forward_test_start')
        if fwd_start:
            stored_entries = [
                s for s in stored_entries
                if s.get('ts') and s['ts'] >= fwd_start
            ]

    if last_n and last_n > 0 and len(stored_entries) > last_n:
        # Sort ascending by ts and take the tail
        stored_entries = sorted(
            stored_entries, key=lambda s: s.get('ts') or '')[-last_n:]

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

    _t = time.time()
    report = _build_report(
        strat, stored_entries, replay_fires, replay_bar_states, meta)
    t_diff = time.time() - _t
    t_total = time.time() - _t_total
    logger.warning(
        "[PARITY:%s] DONE total=%.2fs diff=%.2fs verdict=%s "
        "stored=%d matched=%d stored_only=%d replay_only=%d",
        strategy_id, t_total, t_diff,
        report.get('verdict'),
        report.get('stored_count', 0),
        report.get('matched_count', 0),
        report.get('stored_only_count', 0),
        report.get('replay_only_count', 0))

    report.setdefault('meta', {}).update({
        'last_n_filter': last_n,
        'forward_test_only': forward_test_only,
        'stored_total_before_filter': pre_filter_count,
        'timing_total_s': round(t_total, 2),
        'timing_diff_s': round(t_diff, 2),
    })
    return report
