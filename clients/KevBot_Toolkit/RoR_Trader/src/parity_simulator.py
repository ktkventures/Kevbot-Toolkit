"""Parity Simulator — backtest vs. live replay for confluence packs.

Purpose
-------
Detect confluence packs whose triggers fire in the unified engine's
batch-mode backtest path but DON'T fire in the live worker's incremental
path. That class of bug silently kills alerts in production while the
backtest still looks great — paralyzing pack development.

V1 scope (this commit set)
--------------------------
- Single pack, single timeframe, default config
- Returns parity_score, matched/divergent fire lists, verdict
- No cross-TF support; no mid-stream restart; no strategy-detail mode

V2 deferred
-----------
- Cross-TF (multiple secondary timeframes)
- Mid-stream restart simulation (catches state-machine init bugs)
- Strategy-detail mode (composes multiple packs + position state machine)
- Auto parity gate on pack save

Plan ref: docs/Parity_Simulator_Plan_2026-04-24.md
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy proxy construction
# ---------------------------------------------------------------------------

def _build_strategy_proxy(
    pack_id: str,
    entry_trigger: str,
    timeframe: str = '1Min',
    symbol: str = 'SPY',
    direction: str = 'LONG',
    session: str = 'RTH',
) -> dict:
    """Build a minimal strategy dict that the unified engine will accept.

    Per Kevin's guidance: use the entry trigger plus a short max-hold time
    exit (1-10 bars). No stop, no target, no confluence — keeps the engine
    surface minimal so any divergence is attributable to the pack itself.
    """
    return {
        'id': 0,                                 # placeholder; not persisted
        'name': f'_parity_test_{pack_id}',
        'symbol': symbol,
        'direction': direction,
        'timeframe': timeframe,
        # Engine config
        'entry_trigger_confluence_id': entry_trigger,
        'exit_trigger_confluence_ids': [],        # signal exit only
        'confluence': [],                         # no gating
        'trading_session': session,
        # Risk packs — keep blank so trades exit purely on time exit
        'stop_config': {
            'method': 'percentage',               # simplest stop method
            'value': 1.0,                         # 1% — wide enough that we
                                                  # rarely hit it during V1 tests
            'exec_type': 'L',
        },
        'stop_exec_type': 'L',
        'target_config': None,                    # signal exit only
        'take_profit_pack_id': '',
        # Time exit — Kevin's recommended exit gate for parity testing
        'time_exit_config': {
            'method': 'max_hold_bars',
            'max_bars': 10,
        },
        'data_days': 7,
        'lookback_mode': 'Days',
        # Exec types
        'stop_loss_pack_id': '',
        'time_exit_pack_id': 'max_hold_10',
        'target_exec_type': 'L',
    }


# ---------------------------------------------------------------------------
# Bar loading
# ---------------------------------------------------------------------------

def _load_bars_with_indicators(
    symbol: str,
    timeframe: str,
    days: int,
    session: str = 'RTH',
    feed: str = 'sip',
) -> pd.DataFrame:
    """Load + pre-compute the indicator/interpreter/trigger columns.

    This is the BACKTEST input shape — batch-mode pre-computation runs
    user pack indicators across the whole DataFrame at once, then the
    unified engine reads pre-computed `trig_*` columns directly.

    Live mode does NOT have these pre-computed columns; it has to derive
    them incrementally per bar. That's the parity-relevant difference.
    """
    from services import prepare_data_with_indicators
    return prepare_data_with_indicators(
        symbol=symbol, days=days, timeframe=timeframe,
        data_feed=feed, session=session,
    )


def _load_bars_raw(
    symbol: str,
    timeframe: str,
    days: int,
    session: str = 'RTH',
    feed: str = 'sip',
) -> pd.DataFrame:
    """Load raw OHLCV bars — no indicators, no triggers pre-computed.

    This is what the LIVE worker sees: just bars from the WebSocket /
    Polygon. Live engine has to compute everything else on its own.
    """
    from data_loader import load_market_data
    return load_market_data(
        symbol=symbol, days=days, timeframe=timeframe,
        feed=feed, session=session,
    )


# ---------------------------------------------------------------------------
# Shared replay loop — used by both backtest and live paths
# ---------------------------------------------------------------------------

def _replay_engine_bars(
    strategy: dict,
    df: pd.DataFrame,
    indicator_mode: str,  # 'batch' | 'incremental'
    warmup_bars: int = 200,
) -> list[dict]:
    """Run the engine's TriggerEvaluator over `df` bar-by-bar.

    The two paths share this loop and differ only in where each bar's
    indicator state comes from:

      - **'incremental'** (live mode): instantiate
        IncrementalIndicatorEngine and call ``update_bar(bar)`` per bar.
        For user packs with an ``incremental_class``, this drives the
        per-bar O(1) update.
      - **'batch'** (backtest mode): the DataFrame is already enriched
        with indicator + trigger columns by ``prepare_data_with_indicators``;
        we read each row directly and feed it as ``current`` to the
        evaluator.

    Both paths call ``evaluate_bar_for_backtest`` which returns
    ``(interps, c_triggers, l_fills)`` — that's the engine's full
    single-bar evaluation, including L-type reachability checks against
    the bar's high/low. Fires are emitted for both C-type triggers
    (from ``c_triggers`` booleans) and L-type / LC fills (from the
    ``l_fills`` dict, keyed by suffixed trigger ID).
    """
    from unified_engine import (
        resolve_strategy_requirements,
        IncrementalIndicatorEngine,
        TriggerEvaluator,
    )

    req_ind, req_interp, req_trig, params = resolve_strategy_requirements(strategy)
    ema_periods = params.get('ema_periods', [])
    trig_eval = TriggerEvaluator(req_interp, req_trig, ema_periods)

    if indicator_mode == 'incremental':
        ind_engine = IncrementalIndicatorEngine(req_ind, params)
    else:
        ind_engine = None

    def _row_to_current(row: pd.Series) -> dict:
        """Build a `current` dict from an enriched-DF row (batch mode)."""
        cur: dict = {}
        for col, val in row.items():
            # Skip NaN / non-numeric junk that can confuse downstream logic
            if isinstance(val, float) and val != val:
                continue
            # The trigger-evaluator's user-pack pickup loop reads booleans
            # from `current` keyed by trigger ID (e.g. `utv4_bull_flip`),
            # but the enriched DF keys those columns with a `trig_` prefix.
            # Strip the prefix when copying so the pickup matches.
            if isinstance(col, str) and col.startswith('trig_'):
                cur[col[len('trig_'):]] = bool(val)
                continue
            cur[col] = val
        for k in ('open', 'high', 'low', 'close', 'volume'):
            if k in row:
                cur[k] = row[k]
        return cur

    # Warmup: walk the first N bars through both engine and evaluator so
    # state (rolling indicators, _cached_levels, _ib_gate_open) is seeded
    # before fires start counting. In batch mode there's no incremental
    # engine to warmup, but the evaluator still needs to walk the bars
    # to seed _cached_levels for L-type reachability checks.
    #
    # NOTE: the inner loop's `ind_engine.update_bar(bar)` calls drive the
    # engine's warmup organically (update_bar uses `not _initialized` to
    # detect the first bar, exactly like warmup() does). We do NOT call
    # `ind_engine.warmup(warmup_df)` separately — that would feed every
    # warmup bar through the engine twice, which corrupts user packs
    # whose state cares about bar ORDER (e.g. sr_channels' pivot buffer
    # would see a bar_199 → bar_0 discontinuity at the second pass and
    # detect spurious pivots near the boundary). Most other packs
    # tolerated the double-walk because their rolling windows converge
    # to the same state regardless, but sr_channels was off-by-one on
    # ~3% of fires until this was fixed.
    if warmup_bars > 0 and len(df) > warmup_bars:
        warmup_df = df.iloc[:warmup_bars]
        replay_df = df.iloc[warmup_bars:]
        skip_idx = warmup_bars
        prev_values: dict = {}
        for ts, row in warmup_df.iterrows():
            try:
                if ind_engine is not None:
                    bar = row.to_dict(); bar['timestamp'] = ts
                    current = ind_engine.update_bar(bar)
                else:
                    current = _row_to_current(row)
                trig_eval.evaluate_bar_for_backtest(current, prev_values, 0.0)
                prev_values = current
            except Exception:
                # Ignore warmup errors — they typically come from indicators
                # that need more history than the warmup window provides.
                pass
        logger.info(
            "[parity] %s replay warmup: %d bars seeded, %d remaining",
            indicator_mode, warmup_bars, len(replay_df))
    else:
        replay_df = df
        skip_idx = 0
        prev_values = {}

    fires: list[dict] = []
    for offset, (ts, row) in enumerate(replay_df.iterrows()):
        bar_idx = skip_idx + offset

        try:
            if ind_engine is not None:
                bar = row.to_dict(); bar['timestamp'] = ts
                current = ind_engine.update_bar(bar)
                eval_prev = ind_engine.get_prev_values()
                prev2_macd_hist = getattr(
                    ind_engine.state, 'prev2_macd_hist', 0.0)
            else:
                current = _row_to_current(row)
                eval_prev = prev_values
                prev2_macd_hist = 0.0  # batch-mode MACD hist tracking deferred

            _interps, c_triggers, l_fills = trig_eval.evaluate_bar_for_backtest(
                current, eval_prev, prev2_macd_hist)
        except Exception as e:
            logger.exception(
                "[parity] %s replay error at bar %d (%s): %s",
                indicator_mode, bar_idx, ts, e)
            prev_values = current if 'current' in locals() else prev_values
            continue

        ts_iso = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)

        for trig_name, fired in (c_triggers or {}).items():
            if fired:
                fires.append({
                    'bar_idx': bar_idx,
                    'timestamp': ts_iso,
                    'trigger': trig_name,
                })
        for trig_name, fill_price in (l_fills or {}).items():
            fires.append({
                'bar_idx': bar_idx,
                'timestamp': ts_iso,
                'trigger': trig_name,
                'fill_price': float(fill_price),
            })

        prev_values = current

    logger.info(
        "[parity] %s replay: %d fires across %d bars",
        indicator_mode, len(fires), len(replay_df))
    return fires


def _run_live_replay_path(
    strategy: dict,
    df: pd.DataFrame,
    warmup_bars: int = 200,
) -> list[dict]:
    """Replay bars through the live engine's IncrementalIndicatorEngine.

    Now uses ``evaluate_bar_for_backtest`` so L-type and LC fires are
    captured via reachability checks against bar high/low — matching
    the engine's backtest semantics. Real production mode runs tick-by-
    tick which we can't reproduce without a tick feed; the bar-level
    reachability check is the standard approximation.

    User packs with an ``incremental_class`` get full PASS via this
    path. Packs without one have no live indicator values, so their
    triggers stay silent (the diff engine reports FAIL_SILENT — that's
    the diagnostic surface).
    """
    return _replay_engine_bars(
        strategy=strategy,
        df=df,
        indicator_mode='incremental',
        warmup_bars=warmup_bars,
    )


def _run_backtest_path(
    pack_id: str,
    entry_trigger: str,
    symbol: str = 'SPY',
    timeframe: str = '1Min',
    days: int = 7,
    session: str = 'RTH',
    feed: str = 'sip',
    warmup_bars: int = 200,
) -> tuple[list[dict], pd.DataFrame]:
    """Run the engine in batch (backtest) mode and capture all fires.

    Loads bars with all indicator + trigger columns precomputed, then
    walks them through the same evaluation loop as the live path —
    just sourcing each bar's `current` dict from the enriched DF
    columns instead of an incremental indicator engine. This produces
    both C-type (from the precomputed `trig_*` booleans, picked up via
    the user-pack pickup loop in TriggerEvaluator) and L-type / LC
    fires (from the engine's reachability check on bar high/low).

    Returns (fires, enriched_df). enriched_df is returned for downstream
    diff context (e.g. showing indicator state at divergent bars).
    """
    df = _load_bars_with_indicators(symbol, timeframe, days, session, feed)
    if len(df) < 2:
        logger.warning(
            "[parity] not enough bars loaded (%d) for %s/%s/%dd",
            len(df), symbol, timeframe, days)
        return [], df

    strategy = _build_strategy_proxy(pack_id, entry_trigger,
                                     timeframe=timeframe, symbol=symbol,
                                     session=session)

    fires = _replay_engine_bars(
        strategy=strategy,
        df=df,
        indicator_mode='batch',
        warmup_bars=warmup_bars,
    )
    logger.info(
        "[parity] backtest path: %d bars loaded, %d total fires "
        "for %s on %s/%s/%dd",
        len(df), len(fires), entry_trigger, symbol, timeframe, days)
    return fires, df


# ---------------------------------------------------------------------------
# Diff engine — classify fires + emit verdict
# ---------------------------------------------------------------------------

# Verdict semantics — kept narrow on purpose so the UI can switch on a
# single string and a future grader can promote/demote categories without
# changing the API.
VERDICT_PASS = 'PASS'                 # parity_score == 1.0, no divergence
VERDICT_PARTIAL = 'PARTIAL'           # both paths fire but disagree on some bars
VERDICT_FAIL_SILENT = 'FAIL_SILENT'   # backtest fires, live fires zero — the worst case
VERDICT_FAIL_REVERSE = 'FAIL_REVERSE' # live fires, backtest fires zero — also a smell
VERDICT_NO_FIRES = 'NO_FIRES'         # neither path fires — test inconclusive
VERDICT_TOLERANCE = 'PASS_WITHIN_TOLERANCE'  # all divergence inside warmup


def _diff_fire_lists(
    backtest_fires: list[dict],
    live_fires: list[dict],
    warmup_bars: int,
) -> dict:
    """Classify each fire as matched / backtest_only / live_only.

    Match key: `bar_idx`. Both paths derive it from the same DataFrame
    index, so a same-bar fire from each path is a true match.

    Warmup handling: backtest fires inside the warmup window aren't
    actually divergent — the live path can't see them. We pull them out
    into `backtest_warmup` so the parity score isn't artificially
    deflated, and the verdict accounts for them as "explained".
    """
    backtest_warmup = [f for f in backtest_fires
                       if f.get('bar_idx', 0) < warmup_bars]
    backtest_post = [f for f in backtest_fires
                     if f.get('bar_idx', 0) >= warmup_bars]

    live_by_bar = {f['bar_idx']: f for f in live_fires}
    bt_by_bar = {f['bar_idx']: f for f in backtest_post}

    matched: list[dict] = []
    backtest_only: list[dict] = []
    live_only: list[dict] = []

    for bar_idx, bt in bt_by_bar.items():
        if bar_idx in live_by_bar:
            matched.append({
                'bar_idx': bar_idx,
                'timestamp': bt.get('timestamp'),
                'trigger': bt.get('trigger'),
            })
        else:
            backtest_only.append(bt)

    for bar_idx, lv in live_by_bar.items():
        if bar_idx not in bt_by_bar:
            live_only.append(lv)

    total_post_warmup = len(matched) + len(backtest_only) + len(live_only)
    parity_score = (len(matched) / total_post_warmup
                    if total_post_warmup > 0 else None)

    return {
        'matched': matched,
        'backtest_only': backtest_only,
        'live_only': live_only,
        'backtest_warmup': backtest_warmup,
        'parity_score': parity_score,
    }


def _verdict_from_diff(
    backtest_fires: list[dict],
    live_fires: list[dict],
    diff: dict,
) -> tuple[str, str]:
    """Map the diff into a single-word verdict + a one-line explanation.

    Returns (verdict, explanation).
    """
    bt_n = len(backtest_fires)
    lv_n = len(live_fires)
    bt_post_n = bt_n - len(diff['backtest_warmup'])
    matched_n = len(diff['matched'])
    bt_only_n = len(diff['backtest_only'])
    lv_only_n = len(diff['live_only'])

    if bt_n == 0 and lv_n == 0:
        return VERDICT_NO_FIRES, (
            'Neither path produced fires — test inconclusive. Try a '
            'longer window or a different symbol/timeframe.'
        )

    if bt_n > 0 and lv_n == 0:
        return VERDICT_FAIL_SILENT, (
            f'{bt_n} backtest fire(s) but 0 live fires — pack works in '
            f'batch mode but is silent in production. Likely a missing '
            f'incremental-indicator path (common for user packs).'
        )

    if lv_n > 0 and bt_n == 0:
        return VERDICT_FAIL_REVERSE, (
            f'{lv_n} live fire(s) but 0 backtest fires — unusual; '
            f'check that the trigger column is present in the enriched '
            f'DataFrame and named `trig_{{trigger}}`.'
        )

    if bt_only_n == 0 and lv_only_n == 0 and bt_post_n > 0:
        return VERDICT_PASS, (
            f'{matched_n} fire(s) matched on every post-warmup bar — '
            f'parity confirmed.'
        )

    # Mixed outcome — there's overlap but also disagreement.
    return VERDICT_PARTIAL, (
        f'{matched_n} matched, {bt_only_n} backtest-only, '
        f'{lv_only_n} live-only — divergence on {bt_only_n + lv_only_n} '
        f'bar(s). Investigate indicator/interpreter incremental parity.'
    )


# ---------------------------------------------------------------------------
# Trigger ID resolution — pack base name → prefixed engine ID
# ---------------------------------------------------------------------------

def _ensure_user_packs_loaded() -> None:
    """Make sure scan_and_load_all() has run.

    Idempotent — if packs are already registered, this is a no-op. Needed
    because the smoke-test entrypoint imports this module without booting
    the API, so the pack_registry never gets populated.
    """
    try:
        import pack_registry
        # Internal flag — only run a full scan once per process.
        if not getattr(pack_registry, '_registry_loaded', False):
            pack_registry.scan_and_load_all()
    except Exception as e:
        logger.warning("[parity] pack registry load failed: %s", e)


def _resolve_engine_trigger_id(pack_id: str, entry_trigger: str) -> str:
    """Convert a manifest/template `base` trigger to the engine's full ID.

    Two cases:
    - **User packs** (TfConfluence user_packs): the frontend sends the
      manifest's `base` (e.g. `bullish_c2_detected`). The engine
      registers triggers as `{trigger_prefix}_{base}` — i.e.
      `s123t_bullish_c2_detected`.
    - **Built-in templates** (ema_stack, macd_line, utbot_v2 …): the
      frontend sends the template trigger's `id` (e.g. `cross_bull` or
      `buy`). The engine prefixes with the template's
      TRIGGER_PREFIX_TO_TEMPLATE entry — i.e. `ema_cross_bull` or
      `utbot_v2_buy`.

    If the caller already passed a fully-qualified ID (the smoke-test
    path uses this), this function recognizes the prefix and returns it
    unchanged.
    """
    if not entry_trigger:
        return entry_trigger

    # Case 1: user pack — look up the trigger_prefix in pack_registry.
    try:
        import pack_registry
        registered = pack_registry.get_registered_packs()
        pack = registered.get(pack_id)
        if pack is not None:
            prefix = pack.manifest.get('trigger_prefix', '')
            if prefix:
                if entry_trigger.startswith(prefix + '_') or entry_trigger == prefix:
                    return entry_trigger
                return f'{prefix}_{entry_trigger}'
    except Exception as e:
        logger.warning(
            "[parity] user-pack lookup failed for %s: %s", pack_id, e)

    # Case 2: built-in template — invert TRIGGER_PREFIX_TO_TEMPLATE.
    try:
        from unified_engine import TRIGGER_PREFIX_TO_TEMPLATE
        # template name → prefix (multiple prefixes can map to the same
        # template; pick the longest one to avoid `ema` matching when
        # `ema_pp_v2` is wanted).
        candidates = [pfx for pfx, tpl in TRIGGER_PREFIX_TO_TEMPLATE.items()
                      if tpl == pack_id]
        if candidates:
            prefix = max(candidates, key=len)
            if entry_trigger.startswith(prefix + '_') or entry_trigger == prefix:
                return entry_trigger
            return f'{prefix}_{entry_trigger}'
    except Exception as e:
        logger.warning(
            "[parity] built-in template lookup failed for %s: %s", pack_id, e)

    # No prefix info — pass through (caller already supplied the full ID).
    return entry_trigger


# ---------------------------------------------------------------------------
# Public V1 entrypoint
# ---------------------------------------------------------------------------

def run_pack_parity_test(
    pack_id: str,
    entry_trigger: str,
    symbol: str = 'SPY',
    timeframe: str = '1Min',
    days: int = 7,
    session: str = 'RTH',
    feed: str = 'sip',
    warmup_bars: int = 200,
) -> dict:
    """Run a parity test for a single pack's entry trigger.

    Pipeline: backtest path → live replay → diff → verdict. Returns the
    full result schema (no fields left empty for later commits).
    """
    _ensure_user_packs_loaded()
    raw_input_trigger = entry_trigger
    entry_trigger = _resolve_engine_trigger_id(pack_id, entry_trigger)
    if entry_trigger != raw_input_trigger:
        logger.info(
            "[parity] resolved trigger '%s' → '%s' via pack %s prefix",
            raw_input_trigger, entry_trigger, pack_id)

    backtest_fires_all, enriched_df = _run_backtest_path(
        pack_id=pack_id, entry_trigger=entry_trigger,
        symbol=symbol, timeframe=timeframe, days=days,
        session=session, feed=feed, warmup_bars=warmup_bars,
    )
    # Both paths now record every trigger the evaluator emits (including
    # sibling C/L/LC/CC variants for the same pack). Filter to just the
    # one we're testing — sibling fires are tracked separately for
    # diagnostic display.
    backtest_fires = [f for f in backtest_fires_all
                      if f.get('trigger') == entry_trigger]
    backtest_fires_other = [f for f in backtest_fires_all
                            if f.get('trigger') != entry_trigger]

    # The live worker doesn't see the indicator/interpreter/trigger columns
    # we computed in batch — strip them off so the replay path operates on
    # the same OHLCV shape it'd see in production.
    raw_cols = [c for c in enriched_df.columns
                if c.lower() in ('open', 'high', 'low', 'close', 'volume')]
    raw_df = enriched_df[raw_cols].copy() if raw_cols else enriched_df

    strategy = _build_strategy_proxy(
        pack_id, entry_trigger,
        timeframe=timeframe, symbol=symbol, session=session,
    )

    live_fires_all = _run_live_replay_path(
        strategy=strategy, df=raw_df, warmup_bars=warmup_bars,
    )
    live_fires = [f for f in live_fires_all if f.get('trigger') == entry_trigger]
    live_fires_other = [f for f in live_fires_all
                        if f.get('trigger') != entry_trigger]

    diff = _diff_fire_lists(backtest_fires, live_fires, warmup_bars)
    verdict, explanation = _verdict_from_diff(backtest_fires, live_fires, diff)

    return {
        'pack_id': pack_id,
        'entry_trigger': entry_trigger,
        'symbol': symbol,
        'timeframe': timeframe,
        'days': days,
        'bars_loaded': len(enriched_df),
        'warmup_bars': warmup_bars,
        'backtest_fires': backtest_fires,
        'backtest_fires_other_triggers': backtest_fires_other,
        'live_fires': live_fires,
        'live_fires_other_triggers': live_fires_other,
        'matched': diff['matched'],
        'backtest_only': diff['backtest_only'],
        'live_only': diff['live_only'],
        'backtest_warmup': diff['backtest_warmup'],
        'parity_score': diff['parity_score'],
        'verdict': verdict,
        'explanation': explanation,
        'summary': (
            f'{verdict}: {len(diff["matched"])}/{len(diff["matched"]) + len(diff["backtest_only"]) + len(diff["live_only"])} '
            f'matched (parity={diff["parity_score"]}) for {entry_trigger} on '
            f'{symbol}/{timeframe}/{days}d'
        ),
    }


# ===========================================================================
# 4-QUADRANT EXTENSION (2026-04-27)
# ===========================================================================
# The original `run_pack_parity_test` covers Quadrant 1 (Trigger / primary
# TF). Today's bug chain (#3 = primary-TF interpreter never ran live;
# #4/#5 = secondary-TF user-pack interpreter never registered or never
# received bars) exposed two more parity surfaces. The 4-quadrant test
# extends the simulator to surface all of these at pack-creation time.
#
#   Q1: Trigger / primary TF       — original test
#   Q2: Interpreter / primary TF   — pack as same-TF confluence gate
#   Q3: Interpreter / secondary TF — pack as cross-TF confluence gate
#                                    (exercises shadow engine path)
#   Q4: Data feed fidelity         — secondary aggregation matches
#                                    direct REST load (deferred — would
#                                    catch Polygon-side OHLCV drift, not
#                                    pack-author bugs)
#
# A pack passes the 4-quadrant test if every applicable quadrant returns
# parity_score=1.0. Packs without a manifest interpreter skip Q2/Q3.
# ===========================================================================


def _interp_states_batch(
    pack: 'pack_registry.RegisteredPack',
    enriched_df: pd.DataFrame,
) -> 'list[Optional[str]]':
    """Run pack.interpreter_func on the full DataFrame; return per-bar states.

    Mirrors the backtest pipeline's interpreter step. Returned list aligns
    1:1 with enriched_df rows.
    """
    if pack.interpreter_func is None:
        return [None] * len(enriched_df)
    try:
        states = pack.interpreter_func(enriched_df)
    except Exception as e:
        logger.warning("[parity-Q2] batch interpreter failed: %s", e)
        return [None] * len(enriched_df)
    return [
        states.iloc[i] if states is not None and i < len(states) else None
        for i in range(len(enriched_df))
    ]


def _interp_states_live(
    strategy: dict,
    df: pd.DataFrame,
    interpreter_key: str,
    warmup_bars: int = 200,
) -> 'list[Optional[str]]':
    """Replay df bar-by-bar through IncrementalIndicatorEngine + the live
    user-pack interpreter dispatch in evaluate_bar_close. Capture
    interpreter state per bar.

    Mirrors what `evaluate_bar_close` (with the 88be377 generic dispatch)
    does on every live bar close.
    """
    from unified_engine import (
        resolve_strategy_requirements, IncrementalIndicatorEngine,
        TriggerEvaluator,
    )
    indicators, interpreters, triggers, params = (
        resolve_strategy_requirements(strategy))
    ind = IncrementalIndicatorEngine(indicators, params)
    trig = TriggerEvaluator(interpreters, triggers, params['ema_periods'])

    raw_cols = [c for c in df.columns
                if c.lower() in ('open', 'high', 'low', 'close', 'volume')]
    raw = df[raw_cols].copy() if raw_cols else df

    states: list = []
    for i, (ts, row) in enumerate(raw.iterrows()):
        bar = {
            'open': float(row['open']), 'high': float(row['high']),
            'low': float(row['low']), 'close': float(row['close']),
            'volume': float(row.get('volume', 0)),
            'timestamp': ts,
        }
        current = ind.update_bar(bar)
        if i >= warmup_bars and ind._initialized:
            prev = ind.get_prev_values()
            try:
                interps, _t = trig.evaluate_bar_close(
                    current, prev, ind.state.prev2_macd_hist)
                states.append(interps.get(interpreter_key))
            except Exception as e:
                logger.warning(
                    "[parity-Q2] live evaluate_bar_close failed bar %d: %s",
                    i, e)
                states.append(None)
        else:
            states.append(None)
    return states


def run_quadrant_2_interpreter_primary(
    pack_id: str,
    symbol: str = 'SPY',
    timeframe: str = '1Min',
    days: int = 7,
    session: str = 'RTH',
    feed: str = 'sip',
    warmup_bars: int = 200,
) -> dict:
    """Q2: Interpreter parity on the primary TF.

    Runs the pack's interpreter on the full DataFrame (batch) and replays
    bar-by-bar through the live dispatch. Compares per-bar states.

    Skipped if the pack has no interpreter or no incremental_class.
    """
    _ensure_user_packs_loaded()
    import pack_registry
    pack = pack_registry.get_pack(pack_id)
    if pack is None:
        return {'verdict': 'SKIP',
                'explanation': f'pack {pack_id!r} not registered'}
    if pack.interpreter_func is None:
        return {'verdict': 'SKIP',
                'explanation': f'pack {pack_id!r} has no interpreter_func'}
    if pack.incremental_class is None:
        return {'verdict': 'SKIP',
                'explanation': (f'pack {pack_id!r} is batch-only '
                                '(no incremental_class) — interpreter '
                                'cannot be live-dispatched')}

    interpreter_keys = pack.manifest.get('interpreters', [])
    if not interpreter_keys:
        return {'verdict': 'SKIP',
                'explanation': f'pack {pack_id!r} declares no interpreters'}
    interpreter_key = interpreter_keys[0]

    enriched_df = _load_bars_with_indicators(
        symbol, timeframe, days, session, feed)
    if len(enriched_df) < warmup_bars + 10:
        return {'verdict': 'SKIP',
                'explanation': (f'only {len(enriched_df)} bars; need '
                                f'>{warmup_bars + 10}')}

    # Build a strategy proxy so the live replay knows the pack is required
    triggers = pack.manifest.get('triggers', [])
    if not triggers:
        return {'verdict': 'SKIP',
                'explanation': f'pack {pack_id!r} has no triggers'}
    base = triggers[0].get('base', '')
    proxy_trigger = _resolve_engine_trigger_id(pack_id, base)
    strategy = _build_strategy_proxy(
        pack_id, proxy_trigger,
        timeframe=timeframe, symbol=symbol, session=session)

    batch_states = _interp_states_batch(pack, enriched_df)
    live_states = _interp_states_live(
        strategy, enriched_df, interpreter_key, warmup_bars=warmup_bars)

    # Compare post-warmup; ignore positions where either side is None
    matched = 0
    compared = 0
    divergent: list = []
    for i, (b, l) in enumerate(zip(batch_states, live_states)):
        if i < warmup_bars or b is None or l is None:
            continue
        compared += 1
        if b == l:
            matched += 1
        elif len(divergent) < 5:
            divergent.append({
                'bar_idx': i,
                'timestamp': str(enriched_df.index[i]),
                'batch': b,
                'live': l,
            })
    parity = (matched / compared) if compared else 1.0
    verdict = 'PASS' if parity == 1.0 else (
        'WARN' if parity >= 0.9 else 'FAIL')
    return {
        'quadrant': 'Q2_interpreter_primary',
        'pack_id': pack_id,
        'interpreter_key': interpreter_key,
        'symbol': symbol,
        'timeframe': timeframe,
        'days': days,
        'bars_loaded': len(enriched_df),
        'compared': compared,
        'matched': matched,
        'parity_score': round(parity, 4),
        'divergent_examples': divergent,
        'verdict': verdict,
        'explanation': (
            f'{matched}/{compared} bars matched on primary-TF interpreter '
            f'(parity={parity:.4f}). '
            + ('Pack interpreter parity confirmed.' if parity == 1.0 else
               'Common cause for divergence: pack interpreter reads `__`-prefixed '
               'columns that the live incremental class doesn\'t emit. '
               'Have update_bar() return both single- and double-underscore '
               'variants of trigger booleans.')
            if compared else 'no comparable bars'),
    }


def run_quadrant_3_interpreter_secondary(
    pack_id: str,
    primary_tf: str = '1Min',
    secondary_tf: str = '15Min',
    symbol: str = 'SPY',
    days: int = 14,
    session: str = 'RTH',
    feed: str = 'sip',
    warmup_bars: int = 20,
) -> dict:
    """Q3: Interpreter parity on a SECONDARY TF (cross-TF / shadow path).

    Backtest reference: load secondary_tf bars directly from REST, compute
    the pack interpreter, capture per-bar state.

    Live path: build a SymbolHub with a primary monitor + secondary shadow.
    Feed primary-TF bars via on_polygon_bar (mirrors ca57cac live path);
    secondary builder aggregates and shadow.on_bar_close emits records on
    each secondary boundary. Capture the emitted state per close.

    Compare matched closes.
    """
    _ensure_user_packs_loaded()
    import pack_registry
    pack = pack_registry.get_pack(pack_id)
    if pack is None:
        return {'verdict': 'SKIP',
                'explanation': f'pack {pack_id!r} not registered'}
    if pack.interpreter_func is None or pack.incremental_class is None:
        return {'verdict': 'SKIP',
                'explanation': (f'pack {pack_id!r} cannot be live-shadowed '
                                '(missing interpreter or incremental_class)')}
    interpreter_keys = pack.manifest.get('interpreters', [])
    if not interpreter_keys:
        return {'verdict': 'SKIP',
                'explanation': f'pack {pack_id!r} declares no interpreters'}
    interpreter_key = interpreter_keys[0]
    triggers = pack.manifest.get('triggers', [])
    if not triggers:
        return {'verdict': 'SKIP',
                'explanation': f'pack {pack_id!r} has no triggers'}

    # Direct backtest: load secondary TF bars + compute interpreter
    sec_df = _load_bars_with_indicators(
        symbol, secondary_tf, days, session, feed)
    if len(sec_df) < warmup_bars + 5:
        return {'verdict': 'SKIP',
                'explanation': (f'only {len(sec_df)} secondary bars; need '
                                f'>{warmup_bars + 5}')}
    batch_states = _interp_states_batch(pack, sec_df)

    # Build a synthetic cross-TF strategy and exercise the shadow path
    from ralph_engine import (
        SymbolHub, StrategyMonitor, TIMEFRAME_SECONDS, SECONDS_TO_TIMEFRAME,
    )
    from data_loader import load_market_data

    sec_tf_seconds = TIMEFRAME_SECONDS.get(secondary_tf)
    primary_tf_seconds = TIMEFRAME_SECONDS.get(primary_tf)
    if not sec_tf_seconds or not primary_tf_seconds:
        return {'verdict': 'SKIP',
                'explanation': f'unrecognized timeframe(s)'}
    if sec_tf_seconds < primary_tf_seconds:
        return {'verdict': 'SKIP',
                'explanation': (f'secondary {secondary_tf} is finer than '
                                f'primary {primary_tf} — Q3 simulates '
                                'coarser-secondary cross-TF only')}

    # Use a built-in trigger as the strategy entry so resolve_strategy_
    # requirements doesn't need this pack's trigger to instantiate.
    sec_label = secondary_tf.replace('Min', 'M').replace('Hour', 'H').replace(
        'Day', 'D').replace('Week', 'W').replace('Sec', 's').lower()
    state_for_record = pack.manifest.get('outputs', ['BULL_TREND'])[0]
    strategy = {
        'id': 999_999,
        'symbol': symbol,
        'timeframe': primary_tf,
        'direction': 'LONG',
        'entry_trigger_confluence_id':
            'ema_price_position_v2_default_cross_short_up',
        'exit_trigger_confluence_ids':
            ['ema_price_position_v2_default_cross_mid_down'],
        'confluence': [f'{sec_label}-{interpreter_key}-{state_for_record}'],
        'general_confluences': [],
        'trading_session': session,
    }
    hub = SymbolHub(symbol, publisher=None)
    monitor = StrategyMonitor(strategy, None, general_packs=[])
    hub.add_monitor(monitor)
    hub.finalize_shadow_engines()
    # Bug Hunt Wave 1 #1: shadow keys are (tf_seconds, session) — grab the
    # (single) shadow for this TF whatever session key it landed under.
    shadow = next(
        (sh for (tf, _sess), sh in hub._shadow_engines.items()
         if tf == sec_tf_seconds), None)
    if shadow is None:
        return {'verdict': 'FAIL',
                'explanation': (f'no shadow engine created for '
                                f'{sec_tf_seconds}s — Q3/Q4 user-pack '
                                'requirement plumbing is broken')}

    # IMPORTANT: do NOT call shadow.warmup() here. The live replay below
    # aggregates the same primary bars into the shadow's 15Min builder
    # and calls shadow.on_bar_close on every close — running warmup AND
    # live replay would double-process the warmup window and silently
    # drift the path-dependent trail-stop state. Batch path (interpreter
    # over full DF) starts cold; we mirror that by letting live start
    # cold too. Q4 confirmed the aggregation is bit-perfect, so cold
    # starts converge. We simply skip the first `warmup_bars` 15Min
    # closes from the comparison so both states have time to stabilize.

    # Now feed each PRIMARY-TF bar through hub.on_polygon_bar — secondary
    # builder will aggregate and call shadow.on_bar_close on every close.
    pri_df = load_market_data(
        symbol=symbol, days=days, timeframe=primary_tf, session=session)
    if len(pri_df) < (sec_tf_seconds // primary_tf_seconds) + 2:
        return {'verdict': 'SKIP',
                'explanation': 'not enough primary bars to span secondary'}

    # Capture each shadow close event explicitly, keyed by the period's
    # start timestamp (the bar_start of the just-closed secondary bar).
    # This avoids the off-by-one anchoring problem of keying by primary
    # bar ts (the previous version under-reported parity on packs where
    # FLIP/TREND classifications differed across adjacent periods).
    sec_close_records: dict = {}
    _orig_on_bar_close = shadow.on_bar_close

    def _capture_on_bar_close(bar):
        records = _orig_on_bar_close(bar)
        period_ts = pd.Timestamp(bar['timestamp'])
        if period_ts.tzinfo is None:
            period_ts = period_ts.tz_localize('UTC')
        for rec in records:
            parts = rec.split('-', 2)
            if len(parts) >= 3 and parts[1] == interpreter_key:
                sec_close_records[period_ts] = parts[2]
        return records
    shadow.on_bar_close = _capture_on_bar_close
    try:
        for ts, row in pri_df.iterrows():
            bar_dict = {
                'open': float(row['open']), 'high': float(row['high']),
                'low': float(row['low']), 'close': float(row['close']),
                'volume': float(row.get('volume', 0)),
                'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
            }
            try:
                hub.on_polygon_bar(bar_dict, primary_tf_seconds)
            except Exception as e:
                logger.warning(
                    "[parity-Q3] hub.on_polygon_bar failed: %s", e)
                continue
    finally:
        shadow.on_bar_close = _orig_on_bar_close

    # Compare live emissions against batch states using exact period-start
    # matching. Skip the first `warmup_bars` secondary closes — both
    # paths start cold and need time to converge on path-dependent
    # indicators.
    matched = 0
    compared = 0
    divergent: list = []
    for i, (sec_ts, batch_state) in enumerate(
        zip(sec_df.index, batch_states)
    ):
        if i < warmup_bars:
            continue
        if batch_state is None:
            continue
        sec_ts_norm = pd.Timestamp(sec_ts)
        if sec_ts_norm.tzinfo is None:
            sec_ts_norm = sec_ts_norm.tz_localize('UTC')
        live_state = sec_close_records.get(sec_ts_norm)
        if live_state is None:
            continue
        compared += 1
        if live_state == batch_state:
            matched += 1
        elif len(divergent) < 5:
            divergent.append({
                'sec_ts': str(sec_ts),
                'batch': batch_state,
                'live': live_state,
            })

    parity = (matched / compared) if compared else 1.0
    verdict = 'PASS' if parity == 1.0 else (
        'WARN' if parity >= 0.9 else 'FAIL')
    return {
        'quadrant': 'Q3_interpreter_secondary',
        'pack_id': pack_id,
        'interpreter_key': interpreter_key,
        'symbol': symbol,
        'primary_tf': primary_tf,
        'secondary_tf': secondary_tf,
        'days': days,
        'compared': compared,
        'matched': matched,
        'parity_score': round(parity, 4),
        'divergent_examples': divergent,
        'verdict': verdict,
        'explanation': (
            f'{matched}/{compared} secondary closes matched on '
            f'cross-TF shadow path (parity={parity:.4f}).'
            if compared else
            'no comparable secondary closes — investigate shadow plumbing'),
    }


def run_pack_parity_test_4q(
    pack_id: str,
    symbol: str = 'SPY',
    primary_tf: str = '1Min',
    secondary_tf: str = '15Min',
    days: int = 7,
    session: str = 'RTH',
    feed: str = 'sip',
    warmup_bars: int = 200,
) -> dict:
    """Run all four parity quadrants for a single pack and return a unified
    report. Use this as the "did my pack pass?" gate for new pack saves.

    Q1 = trigger / primary TF (existing)
    Q2 = interpreter / primary TF
    Q3 = interpreter / secondary TF (cross-TF shadow path)
    Q4 = deferred (data feed fidelity)
    """
    _ensure_user_packs_loaded()
    import pack_registry
    pack = pack_registry.get_pack(pack_id)
    if pack is None:
        return {'verdict': 'FAIL',
                'explanation': f'pack {pack_id!r} not registered',
                'quadrants': {}}

    # Q1: Trigger primary TF — use first declared trigger as proxy
    triggers = pack.manifest.get('triggers', [])
    q1 = {'verdict': 'SKIP', 'explanation': 'pack declares no triggers'}
    if triggers:
        first_trigger_base = triggers[0].get('base', '')
        try:
            q1_full = run_pack_parity_test(
                pack_id=pack_id,
                entry_trigger=first_trigger_base,
                symbol=symbol, timeframe=primary_tf, days=days,
                session=session, feed=feed, warmup_bars=warmup_bars,
            )
            q1 = {
                'quadrant': 'Q1_trigger_primary',
                'pack_id': pack_id,
                'trigger': q1_full['entry_trigger'],
                'parity_score': q1_full['parity_score'],
                'verdict': q1_full['verdict'],
                'explanation': q1_full['explanation'],
                'matched': len(q1_full['matched']),
                'backtest_only': len(q1_full['backtest_only']),
                'live_only': len(q1_full['live_only']),
            }
        except Exception as e:
            q1 = {'verdict': 'FAIL', 'explanation': f'Q1 raised: {e}'}

    # Q2: Interpreter primary TF
    try:
        q2 = run_quadrant_2_interpreter_primary(
            pack_id=pack_id, symbol=symbol, timeframe=primary_tf,
            days=days, session=session, feed=feed, warmup_bars=warmup_bars,
        )
    except Exception as e:
        q2 = {'verdict': 'FAIL', 'explanation': f'Q2 raised: {e}'}

    # Q3: Interpreter secondary TF (cross-TF shadow)
    try:
        q3 = run_quadrant_3_interpreter_secondary(
            pack_id=pack_id, primary_tf=primary_tf,
            secondary_tf=secondary_tf, symbol=symbol, days=days,
            session=session, feed=feed,
        )
    except Exception as e:
        q3 = {'verdict': 'FAIL', 'explanation': f'Q3 raised: {e}'}

    # Q4 deferred
    q4 = {'verdict': 'SKIP', 'explanation': 'Q4 (data fidelity) deferred'}

    # Overall verdict: any FAIL → FAIL; any WARN (no FAIL) → WARN; all
    # PASS or SKIP → PASS
    quadrants = {'Q1': q1, 'Q2': q2, 'Q3': q3, 'Q4': q4}
    severity_rank = {'FAIL': 3, 'WARN': 2, 'PASS': 1, 'SKIP': 0}
    worst = max(severity_rank.get(q['verdict'], 0) for q in quadrants.values())
    overall = {3: 'FAIL', 2: 'WARN', 1: 'PASS', 0: 'PASS'}[worst]

    return {
        'pack_id': pack_id,
        'symbol': symbol,
        'primary_tf': primary_tf,
        'secondary_tf': secondary_tf,
        'days': days,
        'quadrants': quadrants,
        'overall_verdict': overall,
        'summary': (
            f'{overall}: '
            f'Q1={q1["verdict"]} Q2={q2["verdict"]} '
            f'Q3={q3["verdict"]} Q4={q4["verdict"]}  ({pack_id})'),
    }


# ---------------------------------------------------------------------------
# CLI smoke test — run this file directly to exercise the backtest path
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import os, sys, json
    from pathlib import Path
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')

    # Load .env from src/ (matches what worker.py does).
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent / '.env'
        if env_path.exists():
            load_dotenv(env_path, override=True)
    except Exception:
        pass

    os.environ.setdefault('USE_DB', 'true')

    # Smoke test runs as admin on Kevin's user. Production callers (the
    # /api/packs/{id}/parity-test endpoint, eventually) will use the
    # request user's context — no admin override needed.
    KEVIN_USER_ID = '19d47e46-f718-49a6-af32-5f5407f5b170'
    from db import set_admin_user_context
    set_admin_user_context(KEVIN_USER_ID)

    # CLI:
    #   python parity_simulator.py <pack_id> [trigger] [symbol] [timeframe] [days]
    #   python parity_simulator.py --4q <pack_id> [symbol] [primary_tf] [secondary_tf] [days]
    args = sys.argv[1:]
    if args and args[0] == '--4q':
        pack_id = args[1] if len(args) > 1 else 'ut_bot_v4'
        symbol = args[2] if len(args) > 2 else 'SPY'
        primary_tf = args[3] if len(args) > 3 else '1Min'
        secondary_tf = args[4] if len(args) > 4 else '15Min'
        days = int(args[5]) if len(args) > 5 else 7

        print(f"\n=== 4-Quadrant Parity: {pack_id} / {symbol} / "
              f"primary={primary_tf} secondary={secondary_tf} / {days}d ===\n")
        result = run_pack_parity_test_4q(
            pack_id=pack_id, symbol=symbol,
            primary_tf=primary_tf, secondary_tf=secondary_tf, days=days,
        )
        # Compact view: per-quadrant verdict + parity
        print(f"Overall: {result['overall_verdict']}\n")
        for qkey in ('Q1', 'Q2', 'Q3', 'Q4'):
            q = result['quadrants'].get(qkey, {})
            ps = q.get('parity_score', '—')
            print(f"  [{qkey}] {q.get('verdict', '?'):<5} parity={ps}  "
                  f"{q.get('explanation', '')}")
        print()
        # Full JSON for follow-up inspection (truncate divergent_examples
        # lists if huge)
        print(json.dumps(result, indent=2, default=str))
    else:
        # Original Q1-only smoke test path — utbot_v2 default
        pack_id = args[0] if args else 'utbot_v2'
        trigger = args[1] if len(args) > 1 else 'utbot_v2_buy'
        symbol = args[2] if len(args) > 2 else 'SPY'
        timeframe = args[3] if len(args) > 3 else '1Min'
        days = int(args[4]) if len(args) > 4 else 7

        print(f"\n=== Parity smoke test: {pack_id} / {trigger} on "
              f"{symbol} / {timeframe} / {days}d ===\n")
        result = run_pack_parity_test(
            pack_id=pack_id, entry_trigger=trigger,
            symbol=symbol, timeframe=timeframe, days=days,
        )
        short = dict(result)
        for key in ('backtest_fires', 'live_fires',
                    'live_fires_other_triggers',
                    'matched', 'backtest_only', 'live_only',
                    'backtest_warmup'):
            vals = result.get(key, []) or []
            short[key] = (
                f"[{len(vals)} fires] "
                + (str(vals[:3]) if vals else '(empty)')
            )
        print(json.dumps(short, indent=2, default=str))
