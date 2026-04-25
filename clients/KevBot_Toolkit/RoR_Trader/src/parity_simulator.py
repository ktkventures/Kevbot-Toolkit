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
# Backtest path — extract trigger fires from enriched DataFrame
# ---------------------------------------------------------------------------

def _extract_trigger_fires_from_enriched(
    enriched_df: pd.DataFrame,
    trigger_name: str,
) -> list[dict]:
    """Pull trigger-fire events from the unified engine's enriched output.

    The unified engine writes one boolean column per trigger named
    `trig_{trigger_name}`. We collect every bar where that column is True.
    """
    col = f'trig_{trigger_name}'
    if col not in enriched_df.columns:
        logger.warning(
            "[parity] backtest path produced no column %s — "
            "trigger may not exist in this engine's registry", col)
        return []

    fires: list[dict] = []
    series = enriched_df[col].fillna(False).astype(bool)
    for idx, fired in enumerate(series):
        if fired:
            ts = enriched_df.index[idx]
            fires.append({
                'bar_idx': int(idx),
                'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                'trigger': trigger_name,
            })
    return fires


def _run_live_replay_path(
    strategy: dict,
    df: pd.DataFrame,
    warmup_bars: int = 200,
) -> list[dict]:
    """Replay bars sequentially through the live engine's classes.

    Mirrors what the worker does in production: fresh
    IncrementalIndicatorEngine + TriggerEvaluator, optional warmup with
    the first N bars, then process the remaining one at a time. The
    result is the set of trigger fires the LIVE worker would have
    produced over this same window.

    KEY ARCHITECTURAL CAVEAT
    ------------------------
    User pack indicators are resolved as `_user_pack_{slug}` markers
    in `resolve_strategy_requirements` (unified_engine.py:578-580) but
    are computed only in batch mode via `run_indicators_for_group`. The
    IncrementalIndicatorEngine has no per-bar code path for them. So
    any user-pack trigger will silently never fire in this replay —
    that's by design at the engine level, and it's the exact root
    cause of strategies like 129 firing in backtest but not live.

    Built-in triggers (utbot, utbot_v2, ema_stack, ema_price_position,
    macd_line, macd_histogram, vwap, rvol) DO have incremental paths
    and will fire normally.
    """
    from unified_engine import (
        resolve_strategy_requirements,
        IncrementalIndicatorEngine,
        TriggerEvaluator,
    )

    req_ind, req_interp, req_trig, params = resolve_strategy_requirements(strategy)
    ema_periods = params.get('ema_periods', [])

    # Pass `_user_pack_<slug>` markers through to the engine — it knows
    # how to instantiate the pack's incremental_class for each one
    # (packs without an incremental_class are batch-only and silently
    # skipped, which surfaces as FAIL_SILENT in the diff).
    ind_engine = IncrementalIndicatorEngine(req_ind, params)
    trig_eval = TriggerEvaluator(req_interp, req_trig, ema_periods)

    # Warmup: prod worker calls monitor.warmup(df) with prior bars to
    # seed indicator state. Matches that here. Fires don't count during
    # warmup — only the post-warmup window.
    bars_to_replay = df
    skip_warmup_idx = 0
    if warmup_bars > 0 and len(df) > warmup_bars:
        try:
            ind_engine.warmup(df.iloc[:warmup_bars])
            bars_to_replay = df.iloc[warmup_bars:]
            skip_warmup_idx = warmup_bars
            logger.info(
                "[parity] live replay warmup: %d bars seeded, %d remaining",
                warmup_bars, len(bars_to_replay))
        except Exception as e:
            logger.warning(
                "[parity] live replay warmup failed (proceeding without): %s", e)

    fires: list[dict] = []
    for offset, (ts, row) in enumerate(bars_to_replay.iterrows()):
        bar_idx = skip_warmup_idx + offset
        bar = row.to_dict()
        # Mirror the worker's bar shape (timestamp + OHLCV)
        bar['timestamp'] = ts

        try:
            current = ind_engine.update_bar(bar)
            prev_values = ind_engine.get_prev_values()
            prev2_macd_hist = getattr(ind_engine.state, 'prev2_macd_hist', 0.0)
            _interps, triggers = trig_eval.evaluate_bar_close(
                current, prev_values, prev2_macd_hist)
        except Exception as e:
            logger.exception(
                "[parity] live replay error at bar %d (%s): %s",
                bar_idx, ts, e)
            continue

        for trig_name, fired in (triggers or {}).items():
            if fired:
                fires.append({
                    'bar_idx': bar_idx,
                    'timestamp': (ts.isoformat() if hasattr(ts, 'isoformat')
                                  else str(ts)),
                    'trigger': trig_name,
                })

    logger.info(
        "[parity] live replay: %d trigger fires across %d bars",
        len(fires), len(bars_to_replay))
    return fires


def _run_backtest_path(
    pack_id: str,
    entry_trigger: str,
    symbol: str = 'SPY',
    timeframe: str = '1Min',
    days: int = 7,
    session: str = 'RTH',
    feed: str = 'sip',
) -> tuple[list[dict], pd.DataFrame]:
    """Run the unified engine in backtest mode and extract fire events.

    Returns (fires, enriched_df). enriched_df is returned for downstream
    diff context (e.g. showing indicator state at divergent bars).
    """
    from unified_engine import run_unified_backtest

    df = _load_bars_with_indicators(symbol, timeframe, days, session, feed)
    if len(df) < 2:
        logger.warning(
            "[parity] not enough bars loaded (%d) for %s/%s/%dd",
            len(df), symbol, timeframe, days)
        return [], df

    strategy = _build_strategy_proxy(pack_id, entry_trigger,
                                     timeframe=timeframe, symbol=symbol,
                                     session=session)

    try:
        trades_df, enriched_df = run_unified_backtest(
            df, strategy, general_packs=[])
    except Exception as e:
        logger.exception(
            "[parity] run_unified_backtest failed for %s/%s: %s",
            pack_id, entry_trigger, e)
        return [], df

    fires = _extract_trigger_fires_from_enriched(enriched_df, entry_trigger)
    logger.info(
        "[parity] backtest path: %d bars loaded, %d trigger fires "
        "for %s on %s/%s/%dd",
        len(df), len(fires), entry_trigger, symbol, timeframe, days)
    return fires, enriched_df


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

    backtest_fires, enriched_df = _run_backtest_path(
        pack_id=pack_id, entry_trigger=entry_trigger,
        symbol=symbol, timeframe=timeframe, days=days,
        session=session, feed=feed,
    )

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
    # Filter to just the trigger we care about — replay records every
    # trigger the evaluator emits, but the test is for one specific one.
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

    # Default smoke test — utbot_v2 (production-validated, fires often)
    pack_id = sys.argv[1] if len(sys.argv) > 1 else 'utbot_v2'
    trigger = sys.argv[2] if len(sys.argv) > 2 else 'utbot_v2_buy'
    symbol = sys.argv[3] if len(sys.argv) > 3 else 'SPY'
    timeframe = sys.argv[4] if len(sys.argv) > 4 else '1Min'
    days = int(sys.argv[5]) if len(sys.argv) > 5 else 7

    print(f"\n=== Parity smoke test: {pack_id} / {trigger} on "
          f"{symbol} / {timeframe} / {days}d ===\n")
    result = run_pack_parity_test(
        pack_id=pack_id, entry_trigger=trigger,
        symbol=symbol, timeframe=timeframe, days=days,
    )
    # Truncate fire lists for printing — just show counts + first 3
    short = dict(result)
    for key in ('backtest_fires', 'live_fires', 'live_fires_other_triggers',
                'matched', 'backtest_only', 'live_only', 'backtest_warmup'):
        vals = result.get(key, []) or []
        short[key] = (
            f"[{len(vals)} fires] "
            + (str(vals[:3]) if vals else '(empty)')
        )
    print(json.dumps(short, indent=2, default=str))
