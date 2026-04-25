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

    # Strip the `_user_pack_{slug}` markers — they're sentinel values
    # that aren't real indicator names. IncrementalIndicatorEngine
    # would crash trying to compute them. Stripping them mirrors what
    # the live worker does at startup (it just doesn't have the
    # incremental path for user packs, full stop).
    real_indicators = {i for i in req_ind if not i.startswith('_user_pack_')}
    user_pack_markers = {i for i in req_ind if i.startswith('_user_pack_')}
    if user_pack_markers:
        logger.info(
            "[parity] live replay will SKIP %d user pack indicator(s): %s — "
            "live engine has no incremental path for these (engine-level "
            "limitation, not a bug in the simulator)",
            len(user_pack_markers), sorted(user_pack_markers))

    ind_engine = IncrementalIndicatorEngine(real_indicators, params)
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
# Public V1 entrypoint (backtest + live replay; diff/verdict in commit 3)
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

    Runs both paths (backtest + live replay) and returns each fire list.
    Diff + verdict are added in commit 3.
    """
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
        'live_fires_other_triggers': [
            f for f in live_fires_all if f.get('trigger') != entry_trigger
        ],
        'matched': [],             # commit 3
        'backtest_only': [],       # commit 3
        'live_only': [],           # commit 3
        'parity_score': None,      # commit 3
        'verdict': 'NOT_YET_RUN',  # commit 3
        'summary': (
            f'backtest={len(backtest_fires)} live={len(live_fires)} for '
            f'{entry_trigger} on {symbol}/{timeframe}/{days}d'
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
    # Truncate fires list for printing — just show counts + first 3
    short = dict(result)
    for key in ('backtest_fires', 'live_fires', 'live_fires_other_triggers'):
        vals = result.get(key, [])
        short[key] = (
            f"[{len(vals)} fires] "
            + (str(vals[:3]) if vals else '(empty)')
        )
    print(json.dumps(short, indent=2, default=str))
