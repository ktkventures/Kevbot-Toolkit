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
# Public V1 entrypoint (backtest only — live replay added in commit 2)
# ---------------------------------------------------------------------------

def run_pack_parity_test(
    pack_id: str,
    entry_trigger: str,
    symbol: str = 'SPY',
    timeframe: str = '1Min',
    days: int = 7,
    session: str = 'RTH',
    feed: str = 'sip',
) -> dict:
    """Run a parity test for a single pack's entry trigger.

    V1 (this commit): runs backtest path only, returns fires.
    Live replay path is added in commit 2; full diff + verdict in commit 3.

    Returns a dict shaped to match the final V1 result schema, with
    `live_fires`, `matched`, `backtest_only`, `live_only` empty until
    later commits.
    """
    backtest_fires, enriched_df = _run_backtest_path(
        pack_id=pack_id, entry_trigger=entry_trigger,
        symbol=symbol, timeframe=timeframe, days=days,
        session=session, feed=feed,
    )

    return {
        'pack_id': pack_id,
        'entry_trigger': entry_trigger,
        'symbol': symbol,
        'timeframe': timeframe,
        'days': days,
        'bars_loaded': len(enriched_df),
        'backtest_fires': backtest_fires,
        'live_fires': [],          # commit 2
        'matched': [],             # commit 3
        'backtest_only': [],       # commit 3
        'live_only': [],           # commit 3
        'parity_score': None,      # commit 3
        'verdict': 'NOT_YET_RUN',  # commit 3
        'summary': (f'V1 backtest-path-only — {len(backtest_fires)} fires for '
                    f'{entry_trigger} on {symbol}/{timeframe}/{days}d'),
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
    # Truncate fires list for printing — just show counts + first 5
    short = dict(result)
    short['backtest_fires'] = (
        f"[{len(result['backtest_fires'])} fires] "
        + (str(result['backtest_fires'][:3]) if result['backtest_fires']
           else '(empty)')
    )
    print(json.dumps(short, indent=2, default=str))
