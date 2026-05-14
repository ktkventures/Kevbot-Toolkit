"""Admin Parity endpoints (Phase C, 2026-05-14).

Provides per-strategy snapshots that join the three execution models
(live alerts, algo trades, backtest trades) onto a common bar timeline
so the frontend Entry Overlay + Divergence Heatmap tabs can render
disagreements visually.

Endpoint shape (Phase C):
  GET /api/admin/parity/snapshot
      ?strategy_id=&start=&end=

Phase E will extend with a /ticks/{symbol} endpoint once flat-file
ingestion is in place.

See docs/Parity_Plan_2026-05-14.md for full plan.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/parity", tags=["admin", "parity"])


def _resolve_user_id(user) -> str:
    uid = user.get("id") if isinstance(user, dict) else getattr(user, "id", None)
    if not uid:
        raise HTTPException(status_code=401, detail="No user context")
    return str(uid)


@router.get("/snapshot")
def get_parity_snapshot(
    strategy_id: int = Query(..., description="Strategy id to compare"),
    start: str = Query(..., description="ISO UTC timestamp — window start (inclusive)"),
    end: Optional[str] = Query(None, description="ISO UTC timestamp — window end (exclusive). Defaults to now."),
    user=Depends(get_current_user),
):
    """3-way snapshot for parity visualization.

    Loads:
      - Strategy config (symbol, timeframe, direction)
      - Live alerts (entry side) inside [start, end)
      - Algo trades (data_source LIKE 'cache_%') with entry_fill_ts inside [start, end)
      - BT trades   (data_source LIKE 'backtest_%') with entry_fill_ts inside [start, end)

    Bars themselves are NOT returned here — the frontend already has
    `useBars` and `useStrategyCacheBars` and will fetch the appropriate
    series independently. This endpoint is the trade-overlay payload.
    """
    from db import (
        get_admin_client, load_trades_admin, get_alerts_for_strategy_db,
    )

    user_id = _resolve_user_id(user)
    end_iso = end or datetime.now(timezone.utc).isoformat()

    if start >= end_iso:
        raise HTTPException(status_code=400, detail="start must be < end")

    sb = get_admin_client()
    strat_row = sb.table('strategies').select(
        'id, name, symbol, timeframe, direction, user_id, config'
    ).eq('id', strategy_id).execute().data
    if not strat_row:
        raise HTTPException(status_code=404, detail=f"strategy {strategy_id} not found")
    strat = strat_row[0]
    if str(strat.get('user_id')) != user_id:
        raise HTTPException(status_code=404, detail=f"strategy {strategy_id} not found")

    def _in_window(ts) -> bool:
        if not ts:
            return False
        s = str(ts)
        return s >= start and s < end_iso

    try:
        algo_all = load_trades_admin(
            strategy_id, user_id, data_source_filter='cache_%') or []
        bt_all = load_trades_admin(
            strategy_id, user_id, data_source_filter='backtest_%') or []
        alerts_all = get_alerts_for_strategy_db(strategy_id, limit=5000) or []
    except Exception as e:
        logger.exception("[ADMIN-PARITY] load failed for sid=%s: %s", strategy_id, e)
        raise HTTPException(status_code=500, detail=f"Load failed: {e}")

    # Window filter. Trades use entry_fill_ts; alerts use fill_ts.
    algo_in = [t for t in algo_all if _in_window(t.get('entry_fill_ts'))]
    bt_in = [t for t in bt_all if _in_window(t.get('entry_fill_ts'))]
    live_entries = [
        a for a in alerts_all
        if a.get('side') == 'entry' and _in_window(a.get('fill_ts'))
    ]
    live_exits = [
        a for a in alerts_all
        if a.get('side') == 'exit' and _in_window(a.get('fill_ts'))
    ]

    def _slim_trade(t: dict) -> dict:
        return {
            'id': t.get('id'),
            'entry_fill_ts': t.get('entry_fill_ts'),
            'exit_fill_ts': t.get('exit_fill_ts'),
            'entry_trigger_ts': t.get('entry_trigger_ts'),
            'exit_trigger_ts': t.get('exit_trigger_ts'),
            'entry_price': t.get('entry_price'),
            'exit_price': t.get('exit_price'),
            'stop_price': t.get('stop_price'),
            'target_price': t.get('target_price'),
            'direction': t.get('direction'),
            'r_multiple': t.get('r_multiple'),
            'exit_reason': t.get('exit_reason'),
            'exec_type': t.get('exec_type'),
            'data_source': t.get('data_source'),
            'hifi_resolved': (t.get('data') or {}).get('hifi_resolved') if isinstance(t.get('data'), dict) else None,
        }

    def _slim_alert(a: dict) -> dict:
        return {
            'id': a.get('id'),
            'fill_ts': a.get('fill_ts'),
            'trigger_ts': a.get('trigger_ts'),
            'bar_time': a.get('bar_time'),
            'side': a.get('side'),
            'price': a.get('price'),
            'actual_price': a.get('actual_price'),
            'event_type': a.get('event_type'),
            'exec_type': a.get('exec_type'),
            'behavior': a.get('behavior'),
            'trigger_id': a.get('trigger_id'),
        }

    return {
        'strategy': {
            'id': strat['id'],
            'name': strat.get('name', ''),
            'symbol': strat.get('symbol', ''),
            'timeframe': strat.get('timeframe', ''),
            'direction': strat.get('direction', ''),
        },
        'window': {'start': start, 'end': end_iso},
        'live_entries': [_slim_alert(a) for a in live_entries],
        'live_exits': [_slim_alert(a) for a in live_exits],
        'algo_trades': [_slim_trade(t) for t in algo_in],
        'bt_trades': [_slim_trade(t) for t in bt_in],
        'counts': {
            'live_entries': len(live_entries),
            'live_exits': len(live_exits),
            'algo_trades': len(algo_in),
            'bt_trades': len(bt_in),
        },
    }
