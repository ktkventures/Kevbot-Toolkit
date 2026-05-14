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


@router.get("/observable")
def get_observable_bars(
    symbol: str = Query(..., description="Ticker symbol (e.g., SPY)"),
    start: str = Query(..., description="ISO UTC window start (inclusive)"),
    end: Optional[str] = Query(None, description="ISO UTC window end (exclusive). Defaults to now."),
    tf_seconds: int = Query(60, ge=1, le=86400,
        description="Target timeframe in seconds. 1=raw 1-sec bars, 60=aggregate to 1Min."),
    user=Depends(get_current_user),
):
    """Return observable bars rebuilt from Polygon flat-file trades.

    Phase E (2026-05-14): observable bars are 1-second OHLCV
    reconstructed by bucketing trades on `sip_timestamp`. They reflect
    what a real-time subscriber WOULD have seen at the moment of
    decision — the diagnostic baseline against which we compare:
      - `live_bars` cache (what our engine actually wrote)
      - Polygon REST aggregates (settled, post-correction view)

    Aggregation: when `tf_seconds > 1`, 1-sec bars are aggregated up
    to the target TF on read. The base table only stores 1-sec bars
    to keep the storage footprint small.
    """
    _resolve_user_id(user)  # auth only — observable data is not user-scoped
    end_iso = end or datetime.now(timezone.utc).isoformat()
    if start >= end_iso:
        raise HTTPException(status_code=400, detail="start must be < end")

    from db import get_admin_client
    sb = get_admin_client()

    # Pull 1-sec bars in the window. PostgREST caps at 1000 rows; for
    # a 1-hour window we'd have up to 3600 rows so we paginate via
    # `.range()` until the page is partial.
    PAGE = 1000
    MAX_ROWS = 200_000
    rows: list[dict] = []
    offset = 0
    while offset < MAX_ROWS:
        page_res = sb.table("polygon_observable_bars").select(
            "ticker, sip_second_ts, open, high, low, close, volume, trade_count"
        ).eq("ticker", symbol.upper()) \
         .gte("sip_second_ts", start) \
         .lt("sip_second_ts", end_iso) \
         .order("sip_second_ts") \
         .range(offset, offset + PAGE - 1) \
         .execute()
        page_data = page_res.data or []
        rows.extend(page_data)
        if len(page_data) < PAGE:
            break
        offset += PAGE

    # Aggregate to requested TF.
    if tf_seconds == 1:
        bars = [
            {
                'timestamp': r['sip_second_ts'],
                'open': r['open'], 'high': r['high'],
                'low': r['low'], 'close': r['close'],
                'volume': r['volume'], 'trade_count': r['trade_count'],
            }
            for r in rows
        ]
    else:
        bars = _aggregate_to_tf(rows, tf_seconds)

    return {
        'symbol': symbol.upper(),
        'window': {'start': start, 'end': end_iso},
        'tf_seconds': tf_seconds,
        'bar_count': len(bars),
        'raw_second_count': len(rows),
        'bars': bars,
    }


def _aggregate_to_tf(rows: list[dict], tf_seconds: int) -> list[dict]:
    """Roll 1-sec bars up to the requested TF.

    Same OHLCV aggregation rules used everywhere else: open = first
    bar's open, close = last bar's close, high/low = max/min,
    volume = sum.
    """
    if not rows:
        return []
    out: list[dict] = []
    current: dict | None = None
    current_bucket_start: int | None = None

    for r in rows:
        # sip_second_ts comes back as ISO string; parse to epoch second.
        ts_str = r['sip_second_ts']
        ts_dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        sec = int(ts_dt.timestamp())
        bucket_start = (sec // tf_seconds) * tf_seconds
        if current is None or bucket_start != current_bucket_start:
            if current is not None:
                out.append(current)
            current_bucket_start = bucket_start
            current = {
                'timestamp': datetime.fromtimestamp(
                    bucket_start, tz=timezone.utc).isoformat(),
                'open': r['open'],
                'high': r['high'],
                'low': r['low'],
                'close': r['close'],
                'volume': int(r['volume'] or 0),
                'trade_count': int(r['trade_count'] or 0),
            }
        else:
            if r['high'] > current['high']:
                current['high'] = r['high']
            if r['low'] < current['low']:
                current['low'] = r['low']
            current['close'] = r['close']
            current['volume'] += int(r['volume'] or 0)
            current['trade_count'] += int(r['trade_count'] or 0)
    if current is not None:
        out.append(current)
    return out
