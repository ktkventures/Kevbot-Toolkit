"""Dashboard router — aggregated summary data for the dashboard page."""

import logging
from collections import defaultdict
from datetime import datetime

from fastapi import APIRouter, Depends

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary")
def get_dashboard_summary(user=Depends(get_current_user)):
    """Get aggregated dashboard data: strategy count, portfolio count, KPIs, positions.

    This endpoint aggregates from existing data — no heavy computation.
    """
    from db import USE_DB

    strategies = []
    portfolios = []

    if USE_DB:
        from db import load_strategies_db, load_portfolios_db
        strategies = load_strategies_db() or []
        portfolios = load_portfolios_db() or []
    else:
        import json, os
        strat_path = os.path.join(os.path.dirname(__file__), '..', '..', 'strategies.json')
        port_path = os.path.join(os.path.dirname(__file__), '..', '..', 'portfolios.json')
        if os.path.exists(strat_path):
            with open(strat_path) as f:
                strategies = json.load(f)
        if os.path.exists(port_path):
            with open(port_path) as f:
                portfolios = json.load(f)

    # Aggregate KPIs from strategies
    total_r = 0.0
    total_trades = 0
    win_counts = 0
    monitored_count = 0

    for s in strategies:
        kpis = s.get('kpis', {})
        total_r += kpis.get('total_r', 0)
        trades = kpis.get('total_trades', 0)
        total_trades += trades
        if kpis.get('win_rate', 0) > 0 and trades > 0:
            win_counts += int(trades * kpis['win_rate'] / 100)
        if s.get('alert_tracking_enabled'):
            monitored_count += 1

    avg_win_rate = (win_counts / total_trades * 100) if total_trades > 0 else 0

    return {
        "strategy_count": len(strategies),
        "portfolio_count": len(portfolios),
        "monitored_count": monitored_count,
        "total_trades": total_trades,
        "total_r": round(total_r, 2),
        "avg_win_rate": round(avg_win_rate, 1),
        "strategies": [
            {
                "id": s.get("id"),
                "name": s.get("name"),
                "symbol": s.get("symbol"),
                "direction": s.get("direction"),
                "kpis": s.get("kpis", {}),
                "alert_tracking_enabled": s.get("alert_tracking_enabled", False),
            }
            for s in strategies
        ],
    }


@router.get("/health")
def get_dashboard_health(user=Depends(get_current_user)):
    """Get per-strategy health indicators (sigma deviation, status)."""
    from db import USE_DB
    import services as svc

    strategies = []
    if USE_DB:
        from db import load_strategies_db
        strategies = load_strategies_db() or []

    health_items = []
    for s in strategies:
        enriched = svc.enrich_strategy(s)
        kpis = s.get('kpis', {})
        health_items.append({
            "id": s.get("id"),
            "name": s.get("name", "--"),
            "symbol": s.get("symbol", "--"),
            "deviation_sd": enriched.get("sigma_fwd"),
            "status": enriched.get("status", "Insufficient Data"),
            "alert_trades": enriched.get("forward_kpis", {}).get("trades", 0) if enriched.get("forward_kpis") else 0,
            "expected_r": kpis.get("total_r", 0),
            "actual_r": enriched.get("forward_kpis", {}).get("total_r", 0) if enriched.get("forward_kpis") else 0,
        })

    return {"strategies": health_items}


@router.get("/positions")
def get_dashboard_positions(user=Depends(get_current_user)):
    """Get open positions from engine state."""
    from db import USE_DB

    if not USE_DB:
        return {"positions": []}

    try:
        from db import load_engine_state_db
        state = load_engine_state_db()
        if not state:
            return {"positions": []}

        positions = []
        for strat_id, pos_data in state.items():
            if isinstance(pos_data, dict) and pos_data.get("status") == "IN_POSITION":
                positions.append({
                    "strategy_id": strat_id,
                    "symbol": pos_data.get("symbol", "--"),
                    "direction": pos_data.get("direction", "--"),
                    "entry_price": pos_data.get("entry_price"),
                    "stop_price": pos_data.get("stop_price"),
                    "entry_time": pos_data.get("entry_time"),
                    "status": "IN_POSITION",
                })
        return {"positions": positions}
    except Exception:
        return {"positions": []}


@router.get("/activity")
def get_dashboard_activity(user=Depends(get_current_user)):
    """Get recent activity feed (alerts, trades, system events)."""
    from db import USE_DB

    if not USE_DB:
        return {"items": []}

    try:
        from db import load_alerts_db
        alerts = load_alerts_db() or []

        # Convert recent alerts to activity items
        items = []
        for alert in alerts[:20]:  # Last 20
            data = alert.get('data', {}) if isinstance(alert.get('data'), dict) else {}
            items.append({
                "time": alert.get("timestamp", "--"),
                "type": "entry" if "entry" in str(alert.get("type", "")).lower() else "exit" if "exit" in str(alert.get("type", "")).lower() else "system",
                "message": f"{data.get('symbol', alert.get('symbol', '--'))}: {alert.get('type', 'Alert')}",
            })
        return {"items": items}
    except Exception:
        return {"items": []}


@router.get("/equity-curve")
def get_dashboard_equity_curve(user=Depends(get_current_user)):
    """Get aggregated equity curve across all strategies.

    Merges each strategy's equity_curve_data (exit_times + cumulative_r)
    into a single time-ordered cumulative R series.
    """
    from db import USE_DB

    if not USE_DB:
        return {"points": []}

    try:
        from db import load_strategies_db
        strategies = load_strategies_db() or []

        # Collect all (time, r_value) data points from every strategy
        all_points: list[tuple[str, float]] = []
        for s in strategies:
            ecd = s.get("equity_curve_data")
            if not ecd:
                continue
            exit_times = ecd.get("exit_times", [])
            cumulative_r = ecd.get("cumulative_r", [])
            if not exit_times or not cumulative_r:
                continue
            # Convert cumulative per-strategy to incremental, then we re-accumulate globally
            prev = 0.0
            for t, cr in zip(exit_times, cumulative_r):
                delta = cr - prev
                prev = cr
                all_points.append((t, delta))

        if not all_points:
            return {"points": []}

        # Sort by timestamp and re-accumulate
        all_points.sort(key=lambda x: x[0])
        cumulative = 0.0
        result = []
        for t, delta in all_points:
            cumulative += delta
            result.append({"time": t, "value": round(cumulative, 4)})

        return {"points": result}
    except Exception as e:
        logger.exception("Error computing dashboard equity curve: %s", e)
        return {"points": []}


@router.get("/daily-pnl")
def get_dashboard_daily_pnl(user=Depends(get_current_user)):
    """Get daily P&L across all strategies.

    Aggregates incremental R from all strategies' equity curves,
    grouped by calendar day.
    """
    from db import USE_DB

    if not USE_DB:
        return {"days": []}

    try:
        from db import load_strategies_db
        strategies = load_strategies_db() or []

        # Collect incremental R with timestamps
        day_totals: dict[str, float] = defaultdict(float)
        for s in strategies:
            ecd = s.get("equity_curve_data")
            if not ecd:
                continue
            exit_times = ecd.get("exit_times", [])
            cumulative_r = ecd.get("cumulative_r", [])
            if not exit_times or not cumulative_r:
                continue
            prev = 0.0
            for t, cr in zip(exit_times, cumulative_r):
                delta = cr - prev
                prev = cr
                try:
                    day = t[:10]  # "YYYY-MM-DD"
                    day_totals[day] += delta
                except (TypeError, IndexError):
                    continue

        if not day_totals:
            return {"days": []}

        # Sort by date and return
        result = [
            {"day": day, "value": round(val, 4)}
            for day, val in sorted(day_totals.items())
        ]

        return {"days": result}
    except Exception as e:
        logger.exception("Error computing dashboard daily P&L: %s", e)
        return {"days": []}
