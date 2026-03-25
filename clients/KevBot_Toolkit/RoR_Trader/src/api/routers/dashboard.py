"""Dashboard router — aggregated summary data for the dashboard page."""

import logging

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
