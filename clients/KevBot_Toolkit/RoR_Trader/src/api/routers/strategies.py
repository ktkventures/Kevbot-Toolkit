"""Strategies router — CRUD + trades + forward test data."""

import copy
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Body, Query

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strategies", tags=["strategies"])


# =============================================================================
# CRUD
# =============================================================================

@router.get("")
def list_strategies(
    enrich: bool = Query(True, description="Include forward KPIs, sigma, status"),
    user=Depends(get_current_user),
):
    """Load all strategies for the current user, optionally enriched."""
    from db import USE_DB
    if USE_DB:
        from db import load_strategies_db
        strategies = load_strategies_db()
    else:
        import json, os
        path = os.path.join(os.path.dirname(__file__), '..', '..', 'strategies.json')
        strategies = []
        if os.path.exists(path):
            with open(path) as f:
                strategies = json.load(f)

    if enrich and strategies:
        import services as svc
        strategies = [svc.enrich_strategy(s) for s in strategies]

    return strategies


@router.post("")
def create_strategy(strategy: dict = Body(...), user=Depends(get_current_user)):
    """Create a new strategy (typically after a backtest).

    The frontend sends the full strategy dict including KPIs and stored_trades.
    """
    strategy['created_at'] = datetime.now(timezone.utc).isoformat()
    strategy['forward_testing'] = True
    strategy['forward_test_start'] = datetime.now(timezone.utc).isoformat()

    if 'confluence' in strategy and isinstance(strategy['confluence'], set):
        strategy['confluence'] = list(strategy['confluence'])

    from db import USE_DB
    if USE_DB:
        from db import save_strategy_db
        result = save_strategy_db(strategy)
        return result if result else {"status": "saved"}

    # JSON fallback
    import json, os
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'strategies.json')
    strategies = []
    if os.path.exists(path):
        with open(path) as f:
            strategies = json.load(f)
    strategy['id'] = max((s.get('id', 0) for s in strategies), default=0) + 1
    strategies.append(strategy)
    with open(path, 'w') as f:
        json.dump(strategies, f, indent=2)
    return strategy


@router.get("/{strategy_id}")
def get_strategy(strategy_id: int, user=Depends(get_current_user)):
    """Get a single strategy by ID, enriched with forward KPIs and sigma."""
    from db import USE_DB
    import services as svc
    if USE_DB:
        from db import get_strategy_by_id_db
        strat = get_strategy_by_id_db(strategy_id)
        if strat is None:
            raise HTTPException(status_code=404, detail="Strategy not found")
        return svc.enrich_strategy(strat, full_compute=True)

    # JSON fallback
    for s in list_strategies(enrich=False, user=user):
        if s.get('id') == strategy_id:
            return svc.enrich_strategy(s, full_compute=True)
    raise HTTPException(status_code=404, detail="Strategy not found")


@router.put("/{strategy_id}")
def update_strategy(
    strategy_id: int,
    updated: dict = Body(...),
    user=Depends(get_current_user),
):
    """Update an existing strategy."""
    from db import USE_DB
    if USE_DB:
        from db import get_strategy_by_id_db, update_strategy_db
        existing = get_strategy_by_id_db(strategy_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Strategy not found")

        # Merge: keep id, created_at from existing
        updated['id'] = strategy_id
        updated['updated_at'] = datetime.now(timezone.utc).isoformat()
        if 'created_at' not in updated:
            updated['created_at'] = existing.get('created_at')
        if 'confluence' in updated and isinstance(updated['confluence'], set):
            updated['confluence'] = list(updated['confluence'])

        update_strategy_db(strategy_id, updated)
        return {"status": "updated"}

    raise HTTPException(status_code=501, detail="JSON update not implemented via API")


@router.delete("/{strategy_id}")
def delete_strategy(strategy_id: int, user=Depends(get_current_user)):
    """Delete a strategy by ID."""
    from db import USE_DB
    if USE_DB:
        from db import delete_strategy_db
        deleted = delete_strategy_db(strategy_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Strategy not found")
        return {"status": "deleted"}

    # JSON fallback
    import json, os
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'strategies.json')
    if os.path.exists(path):
        with open(path) as f:
            strategies = json.load(f)
        original_len = len(strategies)
        strategies = [s for s in strategies if s.get('id') != strategy_id]
        if len(strategies) < original_len:
            with open(path, 'w') as f:
                json.dump(strategies, f, indent=2)
            return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Strategy not found")


@router.post("/{strategy_id}/duplicate")
def duplicate_strategy(strategy_id: int, user=Depends(get_current_user)):
    """Duplicate a strategy. Creates new copy with forward testing disabled."""
    from db import USE_DB
    if USE_DB:
        from db import get_strategy_by_id_db, save_strategy_db
        source = get_strategy_by_id_db(strategy_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Strategy not found")

        new_strategy = copy.deepcopy(source)
        new_strategy.pop('id', None)
        new_strategy['created_at'] = datetime.now(timezone.utc).isoformat()
        new_strategy['name'] = source['name'] + " (Copy)"
        new_strategy['forward_testing'] = False
        new_strategy.pop('forward_test_start', None)
        new_strategy.pop('updated_at', None)
        if 'equity_curve_data' in new_strategy and new_strategy['equity_curve_data']:
            new_strategy['equity_curve_data']['boundary_index'] = None

        result = save_strategy_db(new_strategy)
        return result if result else {"status": "duplicated"}

    raise HTTPException(status_code=501, detail="JSON duplicate not implemented via API")


@router.post("/bulk-delete")
def bulk_delete(
    strategy_ids: list[int] = Body(...),
    user=Depends(get_current_user),
):
    """Delete multiple strategies at once."""
    from db import USE_DB
    if not USE_DB:
        raise HTTPException(status_code=501, detail="Bulk delete requires DB mode")

    from db import delete_strategy_db
    deleted = 0
    for sid in strategy_ids:
        if delete_strategy_db(sid):
            deleted += 1
    return {"status": "deleted", "count": deleted}


# =============================================================================
# TRADE DATA
# =============================================================================

@router.get("/{strategy_id}/trades")
def get_strategy_trades(
    strategy_id: int,
    use_stored: bool = Query(True, description="Use stored_trades fast path when available"),
    user=Depends(get_current_user),
):
    """Get trades for a strategy.

    By default uses stored_trades (instant). Falls back to full backtest
    computation if stored_trades are not available.
    """
    strat = _get_or_404(strategy_id, user)

    # Fast path: return stored trades
    if use_stored and strat.get('stored_trades'):
        import services as svc
        trades_df = svc.trades_df_from_stored(strat['stored_trades'])
        from api.services.backtest_service import _serialize_trades
        return _serialize_trades(trades_df)

    # Slow path: full computation
    import services as svc
    trades_df = svc.get_strategy_trades(strat)
    from api.services.backtest_service import _serialize_trades
    return _serialize_trades(trades_df)


@router.get("/{strategy_id}/forward-test")
def get_forward_test_data(strategy_id: int, user=Depends(get_current_user)):
    """Get forward test split: backtest trades, forward trades, boundary date."""
    strat = _get_or_404(strategy_id, user)

    if not strat.get('forward_testing') or not strat.get('forward_test_start'):
        return {
            "backtest_trades": [],
            "forward_trades": [],
            "forward_test_start": None,
        }

    import services as svc
    from api.services.backtest_service import _serialize_trades

    _, bt, fw, fwd_start = svc.prepare_forward_test_data(strat)
    return {
        "backtest_trades": _serialize_trades(bt),
        "forward_trades": _serialize_trades(fw),
        "forward_test_start": fwd_start.isoformat() if fwd_start else None,
    }


@router.get("/{strategy_id}/kpis")
def get_strategy_kpis(strategy_id: int, user=Depends(get_current_user)):
    """Get KPIs for a strategy (from stored kpis or computed)."""
    strat = _get_or_404(strategy_id, user)

    # Return stored KPIs if available
    if strat.get('kpis'):
        return {"kpis": strat['kpis'], "secondary_kpis": None}

    # Compute from trades
    import services as svc
    trades_df = svc.get_strategy_trades(strat)
    kpis = svc.calculate_kpis(trades_df)
    secondary = svc.calculate_secondary_kpis(trades_df, kpis)
    return {"kpis": kpis, "secondary_kpis": secondary}


# =============================================================================
# TRIGGER ANALYSIS
# =============================================================================

@router.get("/{strategy_id}/trigger-analysis")
def get_trigger_analysis(strategy_id: int, user=Depends(get_current_user)):
    """Per-trigger and per-exit-reason analysis for the Confluence Analysis tab.

    Returns:
    - confluence_groups: Groups used by this strategy
    - exit_breakdown: KPIs grouped by exit_reason
    - trade_distribution: Win/loss counts by exit reason
    """
    strat = _get_or_404(strategy_id, user)

    import services as svc
    import pandas as pd

    # Get confluence groups used
    confluence_ids = strat.get("confluence", [])
    entry_trigger = strat.get("entry_trigger") or strat.get("entry_trigger_confluence_id") or "--"
    exit_triggers = strat.get("exit_trigger_confluence_ids") or strat.get("exit_triggers") or []

    groups = []
    if confluence_ids:
        # Try to load pack names from templates
        try:
            from confluence_groups import TEMPLATES
            for cid in confluence_ids:
                tpl = TEMPLATES.get(cid)
                groups.append({
                    "id": cid,
                    "name": tpl["name"] if tpl else cid.replace("_", " ").title(),
                    "pack": cid,
                })
        except Exception:
            groups = [{"id": cid, "name": cid.replace("_", " ").title(), "pack": cid} for cid in confluence_ids]

    # Analyze trades by exit reason
    exit_breakdown = []
    trade_distribution = []

    stored = strat.get("stored_trades", [])
    if stored:
        trades_df = svc.trades_df_from_stored(stored)
        if len(trades_df) > 0 and "exit_reason" in trades_df.columns:
            for reason, group_df in trades_df.groupby("exit_reason"):
                if reason == "open":
                    continue
                wins = int((group_df["r_multiple"] > 0).sum())
                losses = int((group_df["r_multiple"] <= 0).sum())
                total = len(group_df)
                total_r = float(group_df["r_multiple"].sum())
                avg_r = float(group_df["r_multiple"].mean())
                win_rate = wins / total * 100 if total > 0 else 0

                exit_breakdown.append({
                    "exit_reason": str(reason),
                    "trades": total,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": round(win_rate, 1),
                    "total_r": round(total_r, 4),
                    "avg_r": round(avg_r, 4),
                    "best_trade": round(float(group_df["r_multiple"].max()), 4) if total > 0 else 0,
                    "worst_trade": round(float(group_df["r_multiple"].min()), 4) if total > 0 else 0,
                })

                trade_distribution.append({
                    "exit_reason": str(reason),
                    "wins": wins,
                    "losses": losses,
                })

    return {
        "confluence_groups": groups,
        "entry_trigger": entry_trigger,
        "exit_triggers": exit_triggers if isinstance(exit_triggers, list) else [exit_triggers],
        "exit_breakdown": exit_breakdown,
        "trade_distribution": trade_distribution,
    }


# =============================================================================
# REFRESH / UPDATE DATA
# =============================================================================

@router.post("/{strategy_id}/refresh")
def refresh_strategy(strategy_id: int, user=Depends(get_current_user)):
    """Refresh a strategy's stored_trades and KPIs with current market data.

    Loads fresh bars from Polygon, runs the full backtest, updates stored_trades
    and KPIs in the database. This is the equivalent of Streamlit's 'Update Data' button.
    """
    strat = _get_or_404(strategy_id, user)
    import services as svc

    try:
        # Get full trades (backtest + forward)
        all_trades = svc.get_strategy_trades(strat)

        if len(all_trades) == 0:
            return {"status": "no_trades", "trades": 0}

        # Extract minimal trade records for storage
        stored = []
        for _, row in all_trades.iterrows():
            record = {}
            for col in all_trades.columns:
                val = row[col]
                if hasattr(val, 'isoformat'):
                    record[col] = val.isoformat()
                elif hasattr(val, 'item'):  # numpy types
                    record[col] = val.item()
                elif isinstance(val, set):
                    record[col] = list(val)
                else:
                    record[col] = val
            stored.append(record)

        # Compute KPIs
        trading_days = svc.count_trading_days(all_trades) if hasattr(all_trades, 'index') else 1
        kpis = svc.calculate_kpis(all_trades, total_trading_days=trading_days)

        # Build equity curve data
        from api.services.backtest_service import _build_equity_curve
        eq_data = _build_equity_curve(all_trades)
        boundary_index = None
        if strat.get('forward_test_start'):
            fwd_start = datetime.fromisoformat(strat['forward_test_start'])
            bt_portion, _ = svc.split_trades_at_boundary(all_trades, fwd_start)
            boundary_index = len(bt_portion) if len(bt_portion) > 0 else None

        equity_curve_data = {
            "exit_times": [p.get("timestamp", "") for p in eq_data],
            "cumulative_r": [p.get("cumulative_r", 0) for p in eq_data],
            "boundary_index": boundary_index,
        }

        # Update in database
        from db import USE_DB
        if USE_DB:
            from db import update_strategy_db
            update_strategy_db(strategy_id, {
                **strat,
                "stored_trades": stored,
                "kpis": kpis,
                "equity_curve_data": equity_curve_data,
                "data_refreshed_at": datetime.now(timezone.utc).isoformat(),
            })

        return {
            "status": "refreshed",
            "trades": len(stored),
            "kpis": kpis,
        }
    except Exception as e:
        logger.exception("Strategy refresh failed for %s", strategy_id)
        raise HTTPException(status_code=500, detail=f"Refresh failed: {str(e)}")


# =============================================================================
# HELPERS
# =============================================================================

def _get_or_404(strategy_id: int, user) -> dict:
    """Load strategy or raise 404."""
    from db import USE_DB
    if USE_DB:
        from db import get_strategy_by_id_db
        strat = get_strategy_by_id_db(strategy_id)
    else:
        strat = None
        import json, os
        path = os.path.join(os.path.dirname(__file__), '..', '..', 'strategies.json')
        if os.path.exists(path):
            with open(path) as f:
                for s in json.load(f):
                    if s.get('id') == strategy_id:
                        strat = s
                        break

    if strat is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return strat
