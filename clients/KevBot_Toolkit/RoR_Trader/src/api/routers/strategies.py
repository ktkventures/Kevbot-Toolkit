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
        return svc.enrich_strategy(strat)

    # JSON fallback
    for s in list_strategies(enrich=False, user=user):
        if s.get('id') == strategy_id:
            return svc.enrich_strategy(s)
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
