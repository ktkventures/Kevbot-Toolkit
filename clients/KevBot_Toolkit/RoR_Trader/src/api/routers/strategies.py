"""Strategies router — CRUD + trades + forward test data."""

import copy
import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Body, Query

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strategies", tags=["strategies"])


def _sanitize_json(obj):
    """Recursively replace NaN/Inf floats with None so the response survives
    Starlette's json.dumps. KPIs that divide by zero (profit_factor when
    losses=0, etc.) otherwise blow up the whole endpoint with a 500."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


# =============================================================================
# HELPERS
# =============================================================================

def _filter_trades_by_date_range(strategy: dict, date_range: str) -> dict:
    """Filter stored_trades by date range, returning a shallow copy of the strategy."""
    trades = strategy.get('stored_trades', [])
    if not trades:
        return strategy

    now = datetime.now(timezone.utc)
    fwd_start = strategy.get('forward_test_start')

    filtered = list(trades)  # default: all trades

    if date_range == 'Last 7 Days':
        cutoff = now - timedelta(days=7)
        filtered = [t for t in trades if _trade_after(t, cutoff)]
    elif date_range == 'Last 30 Days':
        cutoff = now - timedelta(days=30)
        filtered = [t for t in trades if _trade_after(t, cutoff)]
    elif date_range == 'Last 90 Days':
        cutoff = now - timedelta(days=90)
        filtered = [t for t in trades if _trade_after(t, cutoff)]
    elif date_range == 'Backtest Only':
        if fwd_start:
            boundary = datetime.fromisoformat(fwd_start)
            if boundary.tzinfo is None:
                boundary = boundary.replace(tzinfo=timezone.utc)
            filtered = [t for t in trades if _trade_before(t, boundary)]
    elif date_range == 'Forward Only':
        if fwd_start:
            boundary = datetime.fromisoformat(fwd_start)
            if boundary.tzinfo is None:
                boundary = boundary.replace(tzinfo=timezone.utc)
            filtered = [t for t in trades if _trade_after(t, boundary)]
        else:
            filtered = []  # No forward test start = no forward trades

    result = dict(strategy)
    result['stored_trades'] = filtered
    return result


def _trade_after(trade: dict, cutoff: datetime) -> bool:
    """Check if a trade's entry_time is after a cutoff datetime."""
    et = trade.get('entry_time')
    if not et:
        return True
    try:
        t = datetime.fromisoformat(str(et))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t >= cutoff
    except (ValueError, TypeError):
        return True


def _trade_before(trade: dict, cutoff: datetime) -> bool:
    """Check if a trade's entry_time is before a cutoff datetime."""
    et = trade.get('entry_time')
    if not et:
        return True
    try:
        t = datetime.fromisoformat(str(et))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t < cutoff
    except (ValueError, TypeError):
        return True


# =============================================================================
# CRUD
# =============================================================================

@router.get("")
def list_strategies(
    enrich: bool = Query(False, description="Include forward KPIs, sigma, status"),
    user=Depends(get_current_user),
):
    """Load all strategies for the current user, optionally enriched.

    Default path (enrich=False) uses load_strategies_db_lite() — excludes
    `stored_trades` and `live_executions` columns. This is what the frontend
    Strategies list page consumes (doesn't use enriched fields). Huge
    Supabase load reduction on fat strategies.

    Opt-in enrichment (?enrich=true) falls back to the full load so forward
    KPIs / sigma / status can be computed. Used by anything that needs those.
    """
    from db import USE_DB
    if USE_DB:
        if enrich:
            from db import load_strategies_db
            strategies = load_strategies_db()
        else:
            from db import load_strategies_db_lite
            strategies = load_strategies_db_lite()
    else:
        import json, os
        path = os.path.join(os.path.dirname(__file__), '..', '..', 'strategies.json')
        strategies = []
        if os.path.exists(path):
            with open(path) as f:
                strategies = json.load(f)

    if enrich and strategies:
        import services as svc
        enriched = []
        for s in strategies:
            try:
                enriched.append(svc.enrich_strategy(s))
            except Exception as e:
                logger.warning(
                    "[LIST] Failed to enrich strategy %s (%s): %s",
                    s.get('id', '?'), s.get('name', '?'), e)
                # Return the raw strategy so the list still loads
                enriched.append(s)
        strategies = enriched

    # Augment every strategy with three cheap count fields so the list-card
    # subtitle can show them alongside backtest counts:
    #   - portfolio_count: how many portfolios include this strategy
    #   - forward_trades_count: trades after forward_test_start (= post-save
    #     algo activity)
    #   - alert_trades_count: total alert rows for this strategy (= live fires)
    if USE_DB and strategies:
        _augment_with_counts(strategies)

    # Strategy Health Badge: attach `health` to every strategy so the list
    # card can render the badge without a per-strategy round trip. Pure
    # function — uses the count fields just attached above.
    _attach_health(strategies)

    return _sanitize_json(strategies)


def _attach_health(strategies: list) -> None:
    """Attach a `health` dict to each strategy via the pure compute_strategy_health
    function. Uses the count fields populated by _augment_with_counts.
    Pack registry is loaded once and reused.
    """
    try:
        import pack_registry
        # scan_and_load_all is cached after first call; safe to invoke here
        pack_registry.scan_and_load_all()
        slugs = set(pack_registry.get_registered_packs().keys())
    except Exception:
        slugs = None
    from services import compute_strategy_health
    for s in strategies:
        try:
            health = compute_strategy_health(
                s,
                n_alerts=s.get('alert_trades_count'),
                n_trades=s.get('forward_trades_count'),
                pack_registry_slugs=slugs,
            )
            s['health'] = health.to_dict()
        except Exception as e:
            logger.warning(
                "[HEALTH] compute_strategy_health failed for sid %s: %s",
                s.get('id', '?'), e)
            s['health'] = {'severity': 'healthy', 'issues': []}


def _augment_with_counts(strategies: list) -> None:
    """Attach portfolio_count / forward_trades_count / alert_trades_count.

    Uses PostgREST count-exact HEAD-style requests (limit=0, Prefer=count=exact)
    — returns only the count header, not the row bodies, so each request is
    tiny. Parallelized via ThreadPoolExecutor to keep total list-endpoint
    latency under a second for typical user strategy counts.

    Mutates `strategies` in place. Failures fall back to 0 so the list
    still renders if any individual count request errors.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from db import (
        get_admin_client, get_current_user_id, load_portfolios_admin,
    )

    user_id = get_current_user_id()

    # Portfolio count — one fetch of portfolios, count in Python.
    port_counts: dict[int, int] = {}
    try:
        ports = load_portfolios_admin(user_id) if user_id else []
        for p in ports:
            for alloc in (p.get('strategies') or []):
                sid = alloc.get('strategy_id')
                if sid is None:
                    continue
                port_counts[int(sid)] = port_counts.get(int(sid), 0) + 1
    except Exception as e:
        logger.warning("[LIST] Portfolio count failed: %s", e)

    client = get_admin_client()

    def _count_alerts(sid: int) -> tuple[int, int]:
        try:
            r = client.table('alerts') \
                .select('id', count='exact') \
                .eq('strategy_id', sid) \
                .limit(0) \
                .execute()
            return sid, (r.count or 0)
        except Exception:
            return sid, 0

    def _count_forward_trades(sid: int, fwd_start) -> tuple[int, int]:
        if not fwd_start:
            return sid, 0
        try:
            r = client.table('trades') \
                .select('id', count='exact') \
                .eq('strategy_id', sid) \
                .gte('entry_fill_ts', fwd_start) \
                .limit(0) \
                .execute()
            return sid, (r.count or 0)
        except Exception:
            return sid, 0

    alert_counts: dict[int, int] = {}
    fwd_counts: dict[int, int] = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        alert_futures = {ex.submit(_count_alerts, s['id']): s['id'] for s in strategies if s.get('id')}
        fwd_futures = {
            ex.submit(_count_forward_trades, s['id'], s.get('forward_test_start')): s['id']
            for s in strategies if s.get('id')
        }
        for fut in as_completed(alert_futures):
            sid, n = fut.result()
            alert_counts[sid] = n
        for fut in as_completed(fwd_futures):
            sid, n = fut.result()
            fwd_counts[sid] = n

    for s in strategies:
        sid = s.get('id')
        if sid is None:
            continue
        s['portfolio_count'] = port_counts.get(int(sid), 0)
        s['alert_trades_count'] = alert_counts.get(sid, 0)
        s['forward_trades_count'] = fwd_counts.get(sid, 0)


@router.post("")
def create_strategy(strategy: dict = Body(...), user=Depends(get_current_user)):
    """Create a new strategy (typically after a backtest).

    The frontend sends the full strategy dict including KPIs and stored_trades.
    """
    strategy['created_at'] = datetime.now(timezone.utc).isoformat()
    strategy['forward_testing'] = True
    strategy['forward_test_start'] = datetime.now(timezone.utc).isoformat()

    # Strategy Health Badge: stamp data_source if the caller didn't already
    # (e.g. Mass Builder paths set 'rapid' explicitly). Frontends that ran
    # a Hi-Fi backtest before saving should send `data_source='hifi'` in
    # the body; full backtest defaults work.
    strategy.setdefault('data_source', 'full')
    strategy.setdefault(
        'kpis_computed_at', datetime.now(timezone.utc).isoformat())
    strategy.setdefault('kpis_stale_since', None)

    if 'confluence' in strategy and isinstance(strategy['confluence'], set):
        strategy['confluence'] = list(strategy['confluence'])

    # Resolve pack IDs → inline configs so the strategy is self-contained for the engine
    from api.services.backtest_service import (
        _resolve_stop_from_pack, _resolve_target_from_pack, _resolve_time_exit_from_pack,
    )
    if strategy.get('stop_loss_pack_id') and not strategy.get('stop_config'):
        strategy['stop_config'] = _resolve_stop_from_pack(strategy['stop_loss_pack_id'])
        if strategy['stop_config'] and 'exec_type' not in strategy['stop_config']:
            strategy['stop_config']['exec_type'] = strategy.get('stop_exec_type', 'L')
    if strategy.get('take_profit_pack_id') and not strategy.get('target_config'):
        strategy['target_config'] = _resolve_target_from_pack(strategy['take_profit_pack_id'])
        if strategy.get('target_config') and 'exec_type' not in strategy['target_config']:
            strategy['target_config']['exec_type'] = strategy.get('target_exec_type', 'L')
    if strategy.get('time_exit_pack_id') and not strategy.get('time_exit_config'):
        strategy['time_exit_config'] = _resolve_time_exit_from_pack(strategy['time_exit_pack_id'])

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
def get_strategy(strategy_id: int, date_range: str = "Strategy Default", user=Depends(get_current_user)):
    """Get a single strategy by ID, enriched with forward KPIs and sigma.

    date_range: 'Strategy Default' | 'All Data' | 'Last 7 Days' | 'Last 30 Days' |
                'Last 90 Days' | 'Backtest Only' | 'Forward Only'
    """
    from db import USE_DB
    import services as svc
    if USE_DB:
        from db import get_strategy_by_id_db
        strat = get_strategy_by_id_db(strategy_id)
        if strat is None:
            raise HTTPException(status_code=404, detail="Strategy not found")
    else:
        strat = None
        for s in list_strategies(enrich=False, user=user):
            if s.get('id') == strategy_id:
                strat = s
                break
        if strat is None:
            raise HTTPException(status_code=404, detail="Strategy not found")

    # Apply date range filter to stored_trades before enrichment
    if date_range and date_range not in ("Strategy Default", "All Data"):
        strat = _filter_trades_by_date_range(strat, date_range)
        # Recompute KPIs from filtered trades so date range actually affects displayed metrics
        filtered_trades = strat.get('stored_trades', [])
        if filtered_trades:
            import services as _svc
            try:
                trades_df = _svc.trades_df_from_stored(filtered_trades)
                strat['kpis'] = _svc.calculate_kpis(trades_df)
                logger.info("[DETAIL] Recomputed KPIs for strategy %s with date_range=%s (%d trades)", strategy_id, date_range, len(filtered_trades))
            except Exception as e:
                logger.warning("[DETAIL] Failed to recompute KPIs for date_range=%s: %s", date_range, e)
        else:
            strat['kpis'] = {}

    try:
        # full_compute=False: rely on stored_trades being kept fresh by the
        # worker's bar-close hook (forward_test_service.recompute_and_persist_
        # stored_trades) instead of re-running the full backtest on every
        # detail load. full_compute=True was hanging Strategy Detail for
        # brand-new strategies (no forward trades yet → triggered a full
        # Polygon+backtest recompute per load, 30s+ per hit). The stored
        # path gives correct data for any monitored strategy; unmonitored
        # strategies with no forward trades return None and the UI shows
        # "Insufficient forward data" which is accurate.
        enriched = svc.enrich_strategy(strat, full_compute=False)
    except Exception as e:
        logger.warning("[DETAIL] Failed to enrich strategy %s: %s", strategy_id, e)
        enriched = strat

    # Strategy Health Badge: attach health to the detail response too. Same
    # pure function as list endpoint; uses already-computed counts when
    # available on the enriched dict.
    try:
        import pack_registry
        pack_registry.scan_and_load_all()
        slugs = set(pack_registry.get_registered_packs().keys())
        from services import compute_strategy_health
        health = compute_strategy_health(
            enriched,
            n_alerts=enriched.get('alert_trades_count'),
            n_trades=enriched.get('forward_trades_count'),
            pack_registry_slugs=slugs,
        )
        enriched['health'] = health.to_dict()
    except Exception as _e:
        logger.warning(
            "[HEALTH] detail compute failed for sid %s: %s", strategy_id, _e)
        enriched['health'] = {'severity': 'healthy', 'issues': []}

    return _sanitize_json(enriched)


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

        # Strategy Health Badge: if any backtest-affecting field changed,
        # stamp kpis_stale_since so the badge surfaces the staleness until
        # a re-backtest clears it. The caller can override by sending
        # `kpis_stale_since=null` along with fresh KPIs (e.g. when this
        # update IS a backtest result).
        if 'kpis_stale_since' not in updated:
            from services import HEALTH_BACKTEST_AFFECTING_FIELDS
            changed_affecting = False
            for field in HEALTH_BACKTEST_AFFECTING_FIELDS:
                old_val = existing.get(field)
                new_val = updated.get(field, old_val)
                if new_val != old_val:
                    changed_affecting = True
                    break
            if changed_affecting:
                updated['kpis_stale_since'] = datetime.now(
                    timezone.utc).isoformat()

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


@router.patch("/{strategy_id}/forward-test-start")
def set_forward_test_start(
    strategy_id: int,
    body: dict = Body(...),
    user=Depends(get_current_user),
):
    """Admin override for ``forward_test_start``.

    Unblocks the "recreate under current schema + restore original start
    date" workflow. Expects body: ``{"forward_test_start": "<ISO8601>"}``
    or ``{"forward_test_start": null}`` to clear. No recompute is triggered
    — caller should hit /refresh afterward if they want stored_trades and
    equity_curve_data regenerated against the new boundary.
    """
    from db import USE_DB
    if not USE_DB:
        raise HTTPException(status_code=501, detail="DB mode required")

    raw = body.get('forward_test_start', ...)
    if raw is ...:
        raise HTTPException(
            status_code=400,
            detail="body must include 'forward_test_start' (ISO8601 string or null)",
        )

    # Validate input. Accept ISO8601 strings or explicit null/empty to clear.
    normalized: str | None
    if raw is None or raw == '':
        normalized = None
    elif isinstance(raw, str):
        try:
            # Parse then re-emit so we store a canonical representation.
            parsed = datetime.fromisoformat(raw.replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            normalized = parsed.isoformat()
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"invalid ISO8601 datetime: {raw!r}",
            )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"forward_test_start must be a string or null, got {type(raw).__name__}",
        )

    from db import get_strategy_by_id_db, update_strategy_db
    existing = get_strategy_by_id_db(strategy_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Strategy not found")

    # Partial update — ONLY touches forward_test_start. Preserves stored_trades,
    # equity_curve_data, kpis, config, etc. update_strategy_db handles the
    # config JSONB partial-update rules correctly as long as stored_trades
    # isn't in the payload (trades-table redirect won't fire, good).
    result = update_strategy_db(strategy_id, {
        'forward_test_start': normalized,
        'forward_testing': bool(normalized),
        'updated_at': datetime.now(timezone.utc).isoformat(),
    })
    if result is None:
        raise HTTPException(status_code=500, detail="Update failed")

    return {
        "status": "updated",
        "forward_test_start": normalized,
        "forward_testing": bool(normalized),
        "id": strategy_id,
    }


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

    # Slow path: full computation (can be slow — 10-30s for 1Min strategies)
    try:
        import services as svc
        trades_df = svc.get_strategy_trades(strat)
        from api.services.backtest_service import _serialize_trades
        return _serialize_trades(trades_df)
    except Exception as e:
        logger.exception("Failed to compute trades for strategy %s: %s", strategy_id, e)
        raise HTTPException(status_code=504, detail=f"Trade computation timed out or failed: {str(e)[:200]}")


@router.get("/{strategy_id}/forward-test")
def get_forward_test_data(strategy_id: int, user=Depends(get_current_user)):
    """Get forward test split: backtest trades, forward trades, boundary date.

    Uses the fast path whenever stored_trades exists: split the cached
    trades at forward_test_start in pure Python — no Polygon round-trip.
    Falls back to the full prepare_forward_test_data re-compute only when
    stored_trades is empty (first ever load / fresh strategy).
    """
    strat = _get_or_404(strategy_id, user)

    if not strat.get('forward_testing') or not strat.get('forward_test_start'):
        return {
            "backtest_trades": [],
            "forward_trades": [],
            "forward_test_start": None,
        }

    import services as svc
    from api.services.backtest_service import _serialize_trades

    stored = strat.get('stored_trades', [])
    fwd_start_dt = datetime.fromisoformat(strat['forward_test_start'])

    if stored:
        try:
            trades_df = svc.trades_df_from_stored(stored)
            bt, fw = svc.split_trades_at_boundary(trades_df, fwd_start_dt)
            return {
                "backtest_trades": _serialize_trades(bt),
                "forward_trades": _serialize_trades(fw),
                "forward_test_start": fwd_start_dt.isoformat(),
            }
        except Exception as e:
            logger.warning("[FORWARD-TEST] Fast split failed for strat %s: %s — "
                           "falling back to full compute",
                           strategy_id, e)

    # Fallback: no stored_trades yet (fresh strategy) or fast split errored.
    # prepare_forward_test_data loads Polygon bars + reruns the engine (slow).
    _, bt, fw, fwd_start = svc.prepare_forward_test_data(strat)
    return {
        "backtest_trades": _serialize_trades(bt),
        "forward_trades": _serialize_trades(fw),
        "forward_test_start": fwd_start.isoformat() if fwd_start else None,
    }


@router.get("/{strategy_id}/kpis")
def get_strategy_kpis(strategy_id: int, date_range: str = "Strategy Default", user=Depends(get_current_user)):
    """Get KPIs for a strategy (from stored kpis or computed). Supports date_range filtering."""
    strat = _get_or_404(strategy_id, user)
    import services as svc

    # Apply date range filter
    if date_range and date_range not in ("Strategy Default", "All Data"):
        strat = _filter_trades_by_date_range(strat, date_range)

    stored = strat.get('stored_trades', [])
    if stored:
        try:
            trades_df = svc.trades_df_from_stored(stored)
            primary_kpis = svc.calculate_kpis(trades_df)
            secondary = svc.calculate_secondary_kpis(trades_df, primary_kpis)
            return {"kpis": primary_kpis, "secondary_kpis": secondary}
        except Exception as e:
            logger.warning("[KPIs] Failed to compute from stored trades: %s", e)

    # No stored trades or computation failed — try live computation
    try:
        trades_df = svc.get_strategy_trades(strat)
        kpis = svc.calculate_kpis(trades_df)
        secondary = svc.calculate_secondary_kpis(trades_df, kpis)
        return {"kpis": kpis, "secondary_kpis": secondary}
    except Exception as e:
        logger.warning("[KPIs] Failed to compute: %s", e)
        return {"kpis": strat.get('kpis', {}), "secondary_kpis": None}


# =============================================================================
# CHART DATA (OHLCV + indicator overlays)
# =============================================================================

@router.get("/{strategy_id}/chart-data")
def get_strategy_chart_data(
    strategy_id: int,
    days: int = Query(None, description="Override data_days"),
    user=Depends(get_current_user),
):
    """Get OHLCV bars with strategy-relevant indicators, classified by pane type.

    Returns:
    - chart_data: OHLCV + indicator values per bar
    - overlay_indicators: indicators to plot ON the price chart (EMA, VWAP, UT Bot)
    - oscillator_indicators: indicators for separate panes below (MACD, RVOL)
    - heatmap_conditions: confluence conditions with per-bar met/unmet states
    """
    strat = _get_or_404(strategy_id, user)
    import services as svc
    import pandas as pd

    OVERLAY_TEMPLATES = {"ema_stack", "ema_price_position", "ema_price_position_v2", "vwap", "utbot", "utbot_v2"}
    OSCILLATOR_TEMPLATES = {"macd_line", "macd_histogram", "rvol"}

    try:
        # Extract required secondary timeframes from confluence records
        from data_loader import get_required_tfs_from_confluence, get_tf_from_label
        req_labels = get_required_tfs_from_confluence(strat.get('confluence', []))
        sec_tfs = tuple(sorted(get_tf_from_label(lbl) for lbl in req_labels if get_tf_from_label(lbl)))

        # If daily secondary TFs are needed, load more history for EMA warmup
        base_days = days or strat.get('data_days', 30)
        if any(tf == '1Day' for tf in sec_tfs):
            chart_days = max(base_days, 365)  # Need ~250 trading days for daily MACD warmup
        else:
            chart_days = base_days

        df = svc.prepare_data_with_indicators(
            strat['symbol'],
            days=chart_days,
            timeframe=strat.get('timeframe', '1Min'),
            session=strat.get('trading_session', 'RTH'),
            secondary_tfs=sec_tfs,
        )
        if len(df) == 0:
            return {"chart_data": [], "overlay_indicators": [], "oscillator_indicators": [], "heatmap_conditions": []}

        from confluence_groups import get_enabled_groups, get_template

        entry_conf_id = strat.get('entry_trigger_confluence_id') or ''
        exit_conf_ids = strat.get('exit_trigger_confluence_ids') or []
        confluence_records = list(strat.get('confluence', []))

        # Extract interpreter keys from confluence records
        confluence_interpreters = set()
        for record in confluence_records:
            if record.startswith("GEN-"):
                continue
            parts = record.split("-")
            if len(parts) >= 2:
                confluence_interpreters.add(parts[1])

        overlay_cols = []
        oscillator_cols = []

        for group in get_enabled_groups():
            gid_prefix = group.id + "_"
            is_relevant = (
                entry_conf_id.startswith(gid_prefix)
                or any(cid.startswith(gid_prefix) for cid in exit_conf_ids)
            )
            if not is_relevant:
                template = get_template(group.base_template)
                if template:
                    for ik in template.get("interpreters", []):
                        if ik in confluence_interpreters:
                            is_relevant = True
                            break
            if not is_relevant:
                continue

            template = get_template(group.base_template)
            if not template:
                continue

            raw_cols = template.get("indicator_columns", [])
            resolved = []
            for col in raw_cols:
                if col in ("ema_short", "ema_mid", "ema_long"):
                    p = group.parameters
                    resolved_name = f"ema_{p.get('short_period', 9)}" if col == "ema_short" else \
                                    f"ema_{p.get('mid_period', 21)}" if col == "ema_mid" else \
                                    f"ema_{p.get('long_period', 200)}"
                    if resolved_name in df.columns:
                        resolved.append(resolved_name)
                elif col in df.columns:
                    resolved.append(col)

            if group.base_template in OVERLAY_TEMPLATES:
                overlay_cols.extend(resolved)
            elif group.base_template in OSCILLATOR_TEMPLATES:
                oscillator_cols.extend(resolved)
            else:
                dt = template.get("display_type", "overlay")
                if dt == "oscillator":
                    oscillator_cols.extend(resolved)
                else:
                    overlay_cols.extend(resolved)

        # Deduplicate
        overlay_cols = list(dict.fromkeys(overlay_cols))
        oscillator_cols = list(dict.fromkeys(oscillator_cols))

        # Ghost overlay: for v2 L-type triggers whose level column ends in
        # `_prev` (e.g. utbot_stop_prev, ema_9_prev), include that column too.
        # The actual fill level is the PREVIOUS bar's indicator value; the
        # solid line shows the CURRENT bar. Plotting both lets the user see
        # where the `+` marker actually landed without optical drift.
        #
        # Only include `_prev` columns whose non-prev sibling is already in
        # overlay_cols — this scopes the ghost line to indicators the strategy
        # actually uses, not every `_prev` column that happens to exist in df.
        try:
            for base_col in list(overlay_cols):
                prev_col = f"{base_col}_prev"
                if prev_col in df.columns and prev_col not in overlay_cols:
                    overlay_cols.append(prev_col)
        except Exception as _e:
            logger.debug("ghost overlay inclusion failed: %s", _e)

        # Apply the same filtering + candle_color_column detection that the
        # /backtest endpoint uses. Without this, Strategy Detail renders
        # boolean/string pack columns as overlay lines and misses candle
        # coloring for packs like Swing 1-2-3.
        from api.services.backtest_service import classify_chart_indicators
        overlay_cols, oscillator_cols, candle_color_column = \
            classify_chart_indicators(df, overlay_cols, oscillator_cols, entry_conf_id)

        all_indicator_cols = overlay_cols + oscillator_cols
        if candle_color_column and candle_color_column not in all_indicator_cols:
            all_indicator_cols.append(candle_color_column)

        # Build heatmap condition data
        from data_loader import get_tf_label
        primary_tf = get_tf_label(strat.get('timeframe', '1Min')).lower()
        all_conditions = list(strat.get('confluence', [])) + list(strat.get('general_confluences', []))

        # Determine which conditions use CB fidelity
        cb_set = set(strat.get('cb_conditions', []))

        heatmap_conditions = []
        for record in all_conditions:
            # Strip [CB] suffix if present (used in confluence array)
            clean_record = record.replace('[CB]', '').replace('[PB]', '')
            parts = clean_record.split('-', 2)
            if len(parts) < 3:
                continue
            rec_tf, interp_key, needed_state = parts
            is_general = rec_tf == 'GEN'
            is_cross_tf = not is_general and rec_tf.lower() != primary_tf

            if is_general:
                col_name = f"GP_{interp_key}"
            elif is_cross_tf:
                col_name = f"{interp_key}__{rec_tf.lower()}"
            else:
                col_name = interp_key

            # Determine fidelity: CB if in cb_conditions set or has [CB] suffix
            is_cb = clean_record in cb_set or '[CB]' in record
            fidelity = 'CB' if is_cb else 'PB'

            heatmap_conditions.append({
                "label": f"{clean_record} [{fidelity}]",
                "column": col_name,
                "needed_state": needed_state,
                "fidelity": fidelity,
                "has_data": col_name in df.columns,
            })

        # Serialize: OHLCV + indicators + interpreter state columns for heatmap
        state_cols = [c["column"] for c in heatmap_conditions if c["has_data"]]
        from api.services.backtest_service import _serialize_chart_data

        # For state columns, serialize separately (they're categorical, not numeric)
        chart_data = _serialize_chart_data(df, all_indicator_cols)

        # Add interpreter state values to each bar for heatmap
        if state_cols:
            reset_df = df.reset_index()
            time_col = reset_df.columns[0]
            # Log what state columns contain for debugging
            for sc in state_cols:
                if sc in reset_df.columns:
                    unique_vals = reset_df[sc].dropna().unique()
                    logger.info("[CHART-DATA] heatmap state_col=%s, unique_values=%s (last 5: %s)",
                                sc, list(unique_vals[:10]),
                                list(reset_df[sc].tail(5).values))
                else:
                    logger.warning("[CHART-DATA] heatmap state_col=%s NOT FOUND in df", sc)
            for i, row in enumerate(chart_data):
                if i < len(reset_df):
                    r = reset_df.iloc[i]
                    for sc in state_cols:
                        val = r.get(sc)
                        row[f"_state_{sc}"] = str(val) if pd.notna(val) else None

        return {
            "chart_data": chart_data,
            "overlay_indicators": overlay_cols,
            "oscillator_indicators": oscillator_cols,
            "heatmap_conditions": heatmap_conditions,
            "candle_color_column": candle_color_column,
        }
    except Exception as e:
        logger.exception("Failed to compute chart data for strategy %s: %s", strategy_id, e)
        raise HTTPException(status_code=504, detail=f"Chart data computation failed: {str(e)[:200]}")


# =============================================================================
# CONFLUENCE CONDITION CHART DATA
# =============================================================================

@router.get("/{strategy_id}/confluence-chart")
def get_confluence_chart(
    strategy_id: int,
    condition: str = Query(..., description="Confluence record e.g. '1d-MACD_LINE-M<S-'"),
    days: int = Query(None),
    user=Depends(get_current_user),
):
    """Get chart data for a specific confluence condition on its native timeframe.

    Returns OHLCV bars + indicator overlay + interpreter state per bar
    for the condition's timeframe (e.g., 1Day for '1d-MACD_LINE-M<S-').
    """
    strat = _get_or_404(strategy_id, user)
    import services as svc
    import pandas as pd

    parts = condition.split('-', 2)
    if len(parts) < 3:
        raise HTTPException(status_code=400, detail="Invalid condition format. Expected 'TF-INTERPRETER-STATE'")

    rec_tf, interp_key, needed_state = parts

    # Resolve timeframe
    from data_loader import get_tf_from_label
    if rec_tf == 'GEN' or rec_tf == '1M':
        chart_tf = strat.get('timeframe', '1Min')
    else:
        chart_tf = get_tf_from_label(rec_tf)
        if not chart_tf:
            chart_tf = strat.get('timeframe', '1Min')

    try:
        # Load 1-minute bars and resample to the target timeframe.
        # This matches how the Chart & Trades heatmap and Streamlit app work:
        # always start from 1-min bars (which are post-split and consistent)
        # rather than loading native daily bars from Polygon (which can have
        # split-adjustment issues and EMA warmup problems).
        base_days = days or strat.get('data_days', 30)
        primary_tf = strat.get('timeframe', '1Min')

        if chart_tf == primary_tf or chart_tf == '1Min':
            # Same timeframe as strategy — load directly
            df = svc.prepare_data_with_indicators(
                strat['symbol'],
                days=base_days,
                timeframe=primary_tf,
                session=strat.get('trading_session', 'RTH'),
            )
        else:
            # Different timeframe — load 1-min bars and resample
            from data_loader import resample_to_timeframe
            df_1m = svc.prepare_data_with_indicators(
                strat['symbol'],
                days=base_days,
                timeframe='1Min',
                session=strat.get('trading_session', 'RTH'),
            )
            if len(df_1m) == 0:
                return {"bars": [], "indicator_columns": [], "overlay_indicators": [], "oscillator_indicators": [], "state_column": None, "needed_state": needed_state, "timeframe": chart_tf, "condition": condition, "matched_template": None}

            # Resample OHLCV to target timeframe
            df_resampled = resample_to_timeframe(
                df_1m[['open', 'high', 'low', 'close', 'volume']].copy(), chart_tf
            )
            if len(df_resampled) == 0:
                return {"bars": [], "indicator_columns": [], "overlay_indicators": [], "oscillator_indicators": [], "state_column": None, "needed_state": needed_state, "timeframe": chart_tf, "condition": condition, "matched_template": None}

            # Run indicators and interpreters on the resampled bars
            from indicators import run_all_indicators, run_indicators_for_group
            from interpreters import run_all_interpreters
            from confluence_groups import get_enabled_groups, load_confluence_groups

            df = run_all_indicators(df_resampled)
            for group in get_enabled_groups(load_confluence_groups()):
                df = run_indicators_for_group(df, group)
            df = run_all_interpreters(df)

        if len(df) == 0:
            return {"bars": [], "indicator_columns": [], "states": [], "timeframe": chart_tf}

        # Find indicator columns for this interpreter's template
        from confluence_groups import get_enabled_groups, get_template

        OVERLAY_TEMPLATES = {"ema_stack", "ema_price_position", "ema_price_position_v2", "vwap", "utbot", "utbot_v2"}
        OSCILLATOR_TEMPLATES = {"macd_line", "macd_histogram", "rvol"}

        overlay_cols = []
        oscillator_cols = []
        matched_template = None

        for group in get_enabled_groups():
            template = get_template(group.base_template)
            if not template:
                continue
            # Match interpreter key: check both exact and case-insensitive
            interp_list = template.get("interpreters", [])
            if not any(ik == interp_key or ik.upper() == interp_key.upper() for ik in interp_list):
                continue

            matched_template = group.base_template
            raw_cols = template.get("indicator_columns", [])
            resolved = []
            for col in raw_cols:
                if col in ("ema_short", "ema_mid", "ema_long"):
                    p = group.parameters
                    name = f"ema_{p.get('short_period', 9)}" if col == "ema_short" else \
                           f"ema_{p.get('mid_period', 21)}" if col == "ema_mid" else \
                           f"ema_{p.get('long_period', 200)}"
                    if name in df.columns:
                        resolved.append(name)
                elif col in df.columns:
                    resolved.append(col)

            # Classify as overlay or oscillator
            if group.base_template in OVERLAY_TEMPLATES:
                overlay_cols.extend(resolved)
            elif group.base_template in OSCILLATOR_TEMPLATES:
                oscillator_cols.extend(resolved)
            else:
                dt = template.get("display_type", "overlay")
                if dt == "oscillator":
                    oscillator_cols.extend(resolved)
                else:
                    overlay_cols.extend(resolved)
            break  # found the matching group

        overlay_cols = list(dict.fromkeys(overlay_cols))
        oscillator_cols = list(dict.fromkeys(oscillator_cols))

        # Strip non-plottable columns (bool flags, hex-string candle colors,
        # pack-internal level/color columns). Without this filter, LWC
        # receives string/bool values for line series and the chart's range
        # calculation breaks — bars squish to one edge.
        from api.services.backtest_service import classify_chart_indicators
        entry_conf_id = strat.get('entry_trigger_confluence_id', '') or ''
        overlay_cols, oscillator_cols, _candle_color_column = \
            classify_chart_indicators(df, overlay_cols, oscillator_cols, entry_conf_id)
        all_cols = overlay_cols + oscillator_cols

        # Get interpreter state column
        state_col = interp_key if interp_key in df.columns else None
        if not state_col:
            # Try case-insensitive match
            for c in df.columns:
                if c.upper() == interp_key.upper():
                    state_col = c
                    break
        if not state_col:
            # Try with group ID prefix (some interpreters use GROUP_ID as column name)
            for c in df.columns:
                if interp_key in c.upper() and not c.startswith('trig_'):
                    state_col = c
                    break

        logger.info("[CONFLUENCE-CHART] condition=%s, interp_key=%s, state_col=%s, needed=%s, cols_sample=%s",
                    condition, interp_key, state_col, needed_state,
                    [c for c in df.columns if interp_key.upper() in c.upper()][:5])

        # Log sample state values for debugging
        if state_col and state_col in df.columns:
            sample_vals = df[state_col].dropna().unique()[:10]
            logger.info("[CONFLUENCE-CHART] state_col=%s, unique_values=%s, needed=%s",
                        state_col, list(sample_vals), needed_state)

        # Serialize
        from api.services.backtest_service import _serialize_chart_data
        chart_data = _serialize_chart_data(df, all_cols)

        # Add state values
        if state_col and state_col in df.columns:
            reset_df = df.reset_index()
            for i, row_data in enumerate(chart_data):
                if i < len(reset_df):
                    val = reset_df.iloc[i].get(state_col)
                    state_str = str(val).strip() if pd.notna(val) else None
                    row_data['_state'] = state_str
                    row_data['_met'] = (state_str == needed_state) if state_str else False
        else:
            logger.warning("[CONFLUENCE-CHART] state_col '%s' NOT FOUND in df.columns. Available: %s",
                          interp_key, [c for c in df.columns if c.isupper()][:20])

        return {
            "bars": chart_data,
            "indicator_columns": all_cols,
            "overlay_indicators": overlay_cols,
            "oscillator_indicators": oscillator_cols,
            "state_column": state_col,
            "needed_state": needed_state,
            "timeframe": chart_tf,
            "condition": condition,
            "matched_template": matched_template,
        }
    except Exception as e:
        logger.exception("Failed to compute confluence chart for %s: %s", condition, e)
        raise HTTPException(status_code=504, detail=f"Confluence chart failed: {str(e)[:200]}")


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

    Thin wrapper around forward_test_service.recompute_and_persist_stored_trades
    so the on-demand button and the worker bar-close hook share a single
    implementation path. stored_trades is always backtest truth (never a
    live append) — see docs for the architecture principle.
    """
    # Ownership check (404 on not-yours or not-found) before dispatching
    # to the shared service, so API responses stay identical to before.
    _get_or_404(strategy_id, user)
    from api.services.forward_test_service import (
        recompute_and_persist_stored_trades,
    )
    try:
        return recompute_and_persist_stored_trades(strategy_id, user['id'])
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


# =============================================================================
# MANUAL EXIT
# =============================================================================

@router.post("/{strategy_id}/manual-exit")
def manual_exit(strategy_id: int, user=Depends(get_current_user)):
    """Fire a manual exit signal for the strategy.

    Creates an exit_signal alert with exit_reason='manual', saves it to
    Supabase, and delivers webhooks to all portfolios that include this
    strategy — same pipeline as the alert monitor.
    """
    strat = _get_or_404(strategy_id, user)

    from alerts import save_alert, load_alert_config
    from alert_monitor import _deliver_alert_to_portfolios

    # Build exit alert dict matching what the alert monitor produces
    alert = {
        "type": "exit_signal",
        "strategy_id": strategy_id,
        "strategy_name": strat.get("name", ""),
        "symbol": strat.get("symbol", ""),
        "direction": strat.get("direction", "LONG"),
        "trigger": "manual_exit",
        "exit_reason": "manual",
        "message": f"Manual exit for {strat.get('name', '')} ({strat.get('symbol', '')})",
    }

    # Try to get current price for the alert
    try:
        from data_loader import load_latest_bars
        latest = load_latest_bars(strat["symbol"], count=1)
        if latest is not None and len(latest) > 0:
            alert["price"] = float(latest.iloc[-1]["close"])
    except Exception:
        pass

    # Enrich with portfolio context so webhook delivery knows which portfolios
    try:
        from alerts import enrich_signal_with_portfolio_context
        alert = enrich_signal_with_portfolio_context(alert)
    except Exception:
        alert["portfolio_context"] = []

    # Save alert to Supabase
    try:
        saved = save_alert(alert)
        alert_id = saved.get("id")
    except Exception as e:
        logger.warning(f"Failed to save manual exit alert: {e}")
        alert_id = None

    # Deliver webhooks
    deliveries = []
    try:
        config = load_alert_config()
        deliveries = _deliver_alert_to_portfolios(alert, config)
    except Exception as e:
        logger.warning(f"Failed to deliver manual exit webhooks: {e}")

    return {
        "status": "ok",
        "alert_id": alert_id,
        "webhooks_delivered": len([d for d in deliveries if d.get("success")]),
        "webhooks_failed": len([d for d in deliveries if not d.get("success")]),
        "price": alert.get("price"),
    }


# =============================================================================
# TRADE DRILL-DOWN (1-second bar zoom)
# =============================================================================

@router.get("/{strategy_id}/trade-zoom")
def trade_zoom(
    strategy_id: int,
    trade_idx: int = Query(..., description="Index into stored_trades"),
    side: str = Query("exit", description="entry or exit — which bar to zoom into"),
    padding_seconds: int = Query(60, description="Seconds of context before/after the bar"),
    user=Depends(get_current_user),
):
    """Fetch 1-second OHLCV bars around a specific trade's entry or exit bar.

    Returns bars for rendering in the trade drill-down modal.
    Uses the shared build_trade_zoom_response() helper from backtest_service.
    """
    from api.services.backtest_service import (
        build_trade_zoom_response, extract_relevant_prefixes,
    )
    import services as svc

    strat = _get_or_404(strategy_id, user)
    stored = strat.get('stored_trades', [])
    if trade_idx < 0 or trade_idx >= len(stored):
        raise HTTPException(status_code=404, detail=f"Trade index {trade_idx} out of range (0-{len(stored)-1})")

    trade = stored[trade_idx]
    symbol = strat.get('symbol', 'SPY')
    timeframe = strat.get('timeframe', '1Min')

    # Load indicator data for stepped overlays
    indicator_df = None
    prefixes = set()
    try:
        indicator_df = svc.prepare_data_with_indicators(
            symbol, days=5, timeframe=timeframe,
            data_feed='sip', session=strat.get('trading_session', 'RTH'),
        )
        # Only use entry/exit triggers for stepped indicators —
        # confluence conditions display via CB heatmap, not overlays.
        prefixes = extract_relevant_prefixes(
            entry_id=strat.get('entry_trigger_confluence_id', ''),
            exit_ids=strat.get('exit_trigger_confluence_ids', []),
        )
    except Exception as e:
        logger.warning("[TRADE-ZOOM] Error loading indicator data: %s", e)

    return build_trade_zoom_response(
        trade=trade,
        symbol=symbol,
        timeframe=timeframe,
        side=side,
        padding_seconds=padding_seconds,
        indicator_df=indicator_df,
        relevant_prefixes=prefixes,
        secondary_tfs=strat.get('secondary_tfs'),
        session=strat.get('trading_session', 'RTH'),
    )
