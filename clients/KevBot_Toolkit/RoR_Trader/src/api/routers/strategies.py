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

    # Strategy Fidelity Badges: attach `algo_history_fidelity` derived from
    # the trades' behavior + hifi_resolved fields. Sampled (not full scan)
    # for performance — list endpoint runs across the user's whole fleet.
    _attach_algo_history_fidelity(strategies)

    # Source filter (Mirror Migration workflow 2026-04-29): attach
    # `template_class` ∈ {user_pack, legacy, mixed, no_mirror, unknown,
    # empty} so the My Strategies page can filter to user-pack-only as
    # the migration progresses. Cheap pure classification per strategy
    # — uses lru_cache'd template indexes loaded once per process.
    from api.services.template_classifier import attach_template_class
    attach_template_class(strategies)

    return _sanitize_json(strategies)


def _attach_algo_history_fidelity(strategies: list) -> None:
    """Attach `algo_history_fidelity` to each strategy: 'Hi-Fi' if any trade
    has hifi_resolved=True, 'Bar' if any has behavior='B' (default backtest),
    'Mixed' if both, 'Unknown' if no trades or no fidelity flags.

    Sampled lookup: queries trades table for the most recent 50 trades per
    strategy and reads the relevant fields. ~50ms per strategy, parallelizable
    if the user has many strategies — for now we run sequentially.
    """
    from db import get_admin_client
    client = get_admin_client()
    for s in strategies:
        sid = s.get('id')
        if not sid:
            s['algo_history_fidelity'] = 'Unknown'
            continue
        try:
            r = (client.table('trades')
                 .select('data')
                 .eq('strategy_id', sid)
                 .order('entry_fill_ts', desc=True)
                 .limit(50)
                 .execute())
            rows = r.data or []
            if not rows:
                s['algo_history_fidelity'] = 'Unknown'
                continue
            has_hifi = False
            has_bar = False
            for row in rows:
                d = row.get('data') or {}
                if isinstance(d, str):
                    import json as _json
                    try:
                        d = _json.loads(d)
                    except Exception:
                        d = {}
                if d.get('hifi_resolved') is True:
                    has_hifi = True
                if d.get('behavior') == 'B':
                    has_bar = True
                if has_hifi and has_bar:
                    break
            if has_hifi and has_bar:
                s['algo_history_fidelity'] = 'Mixed'
            elif has_hifi:
                s['algo_history_fidelity'] = 'Hi-Fi'
            elif has_bar:
                s['algo_history_fidelity'] = 'Bar'
            else:
                s['algo_history_fidelity'] = 'Unknown'
        except Exception as e:
            logger.warning(
                "[FIDELITY] Could not attach for sid %s: %s", sid, e)
            s['algo_history_fidelity'] = 'Unknown'


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


@router.get("/models")
def list_strategy_models():
    """M8.7 (2026-05-02): list available backtest_model and live_model
    values for strategy configuration.

    Used by the frontend strategy edit page to populate the model
    selector dropdowns. Each entry includes label, availability flag,
    and description for tooltip display.

    Currently a placeholder — strategies record their model selection
    but the engine does not yet differentiate behavior. Wiring comes
    when M8.7d (cache read path) ships.
    """
    from strategy_models import (
        BACKTEST_MODELS, LIVE_MODELS,
        get_default_backtest_model, get_default_algo_model, get_default_live_model,
    )
    return {
        "backtest_models": BACKTEST_MODELS,
        "algo_models": BACKTEST_MODELS,  # same registry, different consumer
        "live_models": LIVE_MODELS,
        "defaults": {
            "backtest_model": get_default_backtest_model(),
            "algo_model": get_default_algo_model(),
            "live_model": get_default_live_model(),
        },
    }


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

    # Attach algo history fidelity (Bar / Hi-Fi / Mixed / Unknown). Reuses
    # the list endpoint's helper — operates on a single-strategy list.
    try:
        _attach_algo_history_fidelity([enriched])
    except Exception as _e:
        logger.warning(
            "[FIDELITY] detail attach failed for sid %s: %s", strategy_id, _e)

    # M8.7 (2026-05-02): fill in default model selections for strategies
    # that predate the models placeholder. New strategies will carry
    # these in config; legacy ones get the defaults treatment so the
    # frontend always has a value to render.
    try:
        from strategy_models import (
            get_default_backtest_model, get_default_algo_model, get_default_live_model,
            is_valid_backtest_model, is_valid_algo_model, is_valid_live_model,
        )
        if not enriched.get('backtest_model') or \
                not is_valid_backtest_model(enriched.get('backtest_model')):
            enriched['backtest_model'] = get_default_backtest_model()
        if not enriched.get('algo_model') or \
                not is_valid_algo_model(enriched.get('algo_model')):
            enriched['algo_model'] = get_default_algo_model()
        if not enriched.get('live_model') or \
                not is_valid_live_model(enriched.get('live_model')):
            enriched['live_model'] = get_default_live_model()
    except Exception as _e:
        logger.warning(
            "[MODELS] default fill failed for sid %s: %s", strategy_id, _e)

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


@router.post("/{strategy_id}/run-hifi-pass2")
def run_hifi_pass2(
    strategy_id: int,
    data_source_filter: Optional[str] = Query(
        None,
        description="Optional SQL LIKE pattern (e.g. 'backtest_%', 'cache_%') "
                    "to scope Hi-Fi refinement to a single lane. None = all "
                    "trades (legacy behavior). Added 2026-05-14 Phase D as "
                    "defensive scoping — see docs/Parity_Investigation_2026-"
                    "05-14_Phase_D.md.",
    ),
    user=Depends(get_current_user),
):
    """Run Hi-Fi Pass 2 on a strategy's existing trades — refines entry +
    exit timestamps + prices using 1-second data, without re-running the
    full backtest. ~10x faster than a full hifi_mode backtest because it
    skips the indicator/trigger pipeline.

    For each trade:
      - Entry refinement (L-type only): walks 1-sec bars in entry bar
        window to find when the entry level was crossed. Updates
        entry_fill_ts to the per-second moment.
      - Exit refinement: walks 1-sec bars in exit bar window to find
        which level (stop/target) hit first. Updates exit_reason,
        exit_price, hold_time_seconds.

    Persists refined trades back to the trades table + bumps the
    strategy's `data_source` to 'hifi'.

    Response: counts of entry refinements + exit changes + total trades.

    Phase D (2026-05-14): added optional `data_source_filter` so callers
    can scope Hi-Fi to a specific lane (e.g., 'cache_%' or 'backtest_%').
    Defensive — Phase D investigation confirmed cross-pollination is
    NOT a current bug, but this prevents a future bug from silently
    affecting both lanes via one call site.
    """
    from db import USE_DB
    if not USE_DB:
        raise HTTPException(
            status_code=501, detail="Requires DB mode")

    from db import (get_strategy_by_id_db, get_admin_client,
                    update_strategy_admin)
    strat = get_strategy_by_id_db(strategy_id)
    if strat is None:
        raise HTTPException(status_code=404, detail="Strategy not found")

    user_id = strat.get('user_id') or (
        user.get('id') or user.get('sub')
        if isinstance(user, dict) else None)
    if not user_id:
        raise HTTPException(status_code=401, detail="No user context")

    # Load existing trades from the trades table (the canonical store post-Phase 40).
    # Optional data_source_filter scopes the refinement to one lane.
    from db import load_trades_admin
    trades_list = load_trades_admin(
        strategy_id, str(user_id),
        data_source_filter=data_source_filter,
    ) or []
    if not trades_list:
        return {
            "status": "ok",
            "strategy_id": strategy_id,
            "trades_count": 0,
            "message": "Strategy has no trades to refine",
        }

    import pandas as pd
    trades_df = pd.DataFrame(trades_list)

    # Mirror the column names _hifi_resolve_trades expects. The trades
    # table stores entry_fill_ts/exit_fill_ts; the function falls back to
    # those when entry_time/exit_time are missing (handled in the
    # extension we just shipped).
    symbol = strat.get('symbol', 'SPY')
    timeframe = strat.get('timeframe', '1Min')

    # entry_exec_type is needed by the entry-refinement gate. Pull from
    # exec_type which should be the entry exec_type for L-type strategies.
    if 'entry_exec_type' not in trades_df.columns and 'exec_type' in trades_df.columns:
        trades_df['entry_exec_type'] = trades_df['exec_type']
    if 'direction' not in trades_df.columns or trades_df['direction'].isna().all():
        # Direction not stored on trade row — fall back to strategy direction
        trades_df['direction'] = strat.get('direction', 'LONG')

    # Phase 1 signal-exit refinement (2026-05-07): pre-compute the
    # primary-TF bar DataFrame with indicators so Hi-Fi can read
    # level-column values for signal-exit triggers. Skipped (None) if
    # the strategy has no L-type signal exits — those trades already
    # marked hifi_resolved=False fall through.
    bar_df = None
    has_ib_exits = any(
        ((t.get('exit_trigger') or (t.get('data') or {}).get('exit_trigger') or '').endswith('_ib'))
        for t in trades_list
    )
    if has_ib_exits:
        try:
            import services as _svc
            bar_df = _svc.get_strategy_trades(strat).attrs.get('bar_df_for_hifi')
            # Fallback: re-run prepare_data_with_indicators directly to
            # get the bar DataFrame. get_strategy_trades returns trades,
            # not bars; we need to call prepare_data_with_indicators
            # ourselves to capture the bar DataFrame.
            if bar_df is None:
                from data_loader import (
                    get_required_tfs_from_confluence,
                    get_tf_from_label,
                )
                req_labels = get_required_tfs_from_confluence(strat.get('confluence', []))
                sec_tfs = tuple(sorted(get_tf_from_label(lbl) for lbl in req_labels))
                # Use full data window — we need indicators converged
                # for any trade in trades_df. Hi-Fi runs over the full
                # trade history; bars must cover that.
                # `datetime`/`timedelta`/`timezone` already imported at
                # module level — DON'T re-import locally (would create a
                # local-var shadow that breaks downstream references).
                if strat.get('forward_test_start'):
                    fwd_start = datetime.fromisoformat(strat['forward_test_start'])
                    start_dt = fwd_start - timedelta(days=int(strat.get('data_days', 30)) * 2)
                else:
                    start_dt = datetime.now(timezone.utc) - timedelta(days=int(strat.get('data_days', 30)))
                end_dt = datetime.now(timezone.utc)
                bar_df = _svc.prepare_data_with_indicators(
                    strat['symbol'],
                    start_date=start_dt, end_date=end_dt,
                    timeframe=timeframe,
                    session=strat.get('trading_session', 'RTH'),
                    secondary_tfs=sec_tfs,
                    strat=strat,  # honors cache_locked dispatch
                )
        except Exception as e:
            logger.warning(
                "[HIFI-PASS2] Strategy %s: failed to load bar_df for "
                "signal-exit refinement (will skip signal exits): %s",
                strategy_id, e)
            bar_df = None

    # Run Pass 2 (entries + exits + signal exits)
    from api.services.backtest_service import _hifi_resolve_trades
    try:
        refined_df = _hifi_resolve_trades(
            trades_df, symbol, timeframe, bar_df=bar_df)
    except Exception as e:
        logger.exception("[HIFI-PASS2] Strategy %s crashed: %s",
                         strategy_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Hi-Fi Pass 2 crashed: {type(e).__name__}: {e}")

    # Detect what changed
    entries_changed = 0
    exits_changed = 0
    flag_changes = 0  # rows that got hifi_resolved=True without other changes
    if 'entry_fill_ts' in refined_df.columns:
        for i, row in refined_df.iterrows():
            orig = trades_list[i] if i < len(trades_list) else {}
            had_change = False
            if str(row.get('entry_fill_ts')) != str(orig.get('entry_fill_ts')):
                entries_changed += 1
                had_change = True
            if (str(row.get('exit_price')) != str(orig.get('exit_price'))
                    or str(row.get('exit_reason')) != str(orig.get('exit_reason'))):
                exits_changed += 1
                had_change = True
            # Even if no timestamp/price change, count flag-only changes so
            # persistence iterates and writes them.
            if not had_change and (
                ('hifi_resolved' in row.index and row.get('hifi_resolved') is True)
                or ('behavior' in row.index and row.get('behavior') == 'HIFI')
            ):
                flag_changes += 1

    # Persist refined trades + flip data_source. We update each row that
    # changed; rows that didn't change are skipped.
    client = get_admin_client()
    persisted = 0
    if entries_changed + exits_changed + flag_changes > 0:
        # Update each changed row's columns. Since trades table now has
        # canonical columns we update directly.
        for i, row in refined_df.iterrows():
            tid = row.get('id')
            if not tid:
                continue
            update_payload = {}
            orig = trades_list[i]
            for col in ('entry_fill_ts', 'exit_fill_ts', 'exit_price',
                        'exit_reason', 'r_multiple', 'dollar_pnl',
                        'executed_quantity'):
                if col in row.index:
                    new_val = row[col]
                    # Convert pandas/numpy values for JSON serialization
                    if hasattr(new_val, 'isoformat'):
                        new_val = new_val.isoformat()
                    elif pd.isna(new_val):
                        new_val = None
                    elif hasattr(new_val, 'item'):
                        new_val = new_val.item()
                    if str(new_val) != str(orig.get(col)):
                        update_payload[col] = new_val
            # Also propagate the Hi-Fi flags into the `data` JSONB so the
            # algo_history_fidelity aggregator surfaces the trade as Hi-Fi.
            # _hifi_resolve_trades sets hifi_resolved=True / behavior='HIFI'
            # on the dataframe row when it refines either entry or exit.
            row_hifi = (row.get('hifi_resolved') if 'hifi_resolved' in row.index
                        else None)
            row_behavior = (row.get('behavior') if 'behavior' in row.index
                            else None)
            if row_hifi is True or row_behavior == 'HIFI':
                # 2026-05-12 fix: `orig` came from `_row_to_trade` which
                # SPREADS the data JSONB into top-level keys. So
                # `orig.get('data')` was returning None, the merged dict
                # was starting empty, and the UPDATE was wiping bars_held,
                # hold_time_seconds, pnl, win, behavior, etc. on every row
                # Hi-Fi touched. Reconstruct the JSONB by collecting every
                # non-column field on the orig dict — those are exactly the
                # fields that were originally in the `data` JSONB before
                # _row_to_trade unpacked them.
                from db import TRADE_COLUMN_FIELDS as _COL_FIELDS
                _LEGACY_ALIASES = {'entry_time', 'exit_time', 'data'}
                orig_data = {
                    k: v for k, v in orig.items()
                    if k not in _COL_FIELDS and k not in _LEGACY_ALIASES
                }
                merged = dict(orig_data)
                if row_hifi is True:
                    merged['hifi_resolved'] = True
                if row_behavior == 'HIFI':
                    merged['behavior'] = 'HIFI'
                if merged != orig_data:
                    update_payload['data'] = merged
            if update_payload:
                try:
                    client.table('trades').update(
                        update_payload).eq('id', tid).execute()
                    persisted += 1
                except Exception as e:
                    logger.warning(
                        "[HIFI-PASS2] Failed to update trade %s: %s",
                        tid, e)

    # Mark strategy as Hi-Fi resolved in data_source.
    if entries_changed + exits_changed + flag_changes > 0:
        try:
            update_strategy_admin(strategy_id, str(user_id), {
                'data_source': 'hifi',
                'kpis_computed_at': datetime.now(timezone.utc).isoformat(),
                'kpis_stale_since': None,
            })
        except Exception as e:
            logger.warning(
                "[HIFI-PASS2] Failed to update strategy data_source: %s", e)

    return {
        "status": "ok",
        "strategy_id": strategy_id,
        "trades_count": len(trades_list),
        "entries_refined": entries_changed,
        "exits_refined": exits_changed,
        "flag_only_updates": flag_changes,
        "persisted": persisted,
        "data_source": 'hifi' if entries_changed + exits_changed + flag_changes > 0
                       else strat.get('data_source'),
    }


@router.post("/append-recent-trades-bulk")
def append_recent_trades_bulk(
    body: dict = Body(...),
    user=Depends(get_current_user),
):
    """Manual trigger for the algo-history append cron, scoped to the
    given strategy IDs. Bypasses the recently_processed gate so a user
    click always produces fresh output (vs the cron's 5-min throttle).

    Each strategy: load only the recent window of bars, run engine,
    set-diff INSERT new closed trades. ~30s per strategy worst case
    (full engine replay), <100ms when alerts-gate determines no new
    closed trades possible.

    Body: {"strategy_ids": [int, ...]}.
    Returns per-strategy summary + aggregate counts.
    """
    from db import get_admin_client
    from api.services.forward_test_service import (
        append_new_trades_for_strategy,
    )

    sids = body.get('strategy_ids') or []
    if not isinstance(sids, list) or not sids:
        raise HTTPException(
            status_code=400,
            detail="Provide non-empty strategy_ids list")
    try:
        sids = [int(x) for x in sids]
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="strategy_ids must be integers")

    user_id = user.get("id") if isinstance(user, dict) else getattr(user, "id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="missing user_id")

    # Bypass recently_processed gate by clearing last_recompute_until_ts
    # before each call. Cron will re-stamp it. Safer than adding a
    # bypass parameter (one source of truth for skip logic).
    client = get_admin_client()
    results: list = []
    total_inserted = 0
    total_failed = 0
    total_skipped = 0
    for sid in sids:
        try:
            # Clear last_recompute_until_ts so the recently_processed
            # check falls through (alerts-gate still applies)
            r0 = client.table('strategies').select('config') \
                .eq('id', sid).eq('user_id', user_id).single().execute()
            cfg = dict((r0.data or {}).get('config') or {})
            cfg.pop('last_recompute_until_ts', None)
            client.table('strategies').update({'config': cfg}) \
                .eq('id', sid).eq('user_id', user_id).execute()

            r = append_new_trades_for_strategy(sid, user_id)
            results.append({
                "strategy_id": sid,
                "status": r.get("status"),
                "inserted": r.get("inserted", 0),
                "reason": r.get("reason"),
            })
            total_inserted += int(r.get('inserted') or 0)
            if r.get('status') == 'skipped':
                total_skipped += 1
        except Exception as e:
            total_failed += 1
            results.append({
                "strategy_id": sid,
                "status": "error",
                "detail": str(e),
            })
    return {
        "results": results,
        "total_strategies": len(sids),
        "total_inserted": total_inserted,
        "total_skipped": total_skipped,
        "total_failed": total_failed,
    }


@router.post("/run-hifi-pass2-bulk")
def run_hifi_pass2_bulk(
    body: dict = Body(...),
    user=Depends(get_current_user),
):
    """Run Hi-Fi Pass 2 across multiple strategies in one call.

    Body: {"strategy_ids": [int, ...]}.
    Returns per-strategy summary plus aggregate counts.
    """
    sids = body.get('strategy_ids') or []
    if not isinstance(sids, list) or not sids:
        raise HTTPException(
            status_code=400,
            detail="Provide non-empty strategy_ids list")

    results = []
    total_entries = 0
    total_exits = 0
    total_failed = 0
    for sid in sids:
        try:
            r = run_hifi_pass2(sid, user=user)
            results.append({
                "strategy_id": sid,
                "status": r.get("status"),
                "entries_refined": r.get("entries_refined", 0),
                "exits_refined": r.get("exits_refined", 0),
                "trades_count": r.get("trades_count", 0),
            })
            total_entries += r.get("entries_refined", 0)
            total_exits += r.get("exits_refined", 0)
        except HTTPException as e:
            total_failed += 1
            results.append({
                "strategy_id": sid,
                "status": "error",
                "detail": e.detail,
            })
        except Exception as e:
            total_failed += 1
            results.append({
                "strategy_id": sid,
                "status": "error",
                "detail": str(e),
            })
    return {
        "results": results,
        "total_strategies": len(sids),
        "total_entries_refined": total_entries,
        "total_exits_refined": total_exits,
        "total_failed": total_failed,
    }


@router.post("/{strategy_id}/run-parity")
def run_parity(strategy_id: int, user=Depends(get_current_user)):
    """Queue a background parity replay for one strategy.

    Mirrors the run-hifi-pass2 pattern. Returns immediately with
    parity_status = PENDING; the bg thread writes the final verdict
    when done. Bg threads are serialized via Semaphore(1) so concurrent
    queues don't thrash the API process.

    Use this instead of waiting for auto-parity on refresh: it's the
    user-controlled "compute the live-executable score" action.
    """
    strat = _get_or_404(strategy_id, user)
    user_id = strat.get('user_id') or (
        user.get('id') or user.get('sub')
        if isinstance(user, dict) else None)
    if not user_id:
        raise HTTPException(status_code=401, detail="No user context")

    from api.services.forward_test_service import queue_parity_for_strategy
    return queue_parity_for_strategy(strategy_id, user_id)


@router.get("/{strategy_id}/grace-shadow-comparison")
def grace_shadow_comparison(
    strategy_id: int,
    window_hours: int = Query(6, ge=1, le=48),
    user=Depends(get_current_user),
):
    """Phase 4.5 (LEF spec §5.5) — multi-grace shadow comparison.

    Runs the unified backtest engine against each grace variant's
    recorded shadow bars (grace_2s … grace_7s) and reports per-variant
    trade counts + close-match % vs reference (grace_7s). Returns a
    recommendation ('Fastest' / 'Balanced' / 'Highest-fidelity') based
    on the fastest variant achieving >= 95% close-match.

    Available only for sub-minute strategies on (SPY, TSLA) at 10Sec
    (the grace shadow's recorded scope today). For all other
    strategies, returns `available: false` with an explanation.

    Engine-risk-free: read-only against live_bars_trades, fresh
    UnifiedStrategy per variant, no live-engine touch.
    """
    strat = _get_or_404(strategy_id, user)
    from grace_shadow_compare import compare_strategy
    try:
        return compare_strategy(strat, window_hours=window_hours)
    except Exception as e:
        import logging as _l
        _l.getLogger(__name__).warning(
            "grace_shadow_comparison sid=%s failed: %s", strategy_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"grace shadow comparison failed: {str(e)[:200]}")


@router.post("/run-parity-bulk")
def run_parity_bulk(
    body: dict = Body(...),
    user=Depends(get_current_user),
):
    """Queue parity replays across multiple strategies.

    Body: {"strategy_ids": [int, ...]}.
    Returns per-strategy queue status. The actual parity work runs
    serialized in the background via the global Semaphore(1) — the
    API stays responsive while they process one at a time.
    """
    sids = body.get('strategy_ids') or []
    if not isinstance(sids, list) or not sids:
        raise HTTPException(
            status_code=400,
            detail="Provide non-empty strategy_ids list")

    from db import get_strategy_by_id_db
    from api.services.forward_test_service import queue_parity_for_strategy

    results = []
    queued = 0
    failed = 0
    for sid in sids:
        try:
            strat = get_strategy_by_id_db(sid)
            if strat is None:
                results.append({
                    "strategy_id": sid, "status": "error",
                    "detail": "not found"})
                failed += 1
                continue
            user_id = strat.get('user_id') or (
                user.get('id') or user.get('sub')
                if isinstance(user, dict) else None)
            if not user_id:
                results.append({
                    "strategy_id": sid, "status": "error",
                    "detail": "no user context"})
                failed += 1
                continue
            r = queue_parity_for_strategy(sid, user_id)
            results.append({
                "strategy_id": sid,
                "status": r.get("status", "queued"),
            })
            if r.get("status") == "queued":
                queued += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            results.append({
                "strategy_id": sid, "status": "error",
                "detail": str(e),
            })

    return {
        "results": results,
        "total_strategies": len(sids),
        "total_queued": queued,
        "total_failed": failed,
    }


@router.post("/{strategy_id}/parity-check")
def run_parity_check(
    strategy_id: int,
    last_n: Optional[int] = Query(
        25, ge=0,
        description=(
            "Compare only the most recent N stored trades (0 = all). "
            "Older backtest data was often computed with different pack "
            "code / Polygon snapshots and naturally diverges; the recent "
            "tail is what tells you whether the engine matches today's "
            "backtest output.")),
    forward_test_only: bool = Query(
        False,
        description=(
            "If true, ignore stored trades dated before "
            "strategy.forward_test_start. Pre-forward-test trades are "
            "pure historical backtest and aren't expected to match live.")),
    user=Depends(get_current_user),
):
    """Replay a strategy's stored backtest trades through the live engine
    (StrategyMonitor + shadow engines, same code the worker runs) and
    diff the resulting fires against stored trades.

    Surfaces parity gaps as concrete "this stored trade would NOT fire in
    live, and here's the failing gate" rows for the Strategy Detail UI.

    Response shape: see api.services.parity_service.run_strategy_parity.
    Verdicts: PASS / PARTIAL / FAIL_LIVE_BLOCKED / FAIL_OVER_FIRES /
              NO_TRADES / NO_DATA / ERROR.
    """
    from db import USE_DB
    if not USE_DB:
        raise HTTPException(status_code=501, detail="Requires DB mode")

    from db import (get_strategy_by_id_db, set_admin_user_context,
                    clear_current_user)
    strat = get_strategy_by_id_db(strategy_id)
    if strat is None:
        raise HTTPException(status_code=404, detail="Strategy not found")

    user_id = strat.get('user_id') or (
        user.get('id') or user.get('sub')
        if isinstance(user, dict) else None)
    if not user_id:
        raise HTTPException(status_code=401, detail="No user context")

    # Admin context so the replay can read this user's general packs +
    # trades regardless of which JWT the request arrived with.
    set_admin_user_context(user_id)
    try:
        from api.services.parity_service import run_strategy_parity
        # Treat last_n=0 as "no limit" so the query param can express both.
        effective_last_n = last_n if (last_n and last_n > 0) else None
        report = run_strategy_parity(
            strategy_id, user_id=user_id,
            last_n=effective_last_n,
            forward_test_only=forward_test_only,
        )
    except Exception as e:
        logger.exception(
            "[PARITY] Strategy %s crashed: %s", strategy_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Parity check crashed: {type(e).__name__}: {e}")
    finally:
        clear_current_user()

    return _sanitize_json(report)


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


@router.get("/{strategy_id}/cache-coverage")
def get_strategy_cache_coverage(
    strategy_id: int,
    user=Depends(get_current_user),
):
    """Per-TF freshness snapshot of the worker's live_bars cache.

    For the strategy's primary timeframe + every secondary timeframe
    referenced in its confluence records, returns the most recent
    bar_start in the live_bars table along with seconds_since-now and a
    coarse status (green/yellow/red).

    This is the read side of the diagnostic. If a secondary TF shows
    red, the worker's shadow-engine fan-out for that (symbol, tf) is
    likely broken. Used by Strategy Detail to self-diagnose cross-TF
    cache gaps in 5s instead of running an ad-hoc Python audit.
    """
    from ralph_engine import (
        TIMEFRAME_SECONDS, SECONDS_TO_TIMEFRAME, _LABEL_TO_TF_SECONDS,
    )
    from db import get_admin_client

    strat = _get_or_404(strategy_id, user)
    symbol = strat.get('symbol')
    primary_label = strat.get('timeframe')
    primary_seconds = TIMEFRAME_SECONDS.get(primary_label)
    if not symbol or not primary_seconds:
        raise HTTPException(
            status_code=400,
            detail=f"Strategy {strategy_id} missing symbol or timeframe")

    # `_get_or_404` returns a FLATTENED strat dict (config JSONB keys
    # spread to top level). `strat.get('config')` is therefore None —
    # read 'confluence' from the top level instead. Same gotcha is
    # called out in forward_test_service.py around line 677.
    confluence_records = strat.get('confluence') or []
    if not isinstance(confluence_records, list):
        confluence_records = []

    tfs: dict[int, str] = {primary_seconds: primary_label}
    for rec in confluence_records:
        if not isinstance(rec, str):
            continue
        parts = rec.split('-', 2)
        if len(parts) < 3:
            continue
        sec_seconds = _LABEL_TO_TF_SECONDS.get(parts[0])
        if sec_seconds and sec_seconds not in tfs:
            tfs[sec_seconds] = SECONDS_TO_TIMEFRAME.get(sec_seconds, parts[0])

    sb = get_admin_client()
    now = datetime.now(timezone.utc)
    coverage = []
    for tf_seconds in sorted(tfs.keys()):
        rows = (sb.table('live_bars')
                .select('bar_start,written_at,source')
                .eq('symbol', symbol)
                .eq('timeframe_seconds', tf_seconds)
                .order('bar_start', desc=True)
                .limit(1)
                .execute()
                .data)
        if rows:
            bar_start_raw = rows[0]['bar_start']
            try:
                bar_start_dt = datetime.fromisoformat(
                    bar_start_raw.replace('Z', '+00:00'))
            except Exception:
                bar_start_dt = None
            seconds_since = (
                (now - bar_start_dt).total_seconds()
                if bar_start_dt else None)
            written_at = rows[0].get('written_at')
            source = rows[0].get('source')
        else:
            bar_start_raw = None
            seconds_since = None
            written_at = None
            source = None

        if seconds_since is None:
            status = 'red'
        elif seconds_since < 2 * tf_seconds:
            status = 'green'
        elif seconds_since < 6 * tf_seconds:
            status = 'yellow'
        else:
            status = 'red'

        coverage.append({
            'tf_label': tfs[tf_seconds],
            'tf_seconds': tf_seconds,
            'is_primary': tf_seconds == primary_seconds,
            'latest_bar_start': bar_start_raw,
            'seconds_since': seconds_since,
            'last_written_at': written_at,
            'source': source,
            'status': status,
        })

    return {
        'symbol': symbol,
        'as_of': now.isoformat(),
        'coverage': coverage,
    }


@router.get("/{strategy_id}/algo-trades")
def get_strategy_algo_trades(
    strategy_id: int,
    since: Optional[str] = Query(
        None,
        description="ISO timestamp; only return trades with entry_fill_ts >= this"),
    user=Depends(get_current_user),
):
    """Get algo-lane trades (data_source LIKE 'cache_%') for a strategy.

    Distinct from `/trades` which returns the backtest lane (stored_trades
    JSONB after Phase 41 hydration is filtered to `backtest_%`). This
    endpoint reads the canonical cache-based algo trades from the trades
    table, used by the Chart & Trades "Algo History" + "Price Divergence"
    modules so they show what live algo actually produced rather than
    backtest output.

    Trade records are reconstructed via `_row_to_trade` which merges the
    `data` JSONB into top-level keys, so consumers see `entry_price`,
    `exit_price`, `r_multiple`, `bars_held`, `hold_time_seconds`, etc.
    """
    strat = _get_or_404(strategy_id, user)
    user_id = strat.get('user_id') or (
        user.get('id') or user.get('sub')
        if isinstance(user, dict) else None)
    if not user_id:
        raise HTTPException(status_code=401, detail="No user context")

    from db import load_trades_admin
    trades = load_trades_admin(
        strategy_id, str(user_id),
        data_source_filter='cache_%') or []

    if since:
        trades = [t for t in trades
                  if str(t.get('entry_fill_ts', '')) >= since]

    return _sanitize_json(trades)


@router.get("/{strategy_id}/divergence-data")
def get_strategy_divergence_data(
    strategy_id: int,
    forward_test_only: bool = Query(True, description="Filter to trades after forward_test_start"),
    direction: Optional[str] = Query(None, description="LONG / SHORT filter"),
    tolerance_seconds: float = Query(300.0, ge=10.0, le=3600.0,
        description="Max distance to consider two events 'matched'. Beyond this, lanes split into separate rows."),
    start: Optional[str] = Query(None,
        description="ISO timestamp (UTC); only include trades/alerts with entry_fill_ts >= start. Defaults to 48 hours ago."),
    end: Optional[str] = Query(None,
        description="ISO timestamp (UTC); only include trades/alerts with entry_fill_ts <= end. Defaults to now."),
    max_per_lane: int = Query(2000, ge=10, le=20000,
        description="Cap rows per lane to keep response size + API memory bounded. Most-recent N kept."),
    user=Depends(get_current_user),
):
    """Three-lane comparison data for the Divergence tab.

    Lanes (v1):
      - **backtest** = strategy.stored_trades JSONB (last "Refresh" output;
        the user's official backtest snapshot)
      - **algo** = trades table rows (cron's incremental algo-history;
        reflects current engine output on recent bars)
      - **live** = alerts table rows (what actually fired live)

    All three lanes are read from cached state — no engine runs in this
    endpoint. Drift between backtest↔algo signals backtest staleness;
    drift between algo↔live signals engine→alert divergence.

    Future v2: a "Run Comparison Backtest" endpoint will run both
    `rest_only` and `cache_locked` against the same window and surface
    that as a separate REST↔CACHE divergence view.

    When `forward_test_only=true` (default), each lane is filtered to
    events at/after `forward_test_start` so the comparison only covers
    the period when alerts could have fired.
    """
    strat = _get_or_404(strategy_id, user)
    user_id = user.get("id") if isinstance(user, dict) else getattr(user, "id", None)

    from db import get_alerts_for_strategy_db, load_trades_admin
    from api.services.divergence_service import compute_three_way_divergence
    from datetime import datetime, timezone, timedelta

    fwd_start = strat.get('forward_test_start') if forward_test_only else None

    # 2026-05-12: date-window filter. Default to last 48 hours so the
    # default page load is fast even on strategies with 5000+ trades.
    # Users can explicitly widen via start/end query params from the UI.
    now = datetime.now(timezone.utc)
    end_iso = end or now.isoformat()
    start_iso = start or (now - timedelta(hours=48)).isoformat()

    def _in_window(ts_str):
        if not ts_str:
            return False
        s = str(ts_str)
        return s >= start_iso and s <= end_iso

    # Phase 41 (2026-05-11): both lanes now read from the trades table
    # with data_source filters. Backtest = 'backtest_%', Algo = anything
    # else with a non-null data_source (legacy NULLs were UPDATEd to
    # 'cache_locked' in the Phase 41 schema migration so they're algo
    # by convention).
    try:
        backtest_trades = load_trades_admin(
            strategy_id, user_id,
            data_source_filter='backtest_%') or []
    except Exception as e:
        logger.warning("[DIVERGENCE] backtest lane load failed for sid=%s: %s",
                       strategy_id, e)
        backtest_trades = []

    backtest_total = len(backtest_trades)
    backtest_last_ts = None
    for t in backtest_trades:
        ts = t.get('entry_fill_ts') or t.get('entryFillTime')
        if ts and (backtest_last_ts is None or str(ts) > str(backtest_last_ts)):
            backtest_last_ts = ts

    # Apply date window filter (default 48h), then cap to max_per_lane.
    # Window filter is the primary speedup — for active strategies the
    # 48h slice is typically <100 trades. The cap is a safety net for
    # wide windows.
    backtest_trades = [t for t in backtest_trades if _in_window(t.get('entry_fill_ts'))]
    if len(backtest_trades) > max_per_lane:
        backtest_trades = sorted(
            backtest_trades,
            key=lambda t: str(t.get('entry_fill_ts') or ''),
            reverse=True,
        )[:max_per_lane]

    try:
        # Algo lane = anything NOT backtest_. We use cache_% as the
        # primary filter since that's where the cron writes; falls
        # through to NULL data_source as well via the COALESCE in the
        # schema migration. If users start using other algo_models we
        # may need to broaden this.
        algo_trades = load_trades_admin(
            strategy_id, user_id,
            data_source_filter='cache_%') or []
    except Exception as e:
        logger.warning("[DIVERGENCE] algo trades load failed for sid=%s: %s",
                       strategy_id, e)
        algo_trades = []

    algo_total = len(algo_trades)
    algo_last_ts = None
    for t in algo_trades:
        ts = t.get('entry_fill_ts')
        if ts and (algo_last_ts is None or str(ts) > str(algo_last_ts)):
            algo_last_ts = ts

    algo_trades = [t for t in algo_trades if _in_window(t.get('entry_fill_ts'))]
    if len(algo_trades) > max_per_lane:
        algo_trades = sorted(
            algo_trades,
            key=lambda t: str(t.get('entry_fill_ts') or ''),
            reverse=True,
        )[:max_per_lane]

    try:
        alerts = get_alerts_for_strategy_db(
            strategy_id, limit=max_per_lane) or []
    except Exception as e:
        logger.warning("[DIVERGENCE] alert load failed for sid=%s: %s",
                       strategy_id, e)
        alerts = []
    alerts_total = len(alerts)
    alerts = [a for a in alerts if _in_window(a.get('fill_ts'))]

    last_alert_ts = alerts[0].get('timestamp') if alerts else None

    strat_direction = (
        strat.get('direction')
        or (strat.get('config') or {}).get('direction')
        or 'LONG'
    )

    diverge = compute_three_way_divergence(
        rest_trades=backtest_trades,
        cache_trades=algo_trades,
        alerts=alerts,
        forward_test_start=fwd_start,
        direction_filter=direction,
        default_direction=strat_direction,
        max_match_seconds=tolerance_seconds,
    )

    return {
        'strategy_id': strategy_id,
        'forward_test_start': strat.get('forward_test_start'),
        'backtest_model': (strat.get('config') or {}).get('backtest_model')
                          or strat.get('backtest_model'),
        'algo_model': (strat.get('config') or {}).get('algo_model')
                      or strat.get('algo_model'),
        'live_model': (strat.get('config') or {}).get('live_model')
                      or strat.get('live_model'),
        'backtest': {
            'count': backtest_total,  # total in DB
            'shown': len(backtest_trades),  # after max_per_lane cap
            'last_trade_ts': backtest_last_ts,
            'available': backtest_total > 0,
        },
        'algo': {
            'count': algo_total,
            'shown': len(algo_trades),
            'last_trade_ts': str(algo_last_ts) if algo_last_ts else None,
            'available': algo_total > 0,
        },
        'live': {
            'count': alerts_total,
            'last_alert_ts': last_alert_ts,
        },
        'rows': diverge['rows'],
        'kpis': diverge['kpis'],
        'lane_counts': diverge['lane_counts'],
        'forward_test_only': forward_test_only,
        'direction_filter': direction,
        'max_per_lane': max_per_lane,
        'window_start': start_iso,
        'window_end': end_iso,
    }


@router.get("/admin/divergence-summary")
def get_admin_divergence_summary(
    start: str = Query(..., description="ISO timestamp (UTC); filter trades with entry_fill_ts >= start"),
    end: Optional[str] = Query(None, description="ISO timestamp (UTC); filter trades with entry_fill_ts < end. Defaults to now."),
    max_per_lane: int = Query(1000, ge=10, le=5000,
        description="Cap rows per lane per strategy. Keeps API memory bounded — 20 strategies × multi-thousand rows can OOM the API container."),
    user=Depends(get_current_user),
):
    """Admin: per-strategy divergence summary over a time window.

    Loops over the user's strategies, loads backtest + algo + alert data
    within `[start, end)`, and runs the 3-way joiner. Returns a flat array
    suitable for a sortable summary table — one row per strategy with
    counts, match counts, and drift KPIs (median / p95 / max in seconds).

    Mirrors what the per-strategy `/divergence-data` endpoint produces but
    aggregated across the whole fleet and date-window filtered, so we can
    spot-check fidelity across strategies after any code change.
    """
    import gc
    from db import load_strategies_admin, get_alerts_for_strategy_db, load_trades_admin
    from api.services.divergence_service import compute_three_way_divergence
    from datetime import datetime, timezone

    user_id = user.get("id") if isinstance(user, dict) else getattr(user, "id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="No user context")

    end_iso = end or datetime.now(timezone.utc).isoformat()

    try:
        strategies = load_strategies_admin(str(user_id)) or []
    except Exception as e:
        logger.exception("[ADMIN-DIV] load_strategies failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to load strategies: {e}")

    def _in_window(ts):
        if not ts:
            return False
        s = str(ts)
        return s >= start and (end_iso is None or s < end_iso)

    def _cap_to_recent(trades, key='entry_fill_ts', cap=max_per_lane):
        """Keep only the most-recent N trades to bound memory."""
        if len(trades) <= cap:
            return trades
        return sorted(
            trades, key=lambda t: str(t.get(key) or ''), reverse=True
        )[:cap]

    rows = []
    for s in strategies:
        sid = s.get('id')
        if not sid:
            continue
        cfg = s.get('config') or {}
        if isinstance(cfg, str):
            try:
                import json as _json
                cfg = _json.loads(cfg)
            except Exception:
                cfg = {}
        direction = (s.get('direction') or cfg.get('direction') or 'LONG').upper()

        try:
            bt = load_trades_admin(sid, str(user_id), data_source_filter='backtest_%') or []
            algo = load_trades_admin(sid, str(user_id), data_source_filter='cache_%') or []
            alerts = get_alerts_for_strategy_db(sid, limit=max_per_lane) or []
        except Exception as e:
            logger.warning("[ADMIN-DIV] load failed for sid=%s: %s", sid, e)
            rows.append({
                'strategy_id': sid, 'name': s.get('name', ''),
                'symbol': s.get('symbol', ''), 'timeframe': s.get('timeframe', ''),
                'error': str(e)[:120],
            })
            continue

        # Window filter by entry_fill_ts (trades) / fill_ts (alerts)
        bt_w = [t for t in bt if _in_window(t.get('entry_fill_ts'))]
        algo_w = [t for t in algo if _in_window(t.get('entry_fill_ts'))]
        alerts_w = [a for a in alerts if _in_window(a.get('fill_ts'))]
        # Free the full-strategy lists ASAP — we only need windowed slices
        del bt, algo, alerts
        # Bound the windowed slices too, in case the window is wide
        bt_w = _cap_to_recent(bt_w)
        algo_w = _cap_to_recent(algo_w)
        alerts_w = _cap_to_recent(alerts_w, key='fill_ts')

        if not bt_w and not algo_w and not alerts_w:
            rows.append({
                'strategy_id': sid, 'name': s.get('name', ''),
                'symbol': s.get('symbol', ''), 'timeframe': s.get('timeframe', ''),
                'backtest_count': 0, 'algo_count': 0, 'live_count': 0,
                'no_activity': True,
            })
            continue

        try:
            div = compute_three_way_divergence(
                rest_trades=bt_w,
                cache_trades=algo_w,
                alerts=alerts_w,
                default_direction=direction,
                max_match_seconds=300.0,
            )
            kpis = div.get('kpis') or {}
            e_rc = kpis.get('drift_rest_cache_entry') or {}
            e_cl = kpis.get('drift_cache_live_entry') or {}
            x_cl = kpis.get('drift_cache_live_exit') or {}
            rows.append({
                'strategy_id': sid,
                'name': s.get('name', ''),
                'symbol': s.get('symbol', ''),
                'timeframe': s.get('timeframe', ''),
                'backtest_model': cfg.get('backtest_model'),
                'algo_model': cfg.get('algo_model'),
                'live_model': cfg.get('live_model'),
                'backtest_count': len(bt_w),
                'algo_count': len(algo_w),
                'live_count': len(alerts_w),
                'matched_3way': kpis.get('matched_3way', 0),
                'matched_rest_cache': kpis.get('matched_rest_cache', 0),
                'matched_cache_live': kpis.get('matched_cache_live', 0),
                'entry_rc_median_s': e_rc.get('median_s'),
                'entry_rc_p95_s': e_rc.get('p95_s'),
                'entry_rc_max_s': e_rc.get('max_s'),
                'entry_cl_median_s': e_cl.get('median_s'),
                'entry_cl_p95_s': e_cl.get('p95_s'),
                'exit_cl_median_s': x_cl.get('median_s'),
                'exit_cl_p95_s': x_cl.get('p95_s'),
                'exit_cl_max_s': x_cl.get('max_s'),
            })
        except Exception as e:
            logger.exception("[ADMIN-DIV] joiner failed for sid=%s: %s", sid, e)
            rows.append({
                'strategy_id': sid, 'name': s.get('name', ''),
                'symbol': s.get('symbol', ''), 'timeframe': s.get('timeframe', ''),
                'backtest_count': len(bt_w), 'algo_count': len(algo_w), 'live_count': len(alerts_w),
                'error': str(e)[:120],
            })

        # Free per-strategy intermediates between iterations so memory
        # doesn't pile up. 2026-05-12 incident: this endpoint OOM'd the
        # API container at 20 strategies × ~5k trades each.
        bt_w = algo_w = alerts_w = None
        gc.collect()

    return {
        'start': start,
        'end': end_iso,
        'strategy_count': len(strategies),
        'rows': rows,
        'max_per_lane': max_per_lane,
    }


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

_OVERLAY_TEMPLATES_CHART = {"ema_stack", "ema_price_position", "ema_price_position_v2", "vwap", "utbot", "utbot_v2"}
_OSCILLATOR_TEMPLATES_CHART = {"macd_line", "macd_histogram", "rvol"}


def _build_chart_response_from_df(df, strat, strategy_id, phases=None):
    """M8.7 (2026-05-02) — factored from get_strategy_chart_data.

    Given an enriched DataFrame (post-prepare_data_with_indicators) and the
    strategy dict, builds the chart-data response shape:
        {chart_data, overlay_indicators, oscillator_indicators,
         heatmap_conditions, candle_color_column}

    Used by both /chart-data (REST-derived df) and /chart-data-cache
    (live_bars-derived df). The data-prep stage above this is what
    differs; the indicator classification + serialization is identical.
    """
    import time as _time
    if phases is None:
        phases = {}

    if len(df) == 0:
        return {"chart_data": [], "overlay_indicators": [],
                "oscillator_indicators": [], "heatmap_conditions": []}

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
        template = get_template(group.base_template)

        # Relevance = group's indicator columns should appear on the chart.
        # Three matching paths:
        #   1. group.id is the prefix of the strategy's entry/exit trigger
        #      conf id (works for legacy templates where group.id == trigger_prefix)
        #   2. group's interpreters intersect the strategy's confluence-record
        #      interpreters (catches confluence-only groups)
        #   3. template's `trigger_prefix` is the prefix of an entry/exit
        #      conf id (fixes user packs where group.id ≠ trigger_prefix —
        #      e.g. EPP v4: group `ema_pp_v4_default`, trigger_prefix `eppv4`,
        #      conf id `eppv4_default_cross_short_up`)
        is_relevant = (
            entry_conf_id.startswith(gid_prefix)
            or any(cid.startswith(gid_prefix) for cid in exit_conf_ids)
        )
        if not is_relevant and template:
            for ik in template.get("interpreters", []):
                if ik in confluence_interpreters:
                    is_relevant = True
                    break
        if not is_relevant and template:
            tp = template.get("trigger_prefix", "")
            if tp:
                tp_prefix = tp + "_"
                if entry_conf_id.startswith(tp_prefix) \
                        or any(cid.startswith(tp_prefix) for cid in exit_conf_ids):
                    is_relevant = True
        if not is_relevant:
            continue

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

        if group.base_template in _OVERLAY_TEMPLATES_CHART:
            overlay_cols.extend(resolved)
        elif group.base_template in _OSCILLATOR_TEMPLATES_CHART:
            oscillator_cols.extend(resolved)
        else:
            dt = template.get("display_type", "overlay")
            if dt == "oscillator":
                oscillator_cols.extend(resolved)
            else:
                overlay_cols.extend(resolved)

    overlay_cols = list(dict.fromkeys(overlay_cols))
    oscillator_cols = list(dict.fromkeys(oscillator_cols))

    # Ghost overlay: include `_prev` columns when their non-prev sibling is
    # already in overlay_cols.
    try:
        for base_col in list(overlay_cols):
            prev_col = f"{base_col}_prev"
            if prev_col in df.columns and prev_col not in overlay_cols:
                overlay_cols.append(prev_col)
    except Exception as _e:
        logger.debug("ghost overlay inclusion failed: %s", _e)

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
    cb_set = set(strat.get('cb_conditions', []))

    heatmap_conditions = []
    for record in all_conditions:
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

    _t = _time.time()
    chart_data = _serialize_chart_data(df, all_indicator_cols)
    phases["serialize"] = _time.time() - _t

    # Vectorized state injection
    _t = _time.time()
    if state_cols:
        present_cols = [sc for sc in state_cols if sc in df.columns]
        for sc in state_cols:
            if sc not in df.columns:
                logger.warning("[CHART:%s] heatmap state_col=%s NOT FOUND in df",
                               strategy_id, sc)

        col_values: dict[str, list] = {}
        for sc in present_cols:
            series = df[sc]
            mask = series.notna().tolist()
            vals = series.astype(object).tolist()
            col_values[sc] = [
                str(v) if m else None for v, m in zip(vals, mask)
            ]

        n = min(len(chart_data), len(df))
        for sc in present_cols:
            col_list = col_values[sc]
            key = f"_state_{sc}"
            for i in range(n):
                chart_data[i][key] = col_list[i]
    phases["state_inject"] = _time.time() - _t

    return {
        "chart_data": chart_data,
        "overlay_indicators": overlay_cols,
        "oscillator_indicators": oscillator_cols,
        "heatmap_conditions": heatmap_conditions,
        "candle_color_column": candle_color_column,
    }


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
    import time as _time

    _t_start = _time.time()
    _phases: dict = {}

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

        logger.warning("[CHART-DATA:%s] start symbol=%s tf=%s days=%s sec_tfs=%s",
                       strategy_id, strat.get('symbol'), strat.get('timeframe'),
                       chart_days, sec_tfs)
        _t = _time.time()
        df = svc.prepare_data_with_indicators(
            strat['symbol'],
            days=chart_days,
            timeframe=strat.get('timeframe', '1Min'),
            session=strat.get('trading_session', 'RTH'),
            secondary_tfs=sec_tfs,
        )
        _phases["prepare"] = _time.time() - _t
        logger.warning("[CHART-DATA:%s] prepare_data_with_indicators=%.2fs bars=%d",
                       strategy_id, _phases["prepare"], len(df))

        # M8.7 (2026-05-02): factored indicator classification + serialization
        # into _build_chart_response_from_df, shared with /chart-data-cache.
        result = _build_chart_response_from_df(df, strat, strategy_id, _phases)

        total = _time.time() - _t_start
        logger.warning(
            "[CHART-DATA:%s] DONE total=%.2fs prepare=%.2fs serialize=%.2fs "
            "state_inject=%.2fs bars=%d overlay=%d oscillator=%d",
            strategy_id, total,
            _phases.get("prepare", 0), _phases.get("serialize", 0),
            _phases.get("state_inject", 0),
            len(df),
            len(result.get("overlay_indicators", [])),
            len(result.get("oscillator_indicators", []))
        )
        return result
    except Exception as e:
        elapsed = _time.time() - _t_start
        logger.exception(
            "[CHART-DATA:%s] FAILED after %.2fs phases=%s: %s",
            strategy_id, elapsed, _phases, e
        )
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
# CACHE BARS (M8.7 — live_bars-backed OHLCV for "Live WS" chart view)
# =============================================================================

TF_TO_SECONDS = {
    '10Sec': 10, '30Sec': 30, '1Min': 60, '5Min': 300,
    '10Min': 600, '15Min': 900, '30Min': 1800,
    '1Hour': 3600, '1Day': 86400,
}


@router.get("/{strategy_id}/chart-data-cache")
def get_strategy_chart_data_from_cache(
    strategy_id: int,
    value_type: str = Query("latest", description="'latest' (default — post-rebroadcast WS values) or 'first' (decision-time WS values)"),
    days: int = Query(None, description="Override data_days; capped by cache coverage"),
    user=Depends(get_current_user),
):
    """Same shape as /chart-data, but bars + indicators + heatmap are
    computed from the live_bars cache (WS-derived) instead of Polygon REST.

    M8.7 Phase 2 (2026-05-02): used by the Strategy Detail "Chart & Trades
    (Lab)" tab Alert Lens to render what the live engine actually sees.
    Indicators run on cache OHLCV (primary + cross-TF secondary) so the
    heatmap and overlay lines match what live execution observed —
    closing the gap that Phase 1 left as a caveat.

    value_type:
      - 'latest' = cache.close (post-rebroadcast WS values; 1Min may
        differ from REST by Polygon's 15-min FINRA correction)
      - 'first'  = cache.first_close (decision-time WS values; what the
        engine SAW at the moment the bar first hit)

    Returns same shape as /chart-data:
        {chart_data, overlay_indicators, oscillator_indicators,
         heatmap_conditions, candle_color_column}
    """
    strat = _get_or_404(strategy_id, user)
    import services as svc
    import time as _time
    from datetime import datetime, timedelta, timezone
    from db import set_admin_user_context

    _t_start = _time.time()
    _phases: dict = {}

    try:
        symbol = strat.get('symbol')
        primary_tf = strat.get('timeframe', '1Min')
        primary_tf_seconds = TF_TO_SECONDS.get(primary_tf)
        if primary_tf_seconds is None:
            return {"chart_data": [], "overlay_indicators": [],
                    "oscillator_indicators": [], "heatmap_conditions": [],
                    "note": f"Unknown primary timeframe '{primary_tf}'"}

        # Determine secondary TFs from confluence records
        from data_loader import get_required_tfs_from_confluence, get_tf_from_label
        req_labels = get_required_tfs_from_confluence(strat.get('confluence', []))
        sec_tfs = tuple(sorted(
            get_tf_from_label(lbl) for lbl in req_labels
            if get_tf_from_label(lbl)))

        base_days = days or strat.get('data_days', 30)
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=base_days)

        set_admin_user_context(user.get('id') or strat.get('user_id'))

        # Fetch primary cache as DataFrame.
        # Lab tab "Alert Lens" semantics: ONLY ws + ws_agg sources —
        # exclude rest_backfill rows (Phase D, future) so this view
        # reflects what the LIVE engine actually saw, not what the
        # cache was retroactively patched with.
        _t = _time.time()
        primary_df = fetch_cache_as_df(
            symbol, primary_tf_seconds, start_dt, end_dt, value_type,
            sources=['ws', 'ws_agg'])
        _phases["fetch_primary"] = _time.time() - _t

        if len(primary_df) == 0:
            logger.info(
                "[CHART-CACHE:%s] no cache rows for %s/%ss in window — empty result",
                strategy_id, symbol, primary_tf_seconds)
            return {"chart_data": [], "overlay_indicators": [],
                    "oscillator_indicators": [], "heatmap_conditions": [],
                    "note": (f"No cache data for {symbol}/{primary_tf} in window. "
                             f"Cache started recording 2026-04-30; older periods "
                             f"have no rows.")}

        # Fetch each secondary TF cache as DataFrame
        _t = _time.time()
        sec_tf_dfs: dict = {}
        for sec_tf in sec_tfs:
            sec_tf_seconds = TF_TO_SECONDS.get(sec_tf)
            if sec_tf_seconds is None:
                continue
            sec_df = fetch_cache_as_df(
                symbol, sec_tf_seconds, start_dt, end_dt, value_type,
                sources=['ws', 'ws_agg'])
            if len(sec_df) > 0:
                sec_tf_dfs[sec_tf] = sec_df
        _phases["fetch_secondary"] = _time.time() - _t

        logger.info(
            "[CHART-CACHE:%s] fetched primary=%d bars, secondary=%s rows",
            strategy_id, len(primary_df),
            {tf: len(d) for tf, d in sec_tf_dfs.items()})

        # Run indicator pipeline on injected DataFrames
        _t = _time.time()
        df = svc.prepare_data_with_indicators(
            symbol,
            days=base_days,
            timeframe=primary_tf,
            session=strat.get('trading_session', 'RTH'),
            secondary_tfs=sec_tfs,
            primary_df=primary_df,
            secondary_tf_dfs=sec_tf_dfs if sec_tf_dfs else None,
        )
        _phases["prepare"] = _time.time() - _t

        # Build response via shared helper
        result = _build_chart_response_from_df(df, strat, strategy_id, _phases)
        result["data_source"] = f"cache_{value_type}"
        result["window_start"] = start_dt.isoformat()
        result["window_end"] = end_dt.isoformat()
        result["primary_cache_rows"] = len(primary_df)
        result["secondary_cache_rows"] = {tf: len(d) for tf, d in sec_tf_dfs.items()}

        total = _time.time() - _t_start
        logger.warning(
            "[CHART-CACHE:%s] DONE total=%.2fs fetch_primary=%.2fs "
            "fetch_secondary=%.2fs prepare=%.2fs bars=%d value_type=%s",
            strategy_id, total,
            _phases.get("fetch_primary", 0), _phases.get("fetch_secondary", 0),
            _phases.get("prepare", 0), len(df), value_type)
        return result

    except Exception as e:
        elapsed = _time.time() - _t_start
        logger.exception(
            "[CHART-CACHE:%s] FAILED after %.2fs phases=%s: %s",
            strategy_id, elapsed, _phases, e)
        raise HTTPException(
            status_code=504,
            detail=f"Chart data (cache) computation failed: {str(e)[:200]}")


# fetch_cache_as_df moved to data_loader.py (2026-05-06, Phase E
# preview) so services.py can import without circular dep through
# api.routers.strategies. Kept this re-export for any legacy import
# inside the same module.
from data_loader import fetch_cache_as_df  # noqa: F401


@router.get("/{strategy_id}/cache-bars")
def get_strategy_cache_bars(
    strategy_id: int,
    value_type: str = Query("latest", description="'latest' = post-rebroadcast WS values; 'first' = decision-time values"),
    days: int = Query(None, description="How many days of bars to return (default: strategy.data_days, capped at cache coverage)"),
    source: str = Query("all", description="'all' (default — includes rest_backfill gap fills) or 'live' (ws_agg/ws only — excludes REST backfill, so genuine cache gaps stay visible)"),
    user=Depends(get_current_user),
):
    """OHLCV bars for this strategy's (symbol, tf) sourced from `live_bars`.

    Used by the 'Chart & Trades (Lab)' tab data-source toggle. Returns
    just OHLCV — no indicators or heatmap. Frontend swaps the price-bar
    data while keeping the indicator/heatmap layers REST-derived.

    Two modes:
      - value_type='latest' (default): the WS values currently in cache,
        including any Polygon rebroadcast corrections within the 15-min
        FINRA window. For 1Min bars, this matches REST closely.
      - value_type='first':  the values at FIRST WS write — what the
        live engine actually saw at decision moment. For 1Min, may
        differ from 'latest' if Polygon rebroadcast a correction. For
        sub-minute (we don't reaggregate on per-second corrections),
        first == latest by construction.
    """
    strat = _get_or_404(strategy_id, user)
    from db import get_admin_client, set_admin_user_context
    from datetime import datetime, timedelta, timezone

    symbol = strat.get('symbol')
    tf = strat.get('timeframe', '1Min')
    tf_seconds = TF_TO_SECONDS.get(tf)
    if tf_seconds is None:
        return {"chart_data": [], "value_type": value_type, "tf_seconds": None,
                "note": f"Unknown timeframe '{tf}'"}

    base_days = days or strat.get('data_days', 30)
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=base_days)

    set_admin_user_context(user.get('id') or strat.get('user_id'))
    c = get_admin_client()

    cols = "bar_start,open,high,low,close,volume,first_open,first_high,first_low,first_close,first_volume,source,written_at,last_updated_at"
    rows = []
    page_size = 1000
    offset = 0
    # source='live' excludes rest_backfill rows so genuine cache gaps stay
    # visible (the parity "Cache (decision-time)" view). 'all' keeps them
    # (the gap-free "Cache + backfill" view).
    exclude_backfill = source == 'live'
    while True:
        try:
            q = c.table('live_bars').select(cols) \
                .eq('symbol', symbol) \
                .eq('timeframe_seconds', tf_seconds) \
                .gte('bar_start', start_dt.isoformat()) \
                .lte('bar_start', end_dt.isoformat())
            if exclude_backfill:
                q = q.neq('source', 'rest_backfill')
            r = q.order('bar_start') \
                .range(offset, offset + page_size - 1) \
                .execute()
        except Exception as e:
            logger.warning("cache-bars: fetch failed offset=%d: %s", offset, e)
            break
        batch = r.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    use_first = value_type == 'first'
    chart_data = []
    null_first = 0
    for row in rows:
        # Pick the OHLCV based on requested mode. If first_* is NULL
        # (pre-migration row), fall back to latest values.
        if use_first and row.get('first_close') is not None:
            o, h, l, cl, v = row['first_open'], row['first_high'], row['first_low'], row['first_close'], row['first_volume']
        else:
            if use_first:
                null_first += 1
            o, h, l, cl, v = row['open'], row['high'], row['low'], row['close'], row['volume']
        chart_data.append({
            'timestamp': row['bar_start'],
            'open': float(o or 0),
            'high': float(h or 0),
            'low': float(l or 0),
            'close': float(cl or 0),
            'volume': float(v or 0),
            'source': row.get('source', 'ws'),
        })

    notes = []
    if use_first and null_first:
        notes.append(f"{null_first} bars predate the first_values migration; latest values used as fallback")

    source_counts: dict = {}
    for row in rows:
        s = row.get('source') or 'unknown'
        source_counts[s] = source_counts.get(s, 0) + 1

    return {
        "chart_data": chart_data,
        "value_type": value_type,
        "source_filter": source,
        "source_counts": source_counts,
        "symbol": symbol,
        "timeframe": tf,
        "tf_seconds": tf_seconds,
        "window_start": start_dt.isoformat(),
        "window_end": end_dt.isoformat(),
        "row_count": len(chart_data),
        "notes": notes,
    }


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

@router.post("/{strategy_id}/update")
def update_strategy_lanes(
    strategy_id: int,
    mode: str = Query(..., regex="^(all|new)$",
        description="'all' = full recompute; 'new' = forward append since last stamp"),
    user=Depends(get_current_user),
):
    """Update strategy data on backtest + algo lanes (Option A — both lanes
    fan-out from one button click). All 4 lane×mode combos wired 2026-05-08.

    `mode='all'`:
      - Backtest lane: full recompute via `recompute_and_persist_stored_trades`
        (uses backtest_model, writes stored_trades JSONB)
      - Algo lane:     full recompute via `recompute_and_persist_algo_trades`
        (uses algo_model, wipes + repopulates trades table for strategy)

    `mode='new'`:
      - Backtest lane: forward append via `append_new_backtest_trades_for_strategy`
        (uses backtest_model, extends stored_trades JSONB since latest stamp)
      - Algo lane:     forward append via `append_new_trades_for_strategy`
        (uses algo_model, extends trades table since latest stamp)

    JWT-safety (2026-05-08): each lane call wrapped in
    `set_admin_user_context` so downstream Supabase calls use service-role
    key. Without this, the user JWT can age out during the ~3-5 min
    backtest run, causing PGRST303 on the algo lane that runs after.
    """
    _get_or_404(strategy_id, user)
    user_id = user.get("id") if isinstance(user, dict) else getattr(user, "id", None)

    from api.services.forward_test_service import (
        recompute_and_persist_stored_trades,
        append_new_trades_for_strategy,
        append_new_backtest_trades_for_strategy,
        recompute_and_persist_algo_trades,
    )
    from db import set_admin_user_context, clear_current_user

    backtest_result = None
    algo_result = None

    try:
        if mode == 'all':
            try:
                # JWT-safety (2026-05-08): backtest run can take 3-5 min,
                # exceeding the user JWT's TTL. Force admin context so
                # downstream Supabase calls use the service-role key.
                set_admin_user_context(user_id)
                backtest_result = recompute_and_persist_stored_trades(strategy_id, user_id)
            except Exception as e:
                logger.exception("backtest lane failed for sid=%s", strategy_id)
                backtest_result = {'status': 'error', 'reason': str(e)}

            try:
                set_admin_user_context(user_id)  # re-affirm after long backtest run
                algo_result = recompute_and_persist_algo_trades(strategy_id, user_id)
            except Exception as e:
                logger.exception("algo lane failed for sid=%s", strategy_id)
                algo_result = {'status': 'error', 'reason': str(e)}

        else:  # mode == 'new'
            try:
                set_admin_user_context(user_id)
                backtest_result = append_new_backtest_trades_for_strategy(
                    strategy_id, user_id, force=True)
            except Exception as e:
                logger.exception("backtest lane failed for sid=%s", strategy_id)
                backtest_result = {'status': 'error', 'reason': str(e)}

            try:
                set_admin_user_context(user_id)
                algo_result = append_new_trades_for_strategy(
                    strategy_id, user_id, force=True)
            except Exception as e:
                logger.exception("algo lane failed for sid=%s", strategy_id)
                algo_result = {'status': 'error', 'reason': str(e)}

        return {
            'mode': mode,
            'backtest': backtest_result,
            'algo': algo_result,
        }
    except Exception as e:
        logger.exception("update_strategy_lanes failed for sid=%s", strategy_id)
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")
    finally:
        clear_current_user()


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
