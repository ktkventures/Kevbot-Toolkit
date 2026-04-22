"""Portfolios router — CRUD + compute + account management."""

import copy
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Body, Query

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/portfolios", tags=["portfolios"])


# =============================================================================
# HELPERS
# =============================================================================

def _get_or_404(portfolio_id: int, user) -> dict:
    """Load portfolio or raise 404."""
    from db import USE_DB
    if USE_DB:
        from db import get_portfolio_by_id_db
        portfolio = get_portfolio_by_id_db(portfolio_id)
    else:
        portfolio = None
        import json, os
        path = os.path.join(os.path.dirname(__file__), '..', '..', 'portfolios.json')
        if os.path.exists(path):
            with open(path) as f:
                for p in json.load(f):
                    if p.get('id') == portfolio_id:
                        portfolio = p
                        break

    if portfolio is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return portfolio


def _get_strategy_fn(sid):
    """Load a strategy by ID (passed to portfolio computation functions)."""
    from db import USE_DB
    if USE_DB:
        from db import get_strategy_by_id_db
        return get_strategy_by_id_db(sid)
    import json, os
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'strategies.json')
    if os.path.exists(path):
        with open(path) as f:
            for s in json.load(f):
                if s.get('id') == sid:
                    return s
    return None


def _get_trades_fn(strat):
    """Get trades DataFrame for a strategy (from stored_trades fast path)."""
    import services as svc
    return svc.trades_df_from_stored(strat.get('stored_trades', []))


def _serialize_df(df) -> list[dict]:
    """Convert a pandas DataFrame to JSON-serializable list of dicts."""
    import pandas as pd
    if df is None or (isinstance(df, pd.DataFrame) and len(df) == 0):
        return []
    records = df.copy()
    for col in records.columns:
        if hasattr(records[col], 'dt'):
            records[col] = records[col].apply(
                lambda x: x.isoformat() if pd.notna(x) and hasattr(x, 'isoformat') else x
            )
    return records.where(records.notna(), None).to_dict(orient='records')


def _serialize_series(series) -> list:
    """Convert a pandas Series to JSON-serializable list."""
    import pandas as pd
    if series is None or (isinstance(series, pd.Series) and len(series) == 0):
        return []
    return [
        None if pd.isna(v) else (v.isoformat() if hasattr(v, 'isoformat') else v)
        for v in series.values
    ]


def _sanitize_for_json(value):
    """Recursively replace inf/-inf/nan with None, convert numpy/pandas types to Python natives.

    FastAPI's jsonable_encoder chokes on numpy.bool_, numpy.int64, numpy.float64,
    numpy.datetime64, pandas.Timestamp, pandas.NaT, etc. Walk the structure and
    coerce scalar values to their JSON-safe equivalents.
    """
    import math
    try:
        import numpy as np
    except ImportError:
        np = None
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if value is None:
        return None
    # pandas NaT / NaN
    if pd is not None:
        try:
            if isinstance(value, (pd.Timestamp, )):
                return value.isoformat() if not pd.isna(value) else None
            if value is pd.NaT:
                return None
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    if np is not None:
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            f = float(value)
            if math.isinf(f) or math.isnan(f):
                return None
            return f
        if isinstance(value, np.ndarray):
            return _sanitize_for_json(value.tolist())
        if isinstance(value, np.datetime64):
            try:
                ts = pd.Timestamp(value) if pd is not None else None
                if ts is not None and not pd.isna(ts):
                    return ts.isoformat()
            except Exception:
                pass
            return str(value)
    # Python datetime / date
    import datetime as _dt
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    if isinstance(value, float):
        if math.isinf(value) or math.isnan(value):
            return None
    return value


def _downsample_series(series, max_points: int = 30) -> list:
    """Downsample a pandas Series to at most max_points for preview rendering."""
    import pandas as pd
    if series is None or (isinstance(series, pd.Series) and len(series) == 0):
        return []
    if len(series) <= max_points:
        return _serialize_series(series)
    step = max(1, len(series) // max_points)
    return _serialize_series(series.iloc[::step])


# =============================================================================
# CRUD
# =============================================================================

@router.get("")
def list_portfolios(
    preview: bool = Query(False, description="Include per-portfolio KPIs, equity curve preview, and requirement-set counts"),
    user=Depends(get_current_user),
):
    """Load all portfolios for the current user.

    When ``preview=true``, each portfolio is enriched with:
    - ``kpis`` (from calculate_portfolio_kpis)
    - ``equity_curve_preview`` (downsampled to ~30 points)
    - ``fwd_kpis`` / ``alert_kpis`` (forward test + alert subsets)
    - ``req_passing`` / ``req_total`` (requirement set pass ratio)
    """
    from db import USE_DB
    if USE_DB:
        from db import load_portfolios_db
        portfolios = load_portfolios_db() or []
    else:
        import json, os
        path = os.path.join(os.path.dirname(__file__), '..', '..', 'portfolios.json')
        portfolios = []
        if os.path.exists(path):
            with open(path) as f:
                portfolios = json.load(f)

    if not preview:
        return portfolios

    return [_enrich_portfolio_preview(p) for p in portfolios]


def _enrich_portfolio_preview(portfolio: dict) -> dict:
    """Attach kpis, equity_curve_preview, fwd_kpis, alert_kpis, req_passing/total."""
    import pandas as pd
    from portfolios import (
        get_portfolio_trades, calculate_portfolio_kpis,
        evaluate_requirement_set, get_requirement_set_by_id,
    )

    enriched = dict(portfolio)
    try:
        trade_data = get_portfolio_trades(portfolio, _get_strategy_fn, _get_trades_fn)
        combined = trade_data.get('combined_trades', pd.DataFrame())
        daily_pnl = trade_data.get('daily_pnl', pd.DataFrame())
    except Exception as e:
        logger.warning("preview: get_portfolio_trades failed for portfolio %s: %s",
                       portfolio.get('id'), e)
        enriched['kpis'] = calculate_portfolio_kpis(portfolio, pd.DataFrame(), pd.DataFrame())
        enriched['equity_curve_preview'] = []
        enriched['fwd_kpis'] = None
        enriched['alert_kpis'] = None
        enriched['req_passing'] = None
        enriched['req_total'] = None
        return _sanitize_for_json(enriched)

    enriched['kpis'] = calculate_portfolio_kpis(portfolio, combined, daily_pnl)
    enriched['equity_curve_preview'] = _downsample_series(trade_data.get('equity_curve'), 30)

    # Forward test subset: trades after any strategy's forward_test_start
    fwd_trades, fwd_daily = _filter_forward_test(portfolio, combined, daily_pnl)
    enriched['fwd_kpis'] = calculate_portfolio_kpis(portfolio, fwd_trades, fwd_daily) if len(fwd_trades) > 0 else None

    # Alert-derived KPIs
    try:
        from portfolios import get_portfolio_alert_trades
        alert_data = get_portfolio_alert_trades(portfolio, _get_strategy_fn)
        alert_trades_list = alert_data.get('alert_trades', [])
        if alert_trades_list:
            alert_df = pd.DataFrame(alert_trades_list)
            if 'dollar_pnl' in alert_df.columns:
                alert_df['cumulative_pnl'] = alert_df['dollar_pnl'].cumsum()
                alert_df['win'] = alert_df['dollar_pnl'] > 0
                if 'exit_time' in alert_df.columns:
                    alert_df['exit_time'] = pd.to_datetime(alert_df['exit_time'], errors='coerce')
                    alert_daily = alert_df.groupby(alert_df['exit_time'].dt.date)['dollar_pnl'].sum().reset_index()
                    alert_daily.columns = ['date', 'daily_pnl']
                else:
                    alert_daily = pd.DataFrame()
                enriched['alert_kpis'] = calculate_portfolio_kpis(portfolio, alert_df, alert_daily)
            else:
                enriched['alert_kpis'] = None
        else:
            enriched['alert_kpis'] = None
    except Exception as e:
        logger.warning("preview: alert KPI compute failed for portfolio %s: %s",
                       portfolio.get('id'), e)
        enriched['alert_kpis'] = None

    # Requirement set pass ratio
    req_set_id = portfolio.get('requirement_set_id')
    if req_set_id:
        try:
            req_set = get_requirement_set_by_id(int(req_set_id))
            if req_set:
                eval_result = evaluate_requirement_set(req_set, portfolio, enriched['kpis'], daily_pnl)
                rules = eval_result.get('rules', [])
                enriched['req_total'] = len(rules)
                enriched['req_passing'] = sum(1 for r in rules if r.get('passed'))
            else:
                enriched['req_passing'] = None
                enriched['req_total'] = None
        except Exception as e:
            logger.warning("preview: requirement eval failed for portfolio %s: %s",
                           portfolio.get('id'), e)
            enriched['req_passing'] = None
            enriched['req_total'] = None
    else:
        enriched['req_passing'] = None
        enriched['req_total'] = None

    return _sanitize_for_json(enriched)


def _filter_forward_test(portfolio: dict, combined, daily_pnl):
    """Return trades + daily_pnl restricted to the forward test window.

    The forward test window starts at the max forward_test_start across all
    strategies in the portfolio (any trade before that is backtest-only).
    """
    import pandas as pd
    if len(combined) == 0 or 'entry_time' not in combined.columns:
        return combined.iloc[0:0], daily_pnl.iloc[0:0] if len(daily_pnl) > 0 else daily_pnl

    # Find the earliest forward_test_start across strategies (broadest window)
    starts = []
    for alloc in portfolio.get('strategies', []):
        strat = _get_strategy_fn(alloc.get('strategy_id'))
        if strat and strat.get('forward_test_start'):
            starts.append(strat['forward_test_start'])
    if not starts:
        return combined.iloc[0:0], daily_pnl.iloc[0:0] if len(daily_pnl) > 0 else daily_pnl

    fwd_start = pd.to_datetime(min(starts), utc=True, errors='coerce')
    entry_times = pd.to_datetime(combined['entry_time'], utc=True, errors='coerce')
    mask = entry_times >= fwd_start
    fwd_trades = combined[mask].copy()

    if len(daily_pnl) > 0 and 'date' in daily_pnl.columns:
        fwd_daily = daily_pnl[pd.to_datetime(daily_pnl['date'], utc=True, errors='coerce') >= fwd_start]
    else:
        fwd_daily = daily_pnl.iloc[0:0] if len(daily_pnl) > 0 else daily_pnl

    return fwd_trades, fwd_daily


@router.post("")
def create_portfolio(portfolio: dict = Body(...), user=Depends(get_current_user)):
    """Create a new portfolio."""
    portfolio['created_at'] = datetime.now(timezone.utc).isoformat()

    from db import USE_DB
    if USE_DB:
        from db import save_portfolio_db
        result = save_portfolio_db(portfolio)
        return result if result else {"status": "saved"}

    # JSON fallback
    import json, os
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'portfolios.json')
    portfolios = []
    if os.path.exists(path):
        with open(path) as f:
            portfolios = json.load(f)
    portfolio['id'] = max((p.get('id', 0) for p in portfolios), default=0) + 1
    portfolios.append(portfolio)
    with open(path, 'w') as f:
        json.dump(portfolios, f, indent=2)
    return portfolio


@router.get("/{portfolio_id}")
def get_portfolio(portfolio_id: int, user=Depends(get_current_user)):
    """Get a single portfolio by ID."""
    return _get_or_404(portfolio_id, user)


@router.put("/{portfolio_id}")
def update_portfolio(
    portfolio_id: int,
    updated: dict = Body(...),
    user=Depends(get_current_user),
):
    """Update an existing portfolio."""
    from db import USE_DB
    if USE_DB:
        from db import get_portfolio_by_id_db, update_portfolio_db
        existing = get_portfolio_by_id_db(portfolio_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        updated['id'] = portfolio_id
        updated['updated_at'] = datetime.now(timezone.utc).isoformat()
        if 'created_at' not in updated:
            updated['created_at'] = existing.get('created_at')

        result = update_portfolio_db(portfolio_id, updated)
        return result if result else {"status": "updated"}

    raise HTTPException(status_code=501, detail="JSON update not implemented via API")


@router.delete("/{portfolio_id}")
def delete_portfolio(portfolio_id: int, user=Depends(get_current_user)):
    """Delete a portfolio by ID."""
    from db import USE_DB
    if USE_DB:
        from db import delete_portfolio_db
        deleted = delete_portfolio_db(portfolio_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        return {"status": "deleted"}

    # JSON fallback
    import json, os
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'portfolios.json')
    if os.path.exists(path):
        with open(path) as f:
            portfolios = json.load(f)
        original_len = len(portfolios)
        portfolios = [p for p in portfolios if p.get('id') != portfolio_id]
        if len(portfolios) < original_len:
            with open(path, 'w') as f:
                json.dump(portfolios, f, indent=2)
            return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Portfolio not found")


@router.post("/{portfolio_id}/duplicate")
def duplicate_portfolio(portfolio_id: int, user=Depends(get_current_user)):
    """Duplicate a portfolio. Creates a new copy."""
    from db import USE_DB
    if USE_DB:
        from db import get_portfolio_by_id_db, save_portfolio_db
        source = get_portfolio_by_id_db(portfolio_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        new_portfolio = copy.deepcopy(source)
        new_portfolio.pop('id', None)
        new_portfolio['created_at'] = datetime.now(timezone.utc).isoformat()
        new_portfolio['name'] = source.get('name', 'Portfolio') + " (Copy)"
        new_portfolio.pop('updated_at', None)

        result = save_portfolio_db(new_portfolio)
        return result if result else {"status": "duplicated"}

    raise HTTPException(status_code=501, detail="JSON duplicate not implemented via API")


# =============================================================================
# COMPUTE
# =============================================================================

@router.post("/{portfolio_id}/compute")
def compute_portfolio(
    portfolio_id: int,
    body: dict = Body(...),
    user=Depends(get_current_user),
):
    """Selective portfolio analytics.

    Accepts JSON body with ``include: list[str]`` specifying which computations
    to run. Supported keys: ``kpis``, ``equity_curve``, ``correlation``,
    ``monte_carlo``, ``daily_pnl``, ``benchmark``, ``drawdown``,
    ``equity_curve_with_strategies``.

    Returns only the requested data.
    """
    portfolio = _get_or_404(portfolio_id, user)
    include = body.get('include', [])
    if not include:
        raise HTTPException(status_code=400, detail="'include' list is required")

    from portfolios import get_portfolio_trades, calculate_portfolio_kpis, run_monte_carlo
    import pandas as pd

    # Always need trades for downstream computations
    trade_data = get_portfolio_trades(portfolio, _get_strategy_fn, _get_trades_fn)
    combined_trades = trade_data.get('combined_trades', pd.DataFrame())
    daily_pnl = trade_data.get('daily_pnl', pd.DataFrame())
    starting_balance = portfolio.get('starting_balance', 10000.0)

    result = {}

    if 'kpis' in include:
        result['kpis'] = calculate_portfolio_kpis(portfolio, combined_trades, daily_pnl)

    if 'equity_curve' in include:
        result['equity_curve'] = _serialize_series(trade_data.get('equity_curve'))

    if 'equity_curve_with_strategies' in include:
        # Curve plus per-strategy overlays for the Performance tab.
        # Each strategy's cumulative_pnl is aligned to the combined trade
        # sequence (same length as combined_trades). When a strategy doesn't
        # have a trade at that index, its running sum stays flat (forward-fill).
        # This way overlays span the full portfolio timeline and sum to the combined curve.
        overlays: dict = {}
        if (
            'strategy_id' in combined_trades.columns
            and 'dollar_pnl' in combined_trades.columns
        ):
            n = len(combined_trades)
            strategy_ids = combined_trades['strategy_id'].unique()
            for sid in strategy_ids:
                mask = (combined_trades['strategy_id'] == sid).values
                pnls = combined_trades['dollar_pnl'].where(
                    combined_trades['strategy_id'] == sid, 0
                ).values
                running = pnls.cumsum().tolist()
                overlays[str(sid)] = {
                    'exit_time': _serialize_series(combined_trades.get('exit_time'))
                    if 'exit_time' in combined_trades.columns else [],
                    'cumulative_pnl': running,
                }
        result['equity_curve_with_strategies'] = {
            'combined': {
                'exit_time': _serialize_series(combined_trades.get('exit_time'))
                if 'exit_time' in combined_trades.columns else [],
                'cumulative_pnl': _serialize_series(combined_trades.get('cumulative_pnl'))
                if 'cumulative_pnl' in combined_trades.columns else [],
            },
            'strategies': overlays,
        }

    if 'daily_pnl' in include:
        result['daily_pnl'] = _serialize_df(daily_pnl)

    if 'correlation' in include:
        strategy_daily = trade_data.get('strategy_daily_pnl')
        if strategy_daily is not None and len(strategy_daily) > 0:
            corr = strategy_daily.corr()
            result['correlation'] = corr.where(corr.notna(), None).to_dict()
        else:
            result['correlation'] = {}

    if 'benchmark' in include:
        from portfolios import compute_portfolio_benchmark
        max_trades = body.get('benchmark_max_trades', 200)
        result['benchmark'] = compute_portfolio_benchmark(
            portfolio, _get_strategy_fn, _get_trades_fn,
            max_trades=max_trades,
        )

    if 'drawdown' in include:
        from portfolios import compute_drawdown_series
        dd_df = compute_drawdown_series(combined_trades, starting_balance)
        result['drawdown'] = _serialize_df(dd_df)

    if 'monte_carlo' in include:
        from portfolios import get_daily_limit_thresholds
        thresholds = get_daily_limit_thresholds(portfolio)
        shuffle_mode = body.get('shuffle_mode', 'daily')
        n_simulations = body.get('n_simulations', 1000)
        mc = run_monte_carlo(
            combined_trades, daily_pnl, starting_balance,
            thresholds, n_simulations=n_simulations,
            shuffle_mode=shuffle_mode,
        )
        result['monte_carlo'] = mc

    return _sanitize_for_json(result)


@router.get("/{portfolio_id}/trades")
def get_trades(portfolio_id: int, user=Depends(get_current_user)):
    """Get combined trade data for a portfolio.

    2026-04-22: switched from ``get_portfolio_trades`` (backtest only) to
    ``get_portfolio_combined_view`` which merges backtest + forward-test +
    live alert trades with deduped ``data_source`` tags. Closes a Phase C
    wiring gap — the frontend was already reading ``matched`` / ``phantom``
    / ``data_source`` but the endpoint only returned stored_trades, so
    every row rendered as "Backtest" even when live alerts had fired.
    """
    portfolio = _get_or_404(portfolio_id, user)

    from portfolios import get_portfolio_combined_view

    trade_data = get_portfolio_combined_view(
        portfolio, _get_strategy_fn, _get_trades_fn)
    combined_rows = trade_data.get('combined_trades') or []

    # _sanitize_for_json handles NaN/Inf floats (e.g. open trades have
    # r_multiple=NaN which cascades through dollar_pnl and breaks
    # json.dumps without this pass). Matches the pattern on list/preview
    # endpoints. Kevin observed 500s on portfolio 27 (10s strategy with
    # 800+ trades including open ones) that traced to this.
    return _sanitize_for_json({
        "trades": combined_rows,
        "equity_curve": _serialize_series(trade_data.get('equity_curve')),
    })


# =============================================================================
# REQUIREMENTS CHECK
# =============================================================================

@router.post("/{portfolio_id}/requirements/check")
def check_requirements(
    portfolio_id: int,
    body: dict = Body(...),
    user=Depends(get_current_user),
):
    """Evaluate portfolio against its requirement set.

    Optionally accepts ``requirement_set_id`` in body to override the
    portfolio's default requirement set.
    """
    portfolio = _get_or_404(portfolio_id, user)

    from portfolios import (
        get_portfolio_trades, calculate_portfolio_kpis,
        evaluate_requirement_set, get_requirement_set_by_id,
    )
    import pandas as pd

    # Get requirement set
    req_set_id = body.get('requirement_set_id') or portfolio.get('requirement_set_id')
    if not req_set_id:
        raise HTTPException(status_code=400, detail="No requirement set configured")

    req_set = get_requirement_set_by_id(int(req_set_id))
    if not req_set:
        raise HTTPException(status_code=404, detail="Requirement set not found")

    # Compute trades and KPIs
    trade_data = get_portfolio_trades(portfolio, _get_strategy_fn, _get_trades_fn)
    combined_trades = trade_data.get('combined_trades', pd.DataFrame())
    daily_pnl = trade_data.get('daily_pnl', pd.DataFrame())
    kpis = calculate_portfolio_kpis(portfolio, combined_trades, daily_pnl)

    return _sanitize_for_json(evaluate_requirement_set(req_set, portfolio, kpis, daily_pnl))


# =============================================================================
# HEALTH & ANOMALIES
# =============================================================================

@router.post("/{portfolio_id}/health")
def check_health(
    portfolio_id: int,
    body: dict = Body(default={}),
    user=Depends(get_current_user),
):
    """Classify strategy health for each strategy in the portfolio.

    Optionally accepts ``alert_trades`` and ``benchmark`` in body.
    If not provided, attempts to load from alerts.
    """
    portfolio = _get_or_404(portfolio_id, user)

    from portfolios import (
        classify_strategy_health, get_portfolio_trades,
        calculate_portfolio_kpis,
    )
    from alerts import load_alerts
    import pandas as pd

    alert_trades = body.get('alert_trades')
    if alert_trades is None:
        # Build from alert history
        alerts = load_alerts(limit=5000)
        strategy_ids = {ps['strategy_id'] for ps in portfolio.get('strategies', [])}
        alert_trades = [
            a for a in alerts
            if a.get('strategy_id') in strategy_ids and a.get('type') in ('entry', 'exit')
        ]

    # Build benchmark from backtest KPIs
    trade_data = get_portfolio_trades(portfolio, _get_strategy_fn, _get_trades_fn)
    combined_trades = trade_data.get('combined_trades', pd.DataFrame())
    daily_pnl = trade_data.get('daily_pnl', pd.DataFrame())
    kpis = calculate_portfolio_kpis(portfolio, combined_trades, daily_pnl)

    benchmark = body.get('benchmark', {'overall': kpis, 'per_strategy': {}})
    if 'per_strategy' not in benchmark:
        # Build per-strategy benchmarks from stored KPIs
        per_strat = {}
        for ps in portfolio.get('strategies', []):
            sid = ps['strategy_id']
            strat = _get_strategy_fn(sid)
            if strat and strat.get('kpis'):
                per_strat[sid] = strat['kpis']
        benchmark['per_strategy'] = per_strat

    results = {}
    for ps in portfolio.get('strategies', []):
        sid = ps['strategy_id']
        health = classify_strategy_health(alert_trades, benchmark, sid)
        results[str(sid)] = health

    return results


@router.get("/{portfolio_id}/anomalies")
def get_anomalies(portfolio_id: int, user=Depends(get_current_user)):
    """Detect anomalous conditions in the portfolio's live trading."""
    portfolio = _get_or_404(portfolio_id, user)

    from portfolios import detect_portfolio_anomalies
    from alerts import load_alerts

    # Build alert trades and open positions from alert history
    strategy_ids = {ps['strategy_id'] for ps in portfolio.get('strategies', [])}
    alerts = load_alerts(limit=5000)
    alert_trades = [
        a for a in alerts
        if a.get('strategy_id') in strategy_ids and a.get('type') in ('entry', 'exit')
    ]

    # Open positions from engine state
    open_positions = []
    from db import USE_DB
    if USE_DB:
        from db import load_engine_state_db
        engine_state = load_engine_state_db()
        positions = engine_state.get('positions', {})
        for sym, pos in positions.items():
            if pos.get('status') == 'IN_POSITION':
                open_positions.append(pos)

    anomalies = detect_portfolio_anomalies(
        alert_trades, open_positions, portfolio,
        get_strategy_fn=_get_strategy_fn,
    )
    return anomalies


# =============================================================================
# ACCOUNT
# =============================================================================

@router.get("/{portfolio_id}/account")
def get_account_info(portfolio_id: int, user=Depends(get_current_user)):
    """Get portfolio account (ledger, balance info)."""
    portfolio = _get_or_404(portfolio_id, user)

    from portfolios import get_account
    return get_account(portfolio)


@router.post("/{portfolio_id}/account/deposit")
def deposit(
    portfolio_id: int,
    body: dict = Body(...),
    user=Depends(get_current_user),
):
    """Add a ledger entry (deposit, withdrawal, or trading_pnl)."""
    portfolio = _get_or_404(portfolio_id, user)

    entry_type = body.get('entry_type', 'deposit')
    amount = body.get('amount')
    if amount is None:
        raise HTTPException(status_code=400, detail="'amount' is required")

    note = body.get('note', '')
    date = body.get('date')

    from portfolios import add_ledger_entry
    entry = add_ledger_entry(portfolio, entry_type, float(amount), note=note, date=date)

    # Persist updated portfolio
    from db import USE_DB
    if USE_DB:
        from db import update_portfolio_db
        update_portfolio_db(portfolio_id, portfolio)

    return entry


@router.delete("/{portfolio_id}/account/ledger/{entry_id}")
def delete_ledger_entry(
    portfolio_id: int,
    entry_id: int,
    user=Depends(get_current_user),
):
    """Remove a ledger entry by ID."""
    portfolio = _get_or_404(portfolio_id, user)

    from portfolios import remove_ledger_entry
    removed = remove_ledger_entry(portfolio, entry_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Ledger entry not found")

    # Persist updated portfolio
    from db import USE_DB
    if USE_DB:
        from db import update_portfolio_db
        update_portfolio_db(portfolio_id, portfolio)

    return {"status": "deleted"}


# =============================================================================
# ANALYSIS ENDPOINTS (M8)
# =============================================================================

@router.post("/{portfolio_id}/worst-case")
def worst_case_analysis(
    portfolio_id: int,
    body: dict = Body(default={}),
    user=Depends(get_current_user),
):
    """Compute historical worst-case metrics: worst day, streak, rolling DD, breaches."""
    portfolio = _get_or_404(portfolio_id, user)

    from portfolios import (
        get_portfolio_trades, compute_worst_case_analysis,
        get_daily_limit_thresholds,
    )
    import pandas as pd

    trade_data = get_portfolio_trades(portfolio, _get_strategy_fn, _get_trades_fn)
    daily_pnl = trade_data.get('daily_pnl', pd.DataFrame())
    starting_balance = portfolio.get('starting_balance', 10000.0)
    thresholds = body.get('thresholds') or get_daily_limit_thresholds(portfolio)

    result = compute_worst_case_analysis(daily_pnl, starting_balance, thresholds)
    return _sanitize_for_json(result)


@router.post("/{portfolio_id}/capital-utilization")
def capital_utilization(
    portfolio_id: int,
    body: dict = Body(default={}),
    user=Depends(get_current_user),
):
    """Compute capital utilization timeline from backtest or alert trades.

    Accepts ``source: 'backtest' | 'alerts'`` (default 'backtest').
    """
    portfolio = _get_or_404(portfolio_id, user)
    source = body.get('source', 'backtest')

    from portfolios import get_portfolio_trades, compute_capital_utilization
    import pandas as pd

    if source == 'alerts':
        from portfolios import (
            get_portfolio_alert_trades, compute_alert_buying_power, get_account,
        )
        alert_data = get_portfolio_alert_trades(portfolio, _get_strategy_fn)
        account = get_account(portfolio)
        account_balance = account.get('balance', portfolio.get('starting_balance', 10000.0))
        result = compute_alert_buying_power(
            alert_data.get('alert_trades', []),
            alert_data.get('open_positions', []),
            account_balance,
        )
    else:
        trade_data = get_portfolio_trades(portfolio, _get_strategy_fn, _get_trades_fn)
        combined_trades = trade_data.get('combined_trades', pd.DataFrame())
        starting_balance = portfolio.get('starting_balance', 10000.0)
        result = compute_capital_utilization(combined_trades, starting_balance)
        if result is None:
            return {
                'timeline': [],
                'current_buying_power': starting_balance,
                'peak_capital_deployed': 0,
                'max_concurrent_positions': 0,
                'insufficient_events': [],
                'unavailable_reason': 'Trades missing required columns (entry_price, risk)',
            }

    # Serialize timeline DataFrame if present
    timeline = result.get('timeline')
    if timeline is not None and hasattr(timeline, 'to_dict'):
        result['timeline'] = _serialize_df(timeline)

    return _sanitize_for_json(result)


@router.get("/{portfolio_id}/open-positions")
def open_positions(portfolio_id: int, user=Depends(get_current_user)):
    """Get currently open positions (alert-derived) for a portfolio."""
    portfolio = _get_or_404(portfolio_id, user)

    from portfolios import get_portfolio_alert_trades
    alert_data = get_portfolio_alert_trades(portfolio, _get_strategy_fn)
    return _sanitize_for_json({
        'open_positions': alert_data.get('open_positions', []),
        'strategies_with_data': list(alert_data.get('strategies_with_data', [])),
    })


@router.post("/preview")
def preview_portfolio(
    portfolio: dict = Body(...),
    user=Depends(get_current_user),
):
    """Compute KPIs + equity curve for an unsaved portfolio payload.

    Used by the Portfolio Builder to render real preview data before save.
    """
    from portfolios import get_portfolio_trades, calculate_portfolio_kpis
    import pandas as pd

    if not portfolio.get('strategies'):
        return {
            'kpis': calculate_portfolio_kpis(portfolio, pd.DataFrame(), pd.DataFrame()),
            'equity_curve': [],
            'exit_time': [],
        }

    trade_data = get_portfolio_trades(portfolio, _get_strategy_fn, _get_trades_fn)
    combined = trade_data.get('combined_trades', pd.DataFrame())
    daily_pnl = trade_data.get('daily_pnl', pd.DataFrame())

    result = {
        'kpis': calculate_portfolio_kpis(portfolio, combined, daily_pnl),
        'equity_curve': _serialize_series(trade_data.get('equity_curve')),
        'exit_time': _serialize_series(combined.get('exit_time'))
        if 'exit_time' in combined.columns else [],
    }
    return _sanitize_for_json(result)


@router.post("/recommendations")
def recommend_strategies(
    body: dict = Body(...),
    user=Depends(get_current_user),
):
    """Suggest complementary strategies for a portfolio-in-progress.

    Heuristic: rank candidates by timeframe diversity, exec type diversity, and
    healthy KPIs (PF and win rate). Returns top N scored candidates.
    Body: ``{selected_strategy_ids: list[int], max?: int}``.
    """
    selected_ids = set(body.get('selected_strategy_ids') or [])
    max_results = int(body.get('max', 5))

    from db import USE_DB
    if USE_DB:
        from db import load_strategies_db
        strategies = load_strategies_db() or []
    else:
        import json, os
        path = os.path.join(os.path.dirname(__file__), '..', '..', 'strategies.json')
        strategies = []
        if os.path.exists(path):
            with open(path) as f:
                strategies = json.load(f)

    candidates = [
        s for s in strategies
        if s.get('id') not in selected_ids and s.get('stored_trades')
    ]
    selected_strats = [s for s in strategies if s.get('id') in selected_ids]
    selected_tfs = {s.get('timeframe') for s in selected_strats if s.get('timeframe')}
    selected_exec = {s.get('exec_type') for s in selected_strats if s.get('exec_type')}

    scored = []
    for cand in candidates:
        score = 0
        reasons = []

        cand_tf = cand.get('timeframe')
        if cand_tf and cand_tf not in selected_tfs:
            score += 2
            reasons.append(f"Different timeframe ({cand_tf})")

        cand_exec = cand.get('exec_type')
        if cand_exec and cand_exec not in selected_exec:
            score += 1
            reasons.append(f"Different exec type ({cand_exec})")

        kpis = cand.get('kpis') or {}
        pf = kpis.get('profit_factor')
        if pf and pf != float('inf') and pf > 1.5:
            score += 2
            reasons.append(f"Healthy PF {pf:.2f}")

        wr = kpis.get('win_rate')
        if wr and wr > 50:
            score += 1
            reasons.append(f"Win rate {wr:.0f}%")

        scored.append({
            'strategy_id': cand.get('id'),
            'name': cand.get('name'),
            'symbol': cand.get('symbol'),
            'timeframe': cand_tf,
            'exec_type': cand_exec,
            'score': score,
            'reasons': reasons,
            'kpis': {
                'profit_factor': pf if pf != float('inf') else None,
                'win_rate': wr,
                'total_trades': kpis.get('total_trades'),
            },
        })

    scored.sort(key=lambda x: x['score'], reverse=True)
    return _sanitize_for_json({'recommendations': scored[:max_results]})
