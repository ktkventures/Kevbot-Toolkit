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


# =============================================================================
# CRUD
# =============================================================================

@router.get("")
def list_portfolios(user=Depends(get_current_user)):
    """Load all portfolios for the current user."""
    from db import USE_DB
    if USE_DB:
        from db import load_portfolios_db
        return load_portfolios_db()

    # JSON fallback
    import json, os
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'portfolios.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


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
    ``monte_carlo``, ``daily_pnl``.

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

    result = {}

    if 'kpis' in include:
        result['kpis'] = calculate_portfolio_kpis(portfolio, combined_trades, daily_pnl)

    if 'equity_curve' in include:
        result['equity_curve'] = _serialize_series(trade_data.get('equity_curve'))

    if 'daily_pnl' in include:
        result['daily_pnl'] = _serialize_df(daily_pnl)

    if 'correlation' in include:
        strategy_daily = trade_data.get('strategy_daily_pnl')
        if strategy_daily is not None and len(strategy_daily) > 0:
            corr = strategy_daily.corr()
            result['correlation'] = corr.where(corr.notna(), None).to_dict()
        else:
            result['correlation'] = {}

    if 'monte_carlo' in include:
        from portfolios import get_daily_limit_thresholds
        thresholds = get_daily_limit_thresholds(portfolio)
        starting_balance = portfolio.get('starting_balance', 10000.0)
        shuffle_mode = body.get('shuffle_mode', 'daily')
        n_simulations = body.get('n_simulations', 1000)
        mc = run_monte_carlo(
            combined_trades, daily_pnl, starting_balance,
            thresholds, n_simulations=n_simulations,
            shuffle_mode=shuffle_mode,
        )
        result['monte_carlo'] = mc

    return result


@router.get("/{portfolio_id}/trades")
def get_trades(portfolio_id: int, user=Depends(get_current_user)):
    """Get combined trade data for a portfolio."""
    portfolio = _get_or_404(portfolio_id, user)

    from portfolios import get_portfolio_trades

    trade_data = get_portfolio_trades(portfolio, _get_strategy_fn, _get_trades_fn)
    combined = trade_data.get('combined_trades')

    return {
        "trades": _serialize_df(combined),
        "equity_curve": _serialize_series(trade_data.get('equity_curve')),
    }


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

    return evaluate_requirement_set(req_set, portfolio, kpis, daily_pnl)


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
