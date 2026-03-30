"""Backtest router — run backtests and compute KPIs."""

import logging

from fastapi import APIRouter, Depends

from api.deps import get_current_user
from api.schemas.backtest import BacktestRequest, BacktestResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


@router.post("/run", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest, user=Depends(get_current_user)):
    """Run a full backtest.

    Accepts strategy configuration, loads market data, runs the unified engine,
    computes KPIs, and returns serialized results.

    This is a sync endpoint — FastAPI runs it in a threadpool.
    Typical response time: 2-10 seconds depending on data size.
    """
    from api.services.backtest_service import run_backtest as _run

    try:
        return _run(req)
    except Exception as e:
        logger.exception("Backtest failed for %s", req.symbol)
        # Return empty result on error rather than 500
        return BacktestResponse(
            trades=[], kpis={
                "total_trades": 0, "win_rate": 0, "profit_factor": 0,
                "avg_r": 0, "total_r": 0, "daily_r": 0, "r_squared": 0.0,
                "max_r_drawdown": 0, "final_balance": 10000, "total_pnl": 0,
            },
            secondary_kpis={},
            equity_curve=[],
            total_bars=0,
            total_trading_days=0,
            data_source="Error",
        )


@router.post("/kpis")
def compute_kpis(trades: list[dict], user=Depends(get_current_user)):
    """Calculate KPIs from a list of trade records.

    Useful for recalculating KPIs on stored trades without running
    a full backtest (e.g., after filtering by date range).
    """
    import pandas as pd
    import services as svc

    if not trades:
        return {"kpis": svc.calculate_kpis(pd.DataFrame()), "secondary_kpis": {}}

    df = pd.DataFrame(trades)
    for col in ["entry_time", "exit_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)

    kpis = svc.calculate_kpis(df)
    secondary = svc.calculate_secondary_kpis(df, kpis)
    return {"kpis": kpis, "secondary_kpis": secondary}


@router.post("/analyze")
def analyze_triggers(req: BacktestRequest, user=Depends(get_current_user)):
    """Run per-trigger analysis: backtest each candidate trigger individually.

    Loads market data once, then runs the unified engine with each trigger
    to produce per-trigger KPIs. Used by the Strategy Builder's Analyze
    button to compare triggers side by side.

    Returns list of {trigger_id, trigger_name, exec_type, kpis} dicts.
    """
    import services as svc
    from api.services.backtest_service import _resolve_stop_from_pack, _resolve_target_from_pack

    # Resolve stop/target config from pack IDs
    stop_config = req.stop_config
    target_config = req.target_config
    if req.stop_loss_pack_id and not stop_config:
        stop_config = _resolve_stop_from_pack(req.stop_loss_pack_id)
    if req.take_profit_pack_id and not target_config:
        target_config = _resolve_target_from_pack(req.take_profit_pack_id)
    if not stop_config:
        stop_config = {"method": "atr", "atr_mult": req.stop_atr_mult}

    # Load data once (the expensive part)
    sec_tfs = tuple(sorted(req.secondary_tfs))
    start_date = end_date = None
    if req.lookback_mode == 'Date Range' and req.lookback_start_date:
        from datetime import datetime
        start_date = datetime.fromisoformat(req.lookback_start_date)
        if req.lookback_end_date:
            end_date = datetime.fromisoformat(req.lookback_end_date)

    try:
        df = svc.prepare_data_with_indicators(
            req.symbol, days=req.days, start_date=start_date,
            end_date=end_date, timeframe=req.timeframe,
            data_feed=req.data_feed, session=req.session,
            secondary_tfs=sec_tfs,
        )
    except Exception as e:
        logger.exception("Analyze data load failed for %s", req.symbol)
        return {"results": [], "error": str(e)}

    if len(df) == 0:
        return {"results": []}

    trading_days = svc.count_trading_days(df)

    # Get all available entry triggers for this direction
    from confluence_groups import get_enabled_groups, get_entry_triggers, get_exit_triggers
    groups = get_enabled_groups()
    if req.direction in ('LONG', 'SHORT'):
        candidates = get_entry_triggers(req.direction, groups)
    else:
        candidates = get_exit_triggers(groups)

    # Use the request's exit triggers for all runs
    exit_triggers = req.exit_trigger_confluence_ids

    results = []
    for trigger_id, trigger_name in candidates.items():
        # Build a strategy dict for this specific trigger
        strategy = {
            "id": f"analyze_{trigger_id}",
            "symbol": req.symbol,
            "timeframe": req.timeframe,
            "direction": req.direction,
            "session": req.session,
            "entry_trigger_confluence_id": trigger_id,
            "exit_trigger_confluence_ids": exit_triggers,
            "confluence": list(req.confluence),
            "stop_config": stop_config,
            "target_config": target_config,
            "stop_atr_mult": req.stop_atr_mult,
            "risk_per_trade": req.risk_per_trade,
            "bar_count_exit": req.bar_count_exit,
        }

        try:
            trades_df = svc.unified_trades(df, strategy, include_open_position=False)
            if len(trades_df) == 0:
                continue

            kpis = svc.calculate_kpis(
                trades_df, risk_per_trade=req.risk_per_trade,
                total_trading_days=trading_days,
            )

            # Derive exec type from trigger suffix
            exec_type = 'C'
            if trigger_id.endswith('_lc'): exec_type = 'LC'
            elif trigger_id.endswith('_cc'): exec_type = 'CC'
            elif trigger_id.endswith('_ib'): exec_type = 'L'
            elif trigger_id.endswith('_hm') or trigger_id.endswith('_hl'): exec_type = 'LC'

            results.append({
                "trigger_id": trigger_id,
                "trigger_name": trigger_name,
                "exec_type": exec_type,
                "total_trades": kpis.get("total_trades", 0),
                "profit_factor": round(kpis.get("profit_factor", 0), 2),
                "win_rate": round(kpis.get("win_rate", 0), 1),
                "avg_r": round(kpis.get("avg_r", 0), 3),
                "daily_r": round(kpis.get("daily_r", 0), 3),
                "r_squared": round(kpis.get("r_squared", 0), 3),
            })
        except Exception as e:
            logger.warning("Analyze failed for trigger %s: %s", trigger_id, e)
            continue

    # Sort by daily_r descending (best first)
    results.sort(key=lambda r: r.get("daily_r", 0), reverse=True)
    return {"results": results}
