"""Backtest router — run backtests and compute KPIs."""

import logging

from fastapi import APIRouter, Depends, Query

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
def analyze_triggers(
    req: BacktestRequest,
    mode: str = Query("entry", description="entry|exit|condition|stop|target"),
    user=Depends(get_current_user),
):
    """Run per-component analysis: backtest with each candidate swapped in.

    Modes:
      - entry: test each available entry trigger (default)
      - exit: test each available exit trigger
      - condition: test with/without each TF confluence condition
      - stop: test each available stop loss pack
      - target: test each available take profit pack
    """
    try:
        if mode == "exit":
            return _analyze_exits_impl(req)
        elif mode == "condition":
            return _analyze_conditions_impl(req)
        elif mode == "stop":
            return _analyze_stops_impl(req)
        elif mode == "target":
            return _analyze_targets_impl(req)
        else:
            return _analyze_triggers_impl(req)
    except Exception as e:
        logger.exception("Analyze endpoint failed (mode=%s) for %s", mode, req.symbol)
        return {"results": [], "error": str(e)}


def _analyze_triggers_impl(req: BacktestRequest):
    """Test each available entry trigger while keeping exit/stop/target fixed."""
    import services as svc
    stop_config, target_config = _resolve_configs(req)
    df, trading_days = _load_analyze_data(req)
    if len(df) == 0:
        return {"results": []}

    from confluence_groups import get_enabled_groups, get_entry_triggers
    groups = get_enabled_groups()
    candidates = get_entry_triggers(req.direction, groups) if req.direction in ('LONG', 'SHORT') else {}

    results = []
    for trigger_id, trigger_name in candidates.items():
        strategy = _build_base_strategy(req, stop_config, target_config)
        strategy["id"] = f"analyze_{trigger_id}"
        strategy["entry_trigger_confluence_id"] = trigger_id

        try:
            trades_df = svc.unified_trades(df, strategy, include_open_position=False)
            if len(trades_df) == 0:
                continue
            kpis = svc.calculate_kpis(trades_df, risk_per_trade=req.risk_per_trade,
                                      total_trading_days=trading_days)
            exec_type = 'C'
            if trigger_id.endswith('_lc'): exec_type = 'LC'
            elif trigger_id.endswith('_cc'): exec_type = 'CC'
            elif trigger_id.endswith('_ib'): exec_type = 'L'
            elif trigger_id.endswith('_hm') or trigger_id.endswith('_hl'): exec_type = 'LC'
            results.append(_kpis_to_result(kpis, trigger_id, trigger_name, exec_type))
        except Exception as e:
            logger.warning("Analyze failed for trigger %s: %s", trigger_id, e)

    results.sort(key=lambda r: r.get("daily_r", 0), reverse=True)
    return {"results": results}


def _load_analyze_data(req):
    """Shared data loading for all analyze modes."""
    import services as svc
    from datetime import datetime
    start_date = end_date = None
    if req.lookback_mode == 'Date Range' and req.lookback_start_date:
        start_date = datetime.fromisoformat(req.lookback_start_date)
        if req.lookback_end_date:
            end_date = datetime.fromisoformat(req.lookback_end_date)
    sec_tfs = tuple(sorted(req.secondary_tfs))
    df = svc.prepare_data_with_indicators(
        req.symbol, days=req.days, start_date=start_date,
        end_date=end_date, timeframe=req.timeframe,
        data_feed=req.data_feed, session=req.session,
        secondary_tfs=sec_tfs,
    )
    trading_days = svc.count_trading_days(df) if len(df) > 0 else 1
    return df, trading_days


def _resolve_configs(req):
    """Resolve stop/target configs from pack IDs."""
    from api.services.backtest_service import _resolve_stop_from_pack, _resolve_target_from_pack
    stop_config = req.stop_config
    target_config = req.target_config
    if req.stop_loss_pack_id and not stop_config:
        stop_config = _resolve_stop_from_pack(req.stop_loss_pack_id)
    if req.take_profit_pack_id and not target_config:
        target_config = _resolve_target_from_pack(req.take_profit_pack_id)
    if not stop_config:
        stop_config = {"method": "atr", "atr_mult": req.stop_atr_mult}
    return stop_config, target_config


def _build_base_strategy(req, stop_config, target_config):
    """Build the base strategy dict from a request."""
    return {
        "symbol": req.symbol, "timeframe": req.timeframe,
        "direction": req.direction, "session": req.session,
        "entry_trigger_confluence_id": req.entry_trigger_confluence_id,
        "exit_trigger_confluence_ids": req.exit_trigger_confluence_ids,
        "confluence": list(req.confluence),
        "stop_config": stop_config, "target_config": target_config,
        "stop_atr_mult": req.stop_atr_mult,
        "risk_per_trade": req.risk_per_trade,
        "bar_count_exit": req.bar_count_exit,
    }


def _kpis_to_result(kpis, label, sub_label="", exec_type="C"):
    """Convert KPIs dict to a standard result dict."""
    return {
        "trigger_id": label,
        "trigger_name": sub_label or label,
        "exec_type": exec_type,
        "total_trades": kpis.get("total_trades", 0),
        "profit_factor": round(kpis.get("profit_factor", 0), 2),
        "win_rate": round(kpis.get("win_rate", 0), 1),
        "avg_r": round(kpis.get("avg_r", 0), 3),
        "daily_r": round(kpis.get("daily_r", 0), 3),
        "r_squared": round(kpis.get("r_squared", 0), 3),
    }


def _run_base_trades(req, df, stop_config, target_config):
    """Run ONE backtest with NO confluence filtering to get all possible trades."""
    import services as svc
    strategy = _build_base_strategy(req, stop_config, target_config)
    strategy["confluence"] = []  # No confluence gating → all trades
    strategy["id"] = "analyze_base"
    return svc.unified_trades(df, strategy, include_open_position=False)


def _analyze_exits_impl(req):
    """Test each available exit trigger while keeping entry fixed."""
    import services as svc
    stop_config, target_config = _resolve_configs(req)
    df, trading_days = _load_analyze_data(req)
    if len(df) == 0:
        return {"results": []}

    from confluence_groups import get_enabled_groups, get_exit_triggers
    groups = get_enabled_groups()
    candidates = get_exit_triggers(groups)
    results = []

    for trigger_id, trigger_name in candidates.items():
        strategy = _build_base_strategy(req, stop_config, target_config)
        strategy["id"] = f"analyze_exit_{trigger_id}"
        strategy["exit_trigger_confluence_ids"] = [trigger_id]
        if 'bar_count' in trigger_id:
            strategy["bar_count_exit"] = 4

        try:
            trades_df = svc.unified_trades(df, strategy, include_open_position=False)
            if len(trades_df) == 0:
                continue
            kpis = svc.calculate_kpis(trades_df, risk_per_trade=req.risk_per_trade,
                                      total_trading_days=trading_days)
            exec_type = 'C'
            if trigger_id.endswith('_lc'): exec_type = 'LC'
            elif trigger_id.endswith('_cc'): exec_type = 'CC'
            elif trigger_id.endswith('_ib'): exec_type = 'L'
            results.append(_kpis_to_result(kpis, trigger_id, trigger_name, exec_type))
        except Exception as e:
            logger.warning("Analyze exit failed for %s: %s", trigger_id, e)

    results.sort(key=lambda r: r.get("daily_r", 0), reverse=True)
    return {"results": results}


def _analyze_conditions_impl(req):
    """Per-condition analysis using the fast Streamlit approach.

    Runs ONE backtest with NO confluence filtering, then filters the
    resulting trades by which conditions were active at entry time.
    """
    import services as svc
    import pandas as pd

    stop_config, target_config = _resolve_configs(req)
    df, trading_days = _load_analyze_data(req)
    if len(df) == 0:
        return {"results": []}

    # Run ONE backtest with no confluence gating → get ALL possible trades
    base_trades = _run_base_trades(req, df, stop_config, target_config)
    if len(base_trades) == 0:
        return {"results": []}

    # Extract all unique conditions from confluence_records
    all_conditions = set()
    for _, row in base_trades.iterrows():
        cr = row.get('confluence_records')
        if isinstance(cr, (set, frozenset)):
            all_conditions.update(cr)
        elif isinstance(cr, list):
            all_conditions.update(cr)

    # Filter out GEN- conditions (those belong in General tab)
    tf_conditions = {c for c in all_conditions if not c.startswith('GEN-')}

    results = []
    for cond_id in sorted(tf_conditions):
        # Filter trades where this condition was active at entry
        mask = base_trades['confluence_records'].apply(
            lambda r: isinstance(r, (set, frozenset, list)) and cond_id in r)
        subset = base_trades[mask]
        if len(subset) < 2:
            continue

        try:
            kpis = svc.calculate_kpis(subset, risk_per_trade=req.risk_per_trade,
                                      total_trading_days=trading_days)
            results.append(_kpis_to_result(kpis, cond_id, cond_id))
        except Exception as e:
            logger.warning("Analyze condition failed for %s: %s", cond_id, e)

    results.sort(key=lambda r: r.get("daily_r", 0), reverse=True)
    return {"results": results}


def _analyze_stops_impl(req):
    """Test each available stop loss pack."""
    import services as svc
    _, target_config = _resolve_configs(req)
    df, trading_days = _load_analyze_data(req)
    if len(df) == 0:
        return {"results": []}

    from risk_management_packs import load_risk_management_packs
    packs = load_risk_management_packs()
    results = []

    for pack in packs:
        if not pack.enabled:
            continue
        try:
            sc = pack.get_stop_config()
        except Exception:
            continue
        if not sc:
            continue

        strategy = _build_base_strategy(req, sc, target_config)
        strategy["id"] = f"analyze_stop_{pack.id}"

        try:
            trades_df = svc.unified_trades(df, strategy, include_open_position=False)
            if len(trades_df) == 0:
                continue
            kpis = svc.calculate_kpis(trades_df, risk_per_trade=req.risk_per_trade,
                                      total_trading_days=trading_days)
            label = f"{pack.base_template} ({pack.version})"
            results.append(_kpis_to_result(kpis, pack.id, label))
        except Exception as e:
            logger.warning("Analyze stop failed for %s: %s", pack.id, e)

    results.sort(key=lambda r: r.get("daily_r", 0), reverse=True)
    return {"results": results}


def _analyze_targets_impl(req):
    """Test each available take profit pack."""
    import services as svc
    stop_config, _ = _resolve_configs(req)
    df, trading_days = _load_analyze_data(req)
    if len(df) == 0:
        return {"results": []}

    from risk_management_packs import load_risk_management_packs
    packs = load_risk_management_packs()
    results = []

    for pack in packs:
        if not pack.enabled:
            continue
        try:
            tc = pack.get_target_config()
        except Exception:
            continue
        if not tc:
            continue

        strategy = _build_base_strategy(req, stop_config, tc)
        strategy["id"] = f"analyze_target_{pack.id}"

        try:
            trades_df = svc.unified_trades(df, strategy, include_open_position=False)
            if len(trades_df) == 0:
                continue
            kpis = svc.calculate_kpis(trades_df, risk_per_trade=req.risk_per_trade,
                                      total_trading_days=trading_days)
            label = f"{pack.base_template} ({pack.version})"
            results.append(_kpis_to_result(kpis, pack.id, label))
        except Exception as e:
            logger.warning("Analyze target failed for %s: %s", pack.id, e)

    results.sort(key=lambda r: r.get("daily_r", 0), reverse=True)
    return {"results": results}
    start_date = end_date = None
    if req.lookback_mode == 'Date Range' and req.lookback_start_date:
        start_date = datetime.fromisoformat(req.lookback_start_date)
        if req.lookback_end_date:
            end_date = datetime.fromisoformat(req.lookback_end_date)
    sec_tfs = tuple(sorted(req.secondary_tfs))
    df = svc.prepare_data_with_indicators(
        req.symbol, days=req.days, start_date=start_date,
        end_date=end_date, timeframe=req.timeframe,
        data_feed=req.data_feed, session=req.session,
        secondary_tfs=sec_tfs,
    )
    trading_days = svc.count_trading_days(df) if len(df) > 0 else 1
    return df, trading_days

