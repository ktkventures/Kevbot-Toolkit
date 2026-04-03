"""Backtest router — run backtests and compute KPIs."""

import logging

from fastapi import APIRouter, Depends, Query

from api.deps import get_current_user
from api.schemas.backtest import BacktestRequest, BacktestTradeZoomRequest, BacktestResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


@router.post("/run")
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


@router.post("/trade-zoom")
def backtest_trade_zoom(req: BacktestTradeZoomRequest, user=Depends(get_current_user)):
    """Fetch 1-second bars around a trade from an unsaved backtest.

    Like GET /api/strategies/{id}/trade-zoom but works without a saved strategy —
    accepts the full backtest config, re-runs the engine, and zooms into the
    specified trade.
    """
    from api.services.backtest_service import (
        build_trade_zoom_response, extract_relevant_prefixes,
        _hifi_resolve_trades,
    )
    import services as svc

    try:
        # Load enriched data (same as analyze endpoints)
        df, _trading_days = _load_analyze_data(req)
        if len(df) == 0:
            return {"bars_1s": [], "trade": {}, "indicators": {},
                    "side": req.side, "timeframe": req.timeframe, "symbol": req.symbol}

        # Build strategy dict and run unified engine
        stop_config, target_config = _resolve_configs(req)
        strategy = _build_base_strategy(req, stop_config, target_config)
        strategy["id"] = "trade_zoom_ephemeral"
        trades_df = svc.unified_trades(df, strategy, include_open_position=False)

        # Hi-Fi pass if requested
        if req.hifi_mode and len(trades_df) > 0:
            if 'hifi_resolved' not in trades_df.columns:
                trades_df['hifi_resolved'] = False
            trades_df = _hifi_resolve_trades(trades_df, req.symbol, req.timeframe)

        # Validate trade index
        if len(trades_df) == 0:
            return {"bars_1s": [], "trade": {}, "indicators": {},
                    "side": req.side, "timeframe": req.timeframe, "symbol": req.symbol}
        if req.trade_idx < 0 or req.trade_idx >= len(trades_df):
            return {"bars_1s": [], "trade": {}, "indicators": {},
                    "side": req.side, "timeframe": req.timeframe, "symbol": req.symbol,
                    "error": f"trade_idx {req.trade_idx} out of range (0-{len(trades_df)-1})"}

        trade = trades_df.iloc[req.trade_idx].to_dict()

        # Extract relevant indicator prefixes for stepped overlay
        prefixes = extract_relevant_prefixes(
            entry_id=req.entry_trigger_confluence_id,
            exit_ids=req.exit_trigger_confluence_ids,
            confluence=req.confluence,
        )

        return build_trade_zoom_response(
            trade=trade,
            symbol=req.symbol,
            timeframe=req.timeframe,
            side=req.side,
            padding_seconds=req.padding_seconds,
            indicator_df=df,
            relevant_prefixes=prefixes,
        )

    except Exception as e:
        logger.exception("Backtest trade-zoom failed for %s", req.symbol)
        return {"bars_1s": [], "trade": {}, "indicators": {},
                "side": req.side, "timeframe": req.timeframe, "symbol": req.symbol,
                "error": str(e)}


@router.post("/analyze")
def analyze_triggers(
    req: BacktestRequest,
    mode: str = Query("entry", description="entry|exit|condition|general|combinations|general_combinations|stop|target"),
    depth: int = Query(1, description="Max combination depth for combinations mode"),
    user=Depends(get_current_user),
):
    """Run per-component analysis: backtest with each candidate swapped in.

    Modes:
      - entry: test each available entry trigger (default)
      - exit: test each available exit trigger
      - condition: TF conditions — filter trades by confluence_records (fast)
      - general: General conditions — filter for GEN- prefix conditions
      - stop: test each available stop loss pack
      - target: test each available take profit pack
    """
    try:
        if mode == "exit":
            return _analyze_exits_impl(req)
        elif mode == "condition":
            return _analyze_conditions_impl(req)
        elif mode == "general":
            return _analyze_general_impl(req)
        elif mode == "combinations":
            return _analyze_combinations_impl(req, depth, exclude_prefix='GEN-')
        elif mode == "general_combinations":
            return _analyze_combinations_impl(req, depth, include_prefix='GEN-')
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
            if req.hifi_mode and len(trades_df) > 0:
                from api.services.backtest_service import _hifi_resolve_trades
                if 'hifi_resolved' not in trades_df.columns:
                    trades_df['hifi_resolved'] = False
                trades_df = _hifi_resolve_trades(trades_df, req.symbol, req.timeframe)
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
    """Convert KPIs dict to a standard result dict. Sanitizes inf/nan for JSON."""
    import math
    def _sf(v):
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return 0.0
        return v
    return {
        "trigger_id": label,
        "trigger_name": sub_label or label,
        "exec_type": exec_type,
        "total_trades": kpis.get("total_trades", 0),
        "profit_factor": round(_sf(kpis.get("profit_factor", 0)), 2),
        "win_rate": round(_sf(kpis.get("win_rate", 0)), 1),
        "avg_r": round(_sf(kpis.get("avg_r", 0)), 3),
        "daily_r": round(_sf(kpis.get("daily_r", 0)), 3),
        "r_squared": round(_sf(kpis.get("r_squared", 0)), 3),
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
            if req.hifi_mode and len(trades_df) > 0:
                from api.services.backtest_service import _hifi_resolve_trades
                if 'hifi_resolved' not in trades_df.columns:
                    trades_df['hifi_resolved'] = False
                trades_df = _hifi_resolve_trades(trades_df, req.symbol, req.timeframe)
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
    """TF Conditions analysis — ported from Streamlit analyze_confluences().

    Runs ONE backtest with NO confluence filtering, then uses
    analyze_confluences() to filter trades by confluence_records membership.
    Excludes GEN- prefix conditions (those belong in General tab).

    When hifi_mode is enabled, also generates [CB] variants by recomputing
    secondary-TF indicators at the exact trigger moment.
    """
    import services as svc

    stop_config, target_config = _resolve_configs(req)
    logger.info("[ANALYZE-COND] secondary_tfs=%s, confluence=%s", req.secondary_tfs, req.confluence)
    df, trading_days = _load_analyze_data(req)
    logger.info("[ANALYZE-COND] Data: %d bars, columns with '__': %s",
                len(df), [c for c in df.columns if '__' in c] if len(df) > 0 else [])
    if len(df) == 0:
        return {"results": []}

    base_trades = _run_base_trades(req, df, stop_config, target_config)
    logger.info("[ANALYZE-COND] Base trades: %d", len(base_trades))
    if len(base_trades) == 0:
        return {"results": []}

    # Log sample confluence_records
    if 'confluence_records' in base_trades.columns and len(base_trades) > 0:
        sample_cr = base_trades.iloc[0].get('confluence_records')
        logger.info("[ANALYZE-COND] Sample confluence_records (type=%s): %s",
                    type(sample_cr).__name__, list(sample_cr)[:10] if sample_cr else 'None')

    # PB analysis (standard — using last closed bar's state)
    condition_results = svc.analyze_confluences(
        base_trades,
        required=set(req.confluence) if req.confluence else None,
        min_trades=3,
        risk_per_trade=req.risk_per_trade,
        total_trading_days=trading_days,
        exclude_prefix='GEN-',
    )

    results = []
    for cr in condition_results:
        results.append({
            'trigger_id': cr['confluence'],
            'trigger_name': cr['confluence'],
            'fidelity_type': 'PB',
            'exec_type': 'C',
            'total_trades': cr['total_trades'],
            'profit_factor': cr['profit_factor'],
            'win_rate': cr['win_rate'],
            'avg_r': cr['avg_r'],
            'daily_r': cr['daily_r'],
            'r_squared': cr['r_squared'],
            'pf_change': cr.get('pf_change', 0),
            'wr_change': cr.get('wr_change', 0),
        })

    # CB analysis (Hi-Fi only — recompute secondary-TF state at trigger moment)
    if req.hifi_mode and req.secondary_tfs and len(base_trades) > 0:
        try:
            from api.services.backtest_service import recompute_cb_confluence
            cb_trades = base_trades.copy()
            cb_trades = recompute_cb_confluence(
                cb_trades, req.symbol, req.timeframe,
                secondary_tfs=list(req.secondary_tfs),
                session=req.session,
            )

            if 'cb_confluence_records' in cb_trades.columns:
                cb_condition_results = svc.analyze_confluences(
                    cb_trades,
                    required=set(req.confluence) if req.confluence else None,
                    min_trades=3,
                    risk_per_trade=req.risk_per_trade,
                    total_trading_days=trading_days,
                    exclude_prefix='GEN-',
                    confluence_col='cb_confluence_records',
                )

                for cr in cb_condition_results:
                    results.append({
                        'trigger_id': cr['confluence'] + '[CB]',
                        'trigger_name': cr['confluence'],
                        'fidelity_type': 'CB',
                        'exec_type': 'C',
                        'total_trades': cr['total_trades'],
                        'profit_factor': cr['profit_factor'],
                        'win_rate': cr['win_rate'],
                        'avg_r': cr['avg_r'],
                        'daily_r': cr['daily_r'],
                        'r_squared': cr['r_squared'],
                        'pf_change': cr.get('pf_change', 0),
                        'wr_change': cr.get('wr_change', 0),
                    })
        except Exception as e:
            logger.warning("[ANALYZE-COND] CB recomputation failed: %s", e)

    return {"results": results}


def _analyze_general_impl(req):
    """General conditions analysis — same as TF but only GEN- prefix."""
    import services as svc

    stop_config, target_config = _resolve_configs(req)
    df, trading_days = _load_analyze_data(req)
    if len(df) == 0:
        return {"results": []}

    base_trades = _run_base_trades(req, df, stop_config, target_config)
    if len(base_trades) == 0:
        return {"results": []}

    condition_results = svc.analyze_confluences(
        base_trades,
        required=set(req.confluence) if req.confluence else None,
        min_trades=3,
        risk_per_trade=req.risk_per_trade,
        total_trading_days=trading_days,
        include_prefix='GEN-',  # General tab only shows GEN- conditions
    )

    results = []
    for cr in condition_results:
        results.append({
            'trigger_id': cr['confluence'],
            'trigger_name': cr['confluence'],
            'exec_type': 'C',
            'total_trades': cr['total_trades'],
            'profit_factor': cr['profit_factor'],
            'win_rate': cr['win_rate'],
            'avg_r': cr['avg_r'],
            'daily_r': cr['daily_r'],
            'r_squared': cr['r_squared'],
        })
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
            if req.hifi_mode and len(trades_df) > 0:
                from api.services.backtest_service import _hifi_resolve_trades
                if 'hifi_resolved' not in trades_df.columns:
                    trades_df['hifi_resolved'] = False
                trades_df = _hifi_resolve_trades(trades_df, req.symbol, req.timeframe)
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
            if req.hifi_mode and len(trades_df) > 0:
                from api.services.backtest_service import _hifi_resolve_trades
                if 'hifi_resolved' not in trades_df.columns:
                    trades_df['hifi_resolved'] = False
                trades_df = _hifi_resolve_trades(trades_df, req.symbol, req.timeframe)
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


def _analyze_combinations_impl(req, max_depth: int = 2,
                               exclude_prefix: str = None,
                               include_prefix: str = None):
    """Find best confluence condition combinations using find_best_combinations()."""
    import services as svc
    import traceback

    logger.info("[ANALYZE-COMBO] max_depth=%d, exclude=%s, include=%s", max_depth, exclude_prefix, include_prefix)

    try:
        stop_config, target_config = _resolve_configs(req)
        df, trading_days = _load_analyze_data(req)
        logger.info("[ANALYZE-COMBO] Data loaded: %d bars", len(df))
        if len(df) == 0:
            return {"results": []}

        base_trades = _run_base_trades(req, df, stop_config, target_config)
        logger.info("[ANALYZE-COMBO] Base trades: %d", len(base_trades))
        if len(base_trades) == 0:
            return {"results": []}

        # Log sample confluence_records
        if 'confluence_records' in base_trades.columns and len(base_trades) > 0:
            sample = base_trades.iloc[0].get('confluence_records')
            gen_count = sum(1 for r in (sample or []) if isinstance(r, str) and r.startswith('GEN-'))
            logger.info("[ANALYZE-COMBO] Sample CR: %d total, %d GEN-, type=%s",
                        len(sample) if sample else 0, gen_count, type(sample).__name__)

        effective_exclude = exclude_prefix
        if include_prefix and not exclude_prefix:
            effective_exclude = None

        logger.info("[ANALYZE-COMBO] Calling find_best_combinations with effective_exclude=%s, include_prefix=%s", effective_exclude, include_prefix)
        combo_results = svc.find_best_combinations(
            base_trades,
            max_depth=max_depth,
            min_trades=3,
            top_n=50,
            risk_per_trade=req.risk_per_trade,
            total_trading_days=trading_days,
            exclude_prefix=effective_exclude,
            include_prefix=include_prefix,
        )
        logger.info("[ANALYZE-COMBO] Raw results: %d combos", len(combo_results))

        import math
        def _sf(v):
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                return 0.0
            return v

        results = []
        for cr in combo_results:
            results.append({
                'trigger_id': cr.get('combo_str', ''),
                'trigger_name': cr.get('combo_str', ''),
                'exec_type': 'C',
                'total_trades': cr.get('total_trades', 0),
                'profit_factor': round(_sf(cr.get('profit_factor', 0)), 2),
                'win_rate': round(_sf(cr.get('win_rate', 0)), 1),
                'avg_r': round(_sf(cr.get('avg_r', 0)), 3),
                'daily_r': round(_sf(cr.get('daily_r', 0)), 3),
                'r_squared': round(_sf(cr.get('r_squared', 0)), 3),
                'depth': cr.get('depth', 1),
                'combination': cr.get('combination', []),
            })
        logger.info("[ANALYZE-COMBO] Returning %d results", len(results))
        return {"results": results}

    except Exception as e:
        logger.exception("[ANALYZE-COMBO] FAILED: %s", e)
        return {"results": [], "error": str(e)}
