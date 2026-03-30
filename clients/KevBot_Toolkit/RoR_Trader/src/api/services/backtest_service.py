"""
Backtest service — orchestrates the full backtest pipeline.

Calls services.py functions (extracted from app.py) to:
1. Load + enrich market data
2. Run unified engine
3. Calculate KPIs
4. Serialize results for JSON transport
"""

import logging
from datetime import datetime

import pandas as pd
import numpy as np

import services as svc
from data_loader import get_data_source
from api.schemas.backtest import BacktestRequest, BacktestResponse

logger = logging.getLogger(__name__)


def run_backtest(req: BacktestRequest) -> BacktestResponse:
    """Execute a full backtest and return serialized results."""

    # 1. Determine date range
    start_date = None
    end_date = None
    if req.lookback_mode == "Date Range" and req.lookback_start_date:
        start_date = datetime.fromisoformat(req.lookback_start_date)
        if req.lookback_end_date:
            end_date = datetime.fromisoformat(req.lookback_end_date)

    # 2. Resolve stop/target config from packs if needed
    stop_config = req.stop_config
    target_config = req.target_config

    if req.stop_loss_pack_id and not stop_config:
        stop_config = _resolve_stop_from_pack(req.stop_loss_pack_id)
    if req.take_profit_pack_id and not target_config:
        target_config = _resolve_target_from_pack(req.take_profit_pack_id)

    # Default stop if nothing specified
    if not stop_config:
        stop_config = {"method": "atr", "atr_mult": req.stop_atr_mult}

    # 3. Build strategy dict (matches the format unified_engine expects)
    strategy = {
        "symbol": req.symbol,
        "direction": req.direction,
        "timeframe": req.timeframe,
        "trading_session": req.session,
        "entry_trigger_confluence_id": req.entry_trigger_confluence_id,
        "exit_trigger_confluence_ids": req.exit_trigger_confluence_ids,
        "confluence": req.confluence,
        "stop_config": stop_config,
        "target_config": target_config,
        "stop_atr_mult": req.stop_atr_mult,
        "risk_per_trade": req.risk_per_trade,
        "bar_count_exit": req.bar_count_exit,
    }

    # 4. Load + enrich data
    sec_tfs = tuple(sorted(req.secondary_tfs))
    print(f"[BACKTEST] symbol={req.symbol}, tf={req.timeframe}, dir={req.direction}, days={req.days}")
    print(f"[BACKTEST] entry={req.entry_trigger_confluence_id}")
    print(f"[BACKTEST] exits={req.exit_trigger_confluence_ids}")
    print(f"[BACKTEST] stop_config={stop_config}")
    print(f"[BACKTEST] confluence={req.confluence}")

    df = svc.prepare_data_with_indicators(
        req.symbol, days=req.days, start_date=start_date,
        end_date=end_date, timeframe=req.timeframe,
        data_feed=req.data_feed, session=req.session,
        secondary_tfs=sec_tfs,
    )

    print(f"[BACKTEST] Data loaded: {len(df)} bars, source={get_data_source()}")

    if len(df) == 0:
        return BacktestResponse(
            trades=[], kpis=_empty_kpis(), secondary_kpis={},
            equity_curve=[], total_bars=0, total_trading_days=0,
            data_source=get_data_source(),
        )

    # 5. Run unified engine
    trades_df = svc.unified_trades(df, strategy)
    print(f"[BACKTEST] Unified engine returned {len(trades_df)} trades")
    if len(trades_df) > 0:
        print(f"[BACKTEST] First trade: {trades_df.iloc[0].to_dict()}")
        wins = trades_df['win'].sum() if 'win' in trades_df.columns else 'N/A'
        print(f"[BACKTEST] Wins: {wins} / {len(trades_df)}")

    # 6. Calculate KPIs
    trading_days = svc.count_trading_days(df)
    kpis = svc.calculate_kpis(
        trades_df, starting_balance=10000,
        risk_per_trade=req.risk_per_trade,
        total_trading_days=trading_days,
    )
    secondary_kpis = svc.calculate_secondary_kpis(trades_df, kpis)

    # 7. Build equity curve
    equity_curve = _build_equity_curve(trades_df)

    # 8. Serialize trades
    trades = _serialize_trades(trades_df)

    # 9. Optional chart data with indicator classification
    # Port from Strategy Detail chart-data endpoint (strategies.py lines 320-468)
    chart_data = None
    overlay_indicators = []
    oscillator_indicators = []
    heatmap_conditions = []
    if req.include_chart_data:
        from confluence_groups import get_enabled_groups, get_template, TEMPLATES

        OVERLAY_TEMPLATES = {"ema_stack", "ema_price_position", "ema_price_position_v2", "vwap", "utbot", "utbot_v2", "swing_123"}
        OSCILLATOR_TEMPLATES = {"macd_line", "macd_histogram", "rvol"}

        # Determine which groups are relevant to this strategy
        entry_conf_id = req.entry_trigger_confluence_id or ''
        exit_conf_ids = req.exit_trigger_confluence_ids or []
        confluence_records = list(req.confluence or [])

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

            # Resolve indicator columns (handle parameterized EMA names)
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
        overlay_indicators = list(dict.fromkeys(overlay_cols))
        oscillator_indicators = list(dict.fromkeys(oscillator_cols))
        all_indicator_cols = overlay_indicators + oscillator_indicators

        # Build heatmap conditions from confluence records (same as Strategy Detail)
        from data_loader import get_tf_label
        primary_tf = get_tf_label(req.timeframe).lower()

        for record in confluence_records:
            parts = record.split('-', 2)
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

            heatmap_conditions.append({
                "label": record,
                "column": col_name,
                "needed_state": needed_state,
                "has_data": col_name in df.columns,
            })

        # Serialize chart data: OHLCV + relevant indicators
        chart_data = _serialize_chart_data(df, all_indicator_cols)

        # Add interpreter state values for heatmap
        state_cols = [c["column"] for c in heatmap_conditions if c["has_data"]]
        if state_cols:
            reset_df = df.reset_index()
            for i, row_dict in enumerate(chart_data):
                if i < len(reset_df):
                    r = reset_df.iloc[i]
                    for sc in state_cols:
                        val = r.get(sc)
                        row_dict[f"_state_{sc}"] = str(val) if pd.notna(val) else None

    resp = BacktestResponse(
        trades=trades,
        kpis=kpis,
        secondary_kpis=secondary_kpis,
        equity_curve=equity_curve,
        total_bars=len(df),
        total_trading_days=trading_days,
        data_source=get_data_source(),
        chart_data=chart_data,
    )
    # Attach indicator metadata as extra fields (not in Pydantic model, but JSON serializable)
    resp_dict = resp.model_dump()
    resp_dict['overlay_indicators'] = overlay_indicators
    resp_dict['oscillator_indicators'] = oscillator_indicators
    resp_dict['heatmap_conditions'] = heatmap_conditions
    return resp_dict


# =============================================================================
# HELPERS
# =============================================================================

def _empty_kpis() -> dict:
    return {
        "total_trades": 0, "win_rate": 0, "profit_factor": 0,
        "avg_r": 0, "total_r": 0, "daily_r": 0, "r_squared": 0.0,
        "max_r_drawdown": 0, "final_balance": 10000, "total_pnl": 0,
    }


def _serialize_trades(trades_df: pd.DataFrame) -> list[dict]:
    """Convert trades DataFrame to JSON-serializable list."""
    if len(trades_df) == 0:
        return []

    records = trades_df.copy()
    for col in ["entry_time", "exit_time"]:
        if col in records.columns:
            records[col] = records[col].apply(
                lambda x: x.isoformat() if pd.notna(x) and hasattr(x, 'isoformat') else None
            )

    # Convert numpy types to native Python for JSON serialization
    result = []
    for _, row in records.iterrows():
        d = {}
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                d[k] = int(v)
            elif isinstance(v, (np.floating,)):
                d[k] = float(v) if not np.isnan(v) else None
            elif isinstance(v, (np.bool_,)):
                d[k] = bool(v)
            elif isinstance(v, set):
                d[k] = list(v)
            elif isinstance(v, (list, tuple, frozenset)):
                d[k] = list(v)
            elif isinstance(v, dict):
                d[k] = v
            else:
                try:
                    if pd.isna(v):
                        d[k] = None
                    else:
                        d[k] = v
                except (ValueError, TypeError):
                    # pd.isna fails on arrays/collections — keep as-is
                    d[k] = v
        result.append(d)
    return result


def _build_equity_curve(trades_df: pd.DataFrame) -> list[dict]:
    """Build equity curve points from trades."""
    if len(trades_df) == 0:
        return []

    # Filter out open positions
    closed = trades_df
    if "exit_reason" in closed.columns:
        closed = closed[closed["exit_reason"] != "open"]
    if len(closed) == 0:
        return []

    cum_r = closed["r_multiple"].cumsum()
    points = []
    for i, (idx, row) in enumerate(closed.iterrows()):
        ts = row.get("exit_time")
        if ts is not None and hasattr(ts, 'isoformat'):
            ts = ts.isoformat()
        points.append({
            "trade_number": i + 1,
            "timestamp": ts,
            "cumulative_r": round(float(cum_r.iloc[i]), 4),
        })
    return points


def _serialize_chart_data(
    df: pd.DataFrame,
    indicators: list[str] | None = None,
) -> list[dict]:
    """Serialize OHLCV + selected indicator columns for chart rendering."""
    cols = ["open", "high", "low", "close", "volume"]
    if indicators:
        cols += [c for c in indicators if c in df.columns]

    subset = df[cols].copy()
    records = []
    for ts, row in subset.iterrows():
        d = {"timestamp": ts.isoformat()}
        for col in cols:
            val = row[col]
            if isinstance(val, (np.floating,)):
                d[col] = round(float(val), 6) if not np.isnan(val) else None
            elif isinstance(val, (np.integer,)):
                d[col] = int(val)
            else:
                d[col] = val
        records.append(d)
    return records


def _resolve_stop_from_pack(pack_id: str) -> dict:
    """Load a risk management pack and extract its stop config."""
    from risk_management_packs import load_risk_management_packs
    packs = load_risk_management_packs()
    for pack in packs:
        if pack.id == pack_id:
            return pack.get_stop_config()
    return {"method": "atr", "atr_mult": 1.5}


def _resolve_target_from_pack(pack_id: str) -> dict | None:
    """Load a risk management pack and extract its target config."""
    from risk_management_packs import load_risk_management_packs
    packs = load_risk_management_packs()
    for pack in packs:
        if pack.id == pack_id:
            return pack.get_target_config()
    return None
