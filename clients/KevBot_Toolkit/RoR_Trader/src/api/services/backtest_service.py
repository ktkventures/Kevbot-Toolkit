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
    print(f"[BACKTEST] hifi_mode={req.hifi_mode}")

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

    # 5. Run unified engine (Pass 1)
    trades_df = svc.unified_trades(df, strategy)
    print(f"[BACKTEST] Unified engine returned {len(trades_df)} trades")

    # 5b. Hi-Fi Pass 2: Resolve every entry/exit with 1-second data
    if req.hifi_mode and len(trades_df) > 0:
        if 'hifi_resolved' not in trades_df.columns:
            trades_df['hifi_resolved'] = False
        trades_df = _hifi_resolve_trades(trades_df, req.symbol, req.timeframe)

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


# =============================================================================
# HI-FI RESOLUTION (PASS 2)
# =============================================================================

def _hifi_resolve_trades(trades_df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    """Resolve every entry/exit with 1-second data for precise timing.

    For each trade:
    1. Fetch 1-second bars for the exit bar window
    2. Walk 1-second bars to find which level (stop or target) was hit first
    3. Update exit_reason, exit_price, and hold_time_seconds if outcome changed

    This is Pass 2 of the Hi-Fi backtest — runs after the normal engine Pass 1.
    """
    from data_loader import fetch_1s_bars_for_window
    from datetime import datetime as _dt, timedelta, timezone

    outcomes_changed = 0
    total_resolved = 0

    # Get timeframe duration in seconds for window calculation
    tf_seconds = _tf_to_seconds(timeframe)

    for idx in trades_df.index:
        trade = trades_df.loc[idx]
        entry_time_str = trade.get('entry_time')
        exit_time_str = trade.get('exit_time')
        stop_price = trade.get('stop_price')
        target_price = trade.get('target_price')
        direction = trade.get('direction', 'LONG')

        if not exit_time_str or not stop_price or not target_price:
            continue
        if target_price == 0:
            continue  # No target set (signal-only exit)

        try:
            exit_dt = _parse_dt(exit_time_str)
            if exit_dt is None:
                continue

            # Fetch 1-second bars for the exit bar window
            window_start = exit_dt
            window_end = exit_dt + timedelta(seconds=tf_seconds)
            bars_1s = fetch_1s_bars_for_window(symbol, window_start, window_end, padding_seconds=5)

            if bars_1s is None or len(bars_1s) == 0:
                continue

            total_resolved += 1

            # Walk 1-second bars to find which level hit first
            original_reason = trade.get('exit_reason', '')
            resolved = _walk_1s_for_exit(bars_1s, stop_price, target_price, direction)

            if resolved:
                new_reason = resolved['exit_reason']
                new_price = resolved['exit_price']
                new_time = resolved['exit_time']

                if new_reason != original_reason:
                    outcomes_changed += 1
                    logger.info("[HIFI] Trade %s: %s → %s (price %.2f → %.2f)",
                                idx, original_reason, new_reason,
                                trade.get('exit_price', 0), new_price)

                trades_df.at[idx, 'exit_reason'] = new_reason
                trades_df.at[idx, 'exit_price'] = new_price
                trades_df.at[idx, 'hifi_resolved'] = True

                # Recompute R-multiple with resolved exit price
                entry_price = trade.get('entry_price', 0)
                initial_stop = trade.get('initial_stop_price', stop_price)
                risk = abs(entry_price - initial_stop) if initial_stop else abs(entry_price * 0.01)
                if risk <= 0:
                    risk = entry_price * 0.01
                if direction == 'LONG':
                    pnl = new_price - entry_price
                else:
                    pnl = entry_price - new_price
                trades_df.at[idx, 'r_multiple'] = pnl / risk if risk > 0 else 0
                trades_df.at[idx, 'win'] = pnl > 0
                trades_df.at[idx, 'pnl'] = pnl

                # Update hold time with resolved exit time
                if entry_time_str and new_time:
                    entry_dt = _parse_dt(entry_time_str)
                    exit_dt_resolved = _parse_dt(str(new_time))
                    if entry_dt and exit_dt_resolved:
                        trades_df.at[idx, 'hold_time_seconds'] = (exit_dt_resolved - entry_dt).total_seconds()

        except Exception as e:
            logger.warning("[HIFI] Error resolving trade %s: %s", idx, e)

    logger.info("[HIFI] Resolved %d trades, %d outcomes changed", total_resolved, outcomes_changed)
    return trades_df


def _walk_1s_for_exit(bars_1s: pd.DataFrame, stop: float, target: float, direction: str) -> dict | None:
    """Walk 1-second bars to find which exit level was hit first.

    Returns dict with exit_reason, exit_price, exit_time, or None if neither hit.
    """
    for ts, bar in bars_1s.iterrows():
        high = bar.get('high', 0)
        low = bar.get('low', 0)

        if direction == 'LONG':
            if low <= stop:
                return {'exit_reason': 'stop_loss', 'exit_price': stop, 'exit_time': ts}
            if high >= target:
                return {'exit_reason': 'target', 'exit_price': target, 'exit_time': ts}
        else:  # SHORT
            if high >= stop:
                return {'exit_reason': 'stop_loss', 'exit_price': stop, 'exit_time': ts}
            if low <= target:
                return {'exit_reason': 'target', 'exit_price': target, 'exit_time': ts}

    return None


def _tf_to_seconds(timeframe: str) -> int:
    """Convert timeframe string to seconds."""
    if 'Min' in timeframe:
        return int(timeframe.replace('Min', '')) * 60
    if 'Hour' in timeframe or timeframe == '1H':
        return int(timeframe.replace('Hour', '').replace('H', '') or '1') * 3600
    if 'Day' in timeframe or timeframe == '1D':
        return 86400
    return 60  # default 1 minute


def _parse_dt(dt_str: str):
    """Parse a datetime string, handling various formats."""
    from datetime import datetime as _dt, timezone
    if not dt_str or dt_str == '--':
        return None
    try:
        s = str(dt_str).replace('Z', '+00:00')
        d = _dt.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d
    except (ValueError, TypeError):
        return None
