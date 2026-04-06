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

    # 5. Run unified engine (single path for all packs — built-in and user)
    # User pack indicators/interpreters/triggers are pre-computed in the
    # DataFrame by the batch pipeline. The unified engine reads these columns
    # alongside its incremental computation for built-in packs.
    trades_df = svc.unified_trades(df, strategy)
    print(f"[BACKTEST] Unified engine returned {len(trades_df)} trades")

    # 5b. Hi-Fi Pass 2: Resolve every entry/exit with 1-second data
    if req.hifi_mode and len(trades_df) > 0:
        if 'hifi_resolved' not in trades_df.columns:
            trades_df['hifi_resolved'] = False
        trades_df = _hifi_resolve_trades(trades_df, req.symbol, req.timeframe)

    # 5c. CB Fidelity Pass 3: Recompute secondary-TF confluence at trigger moment
    if req.hifi_mode and req.cb_conditions and req.secondary_tfs and len(trades_df) > 0:
        trades_df = recompute_cb_confluence(
            trades_df, req.symbol, req.timeframe,
            secondary_tfs=list(req.secondary_tfs),
            session=req.session,
        )
        # Filter trades: for CB conditions, check cb_confluence_records instead of confluence_records
        cb_set = set(req.cb_conditions)
        if 'cb_confluence_records' in trades_df.columns:
            def _cb_filter(row):
                cb_recs = row.get('cb_confluence_records')
                if cb_recs is None or not isinstance(cb_recs, set):
                    return False
                return cb_set.issubset(cb_recs)
            cb_mask = trades_df.apply(_cb_filter, axis=1)
            before_count = len(trades_df)
            trades_df = trades_df[cb_mask].copy()
            print(f"[BACKTEST] CB filter: {before_count} → {len(trades_df)} trades")

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

        # Track which templates we've already classified
        _classified_templates = set()

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
            _classified_templates.add(group.base_template)

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

        # Fallback: check user packs from registry if the entry trigger's pack
        # wasn't found in enabled DB groups (e.g., newly installed pack)
        try:
            import pack_registry
            for slug, pack in pack_registry.get_registered_packs().items():
                if slug in _classified_templates:
                    continue
                trigger_prefix = pack.manifest.get("trigger_prefix", "")
                if not (entry_conf_id.startswith(slug) or
                        (trigger_prefix and f"_{trigger_prefix}_" in entry_conf_id)):
                    continue
                raw_cols = pack.manifest.get("indicator_columns", [])
                resolved = [c for c in raw_cols if c in df.columns]
                dt = pack.manifest.get("display_type", "overlay")
                if dt == "oscillator":
                    oscillator_cols.extend(resolved)
                else:
                    overlay_cols.extend(resolved)
        except Exception:
            pass

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

        if not exit_time_str:
            continue

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


def _walk_1s_for_exit(bars_1s: pd.DataFrame, stop: float | None, target: float | None, direction: str) -> dict | None:
    """Walk 1-second bars to find which exit level was hit first.

    Handles cases where only stop, only target, or both are set.
    Returns dict with exit_reason, exit_price, exit_time, or None if neither hit.
    """
    for ts, bar in bars_1s.iterrows():
        high = bar.get('high', 0)
        low = bar.get('low', 0)

        if direction == 'LONG':
            if stop is not None and low <= stop:
                return {'exit_reason': 'stop_loss', 'exit_price': stop, 'exit_time': ts}
            if target is not None and target > 0 and high >= target:
                return {'exit_reason': 'target', 'exit_price': target, 'exit_time': ts}
        else:  # SHORT
            if stop is not None and high >= stop:
                return {'exit_reason': 'stop_loss', 'exit_price': stop, 'exit_time': ts}
            if target is not None and target > 0 and low <= target:
                return {'exit_reason': 'target', 'exit_price': target, 'exit_time': ts}

    return None


# =============================================================================
# CB FIDELITY — Current Bar Confluence Recomputation
# =============================================================================

def recompute_cb_confluence(
    trades_df: pd.DataFrame,
    symbol: str,
    primary_tf: str,
    secondary_tfs: list[str],
    session: str = 'RTH',
) -> pd.DataFrame:
    """Recompute confluence state at the exact trigger moment for CB conditions.

    For each trade, takes the entry timestamp and:
    1. Identifies which secondary-TF bar was forming at that moment
    2. Loads the completed secondary-TF bars leading up to the forming bar
    3. Builds a partial bar from the primary-TF bars up to the trigger moment
    4. Runs the full indicator + interpreter pipeline on this augmented series
    5. Extracts the interpreter state at the partial bar → CB state
    6. Stores as cb_confluence_records on each trade

    PB state comes from the last CLOSED secondary-TF bar (already in confluence_records).
    CB state comes from the FORMING secondary-TF bar at trigger time.
    """
    import services as svc
    from data_loader import resample_to_timeframe, get_tf_label
    from interpreters import INTERPRETERS

    if len(trades_df) == 0 or not secondary_tfs:
        return trades_df

    # Ensure cb_confluence_records column exists
    if 'cb_confluence_records' not in trades_df.columns:
        trades_df['cb_confluence_records'] = None

    # Load 1-min bars for resampling (generous window for indicator lookback)
    primary_df = svc.load_market_data(symbol, days=10, timeframe='1Min',
                                       feed='sip', session=session)
    if len(primary_df) == 0:
        return trades_df

    for sec_tf in secondary_tfs:
        tf_label = get_tf_label(sec_tf)
        sec_period_seconds = _tf_to_seconds(sec_tf)

        # Build complete secondary-TF series with indicators (ONCE — cached for all trades)
        sec_df = resample_to_timeframe(
            primary_df[['open', 'high', 'low', 'close', 'volume']].copy(), sec_tf)
        if len(sec_df) < 5:
            continue

        sec_df = svc.run_all_indicators(sec_df)
        groups = svc.get_enabled_groups(svc.load_confluence_groups())
        for group in groups:
            sec_df = svc.run_indicators_for_group(sec_df, group)
        sec_df = svc.run_all_interpreters(sec_df)

        interp_cols = [c for c in sec_df.columns if c in INTERPRETERS]

        # Pre-cache: completed OHLCV bars for each secondary-TF bar start
        # This avoids re-slicing sec_df per trade
        sec_ohlcv = sec_df[['open', 'high', 'low', 'close', 'volume']]

        for idx in trades_df.index:
            entry_time_str = trades_df.at[idx, 'entry_time']
            if not entry_time_str:
                continue

            try:
                entry_dt = _parse_dt(str(entry_time_str))
                if entry_dt is None:
                    continue
                entry_ts = pd.Timestamp(entry_dt)
                if entry_ts.tzinfo is None:
                    entry_ts = entry_ts.tz_localize('UTC')

                # Find which secondary-TF bar was forming at entry time
                sec_bar_start = _find_forming_bar_start(
                    sec_df.index, entry_ts, sec_period_seconds)
                if sec_bar_start is None:
                    continue

                # Get primary-TF bars within the forming bar, up to trigger moment
                primary_mask = (primary_df.index >= sec_bar_start) & (primary_df.index <= entry_ts)
                partial_1min = primary_df.loc[primary_mask, ['open', 'high', 'low', 'close', 'volume']]
                if len(partial_1min) == 0:
                    continue

                # Build partial OHLCV bar
                partial_bar = pd.DataFrame([{
                    'open': float(partial_1min.iloc[0]['open']),
                    'high': float(partial_1min['high'].max()),
                    'low': float(partial_1min['low'].min()),
                    'close': float(partial_1min.iloc[-1]['close']),
                    'volume': float(partial_1min['volume'].sum()),
                }], index=[sec_bar_start])

                # Append partial bar to completed secondary-TF bars (cached slice)
                completed_bars = sec_ohlcv.loc[sec_ohlcv.index < sec_bar_start]
                lookback = min(250, len(completed_bars))
                cb_input = pd.concat([completed_bars.tail(lookback), partial_bar])

                # Run indicator + interpreter pipeline on the small augmented series
                # This is fast: only ~251 bars (250 lookback + 1 partial)
                cb_input = svc.run_all_indicators(cb_input)
                for group in groups:
                    cb_input = svc.run_indicators_for_group(cb_input, group)
                cb_input = svc.run_all_interpreters(cb_input)
                cb_input = svc.run_all_indicators(cb_input)
                for group in svc.get_enabled_groups(svc.load_confluence_groups()):
                    cb_input = svc.run_indicators_for_group(cb_input, group)
                cb_input = svc.run_all_interpreters(cb_input)

                # Extract CB state from the partial bar (last row)
                cb_records = set()
                if len(cb_input) > 0:
                    last_row = cb_input.iloc[-1]
                    for interp_col in interp_cols:
                        val = last_row.get(interp_col)
                        if val is not None and pd.notna(val):
                            cb_records.add(f"{tf_label}-{interp_col}-{val}")

                # Store CB confluence records
                existing = trades_df.at[idx, 'cb_confluence_records']
                if existing is None or not isinstance(existing, set):
                    trades_df.at[idx, 'cb_confluence_records'] = cb_records
                else:
                    existing.update(cb_records)

            except Exception as e:
                logger.warning("[CB-RECOMPUTE] Error for trade %s, tf=%s: %s", idx, sec_tf, e)

    logger.info("[CB-RECOMPUTE] Processed %d trades across %d secondary TFs",
                len(trades_df), len(secondary_tfs))
    return trades_df


def _find_forming_bar_start(
    sec_index: pd.DatetimeIndex,
    entry_ts: pd.Timestamp,
    period_seconds: int,
) -> pd.Timestamp | None:
    """Find the start of the secondary-TF bar forming at entry_ts."""
    period = pd.Timedelta(seconds=period_seconds)
    candidates = sec_index[sec_index <= entry_ts]
    if len(candidates) == 0:
        return None
    last_bar_start = candidates[-1]
    if entry_ts < last_bar_start + period:
        return last_bar_start
    return last_bar_start + period


def _tf_to_seconds(timeframe: str) -> int:
    """Convert timeframe string to seconds."""
    if 'Min' in timeframe:
        return int(timeframe.replace('Min', '')) * 60
    if 'Hour' in timeframe or timeframe == '1H':
        return int(timeframe.replace('Hour', '').replace('H', '') or '1') * 3600
    if 'Day' in timeframe or timeframe == '1D':
        return 86400
    return 60  # default 1 minute


def _parse_dt(dt_val):
    """Parse a datetime value, handling strings, Timestamps, and datetimes."""
    from datetime import datetime as _dt, timezone
    import pandas as _pd
    if dt_val is None or (isinstance(dt_val, str) and (not dt_val or dt_val == '--')):
        return None
    try:
        # Handle pandas Timestamp directly
        if isinstance(dt_val, _pd.Timestamp):
            return dt_val.to_pydatetime()
        if isinstance(dt_val, _dt):
            if dt_val.tzinfo is None:
                return dt_val.replace(tzinfo=timezone.utc)
            return dt_val
        # String parsing
        s = str(dt_val).replace('Z', '+00:00')
        d = _dt.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d
    except (ValueError, TypeError):
        return None


# =============================================================================
# TRADE ZOOM HELPER (shared by strategy + backtest trade-zoom endpoints)
# =============================================================================

def build_trade_zoom_response(
    trade: dict,
    symbol: str,
    timeframe: str,
    side: str,
    padding_seconds: int,
    indicator_df: pd.DataFrame | None = None,
    relevant_prefixes: set | None = None,
    secondary_tfs: list[str] | None = None,
    session: str = 'RTH',
) -> dict:
    """Build a trade-zoom response with 1-second bars + stepped indicators.

    This is the shared logic used by both:
    - GET /api/strategies/{id}/trade-zoom  (saved strategy)
    - POST /api/backtest/trade-zoom        (unsaved backtest)

    Args:
        trade: Trade dict with entry_time, exit_time, etc.
        symbol: Ticker symbol.
        timeframe: Strategy timeframe (e.g., "5Min").
        side: "entry" or "exit" — which bar to zoom into.
        padding_seconds: Seconds of context before/after the bar.
        indicator_df: Optional enriched DataFrame with indicator columns.
        relevant_prefixes: Optional set of prefix strings to filter indicator columns.

    Returns:
        Dict with bars_1s, trade, indicators, side, timeframe, symbol.
    """
    from data_loader import fetch_1s_bars_for_window
    from datetime import timedelta

    # Determine which timestamp to zoom into
    ts_str = trade.get('entry_time') if side == 'entry' else trade.get('exit_time')
    if not ts_str:
        return {
            "bars_1s": [], "trade": _serialize_trade_for_zoom(trade),
            "indicators": {}, "side": side, "timeframe": timeframe, "symbol": symbol,
        }

    # Parse the timestamp
    ts = pd.Timestamp(ts_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    ts_dt = ts.to_pydatetime()

    # Determine bar duration for the window
    tf_seconds = _tf_to_seconds(timeframe)
    bar_end = ts_dt + timedelta(seconds=tf_seconds)

    # Fetch 1-second bars
    try:
        bars_df = fetch_1s_bars_for_window(symbol, ts_dt, bar_end, padding_seconds=padding_seconds)
    except Exception as e:
        logger.warning("[TRADE-ZOOM] Error fetching 1s bars: %s", e)
        return {
            "bars_1s": [], "trade": _serialize_trade_for_zoom(trade),
            "indicators": {}, "side": side, "timeframe": timeframe, "symbol": symbol,
        }

    # Convert to JSON-serializable list
    bars_list = []
    for idx_ts, row in bars_df.iterrows():
        bars_list.append({
            "time": idx_ts.isoformat(),
            "open": float(row.get('open', 0)),
            "high": float(row.get('high', 0)),
            "low": float(row.get('low', 0)),
            "close": float(row.get('close', 0)),
            "volume": float(row.get('volume', 0)),
        })

    # Compute stepped indicator values from the original timeframe
    stepped_indicators = {}
    if indicator_df is not None and len(indicator_df) > 0 and relevant_prefixes:
        try:
            standard_cols = {'open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count'}
            indicator_cols = []
            for c in indicator_df.columns:
                if c in standard_cols or c.startswith('__'):
                    continue
                if indicator_df[c].dtype not in ('float64', 'float32', 'int64'):
                    continue
                c_lower = c.lower()
                if any(prefix in c_lower for prefix in relevant_prefixes):
                    indicator_cols.append(c)

            # Filter to columns that have values in our window
            window_start = ts_dt - timedelta(seconds=padding_seconds + tf_seconds * 3)
            window_end = ts_dt + timedelta(seconds=padding_seconds + tf_seconds * 3)

            def _to_utc_ts(dt):
                t = pd.Timestamp(dt)
                return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')

            for col in indicator_cols[:20]:
                mask = (indicator_df.index >= _to_utc_ts(window_start)) & \
                       (indicator_df.index <= _to_utc_ts(window_end))
                window_data = indicator_df.loc[mask, col].dropna()
                if len(window_data) == 0:
                    continue

                steps = []
                for bar_ts, val in window_data.items():
                    steps.append({"time": bar_ts.isoformat(), "value": float(val)})

                if steps:
                    stepped_indicators[col] = steps

        except Exception as e:
            logger.warning("[TRADE-ZOOM] Error computing indicators: %s", e)

    # Compute CB confluence timeline if secondary TFs are provided
    cb_timeline = {}
    pb_states = {}
    if secondary_tfs and ts_dt:
        try:
            logger.info("[TRADE-ZOOM] Computing CB timeline for %s, secondary_tfs=%s", symbol, secondary_tfs)
            cb_timeline, pb_states = _compute_cb_timeline(
                symbol, ts_dt, tf_seconds, secondary_tfs, session, padding_seconds)
            logger.info("[TRADE-ZOOM] CB timeline: %d TFs, pb_states: %s", len(cb_timeline), list(pb_states.keys()))
        except Exception as e:
            logger.exception("[TRADE-ZOOM] Error computing CB timeline: %s", e)

    result = {
        "bars_1s": bars_list,
        "trade": _serialize_trade_for_zoom(trade),
        "indicators": stepped_indicators,
        "side": side,
        "timeframe": timeframe,
        "symbol": symbol,
    }
    if cb_timeline:
        result["cb_confluence_timeline"] = cb_timeline
    if pb_states:
        result["pb_states"] = pb_states
    return result


def _compute_cb_timeline(
    symbol: str,
    bar_dt: 'datetime',
    tf_seconds: int,
    secondary_tfs: list[str],
    session: str,
    padding_seconds: int,
) -> tuple[dict, dict]:
    """Compute second-by-second CB confluence state within the zoom window.

    For each secondary TF, computes what the interpreter state would be
    at each primary-TF bar within the window. Returns:
      cb_timeline: {tf_label: [{time, state}, ...]} — state at each bar
      pb_states: {tf_label: state} — the PB state (from last closed bar)
    """
    import services as svc
    from data_loader import resample_to_timeframe, get_tf_label
    from interpreters import INTERPRETERS
    from datetime import timedelta

    # Load 1-min bars covering the zoom window + indicator lookback
    # Use 30 days to ensure we cover the trade's timestamp (backtest may span up to 90 days)
    logger.info("[CB-TIMELINE] Loading 1-min bars for %s (bar_dt=%s)", symbol, bar_dt)
    primary_df = svc.load_market_data(symbol, days=30, timeframe='1Min',
                                       feed='sip', session=session)
    logger.info("[CB-TIMELINE] Loaded %d 1-min bars, range: %s to %s",
                len(primary_df),
                primary_df.index[0] if len(primary_df) > 0 else 'N/A',
                primary_df.index[-1] if len(primary_df) > 0 else 'N/A')
    if len(primary_df) == 0:
        return {}, {}

    cb_timeline = {}
    pb_states = {}

    for sec_tf in secondary_tfs:
        tf_label = get_tf_label(sec_tf)
        sec_period = _tf_to_seconds(sec_tf)
        logger.info("[CB-TIMELINE] Processing sec_tf=%s (label=%s, period=%ds)", sec_tf, tf_label, sec_period)

        # Build complete secondary-TF series
        sec_df = resample_to_timeframe(
            primary_df[['open', 'high', 'low', 'close', 'volume']].copy(), sec_tf)
        if len(sec_df) < 5:
            logger.warning("[CB-TIMELINE] Only %d secondary bars for %s, skipping", len(sec_df), sec_tf)
            continue

        sec_df = svc.run_all_indicators(sec_df)
        for group in svc.get_enabled_groups(svc.load_confluence_groups()):
            sec_df = svc.run_indicators_for_group(sec_df, group)
        sec_df = svc.run_all_interpreters(sec_df)

        interp_cols = [c for c in sec_df.columns if c in INTERPRETERS]
        if not interp_cols:
            continue

        # PB state: the last completed secondary-TF bar before the zoom bar
        pb_ts = pd.Timestamp(bar_dt)
        if pb_ts.tzinfo is None:
            pb_ts = pb_ts.tz_localize('UTC')
        completed = sec_df[sec_df.index < pb_ts]
        if len(completed) > 0:
            # PB is shifted forward by one period — the last completed bar's state
            last_row = completed.iloc[-1]
            for ic in interp_cols:
                val = last_row.get(ic)
                if val is not None and pd.notna(val):
                    pb_states[f"{tf_label}-{ic}"] = str(val)

        # CB timeline: at each 1-min bar within the zoom window, compute the forming bar state
        window_start = bar_dt - timedelta(seconds=padding_seconds)
        window_end = bar_dt + timedelta(seconds=tf_seconds + padding_seconds)

        ws_ts = pd.Timestamp(window_start)
        we_ts = pd.Timestamp(window_end)
        if ws_ts.tzinfo is None:
            ws_ts = ws_ts.tz_localize('UTC')
        if we_ts.tzinfo is None:
            we_ts = we_ts.tz_localize('UTC')

        window_1min = primary_df[(primary_df.index >= ws_ts) & (primary_df.index <= we_ts)]

        if len(window_1min) == 0:
            logger.info("[CB-TIMELINE] No 1-min bars in window for %s, skipping timeline", sec_tf)
            continue

        timeline_entries = []
        # Sample at ~30 points to avoid excessive computation
        sample_indices = list(range(0, len(window_1min), max(1, len(window_1min) // 30)))
        if len(window_1min) > 0 and (len(window_1min) - 1) not in sample_indices:
            sample_indices.append(len(window_1min) - 1)

        for si in sample_indices:
            sample_ts = window_1min.index[si]

            # Find the forming secondary bar
            sec_bar_start = _find_forming_bar_start(sec_df.index, sample_ts, sec_period)
            if sec_bar_start is None:
                continue

            # Build partial bar up to this moment
            partial_mask = (primary_df.index >= sec_bar_start) & (primary_df.index <= sample_ts)
            partial_bars = primary_df.loc[partial_mask, ['open', 'high', 'low', 'close', 'volume']]
            if len(partial_bars) == 0:
                continue

            partial_ohlcv = pd.DataFrame([{
                'open': float(partial_bars.iloc[0]['open']),
                'high': float(partial_bars['high'].max()),
                'low': float(partial_bars['low'].min()),
                'close': float(partial_bars.iloc[-1]['close']),
                'volume': float(partial_bars['volume'].sum()),
            }], index=[sec_bar_start])

            # Append to completed bars and run pipeline
            completed_sec = sec_df.loc[sec_df.index < sec_bar_start,
                                       ['open', 'high', 'low', 'close', 'volume']]
            lookback = min(250, len(completed_sec))
            cb_input = pd.concat([completed_sec.tail(lookback), partial_ohlcv])

            cb_input = svc.run_all_indicators(cb_input)
            for group in svc.get_enabled_groups(svc.load_confluence_groups()):
                cb_input = svc.run_indicators_for_group(cb_input, group)
            cb_input = svc.run_all_interpreters(cb_input)

            if len(cb_input) > 0:
                last_row = cb_input.iloc[-1]
                states = {}
                for ic in interp_cols:
                    val = last_row.get(ic)
                    if val is not None and pd.notna(val):
                        states[ic] = str(val)
                timeline_entries.append({
                    "time": sample_ts.isoformat(),
                    "states": states,
                })

        if timeline_entries:
            cb_timeline[tf_label] = timeline_entries

    return cb_timeline, pb_states


def _serialize_trade_for_zoom(trade: dict) -> dict:
    """Serialize a trade dict for the zoom response, handling Timestamps."""
    result = {}
    for k, v in trade.items():
        if isinstance(v, pd.Timestamp):
            result[k] = v.isoformat()
        elif isinstance(v, set):
            result[k] = list(v)
        elif isinstance(v, float) and (v != v):  # NaN check
            result[k] = None
        else:
            result[k] = v
    return result


def _tf_to_seconds(timeframe: str) -> int:
    """Convert timeframe string to seconds."""
    if 'Min' in timeframe:
        return int(timeframe.replace('Min', '')) * 60
    if 'Hour' in timeframe or timeframe == '1H':
        return int(timeframe.replace('Hour', '').replace('H', '') or '1') * 3600
    if 'Day' in timeframe or timeframe == '1D':
        return 86400
    return 60


def extract_relevant_prefixes(
    entry_id: str = '',
    exit_ids: list[str] | None = None,
    confluence: list[str] | None = None,
) -> set:
    """Extract indicator prefixes from trigger IDs and confluence conditions.

    Used to filter indicator columns to only those relevant to a strategy.
    """
    prefixes = set()
    for tid in [entry_id] + (exit_ids or []):
        if tid:
            parts = tid.split('_')
            if len(parts) >= 2:
                prefixes.add(parts[0])
                prefixes.add('_'.join(parts[:2]))
    for cond in (confluence or []):
        cond_parts = cond.split('-')
        if len(cond_parts) >= 2:
            prefixes.add(cond_parts[1].lower())
    return prefixes
