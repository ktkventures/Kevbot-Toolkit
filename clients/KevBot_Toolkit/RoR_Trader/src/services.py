"""
Core business logic for RoR Trader — importable from both Streamlit and FastAPI.

Extracted from app.py (Phase 38) to remove Streamlit dependencies (@st.cache_data,
st.session_state). All functions here are pure computation — no UI, no caching decorators.

Usage:
    # From FastAPI:
    from services import prepare_data_with_indicators, calculate_kpis

    # From Streamlit (app.py) — thin wrappers add @st.cache_data:
    @st.cache_data(...)
    def prepare_data_with_indicators(...):
        return services.prepare_data_with_indicators(...)
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from data_loader import load_market_data, get_data_source, is_crypto
from indicators import run_all_indicators, run_indicators_for_group
from interpreters import INTERPRETERS, run_all_interpreters, detect_all_triggers
from triggers import generate_trades
import general_packs as gp_module
from confluence_groups import load_confluence_groups, get_enabled_groups

_logger = logging.getLogger('ror_trader')


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data_with_indicators(
    symbol: str, days: int = 30, seed: int = 42,
    start_date=None, end_date=None,
    timeframe: str = "1Min", data_feed: str = "sip",
    session: str = "RTH", secondary_tfs: tuple = ()
) -> pd.DataFrame:
    """Load market data and run all indicators, interpreters, and trigger detection.

    Extracted from app.py:634 — identical logic, no @st.cache_data.

    Args:
        symbol: Stock symbol
        days: Number of days (used if start_date/end_date not provided)
        seed: Random seed for mock data
        start_date: Explicit start date (overrides days)
        end_date: Explicit end date (overrides days)
        timeframe: Bar timeframe (e.g., "1Min", "5Min", "1Hour")
        data_feed: Alpaca data feed — "sip" or "iex" (also used as cache key)
        session: Trading session — "RTH", "Pre-Market", "After Hours", "Extended Hours"
        secondary_tfs: Tuple of secondary timeframes to compute for MTF confluence

    Returns DataFrame ready for trade generation and analysis.
    """
    from data_loader import resample_to_timeframe, get_tf_label

    # Load raw bars
    df = load_market_data(symbol, days=days, seed=seed,
                          start_date=start_date, end_date=end_date,
                          timeframe=timeframe, feed=data_feed,
                          session=session)

    if len(df) == 0:
        return df

    # Run indicators
    df = run_all_indicators(df)

    # Run group-specific indicators for custom parameters
    for group in get_enabled_groups(load_confluence_groups()):
        df = run_indicators_for_group(df, group)

    # Run interpreters
    df = run_all_interpreters(df)

    # Detect triggers
    df = detect_all_triggers(df)

    # Evaluate enabled general packs
    gen_packs = gp_module.load_general_packs()
    enabled_gen = gp_module.get_enabled_general_packs(gen_packs)
    for gpack in enabled_gen:
        col_name = gpack.get_condition_column()
        df[col_name] = gp_module.evaluate_condition(df, gpack)

    # Multi-Timeframe: resample + run pipeline on secondary TFs
    if secondary_tfs and len(df) > 0:
        interp_keys = list(INTERPRETERS.keys())
        for sec_tf in secondary_tfs:
            try:
                sec_df = resample_to_timeframe(
                    df[['open', 'high', 'low', 'close', 'volume']].copy(), sec_tf)
                if len(sec_df) == 0:
                    continue
                sec_df = run_all_indicators(sec_df)
                for group in get_enabled_groups(load_confluence_groups()):
                    sec_df = run_indicators_for_group(sec_df, group)
                sec_df = run_all_interpreters(sec_df)

                tf_label = get_tf_label(sec_tf)
                from unified_engine import TIMEFRAME_SECONDS
                period_seconds = TIMEFRAME_SECONDS.get(sec_tf, 60)
                period_offset = pd.Timedelta(seconds=period_seconds)
                for interp_col in interp_keys:
                    if interp_col in sec_df.columns:
                        suffixed = f"{interp_col}__{tf_label}"
                        shifted = sec_df[interp_col].copy()
                        shifted.index = shifted.index + period_offset
                        df[suffixed] = shifted.reindex(df.index, method='ffill')
                        # Speculative (unshifted) values for heatmap yellow detection
                        df[f"_spec_{interp_col}__{tf_label}"] = sec_df[interp_col].reindex(
                            df.index, method='ffill')
            except Exception:
                pass

    return df


def get_secondary_tf_map(df: pd.DataFrame) -> dict:
    """Extract secondary TF map from column names containing '__'.

    Returns ``{tf_label: [suffixed_col_names]}`` suitable for passing to
    ``get_mtf_confluence_records()`` or ``generate_trades(secondary_tf_map=...)``.
    """
    tf_map: dict = {}
    for col in df.columns:
        if "__" in col and not col.startswith("_spec_"):
            parts = col.rsplit("__", 1)
            if len(parts) == 2:
                tf_label = parts[1]
                tf_map.setdefault(tf_label, []).append(col)
    return tf_map


# =============================================================================
# TRADE GENERATION
# =============================================================================

def unified_trades(
    df: pd.DataFrame, strategy: dict,
    include_open_position: bool = True,
    last_bar_partial: bool = False,
    bar_cache=None, cache_metadata=None
) -> pd.DataFrame:
    """Trade generation via unified engine with MTF support.

    Extracted from app.py:748 (_unified_trades).

    Args:
        df: Enriched DataFrame from prepare_data_with_indicators().
        strategy: Strategy config dict (saved format OR builder format).
        include_open_position: Include synthetic row for open position at end of data.
        last_bar_partial: If True, suppress signals on last bar (forming).
        bar_cache: Optional pre-computed bar cache from precompute_bar_cache().
        cache_metadata: Required when bar_cache is provided.

    Returns:
        trades_df matching generate_trades() schema.
    """
    # Fast path: replay from cache
    if bar_cache is not None and cache_metadata is not None:
        try:
            from unified_engine import run_trades_from_cache
            return run_trades_from_cache(
                bar_cache, strategy, cache_metadata,
                include_open_position=include_open_position)
        except Exception as exc:
            _logger.warning("cache replay failed (%s), falling back to full backtest", exc)

    sec_tf_map = get_secondary_tf_map(df)

    enabled_gen = gp_module.get_enabled_general_packs(
        gp_module.load_general_packs())

    try:
        from unified_engine import run_unified_backtest
        trades_df, _ = run_unified_backtest(
            df, strategy, general_packs=enabled_gen,
            secondary_tf_map=sec_tf_map if sec_tf_map else None,
            include_open_position=include_open_position,
            last_bar_partial=last_bar_partial)
    except Exception as exc:
        _logger.warning("unified engine failed (%s), falling back", exc)
        confluence_set = (
            set(strategy.get('confluence', []))
            | set(strategy.get('general_confluences', []))
        )
        confluence_set = confluence_set if confluence_set else None
        general_cols = [c for c in df.columns if c.startswith("GP_")]
        return generate_trades(
            df,
            direction=strategy['direction'],
            entry_trigger=strategy['entry_trigger'],
            exit_trigger=strategy.get('exit_trigger'),
            exit_triggers=strategy.get('exit_triggers'),
            confluence_required=confluence_set,
            risk_per_trade=strategy.get('risk_per_trade', 100.0),
            stop_atr_mult=strategy.get('stop_atr_mult', 1.5),
            stop_config=strategy.get('stop_config'),
            target_config=strategy.get('target_config'),
            bar_count_exit=strategy.get('bar_count_exit'),
            general_columns=general_cols if general_cols else None,
            secondary_tf_map=sec_tf_map if sec_tf_map else None,
        )

    if not isinstance(trades_df, pd.DataFrame):
        trades_df = pd.DataFrame()
    return trades_df


# =============================================================================
# STRATEGY TRADE LOADING
# =============================================================================

def trades_df_from_stored(stored_trades: list) -> pd.DataFrame:
    """Reconstruct a trades DataFrame from stored minimal records.

    Extracted from app.py:1475.
    """
    if not stored_trades:
        return pd.DataFrame(columns=["entry_time", "exit_time", "r_multiple", "win"])
    df = pd.DataFrame(stored_trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    return df


def split_trades_at_boundary(trades_df: pd.DataFrame, boundary_dt: datetime):
    """Split trades into backtest (before boundary) and forward (at/after boundary).

    Extracted from app.py:1587.
    """
    if len(trades_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    boundary_ts = pd.Timestamp(boundary_dt)
    col_tz = getattr(trades_df['entry_time'].dtype, 'tz', None)

    if col_tz is not None and boundary_ts.tzinfo is None:
        boundary_ts = boundary_ts.tz_localize(col_tz)
    elif col_tz is not None and boundary_ts.tzinfo is not None:
        boundary_ts = boundary_ts.tz_convert(col_tz)
    elif col_tz is None and boundary_ts.tzinfo is not None:
        boundary_ts = boundary_ts.tz_localize(None)

    backtest = trades_df[trades_df['entry_time'] < boundary_ts].copy()
    forward = trades_df[trades_df['entry_time'] >= boundary_ts].copy()
    return backtest, forward


def get_strategy_trades(strat: dict, data_feed: str = "sip") -> pd.DataFrame:
    """Get trades for any modern strategy (backtest-only or forward-testing).

    Extracted from app.py:892. The data_feed parameter replaces the
    Streamlit session-state lookup (_get_data_feed).
    """
    # Webhook origin: use stored trades only
    if strat.get('strategy_origin') == 'webhook_inbound':
        return trades_df_from_stored(strat.get('stored_trades', []))

    if 'entry_trigger_confluence_id' not in strat:
        return pd.DataFrame()

    if strat.get('forward_testing') and strat.get('forward_test_start'):
        _, bt, fw, _ = prepare_forward_test_data(strat, data_feed=data_feed)
        return pd.concat([bt, fw], ignore_index=True)
    else:
        data_days = strat.get('data_days', 30)
        data_seed = strat.get('data_seed', 42)
        gst_start = None
        gst_end = None
        if strat.get('lookback_mode') == 'Date Range' and strat.get('lookback_start_date'):
            gst_start = datetime.fromisoformat(strat['lookback_start_date'])
            gst_end = datetime.fromisoformat(strat['lookback_end_date'])
        from data_loader import get_required_tfs_from_confluence, get_tf_from_label
        req_labels = get_required_tfs_from_confluence(strat.get('confluence', []))
        sec_tfs = tuple(sorted(get_tf_from_label(lbl) for lbl in req_labels))
        df = prepare_data_with_indicators(
            strat['symbol'], data_days, data_seed,
            start_date=gst_start, end_date=gst_end,
            timeframe=strat.get('timeframe', '1Min'),
            data_feed=data_feed,
            session=strat.get('trading_session', 'RTH'),
            secondary_tfs=sec_tfs)
        if len(df) == 0:
            return pd.DataFrame()
        return unified_trades(df, strat)


def prepare_forward_test_data(
    strat: dict, data_feed: str = "sip",
    data_days_override: int = None
):
    """Load continuous data and split trades at forward test boundary.

    Extracted from app.py:825. The data_feed parameter replaces _get_data_feed().

    Returns (df, backtest_trades, forward_trades, forward_test_start_dt)
    """
    forward_test_start_dt = datetime.fromisoformat(strat['forward_test_start'])

    # Webhook origin: use stored trades directly
    if strat.get('strategy_origin') == 'webhook_inbound':
        trades = trades_df_from_stored(strat.get('stored_trades', []))
        if len(trades) == 0:
            empty = pd.DataFrame()
            return pd.DataFrame(), empty, empty, forward_test_start_dt
        backtest_trades, forward_trades = split_trades_at_boundary(trades, forward_test_start_dt)
        return pd.DataFrame(), backtest_trades, forward_trades, forward_test_start_dt

    data_days = data_days_override if data_days_override is not None else strat.get('data_days', 30)
    data_seed = strat.get('data_seed', 42)

    start_date = forward_test_start_dt - timedelta(days=data_days * 2)
    end_date = datetime.now(timezone.utc).replace(hour=23, minute=59, second=0, microsecond=0)

    strat_timeframe = strat.get('timeframe', '1Min')
    from data_loader import get_required_tfs_from_confluence, get_tf_from_label
    required_tf_labels = get_required_tfs_from_confluence(strat.get('confluence', []))
    sec_tfs = tuple(sorted(get_tf_from_label(lbl) for lbl in required_tf_labels))

    df = prepare_data_with_indicators(
        strat['symbol'], seed=data_seed,
        start_date=start_date, end_date=end_date,
        timeframe=strat_timeframe, data_feed=data_feed,
        session=strat.get('trading_session', 'RTH'),
        secondary_tfs=sec_tfs,
    )

    if len(df) == 0:
        empty = pd.DataFrame()
        return df, empty, empty, forward_test_start_dt

    trades = unified_trades(df, strat)

    backtest_trades, forward_trades = split_trades_at_boundary(trades, forward_test_start_dt)

    # Trim backtest trades to pinned backtest window (exclude warmup-period trades)
    _bt_start = strat.get('backtest_start_date')
    if _bt_start and len(backtest_trades) > 0:
        _bt_start_ts = pd.Timestamp(_bt_start)
        if _bt_start_ts.tz is None and backtest_trades['entry_time'].dt.tz is not None:
            _bt_start_ts = _bt_start_ts.tz_localize('UTC')
        backtest_trades = backtest_trades[backtest_trades['entry_time'] >= _bt_start_ts]

    return df, backtest_trades, forward_trades, forward_test_start_dt


# =============================================================================
# KPI CALCULATIONS
# =============================================================================

def count_trading_days(df: pd.DataFrame) -> int:
    """Count unique trading days in a DataFrame with a DatetimeIndex."""
    if len(df) == 0 or not hasattr(df.index, 'normalize'):
        return 1
    return max(df.index.normalize().nunique(), 1)


def calculate_kpis(
    trades_df: pd.DataFrame,
    starting_balance: float = 10000,
    risk_per_trade: float = 100,
    total_trading_days: int = None
) -> dict:
    """Calculate strategy KPIs.

    Extracted from app.py:1730.

    Args:
        total_trading_days: Total trading days in the data period (all market days,
            not just days with trades). When provided, Daily R = total_r / total_trading_days.
    """
    if len(trades_df) == 0:
        return {
            "total_trades": 0, "win_rate": 0, "profit_factor": 0,
            "avg_r": 0, "total_r": 0, "daily_r": 0, "r_squared": 0.0,
            "max_r_drawdown": 0, "final_balance": starting_balance, "total_pnl": 0
        }

    # Exclude open-position rows
    if 'exit_reason' in trades_df.columns:
        trades_df = trades_df[trades_df['exit_reason'] != 'open']
    if len(trades_df) == 0:
        return {
            "total_trades": 0, "win_rate": 0, "profit_factor": 0,
            "avg_r": 0, "total_r": 0, "daily_r": 0, "r_squared": 0.0,
            "max_r_drawdown": 0, "final_balance": starting_balance, "total_pnl": 0
        }

    wins = trades_df[trades_df["win"] == True]
    losses = trades_df[trades_df["win"] == False]

    gross_profit = wins["r_multiple"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["r_multiple"].sum()) if len(losses) > 0 else 0
    total_r = trades_df["r_multiple"].sum()

    if total_trading_days is not None and total_trading_days > 0:
        trading_days = total_trading_days
    elif "exit_time" in trades_df.columns:
        trading_days = trades_df["exit_time"].dt.date.nunique()
    else:
        trading_days = 1
    trading_days = max(trading_days, 1)

    if len(trades_df) >= 2:
        cumulative_r = trades_df["r_multiple"].cumsum().values
        x = np.arange(len(cumulative_r))
        correlation = np.corrcoef(x, cumulative_r)[0, 1]
        r_squared = round(correlation ** 2, 4) if not np.isnan(correlation) else 0.0
        running_max = np.maximum.accumulate(cumulative_r)
        drawdown = cumulative_r - running_max
        max_r_drawdown = round(float(drawdown.min()), 2)
    else:
        r_squared = 0.0
        max_r_drawdown = 0.0

    total_pnl = total_r * risk_per_trade
    final_balance = starting_balance + total_pnl

    return {
        "total_trades": len(trades_df),
        "win_rate": len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        "avg_r": trades_df["r_multiple"].mean(),
        "total_r": total_r,
        "daily_r": total_r / trading_days,
        "r_squared": r_squared,
        "max_r_drawdown": max_r_drawdown,
        "final_balance": final_balance,
        "total_pnl": total_pnl,
        "_risk_per_trade": risk_per_trade,
        "_starting_balance": starting_balance,
    }


def calculate_secondary_kpis(trades_df: pd.DataFrame, kpis: dict) -> dict:
    """Calculate secondary/extended KPIs from trade data.

    Extracted from app.py:1808. Includes basic trade stats, risk-adjusted ratios,
    distribution metrics, and drawdown analytics.
    """
    _EMPTY = {
        "win_count": 0, "loss_count": 0,
        "best_trade_r": 0, "worst_trade_r": 0,
        "avg_win_r": 0, "avg_loss_r": 0,
        "max_consec_wins": 0, "max_consec_losses": 0,
        "payoff_ratio": 0, "recovery_factor": 0,
        "longest_dd_trades": 0,
        "sharpe_ratio": None, "sortino_ratio": None, "calmar_ratio": None,
        "kelly_criterion": None, "daily_var_95": None, "cvar_95": None,
        "gain_pain_ratio": None, "common_sense_ratio": None,
        "tail_ratio": None, "outlier_win_ratio": None, "outlier_loss_ratio": None,
        "ulcer_index": None, "serenity_index": None,
        "skewness": None, "kurtosis": None,
        "expected_daily": None, "expected_monthly": None, "expected_yearly": None,
        "volatility": None, "longest_dd_days": None,
    }
    if len(trades_df) == 0:
        return _EMPTY

    wins_mask = trades_df["win"].values
    r_mult = trades_df["r_multiple"].values
    n = len(r_mult)

    win_count = int(wins_mask.sum())
    loss_count = n - win_count

    best_trade_r = float(r_mult.max())
    worst_trade_r = float(r_mult.min())

    avg_win_r = float(r_mult[wins_mask].mean()) if win_count > 0 else 0
    avg_loss_r = float(r_mult[~wins_mask].mean()) if loss_count > 0 else 0

    max_consec_wins = max_consec_losses = 0
    current_wins = current_losses = 0
    for w in wins_mask:
        if w:
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)

    abs_avg_loss = abs(avg_loss_r)
    payoff_ratio = avg_win_r / abs_avg_loss if abs_avg_loss > 0 else float('inf')

    max_r_dd_abs = abs(kpis.get("max_r_drawdown", 0))
    total_r = kpis.get("total_r", 0)
    recovery_factor = total_r / max_r_dd_abs if max_r_dd_abs > 0 else float('inf')

    cumulative = np.cumsum(r_mult) if n >= 2 else np.array([0.0])
    if n >= 2:
        running_max = np.maximum.accumulate(cumulative)
        in_dd = cumulative < running_max
        longest_dd = 0
        current_dd = 0
        for d in in_dd:
            if d:
                current_dd += 1
                longest_dd = max(longest_dd, current_dd)
            else:
                current_dd = 0
    else:
        longest_dd = 0

    # ── Advanced Metrics ──────────────────────────────────────────────────
    risk_per_trade = kpis.get("_risk_per_trade", 100.0)
    starting_balance = kpis.get("_starting_balance", 10000.0)

    daily_r = None
    if "exit_time" in trades_df.columns and n >= 5:
        try:
            daily_r = trades_df.groupby(trades_df["exit_time"].dt.date)["r_multiple"].sum().values
        except Exception:
            daily_r = None

    sharpe_ratio = None
    if daily_r is not None and len(daily_r) >= 5:
        dr_mean = np.mean(daily_r)
        dr_std = np.std(daily_r, ddof=1)
        if dr_std > 0:
            sharpe_ratio = round(float(dr_mean / dr_std * np.sqrt(252)), 2)

    sortino_ratio = None
    if daily_r is not None and len(daily_r) >= 5:
        dr_mean = np.mean(daily_r)
        downside = daily_r[daily_r < 0]
        if len(downside) > 0:
            downside_std = np.std(downside, ddof=1)
            if downside_std > 0:
                sortino_ratio = round(float(dr_mean / downside_std * np.sqrt(252)), 2)

    calmar_ratio = None
    if n >= 10 and max_r_dd_abs > 0 and daily_r is not None and len(daily_r) >= 5:
        trading_years = len(daily_r) / 252
        if trading_years > 0:
            annualized_return = (total_r * risk_per_trade / starting_balance) / trading_years
            max_dd_pct = max_r_dd_abs * risk_per_trade / starting_balance
            if max_dd_pct > 0:
                calmar_ratio = round(float(annualized_return / max_dd_pct), 2)

    kelly_criterion = None
    if n >= 5 and win_count > 0 and loss_count > 0:
        win_rate_frac = win_count / n
        pr = avg_win_r / abs_avg_loss if abs_avg_loss > 0 else 0
        if pr > 0:
            kelly = win_rate_frac - (1 - win_rate_frac) / pr
            kelly_criterion = round(float(max(0.0, min(1.0, kelly))), 3)

    daily_var_95 = None
    if daily_r is not None and len(daily_r) >= 10:
        var_r = float(np.percentile(daily_r, 5))
        daily_var_95 = round(var_r * risk_per_trade, 2)

    cvar_95 = None
    if daily_r is not None and len(daily_r) >= 10:
        var_threshold = np.percentile(daily_r, 5)
        tail = daily_r[daily_r <= var_threshold]
        if len(tail) > 0:
            cvar_95 = round(float(np.mean(tail) * risk_per_trade), 2)

    gross_profit = float(r_mult[r_mult > 0].sum()) if np.any(r_mult > 0) else 0
    gross_loss_abs = float(abs(r_mult[r_mult < 0].sum())) if np.any(r_mult < 0) else 0
    gain_pain_ratio = round(gross_profit / gross_loss_abs, 2) if gross_loss_abs > 0 else None

    common_sense_ratio = None
    if n >= 5:
        pf = kpis.get("profit_factor", 0)
        if pf != float('inf') and pf > 0:
            common_sense_ratio = round(float(pf * (1 - 1 / n)), 2)

    tail_ratio = None
    if n >= 10:
        p95 = abs(float(np.percentile(r_mult, 95)))
        p5 = abs(float(np.percentile(r_mult, 5)))
        if p5 > 0:
            tail_ratio = round(p95 / p5, 2)

    outlier_win_ratio = None
    if win_count >= 2 and avg_win_r > 0:
        max_win = float(r_mult[wins_mask].max())
        outlier_win_ratio = round(max_win / avg_win_r, 2)

    outlier_loss_ratio = None
    if loss_count >= 2 and abs_avg_loss > 0:
        max_loss_abs = float(abs(r_mult[~wins_mask].min()))
        outlier_loss_ratio = round(max_loss_abs / abs_avg_loss, 2)

    ulcer_index = None
    if n >= 5:
        dd_series = cumulative - np.maximum.accumulate(cumulative)
        ulcer_index = round(float(np.sqrt(np.mean(dd_series ** 2))), 3)

    serenity_index = None
    if ulcer_index is not None and ulcer_index > 0:
        serenity_index = round(float(total_r / ulcer_index), 2)

    skewness = None
    if n >= 5:
        m = np.mean(r_mult)
        s = np.std(r_mult, ddof=1)
        if s > 0:
            skewness = round(float(np.mean(((r_mult - m) / s) ** 3) * n / max((n - 1) * (n - 2), 1) * n), 2)

    kurtosis = None
    if n >= 5:
        m = np.mean(r_mult)
        s = np.std(r_mult, ddof=1)
        if s > 0:
            kurt_raw = float(np.mean(((r_mult - m) / s) ** 4))
            kurtosis = round(kurt_raw - 3.0, 2)

    expected_daily = None
    expected_monthly = None
    expected_yearly = None
    if daily_r is not None and len(daily_r) >= 5:
        ed = float(np.mean(daily_r) * risk_per_trade)
        expected_daily = round(ed, 2)
        expected_monthly = round(ed * 21, 2)
        expected_yearly = round(ed * 252, 2)

    volatility = None
    if daily_r is not None and len(daily_r) >= 5:
        vol = float(np.std(daily_r, ddof=1) * np.sqrt(252) * 100)
        volatility = round(vol, 1)

    longest_dd_days = None
    if n >= 5 and "exit_time" in trades_df.columns:
        try:
            _exit_times = pd.to_datetime(trades_df["exit_time"])
            _cum = trades_df["r_multiple"].cumsum().values
            _rm = np.maximum.accumulate(_cum)
            _in_dd = _cum < _rm
            _max_dd_dur = 0
            _dd_start = None
            for idx_i in range(len(_in_dd)):
                if _in_dd[idx_i]:
                    if _dd_start is None:
                        _dd_start = _exit_times.iloc[idx_i]
                else:
                    if _dd_start is not None:
                        dur = (_exit_times.iloc[idx_i] - _dd_start).days
                        _max_dd_dur = max(_max_dd_dur, dur)
                        _dd_start = None
            if _dd_start is not None:
                dur = (_exit_times.iloc[-1] - _dd_start).days
                _max_dd_dur = max(_max_dd_dur, dur)
            longest_dd_days = _max_dd_dur
        except Exception:
            longest_dd_days = None

    return {
        "win_count": win_count,
        "loss_count": loss_count,
        "best_trade_r": round(best_trade_r, 2),
        "worst_trade_r": round(worst_trade_r, 2),
        "avg_win_r": round(avg_win_r, 2),
        "avg_loss_r": round(avg_loss_r, 2),
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "payoff_ratio": round(payoff_ratio, 2) if payoff_ratio != float('inf') else float('inf'),
        "recovery_factor": round(recovery_factor, 1) if recovery_factor != float('inf') else float('inf'),
        "longest_dd_trades": longest_dd,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "kelly_criterion": kelly_criterion,
        "daily_var_95": daily_var_95,
        "cvar_95": cvar_95,
        "gain_pain_ratio": gain_pain_ratio,
        "common_sense_ratio": common_sense_ratio,
        "tail_ratio": tail_ratio,
        "outlier_win_ratio": outlier_win_ratio,
        "outlier_loss_ratio": outlier_loss_ratio,
        "ulcer_index": ulcer_index,
        "serenity_index": serenity_index,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "expected_daily": expected_daily,
        "expected_monthly": expected_monthly,
        "expected_yearly": expected_yearly,
        "volatility": volatility,
        "longest_dd_days": longest_dd_days,
    }
