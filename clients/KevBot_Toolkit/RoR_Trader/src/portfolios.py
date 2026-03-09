"""
Portfolio Management for RoR Trader
====================================

Handles portfolio CRUD, combined trade computation with compounding,
drawdown analysis, correlation, and prop firm compliance checking.
"""

import json
import os
import copy
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Callable

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PORTFOLIOS_FILE = os.path.join(_SCRIPT_DIR, "portfolios.json")
REQUIREMENTS_FILE = os.path.join(_SCRIPT_DIR, "requirements.json")


# =============================================================================
# PROP FIRM RULE DEFINITIONS
# =============================================================================

PROP_FIRM_RULES = {
    "ttp": {
        "name": "Trade The Pool",
        "rules": [
            {"name": "Profit Target", "type": "min_profit_pct", "value": 6.0,
             "description": "Minimum 6% profit on account"},
            {"name": "Max Daily Loss", "type": "max_daily_loss_pct", "value": 2.0,
             "description": "Maximum 2% loss in a single day"},
            {"name": "Max Total Drawdown", "type": "max_total_drawdown_pct", "value": 4.0,
             "description": "Maximum 4% total drawdown from peak"},
            {"name": "Min Profitable Days", "type": "min_profitable_days", "value": 3,
             "threshold_pct": 0.5,
             "description": "At least 3 days with 0.5%+ gain"},
            {"name": "Daily Pause", "type": "daily_pause_pct", "value": 1.5,
             "description": "Pause trading if daily loss exceeds 1.5%"},
        ]
    },
    "ftmo": {
        "name": "FTMO",
        "rules": [
            {"name": "Profit Target", "type": "min_profit_pct", "value": 10.0,
             "description": "Minimum 10% profit on account"},
            {"name": "Max Daily Loss", "type": "max_daily_loss_pct", "value": 5.0,
             "description": "Maximum 5% loss in a single day"},
            {"name": "Max Total Drawdown", "type": "max_total_drawdown_pct", "value": 10.0,
             "description": "Maximum 10% total drawdown from peak"},
            {"name": "Min Trading Days", "type": "min_trading_days", "value": 4,
             "description": "Minimum 4 trading days"},
        ]
    },
}


# =============================================================================
# PORTFOLIO CRUD
# =============================================================================

def load_portfolios() -> list:
    """Load saved portfolios from file. Migrates legacy prop_firm fields."""
    if not os.path.exists(PORTFOLIOS_FILE):
        return []
    try:
        with open(PORTFOLIOS_FILE, 'r') as f:
            portfolios = json.load(f)
    except (json.JSONDecodeError, Exception):
        return []

    # Migrate legacy prop_firm/custom_rules → requirement_set_id
    migrated = False
    for i, p in enumerate(portfolios):
        if 'requirement_set_id' not in p:
            portfolios[i] = _migrate_portfolio_prop_firm(p)
            migrated = True
    if migrated:
        _save_all(portfolios)

    return portfolios


def _save_all(portfolios: list):
    """Write portfolios list to file."""
    with open(PORTFOLIOS_FILE, 'w') as f:
        json.dump(portfolios, f, indent=2)


def save_portfolio(portfolio: dict) -> dict:
    """Save a new portfolio. Assigns ID and created_at. Returns the saved portfolio."""
    portfolios = load_portfolios()
    portfolio['id'] = max((p.get('id', 0) for p in portfolios), default=0) + 1
    portfolio['created_at'] = datetime.now().isoformat()
    portfolios.append(portfolio)
    _save_all(portfolios)
    return portfolio


def get_portfolio_by_id(portfolio_id: int) -> Optional[dict]:
    """Get a single portfolio by ID."""
    for p in load_portfolios():
        if p.get('id') == portfolio_id:
            return p
    return None


def update_portfolio(portfolio_id: int, updated: dict) -> bool:
    """Update an existing portfolio. Preserves id and created_at."""
    portfolios = load_portfolios()
    for i, p in enumerate(portfolios):
        if p.get('id') == portfolio_id:
            updated['id'] = portfolio_id
            updated['created_at'] = p['created_at']
            updated['updated_at'] = datetime.now().isoformat()
            portfolios[i] = updated
            _save_all(portfolios)
            return True
    return False


def delete_portfolio(portfolio_id: int) -> bool:
    """Delete a portfolio by ID."""
    portfolios = load_portfolios()
    original_len = len(portfolios)
    portfolios = [p for p in portfolios if p.get('id') != portfolio_id]
    if len(portfolios) < original_len:
        _save_all(portfolios)
        return True
    return False


def duplicate_portfolio(portfolio_id: int) -> Optional[dict]:
    """Duplicate a portfolio with new ID and '(Copy)' suffix."""
    portfolios = load_portfolios()
    source = None
    for p in portfolios:
        if p.get('id') == portfolio_id:
            source = p
            break
    if source is None:
        return None

    new = copy.deepcopy(source)
    new['id'] = max((p.get('id', 0) for p in portfolios), default=0) + 1
    new['created_at'] = datetime.now().isoformat()
    new['name'] = source['name'] + " (Copy)"
    new.pop('updated_at', None)
    portfolios.append(new)
    _save_all(portfolios)
    return new


# =============================================================================
# REQUIREMENT SET CRUD
# =============================================================================

def _seed_built_in_requirements() -> list:
    """Create initial requirement sets from PROP_FIRM_RULES constant."""
    seeds = []
    for i, (firm_key, firm_def) in enumerate(PROP_FIRM_RULES.items(), start=1):
        seeds.append({
            'id': i,
            'name': firm_def['name'],
            'built_in': True,
            'firm_key': firm_key,
            'rules': copy.deepcopy(firm_def['rules']),
            'created_at': datetime.now().isoformat(),
        })
    return seeds


def load_requirements() -> list:
    """Load requirement sets from file. Seeds built-in templates on first call."""
    if not os.path.exists(REQUIREMENTS_FILE):
        seeds = _seed_built_in_requirements()
        _save_all_requirements(seeds)
        return seeds
    with open(REQUIREMENTS_FILE, 'r') as f:
        return json.load(f)


def _save_all_requirements(requirements: list):
    """Write requirements list to file."""
    with open(REQUIREMENTS_FILE, 'w') as f:
        json.dump(requirements, f, indent=2)


def save_requirement_set(req_set: dict) -> dict:
    """Save a new requirement set. Assigns ID and created_at."""
    requirements = load_requirements()
    req_set['id'] = max((r.get('id', 0) for r in requirements), default=0) + 1
    req_set['created_at'] = datetime.now().isoformat()
    req_set.setdefault('built_in', False)
    req_set.setdefault('firm_key', None)
    requirements.append(req_set)
    _save_all_requirements(requirements)
    return req_set


def get_requirement_set_by_id(req_id: int) -> Optional[dict]:
    """Get a single requirement set by ID."""
    for r in load_requirements():
        if r.get('id') == req_id:
            return r
    return None


def update_requirement_set(req_id: int, updated: dict) -> bool:
    """Update an existing requirement set. Preserves id, created_at, built_in."""
    requirements = load_requirements()
    for i, r in enumerate(requirements):
        if r.get('id') == req_id:
            updated['id'] = req_id
            updated['created_at'] = r['created_at']
            updated['built_in'] = r.get('built_in', False)
            updated['firm_key'] = r.get('firm_key')
            updated['updated_at'] = datetime.now().isoformat()
            requirements[i] = updated
            _save_all_requirements(requirements)
            return True
    return False


def delete_requirement_set(req_id: int) -> bool:
    """Delete a requirement set by ID. Blocks deletion of built_in sets."""
    requirements = load_requirements()
    original_len = len(requirements)
    requirements = [r for r in requirements if not (r.get('id') == req_id and not r.get('built_in'))]
    if len(requirements) < original_len:
        _save_all_requirements(requirements)
        return True
    return False


def duplicate_requirement_set(req_id: int) -> Optional[dict]:
    """Duplicate a requirement set. Always creates non-built_in copy."""
    requirements = load_requirements()
    source = None
    for r in requirements:
        if r.get('id') == req_id:
            source = r
            break
    if source is None:
        return None

    new = copy.deepcopy(source)
    new['id'] = max((r.get('id', 0) for r in requirements), default=0) + 1
    new['created_at'] = datetime.now().isoformat()
    new['name'] = source['name'] + " (Copy)"
    new['built_in'] = False
    new['firm_key'] = None
    new.pop('updated_at', None)
    requirements.append(new)
    _save_all_requirements(requirements)
    return new


# =============================================================================
# PORTFOLIO MIGRATION
# =============================================================================

def _migrate_portfolio_prop_firm(portfolio: dict) -> dict:
    """Migrate legacy prop_firm/custom_rules to requirement_set_id."""
    if 'requirement_set_id' in portfolio:
        return portfolio

    firm_key = portfolio.get('prop_firm')
    custom_rules = portfolio.get('custom_rules', [])

    if custom_rules:
        req_set = save_requirement_set({
            'name': f"{portfolio.get('name', 'Unknown')} - Custom Rules",
            'rules': custom_rules,
        })
        portfolio['requirement_set_id'] = req_set['id']
    elif firm_key:
        for rs in load_requirements():
            if rs.get('firm_key') == firm_key:
                portfolio['requirement_set_id'] = rs['id']
                break
        else:
            portfolio['requirement_set_id'] = None
    else:
        portfolio['requirement_set_id'] = None

    portfolio.pop('prop_firm', None)
    portfolio.pop('custom_rules', None)
    return portfolio


# =============================================================================
# PORTFOLIO COMPUTATION ENGINE
# =============================================================================

def get_portfolio_trades(
    portfolio: dict,
    get_strategy_fn: Callable,
    get_trades_fn: Callable,
) -> dict:
    """
    Compute combined trade data for a portfolio with compounding support.

    Returns dict with:
        - 'strategy_trades': {strategy_id: DataFrame}
        - 'combined_trades': DataFrame sorted by exit_time with dollar_pnl
        - 'equity_curve': Series of cumulative dollar P&L
        - 'daily_pnl': DataFrame with date and daily_pnl columns
        - 'strategy_daily_pnl': DataFrame pivoted with one column per strategy
    """
    starting_balance = portfolio.get('starting_balance', 10000.0)
    compound_rate = portfolio.get('compound_rate', 0.0)

    strategy_trades = {}
    all_trades = []

    for ps in portfolio.get('strategies', []):
        sid = ps['strategy_id']
        base_risk = ps['risk_per_trade']
        strat = get_strategy_fn(sid)
        if strat is None:
            continue

        trades = get_trades_fn(strat)
        if trades is None or len(trades) == 0:
            continue

        trades = trades.copy()
        trades['strategy_id'] = sid
        trades['strategy_name'] = strat.get('name', f'Strategy {sid}')
        trades['base_risk_per_trade'] = base_risk
        strategy_trades[sid] = trades
        all_trades.append(trades)

    if not all_trades:
        empty = pd.DataFrame()
        return {
            'strategy_trades': strategy_trades,
            'combined_trades': empty,
            'equity_curve': pd.Series(dtype=float),
            'daily_pnl': empty,
            'strategy_daily_pnl': empty,
        }

    combined = pd.concat(all_trades, ignore_index=True).sort_values('exit_time').reset_index(drop=True)

    # Compute dollar P&L sequentially for compounding
    current_balance = starting_balance
    dollar_pnls = []
    scaled_risks = []

    for _, trade in combined.iterrows():
        account_growth_pct = (current_balance - starting_balance) / starting_balance
        scaled_risk = trade['base_risk_per_trade'] * (1 + account_growth_pct * compound_rate)
        scaled_risk = max(scaled_risk, 0)  # Don't allow negative risk
        dollar_pnl = trade['r_multiple'] * scaled_risk
        current_balance += dollar_pnl
        dollar_pnls.append(dollar_pnl)
        scaled_risks.append(scaled_risk)

    combined['dollar_pnl'] = dollar_pnls
    combined['scaled_risk'] = scaled_risks
    combined['cumulative_pnl'] = combined['dollar_pnl'].cumsum()

    # Equity curve
    equity_curve = combined.set_index('exit_time')['cumulative_pnl']

    # Daily P&L (combined)
    combined['exit_date'] = combined['exit_time'].dt.date
    daily_pnl = combined.groupby('exit_date')['dollar_pnl'].sum().reset_index()
    daily_pnl.columns = ['date', 'daily_pnl']

    # Per-strategy daily P&L (for correlation)
    strategy_daily = combined.groupby(['exit_date', 'strategy_name'])['dollar_pnl'].sum().unstack(fill_value=0)
    strategy_daily.index.name = 'date'

    return {
        'strategy_trades': strategy_trades,
        'combined_trades': combined,
        'equity_curve': equity_curve,
        'daily_pnl': daily_pnl,
        'strategy_daily_pnl': strategy_daily,
    }


def calculate_portfolio_kpis(portfolio: dict, combined_trades: pd.DataFrame,
                              daily_pnl: pd.DataFrame) -> dict:
    """Calculate portfolio-level KPIs including dollar metrics."""
    starting_balance = portfolio.get('starting_balance', 10000.0)

    if len(combined_trades) == 0:
        return {
            'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
            'total_pnl': 0, 'final_balance': starting_balance,
            'max_drawdown_pct': 0, 'max_drawdown_dollars': 0,
            'avg_daily_pnl': 0, 'daily_pnl_std': 0,
            'profitable_days_count': 0, 'total_trading_days': 0,
            'profitable_days_pct': 0,
        }

    wins = combined_trades[combined_trades['win'] == True]
    losses = combined_trades[combined_trades['win'] == False]

    gross_profit = wins['dollar_pnl'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['dollar_pnl'].sum()) if len(losses) > 0 else 0

    total_pnl = combined_trades['dollar_pnl'].sum()
    final_balance = starting_balance + total_pnl

    # Drawdown from equity curve
    cumulative = combined_trades['cumulative_pnl'] + starting_balance
    peak = cumulative.cummax()
    drawdown = cumulative - peak
    max_dd_dollars = drawdown.min()
    max_dd_pct = (drawdown / peak).min() * 100 if peak.max() > 0 else 0

    # Daily stats
    if len(daily_pnl) > 0:
        avg_daily = daily_pnl['daily_pnl'].mean()
        std_daily = daily_pnl['daily_pnl'].std()
        profitable_days = (daily_pnl['daily_pnl'] > 0).sum()
        total_days = len(daily_pnl)
    else:
        avg_daily = std_daily = 0
        profitable_days = total_days = 0

    return {
        'total_trades': len(combined_trades),
        'win_rate': len(wins) / len(combined_trades) * 100 if len(combined_trades) > 0 else 0,
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        'total_pnl': total_pnl,
        'final_balance': final_balance,
        'max_drawdown_pct': max_dd_pct,
        'max_drawdown_dollars': max_dd_dollars,
        'avg_daily_pnl': avg_daily,
        'daily_pnl_std': std_daily if not pd.isna(std_daily) else 0,
        'profitable_days_count': int(profitable_days),
        'total_trading_days': total_days,
        'profitable_days_pct': profitable_days / total_days * 100 if total_days > 0 else 0,
    }


def compute_drawdown_series(combined_trades: pd.DataFrame, starting_balance: float) -> pd.DataFrame:
    """Compute drawdown series from combined trades."""
    if len(combined_trades) == 0:
        return pd.DataFrame(columns=['exit_time', 'cumulative_pnl', 'peak', 'drawdown', 'drawdown_pct'])

    df = combined_trades[['exit_time', 'cumulative_pnl']].copy()
    df['balance'] = df['cumulative_pnl'] + starting_balance
    df['peak'] = df['balance'].cummax()
    df['drawdown'] = df['balance'] - df['peak']
    df['drawdown_pct'] = (df['drawdown'] / df['peak']) * 100
    return df


def compute_strategy_correlation(strategy_daily_pnl: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix of daily P&L across strategies."""
    if strategy_daily_pnl.empty or len(strategy_daily_pnl.columns) < 2:
        return pd.DataFrame()
    return strategy_daily_pnl.corr()


# =============================================================================
# PROP FIRM RULE EVALUATION
# =============================================================================

def evaluate_prop_firm_rules(
    firm_key: str,
    portfolio: dict,
    kpis: dict,
    daily_pnl: pd.DataFrame,
    custom_rules: list = None,
) -> dict:
    """
    Evaluate portfolio against a prop firm's rules.

    Returns dict with:
        - firm_name: str
        - rules: list of evaluation results
        - overall_pass: bool
    """
    starting_balance = portfolio.get('starting_balance', 10000.0)

    if firm_key == "custom":
        rules = custom_rules or []
        firm_name = "Custom Rules"
    else:
        firm_def = PROP_FIRM_RULES.get(firm_key, {})
        rules = firm_def.get('rules', [])
        firm_name = firm_def.get('name', firm_key)

    results = []

    for rule in rules:
        result = _evaluate_single_rule(rule, starting_balance, kpis, daily_pnl)
        results.append(result)

    overall_pass = all(r['passed'] for r in results) if results else True

    return {
        'firm_name': firm_name,
        'rules': results,
        'overall_pass': overall_pass,
    }


def _evaluate_single_rule(rule: dict, starting_balance: float, kpis: dict,
                           daily_pnl: pd.DataFrame) -> dict:
    """Evaluate a single prop firm rule."""
    rule_type = rule['type']
    rule_value = rule['value']
    name = rule['name']

    if rule_type == 'min_profit_pct':
        actual_pct = kpis['total_pnl'] / starting_balance * 100
        passed = actual_pct >= rule_value
        limit_display = f"+{rule_value}%"
        value_display = f"{actual_pct:+.1f}%"
        margin = actual_pct - rule_value

    elif rule_type == 'max_daily_loss_pct':
        if len(daily_pnl) > 0:
            worst_day = daily_pnl['daily_pnl'].min()
            worst_day_pct = abs(worst_day) / starting_balance * 100
        else:
            worst_day_pct = 0
        passed = worst_day_pct <= rule_value
        limit_display = f"-{rule_value}%"
        value_display = f"-{worst_day_pct:.1f}%"
        margin = rule_value - worst_day_pct

    elif rule_type == 'max_total_drawdown_pct':
        max_dd = abs(kpis['max_drawdown_pct'])
        passed = max_dd <= rule_value
        limit_display = f"-{rule_value}%"
        value_display = f"-{max_dd:.1f}%"
        margin = rule_value - max_dd

    elif rule_type == 'min_profitable_days':
        threshold_pct = rule.get('threshold_pct', 0.5)
        threshold_dollars = starting_balance * threshold_pct / 100
        if len(daily_pnl) > 0:
            count = (daily_pnl['daily_pnl'] >= threshold_dollars).sum()
        else:
            count = 0
        passed = count >= rule_value
        limit_display = f"{rule_value} days"
        value_display = f"{count} days"
        margin = count - rule_value

    elif rule_type == 'min_trading_days':
        days = kpis.get('total_trading_days', 0)
        passed = days >= rule_value
        limit_display = f"{rule_value} days"
        value_display = f"{days} days"
        margin = days - rule_value

    elif rule_type == 'daily_pause_pct':
        # Same calc as max_daily_loss_pct — semantic difference:
        # daily_pause = soft limit (pause, resume next day)
        # max_daily_loss = hard limit (potential disqualification)
        if len(daily_pnl) > 0:
            worst_day = daily_pnl['daily_pnl'].min()
            worst_day_pct = abs(worst_day) / starting_balance * 100
        else:
            worst_day_pct = 0
        passed = worst_day_pct <= rule_value
        limit_display = f"-{rule_value}%"
        value_display = f"-{worst_day_pct:.1f}%"
        margin = rule_value - worst_day_pct

    else:
        return {'name': name, 'limit_display': '?', 'value_display': '?',
                'passed': True, 'margin': 0}

    return {
        'name': name,
        'limit_display': limit_display,
        'value_display': value_display,
        'passed': passed,
        'margin': margin,
    }


def evaluate_requirement_set(
    requirement_set: dict,
    portfolio: dict,
    kpis: dict,
    daily_pnl: pd.DataFrame,
) -> dict:
    """Evaluate portfolio against a requirement set's rules."""
    starting_balance = portfolio.get('starting_balance', 10000.0)
    rules = requirement_set.get('rules', [])
    firm_name = requirement_set.get('name', 'Unknown')

    results = []
    for rule in rules:
        result = _evaluate_single_rule(rule, starting_balance, kpis, daily_pnl)
        results.append(result)

    overall_pass = all(r['passed'] for r in results) if results else True

    return {
        'firm_name': firm_name,
        'rules': results,
        'overall_pass': overall_pass,
    }


def get_daily_limit_thresholds(portfolio: dict) -> dict:
    """
    Extract risk limit thresholds from the portfolio's requirement set.

    Returns dict with optional keys:
        - 'max_daily_loss_pct': float (hard limit)
        - 'daily_pause_pct': float (soft limit)
        - 'max_total_drawdown_pct': float
    """
    req_id = portfolio.get('requirement_set_id')
    if not req_id:
        return {}
    req_set = get_requirement_set_by_id(req_id)
    if not req_set:
        return {}
    thresholds = {}
    for rule in req_set.get('rules', []):
        if rule['type'] == 'max_daily_loss_pct':
            thresholds['max_daily_loss_pct'] = rule['value']
        elif rule['type'] == 'daily_pause_pct':
            thresholds['daily_pause_pct'] = rule['value']
        elif rule['type'] == 'max_total_drawdown_pct':
            thresholds['max_total_drawdown_pct'] = rule['value']
    return thresholds


def compute_worst_case_analysis(
    daily_pnl: pd.DataFrame,
    starting_balance: float,
    thresholds: dict,
) -> dict:
    """
    Compute historical worst-case metrics from portfolio daily P&L data.

    Returns dict with worst_day, streak, rolling DD, breach counts, and top 5 worst days.
    """
    if len(daily_pnl) == 0:
        return {
            'worst_day_dollars': 0, 'worst_day_pct': 0, 'worst_day_date': None,
            'worst_streak_days': 0, 'worst_streak_dollars': 0,
            'worst_5day_rolling_dd': 0,
            'days_breach_daily_pause': 0, 'days_breach_max_daily_loss': 0,
            'top_5_worst_days': [],
        }

    dp = daily_pnl.copy()
    dp['pnl_pct'] = dp['daily_pnl'] / starting_balance * 100
    dp['cumulative_pnl'] = dp['daily_pnl'].cumsum()
    dp['balance'] = dp['cumulative_pnl'] + starting_balance
    dp['peak'] = dp['balance'].cummax()
    dp['dd_pct'] = (dp['balance'] - dp['peak']) / dp['peak'] * 100

    # Worst single day
    worst_idx = dp['daily_pnl'].idxmin()
    worst_day_dollars = dp.loc[worst_idx, 'daily_pnl']
    worst_day_pct = dp.loc[worst_idx, 'pnl_pct']
    worst_day_date = dp.loc[worst_idx, 'date']

    # Worst consecutive losing streak
    is_loss = (dp['daily_pnl'] < 0).astype(int)
    streak_id = (is_loss != is_loss.shift()).cumsum()
    loss_groups = dp[is_loss == 1].groupby(streak_id[is_loss == 1])
    if len(loss_groups) > 0:
        streak_stats = loss_groups.agg(
            days=('daily_pnl', 'count'),
            total_loss=('daily_pnl', 'sum'),
        )
        worst_streak_row = streak_stats.loc[streak_stats['days'].idxmax()]
        worst_streak_days = int(worst_streak_row['days'])
        worst_streak_dollars = float(worst_streak_row['total_loss'])
    else:
        worst_streak_days = 0
        worst_streak_dollars = 0.0

    # Worst 5-day rolling drawdown
    if len(dp) >= 5:
        rolling_5d = dp['daily_pnl'].rolling(5).sum()
        worst_5day_rolling_dd = float(rolling_5d.min())
    else:
        _total = dp['daily_pnl'].sum()
        worst_5day_rolling_dd = float(_total) if _total < 0 else 0.0

    # Threshold breach counts
    max_daily_loss = thresholds.get('max_daily_loss_pct')
    daily_pause = thresholds.get('daily_pause_pct')

    days_breach_max = 0
    if max_daily_loss is not None:
        days_breach_max = int(((dp['pnl_pct'] < 0) & (dp['pnl_pct'].abs() >= max_daily_loss)).sum())

    days_breach_pause = 0
    if daily_pause is not None:
        days_breach_pause = int(((dp['pnl_pct'] < 0) & (dp['pnl_pct'].abs() >= daily_pause)).sum())

    # Top 5 worst days
    worst_5 = dp.nsmallest(5, 'daily_pnl')
    top_5 = []
    for _, row in worst_5.iterrows():
        breach = "None"
        abs_pct = abs(row['pnl_pct'])
        if max_daily_loss is not None and abs_pct >= max_daily_loss:
            breach = "Max Daily Loss"
        elif daily_pause is not None and abs_pct >= daily_pause:
            breach = "Daily Pause"
        top_5.append({
            'date': row['date'],
            'pnl_dollars': float(row['daily_pnl']),
            'pnl_pct': float(row['pnl_pct']),
            'cumulative_dd_pct': float(row['dd_pct']),
            'breach_status': breach,
        })

    return {
        'worst_day_dollars': float(worst_day_dollars),
        'worst_day_pct': float(worst_day_pct),
        'worst_day_date': worst_day_date,
        'worst_streak_days': worst_streak_days,
        'worst_streak_dollars': float(worst_streak_dollars),
        'worst_5day_rolling_dd': float(worst_5day_rolling_dd),
        'days_breach_daily_pause': days_breach_pause,
        'days_breach_max_daily_loss': days_breach_max,
        'top_5_worst_days': top_5,
    }


def compute_capital_utilization(
    combined_trades: pd.DataFrame,
    starting_balance: float,
) -> dict | None:
    """
    Compute capital utilization timeline showing buying power over time.

    For each trade, calculates position size (quantity * entry_price) and
    tracks when capital is tied up in open positions vs available.

    Returns dict with timeline DataFrame and summary metrics, or None if
    the required columns (entry_price, risk) are not available.
    """
    required = {'entry_price', 'risk', 'scaled_risk', 'entry_time', 'exit_time', 'dollar_pnl'}
    if len(combined_trades) == 0 or not required.issubset(combined_trades.columns):
        return None

    events = []
    for _, trade in combined_trades.iterrows():
        risk = trade['risk']
        scaled = trade['scaled_risk']
        entry_px = trade['entry_price']
        if pd.isna(risk) or pd.isna(scaled) or pd.isna(entry_px) or risk <= 0 or scaled <= 0:
            continue
        quantity = int(scaled / risk)
        if quantity <= 0:
            continue
        capital = quantity * entry_px

        events.append({
            'time': trade['entry_time'],
            'capital_change': capital,
            'positions_change': 1,
            'pnl': 0.0,
        })
        events.append({
            'time': trade['exit_time'],
            'capital_change': -capital,
            'positions_change': -1,
            'pnl': trade['dollar_pnl'],
        })

    if not events:
        return None

    tl = pd.DataFrame(events).sort_values('time').reset_index(drop=True)
    tl['capital_deployed'] = tl['capital_change'].cumsum()
    tl['concurrent_positions'] = tl['positions_change'].cumsum()
    tl['realized_pnl'] = tl['pnl'].cumsum()
    tl['available_buying_power'] = starting_balance + tl['realized_pnl'] - tl['capital_deployed']

    peak_capital = float(tl['capital_deployed'].max())
    min_bp = float(tl['available_buying_power'].min())

    # Count transitions to insufficient capital (buying power ≤ 0)
    bp_negative = tl['available_buying_power'] <= 0
    insufficient_events = int((bp_negative & ~bp_negative.shift(fill_value=False)).sum())

    # Time-weighted average concurrent positions
    if len(tl) >= 2:
        durations = tl['time'].diff().dt.total_seconds().fillna(0)
        total_time = durations.sum()
        if total_time > 0:
            avg_conc = float((tl['concurrent_positions'].shift(fill_value=0) * durations).sum() / total_time)
        else:
            avg_conc = float(tl['concurrent_positions'].mean())
    else:
        avg_conc = float(tl['concurrent_positions'].mean())

    return {
        'timeline': tl[['time', 'capital_deployed', 'concurrent_positions', 'available_buying_power']],
        'peak_capital_deployed': peak_capital,
        'peak_capital_pct': peak_capital / starting_balance * 100 if starting_balance > 0 else 0,
        'min_buying_power': min_bp,
        'min_buying_power_pct': min_bp / starting_balance * 100 if starting_balance > 0 else 0,
        'max_concurrent_positions': int(tl['concurrent_positions'].max()),
        'avg_concurrent_positions': round(avg_conc, 1),
        'insufficient_capital_events': insufficient_events,
    }


def run_monte_carlo(
    combined_trades: pd.DataFrame,
    daily_pnl: pd.DataFrame,
    starting_balance: float,
    thresholds: dict,
    n_simulations: int = 1000,
    shuffle_mode: str = 'daily',
) -> dict:
    """
    Run Monte Carlo simulation by shuffling historical trade data.

    Shuffle modes:
        - 'daily': shuffle order of entire trading days (preserves intraday correlation)
        - 'weekly': shuffle order of entire weeks
        - 'individual': shuffle individual trade P&Ls (breaks all time correlation)

    Returns dict with bust/pause probabilities, DD distributions, equity percentile bands.
    """
    max_total_dd = thresholds.get('max_total_drawdown_pct')
    max_daily_loss = thresholds.get('max_daily_loss_pct')
    daily_pause = thresholds.get('daily_pause_pct')

    rng = np.random.default_rng()

    if shuffle_mode == 'individual':
        trade_pnls = combined_trades['dollar_pnl'].values.copy()
        n_trades = len(trade_pnls)
        if n_trades == 0:
            return _empty_mc_result(n_simulations, shuffle_mode)

        all_equity = np.zeros((n_simulations, n_trades))
        max_dd_values = np.zeros(n_simulations)
        worst_day_values = np.zeros(n_simulations)
        bust_count = 0
        pause_count = 0
        max_loss_count = 0

        for i in range(n_simulations):
            shuffled = rng.permutation(trade_pnls)
            cumulative = np.cumsum(shuffled)
            all_equity[i] = cumulative

            balance = cumulative + starting_balance
            peak = np.maximum.accumulate(balance)
            dd_pct = (balance - peak) / peak * 100
            max_dd_values[i] = dd_pct.min()

            # For individual trade mode, use worst single trade as proxy for worst day
            worst_trade_pct = (shuffled.min() / starting_balance) * 100
            worst_day_values[i] = worst_trade_pct

            if max_total_dd is not None and abs(dd_pct.min()) >= max_total_dd:
                bust_count += 1
            if daily_pause is not None and abs(worst_trade_pct) >= daily_pause:
                pause_count += 1
            if max_daily_loss is not None and abs(worst_trade_pct) >= max_daily_loss:
                max_loss_count += 1

    else:
        # Block-based shuffle (daily or weekly)
        dp = daily_pnl.copy()
        dp['date'] = pd.to_datetime(dp['date'])

        if shuffle_mode == 'weekly':
            iso = dp['date'].dt.isocalendar()
            dp['block_key'] = iso.year.astype(str) + '_W' + iso.week.astype(str)
        else:  # daily (default)
            dp['block_key'] = dp['date'].astype(str)

        # Group P&L into blocks
        blocks = []
        for _, group in dp.groupby('block_key', sort=False):
            blocks.append(group['daily_pnl'].values)

        n_blocks = len(blocks)
        if n_blocks == 0:
            return _empty_mc_result(n_simulations, shuffle_mode)

        block_indices = np.arange(n_blocks)
        total_days = sum(len(b) for b in blocks)

        all_equity = np.zeros((n_simulations, total_days))
        max_dd_values = np.zeros(n_simulations)
        worst_day_values = np.zeros(n_simulations)
        bust_count = 0
        pause_count = 0
        max_loss_count = 0

        for i in range(n_simulations):
            shuffled_indices = rng.permutation(block_indices)
            daily_sequence = np.concatenate([blocks[idx] for idx in shuffled_indices])
            cumulative = np.cumsum(daily_sequence)
            all_equity[i] = cumulative

            balance = cumulative + starting_balance
            peak = np.maximum.accumulate(balance)
            dd_pct = (balance - peak) / peak * 100
            max_dd_values[i] = dd_pct.min()

            worst_day_pct = (daily_sequence.min() / starting_balance) * 100
            worst_day_values[i] = worst_day_pct

            if max_total_dd is not None and abs(dd_pct.min()) >= max_total_dd:
                bust_count += 1
            if daily_pause is not None and abs(worst_day_pct) >= daily_pause:
                pause_count += 1
            if max_daily_loss is not None and abs(worst_day_pct) >= max_daily_loss:
                max_loss_count += 1

    # Percentile bands for equity curves
    percentiles = {}
    for p in [5, 25, 50, 75, 95]:
        percentiles[str(p)] = np.percentile(all_equity, p, axis=0)

    return {
        'bust_probability': bust_count / n_simulations * 100,
        'daily_pause_probability': pause_count / n_simulations * 100,
        'max_daily_loss_probability': max_loss_count / n_simulations * 100,
        'max_dd_values': max_dd_values,
        'worst_day_values': worst_day_values,
        'equity_percentiles': percentiles,
        'median_max_dd': float(np.median(max_dd_values)),
        'p95_max_dd': float(np.percentile(max_dd_values, 5)),  # 5th percentile = worst case
        'expected_worst_day': float(np.median(worst_day_values)),
        'n_simulations': n_simulations,
        'shuffle_mode': shuffle_mode,
    }


def _empty_mc_result(n_simulations: int, shuffle_mode: str) -> dict:
    """Return empty Monte Carlo result when no data available."""
    return {
        'bust_probability': 0.0,
        'daily_pause_probability': 0.0,
        'max_daily_loss_probability': 0.0,
        'max_dd_values': np.zeros(0),
        'worst_day_values': np.zeros(0),
        'equity_percentiles': {str(p): np.zeros(0) for p in [5, 25, 50, 75, 95]},
        'median_max_dd': 0.0,
        'p95_max_dd': 0.0,
        'expected_worst_day': 0.0,
        'n_simulations': n_simulations,
        'shuffle_mode': shuffle_mode,
    }


# =============================================================================
# STRATEGY RECOMMENDATION ENGINE
# =============================================================================

def _score_candidate(current_kpis: dict, hypo_kpis: dict,
                     hypo_corr: pd.DataFrame) -> float:
    """Score a candidate strategy addition. Higher = better complement."""
    score = 0.0

    # P&L improvement (30% weight)
    current_pnl = current_kpis.get('total_pnl', 0)
    pnl_improvement = hypo_kpis['total_pnl'] - current_pnl
    pnl_score = pnl_improvement / max(abs(current_pnl), 1)
    score += pnl_score * 30

    # Drawdown reduction (25% weight)
    dd_current = abs(current_kpis.get('max_drawdown_pct', 0))
    dd_new = abs(hypo_kpis['max_drawdown_pct'])
    dd_improvement = dd_current - dd_new
    dd_score = dd_improvement / max(dd_current, 0.1)
    score += dd_score * 25

    # Profit Factor improvement (20% weight)
    pf_current = current_kpis.get('profit_factor', 0)
    if pf_current == float('inf'):
        pf_current = 10
    pf_new = hypo_kpis['profit_factor']
    if pf_new == float('inf'):
        pf_new = 10
    pf_improvement = pf_new - pf_current
    score += min(pf_improvement, 5) * 4

    # Low correlation bonus (15% weight)
    if hypo_corr is not None and len(hypo_corr) >= 2:
        mask = ~np.eye(len(hypo_corr), dtype=bool)
        avg_corr = hypo_corr.values[mask].mean()
        score += (1 - avg_corr) * 15

    # Win rate improvement (10% weight)
    wr_change = hypo_kpis['win_rate'] - current_kpis.get('win_rate', 0)
    score += min(wr_change, 10) * 1

    return score


def compute_strategy_recommendations(
    current_portfolio: dict,
    current_data: dict,
    candidate_strategies: list,
    get_strategy_fn: Callable,
    get_trades_fn: Callable,
    top_n: int = 5,
) -> list:
    """
    For each candidate strategy, compute what portfolio KPIs would be
    if that strategy were added. Return ranked list of recommendations.
    """
    current_kpis = calculate_portfolio_kpis(
        current_portfolio,
        current_data['combined_trades'],
        current_data['daily_pnl']
    )

    recommendations = []
    for strat in candidate_strategies:
        hypo_strategies = copy.deepcopy(current_portfolio.get('strategies', [])) + [{
            'strategy_id': strat['id'],
            'risk_per_trade': strat.get('risk_per_trade', 100.0),
        }]
        hypo_portfolio = {
            'starting_balance': current_portfolio.get('starting_balance', 10000.0),
            'compound_rate': current_portfolio.get('compound_rate', 0.0),
            'strategies': hypo_strategies,
        }

        hypo_data = get_portfolio_trades(hypo_portfolio, get_strategy_fn, get_trades_fn)
        if len(hypo_data['combined_trades']) == 0:
            continue

        hypo_kpis = calculate_portfolio_kpis(
            hypo_portfolio, hypo_data['combined_trades'], hypo_data['daily_pnl']
        )
        hypo_corr = compute_strategy_correlation(hypo_data['strategy_daily_pnl'])

        score = _score_candidate(current_kpis, hypo_kpis, hypo_corr)

        avg_correlation = 0.0
        if hypo_corr is not None and len(hypo_corr) >= 2:
            mask = ~np.eye(len(hypo_corr), dtype=bool)
            avg_correlation = float(hypo_corr.values[mask].mean())

        recommendations.append({
            'strategy_id': strat['id'],
            'strategy_name': strat.get('name', f"Strategy {strat['id']}"),
            'score': score,
            'pnl_change': hypo_kpis['total_pnl'] - current_kpis.get('total_pnl', 0),
            'dd_change': hypo_kpis['max_drawdown_pct'] - current_kpis.get('max_drawdown_pct', 0),
            'pf_change': (hypo_kpis['profit_factor'] if hypo_kpis['profit_factor'] != float('inf') else 10)
                       - (current_kpis.get('profit_factor', 0) if current_kpis.get('profit_factor', 0) != float('inf') else 10),
            'wr_change': hypo_kpis['win_rate'] - current_kpis.get('win_rate', 0),
            'avg_correlation': avg_correlation,
        })

    recommendations.sort(key=lambda r: r['score'], reverse=True)
    return recommendations[:top_n]


# =============================================================================
# ALERT CONTEXT
# =============================================================================

# =============================================================================
# ACCOUNT MANAGEMENT
# =============================================================================

def get_account(portfolio: dict) -> dict:
    """Get or initialize the account sub-dict for a portfolio."""
    if 'account' not in portfolio:
        portfolio['account'] = {
            'starting_balance': portfolio.get('starting_balance', 10000.0),
            'ledger': [],
            'notes': '',
            'notes_updated_at': None,
        }
    return portfolio['account']


def compute_account_balance(account: dict) -> float:
    """Compute current balance from ledger entries."""
    return sum(entry.get('amount', 0) for entry in account.get('ledger', []))


def add_ledger_entry(portfolio: dict, entry_type: str, amount: float,
                     note: str = '', date: str = None, auto: bool = False) -> dict:
    """Add a ledger entry to the portfolio's account.

    Args:
        portfolio: Portfolio dict (modified in place).
        entry_type: 'deposit', 'withdrawal', or 'trading_pnl'.
        amount: Dollar amount (positive for deposit/profit, negative for withdrawal/loss).
        note: Optional note.
        date: ISO date string. Defaults to today.
        auto: True for auto-generated trading P&L entries.

    Returns:
        The new ledger entry.
    """
    account = get_account(portfolio)
    ledger = account.setdefault('ledger', [])
    entry_id = max((e.get('id', 0) for e in ledger), default=0) + 1
    entry = {
        'id': entry_id,
        'date': date or datetime.now().strftime('%Y-%m-%d'),
        'type': entry_type,
        'amount': amount,
        'note': note,
        'auto': auto,
    }
    ledger.append(entry)
    return entry


def remove_ledger_entry(portfolio: dict, entry_id: int) -> bool:
    """Remove a ledger entry by ID."""
    account = get_account(portfolio)
    ledger = account.get('ledger', [])
    original_len = len(ledger)
    account['ledger'] = [e for e in ledger if e.get('id') != entry_id]
    return len(account['ledger']) < original_len


def get_balance_history(account: dict) -> list:
    """Compute running balance from ledger, sorted by date.

    Returns list of {'date': str, 'balance': float, 'type': str}.
    """
    ledger = sorted(account.get('ledger', []), key=lambda e: e.get('date', ''))
    running = 0.0
    history = []
    for entry in ledger:
        running += entry.get('amount', 0)
        history.append({
            'date': entry.get('date', ''),
            'balance': running,
            'type': entry.get('type', ''),
        })
    return history


def get_portfolio_alert_context(strategy_id: int) -> list:
    """
    Find all portfolios containing a strategy and return context for alerts.

    Returns list of dicts with portfolio_id, portfolio_name, risk_per_trade,
    and requirement_set_id for each portfolio that includes this strategy.
    """
    portfolios = load_portfolios()
    context = []

    for port in portfolios:
        for alloc in port.get('strategies', []):
            if alloc.get('strategy_id') == strategy_id:
                context.append({
                    "portfolio_id": port['id'],
                    "portfolio_name": port.get('name', f"Portfolio {port['id']}"),
                    "risk_per_trade": alloc.get('risk_per_trade', 100.0),
                    "requirement_set_id": port.get('requirement_set_id'),
                })
                break

    return context
