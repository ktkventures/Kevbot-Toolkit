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
    """Load saved portfolios from file."""
    if not os.path.exists(PORTFOLIOS_FILE):
        return []
    with open(PORTFOLIOS_FILE, 'r') as f:
        return json.load(f)


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
