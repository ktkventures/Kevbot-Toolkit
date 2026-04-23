"""
Portfolio Management for RoR Trader
====================================

Handles portfolio CRUD, combined trade computation with compounding,
drawdown analysis, correlation, and prop firm compliance checking.
"""

import json
import os
import copy
import math
import pandas as pd
import numpy as np
from datetime import datetime, timezone
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
    """Load saved portfolios from file or database. Migrates legacy prop_firm fields."""
    from db import USE_DB
    if USE_DB:
        from db import load_portfolios_db
        return load_portfolios_db()

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
    from db import USE_DB
    if USE_DB:
        from db import save_portfolio_db
        portfolio['created_at'] = datetime.now().isoformat()
        return save_portfolio_db(portfolio)

    portfolios = load_portfolios()
    portfolio['id'] = max((p.get('id', 0) for p in portfolios), default=0) + 1
    portfolio['created_at'] = datetime.now().isoformat()
    portfolios.append(portfolio)
    _save_all(portfolios)
    return portfolio


def get_portfolio_by_id(portfolio_id: int) -> Optional[dict]:
    """Get a single portfolio by ID."""
    from db import USE_DB
    if USE_DB:
        from db import get_portfolio_by_id_db
        return get_portfolio_by_id_db(portfolio_id)

    for p in load_portfolios():
        if p.get('id') == portfolio_id:
            return p
    return None


def update_portfolio(portfolio_id: int, updated: dict) -> bool:
    """Update an existing portfolio. Preserves id and created_at."""
    from db import USE_DB
    if USE_DB:
        from db import get_portfolio_by_id_db, update_portfolio_db
        old = get_portfolio_by_id_db(portfolio_id)
        if not old:
            return False
        updated['id'] = portfolio_id
        updated['created_at'] = old['created_at']
        updated['updated_at'] = datetime.now().isoformat()
        update_portfolio_db(portfolio_id, updated)
        return True

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
    from db import USE_DB
    if USE_DB:
        from db import delete_portfolio_db
        return delete_portfolio_db(portfolio_id)

    portfolios = load_portfolios()
    original_len = len(portfolios)
    portfolios = [p for p in portfolios if p.get('id') != portfolio_id]
    if len(portfolios) < original_len:
        _save_all(portfolios)
        return True
    return False


def duplicate_portfolio(portfolio_id: int) -> Optional[dict]:
    """Duplicate a portfolio with new ID and '(Copy)' suffix."""
    from db import USE_DB
    if USE_DB:
        from db import get_portfolio_by_id_db, save_portfolio_db
        source = get_portfolio_by_id_db(portfolio_id)
        if source is None:
            return None
        new = copy.deepcopy(source)
        new.pop('id', None)
        new['created_at'] = datetime.now().isoformat()
        new['name'] = source['name'] + " (Copy)"
        new.pop('updated_at', None)
        return save_portfolio_db(new)

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
    """Load requirement sets from file or database. Seeds built-in templates on first call."""
    from db import USE_DB
    if USE_DB:
        from db import load_requirements_db
        return load_requirements_db()

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
    from db import USE_DB
    if USE_DB:
        from db import save_requirement_set_db
        req_set['created_at'] = datetime.now().isoformat()
        req_set.setdefault('built_in', False)
        req_set.setdefault('firm_key', None)
        return save_requirement_set_db(req_set)

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
    from db import USE_DB
    if USE_DB:
        from db import get_client
        client = get_client()
        result = client.table('requirement_sets') \
            .select('*') \
            .eq('id', req_id) \
            .maybe_single() \
            .execute()
        return result.data

    for r in load_requirements():
        if r.get('id') == req_id:
            return r
    return None


def update_requirement_set(req_id: int, updated: dict) -> bool:
    """Update an existing requirement set. Preserves id, created_at, built_in."""
    from db import USE_DB
    if USE_DB:
        from db import update_requirement_set_db
        old = get_requirement_set_by_id(req_id)
        if not old:
            return False
        updated['id'] = req_id
        updated['created_at'] = old['created_at']
        updated['built_in'] = old.get('built_in', False)
        updated['firm_key'] = old.get('firm_key')
        updated['updated_at'] = datetime.now().isoformat()
        update_requirement_set_db(req_id, updated)
        return True

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
    from db import USE_DB
    if USE_DB:
        from db import delete_requirement_set_db
        old = get_requirement_set_by_id(req_id)
        if not old or old.get('built_in'):
            return False
        return delete_requirement_set_db(req_id)

    requirements = load_requirements()
    original_len = len(requirements)
    requirements = [r for r in requirements if not (r.get('id') == req_id and not r.get('built_in'))]
    if len(requirements) < original_len:
        _save_all_requirements(requirements)
        return True
    return False


def duplicate_requirement_set(req_id: int) -> Optional[dict]:
    """Duplicate a requirement set. Always creates non-built_in copy."""
    from db import USE_DB
    if USE_DB:
        from db import save_requirement_set_db
        source = get_requirement_set_by_id(req_id)
        if source is None:
            return None
        new = copy.deepcopy(source)
        new.pop('id', None)
        new['created_at'] = datetime.now().isoformat()
        new['name'] = source['name'] + " (Copy)"
        new['built_in'] = False
        new['firm_key'] = None
        new.pop('updated_at', None)
        return save_requirement_set_db(new)

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
            'profitable_days_pct': 0, 'trades_per_day': 0,
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

    trades_per_day = len(combined_trades) / total_days if total_days > 0 else 0

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
        'trades_per_day': trades_per_day,
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
    actual_raw = None  # raw comparable value for UI progress bars

    if rule_type == 'min_profit_pct':
        actual_pct = kpis['total_pnl'] / starting_balance * 100
        actual_raw = actual_pct
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
        actual_raw = worst_day_pct
        passed = worst_day_pct <= rule_value
        limit_display = f"-{rule_value}%"
        value_display = f"-{worst_day_pct:.1f}%"
        margin = rule_value - worst_day_pct

    elif rule_type == 'max_total_drawdown_pct':
        max_dd = abs(kpis['max_drawdown_pct'])
        actual_raw = max_dd
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
        actual_raw = int(count)
        passed = count >= rule_value
        limit_display = f"{rule_value} days"
        value_display = f"{count} days"
        margin = count - rule_value

    elif rule_type == 'min_trading_days':
        days = kpis.get('total_trading_days', 0)
        actual_raw = days
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
        actual_raw = worst_day_pct
        passed = worst_day_pct <= rule_value
        limit_display = f"-{rule_value}%"
        value_display = f"-{worst_day_pct:.1f}%"
        margin = rule_value - worst_day_pct

    else:
        return {'name': name, 'type': rule_type, 'threshold': rule_value,
                'actual': None, 'limit_display': '?', 'value_display': '?',
                'passed': True, 'margin': 0}

    return {
        'name': name,
        'type': rule_type,
        'threshold': rule_value,
        'actual': actual_raw,
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
    if not portfolio.get('account'):
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


# =============================================================================
# PHASE 37: PORTFOLIO LIVE DASHBOARD — DATA & COMPUTATION
# =============================================================================


def compute_strategy_r_distribution(stored_trades: list,
                                     forward_test_start: str = None) -> dict:
    """Extract backtest R distribution statistics from stored trades.

    Filters to trades before forward_test_start (backtest portion only).
    Returns {avg_r, std_r, var_r, n_trades, median_r}.
    """
    if not stored_trades:
        return {'avg_r': 0, 'std_r': 0, 'var_r': 0, 'n_trades': 0, 'median_r': 0}

    r_values = []
    ft_dt = None
    if forward_test_start:
        try:
            ft_dt = datetime.fromisoformat(forward_test_start)
            if ft_dt.tzinfo:
                ft_dt = ft_dt.astimezone(timezone.utc).replace(tzinfo=None)
        except (ValueError, TypeError):
            ft_dt = None

    for t in stored_trades:
        if ft_dt:
            try:
                entry_dt = datetime.fromisoformat(t.get('entry_time', ''))
                if entry_dt.tzinfo:
                    entry_dt = entry_dt.astimezone(timezone.utc).replace(tzinfo=None)
                if entry_dt >= ft_dt:
                    continue  # skip forward-test trades
            except (ValueError, TypeError):
                continue
        r_values.append(t.get('r_multiple', 0))

    if not r_values:
        return {'avg_r': 0, 'std_r': 0, 'var_r': 0, 'n_trades': 0, 'median_r': 0}

    arr = np.array(r_values, dtype=float)
    return {
        'avg_r': float(np.mean(arr)),
        'std_r': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'var_r': float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'n_trades': len(arr),
        'median_r': float(np.median(arr)),
    }


def _pair_raw_alerts_for_strategy(alerts: list, strategy: dict,
                                   risk_per_trade: float,
                                   portfolio_id: int = None) -> tuple:
    """Pair raw entry+exit alerts chronologically into trade records.

    Used as a fallback when live_executions aren't available.
    Returns (trades_list, open_positions_list).

    portfolio_id (optional): used to locate the matching portfolio_context
    entry on each alert so we can copy through rpt/bp/executed quantity
    fields for the Trade History display. Falls back to position_risk
    matching if portfolio_id isn't supplied or doesn't match.
    """
    sid = strategy.get('id')
    strategy_name = strategy.get('name', f'Strategy {sid}')
    symbol = strategy.get('symbol', '')
    direction = strategy.get('direction', 'LONG')

    # Filter to this strategy's entry/exit signals
    strat_alerts = [a for a in alerts
                    if a.get('strategy_id') == sid
                    and a.get('type') in ('entry_signal', 'exit_signal')]
    strat_alerts.sort(key=lambda a: a.get('timestamp', ''))

    trades = []
    open_positions = []
    pending_entry = None

    def _safe_float(val, default=0.0):
        """Convert value to float safely (handles strings, None)."""
        if val is None or val == '':
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    for alert in strat_alerts:
        if alert.get('type') == 'entry_signal':
            pending_entry = alert
        elif alert.get('type') == 'exit_signal' and pending_entry is not None:
            entry_price = _safe_float(pending_entry.get('price'))
            exit_price = _safe_float(alert.get('price'))
            stop_price = _safe_float(pending_entry.get('stop_price'))
            per_share_risk = abs(entry_price - stop_price) if stop_price and entry_price else 0

            if per_share_risk > 0:
                if direction == 'LONG':
                    r_multiple = (exit_price - entry_price) / per_share_risk
                else:
                    r_multiple = (entry_price - exit_price) / per_share_risk
            else:
                r_multiple = 0

            quantity = int(risk_per_trade / per_share_risk) if per_share_risk > 0 else 1
            buying_power_used = quantity * entry_price

            # Exit reason: try multiple fields
            exit_reason = (alert.get('trigger', '')
                          or alert.get('exit_reason', '')
                          or pending_entry.get('trigger', '')  # use entry trigger as hint
                          or 'signal')

            # Quantity taxonomy — raw-alerts fallback path. We have the
            # raw entry alert here (`pending_entry`); its portfolio_context
            # (if populated by the worker enrichment) carries the real
            # rpt/bp/executed quantities for this portfolio. Fall back
            # to RPT math if absent (older alerts, misc paths).
            entry_pc = None
            for pc in (pending_entry.get('portfolio_context') or []):
                if (portfolio_id is not None and pc.get('portfolio_id') == portfolio_id) \
                        or pc.get('position_risk') == risk_per_trade:
                    entry_pc = pc
                    break
            rpt_quantity = (entry_pc.get('rpt_quantity')
                            if entry_pc and entry_pc.get('rpt_quantity') is not None
                            else quantity)
            bp_quantity = (entry_pc.get('bp_quantity')
                           if entry_pc else None)
            executed_quantity = (entry_pc.get('executed_quantity')
                                 if entry_pc and entry_pc.get('executed_quantity') is not None
                                 else quantity)

            # Dollar P&L is (price_diff × executed_quantity), sign-flipped for
            # SHORT. r_multiple × risk_per_trade was the old formula — it's
            # only equivalent to reality when executed == RPT-planned. With
            # BP capping, executed can be much smaller; using the planned
            # risk budget overstates P&L by the cap ratio.
            if direction == 'LONG':
                dollar_pnl = (exit_price - entry_price) * executed_quantity
            else:
                dollar_pnl = (entry_price - exit_price) * executed_quantity

            trades.append({
                'strategy_id': sid,
                'strategy_name': strategy_name,
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'theoretical_entry': entry_price,
                'theoretical_exit': exit_price,
                'entry_time': pending_entry.get('timestamp', ''),
                'exit_time': alert.get('timestamp', ''),
                'exit_reason': exit_reason,
                'r_multiple': round(r_multiple, 4),
                'entry_slippage_r': 0,
                'exit_slippage_r': 0,
                'risk_per_trade': risk_per_trade,
                'dollar_pnl': round(dollar_pnl, 2),
                'planned_quantity': quantity,  # legacy alias
                'quantity': executed_quantity,
                'rpt_quantity': rpt_quantity,
                'bp_quantity': bp_quantity,
                'executed_quantity': executed_quantity,
                'buying_power_used': buying_power_used,
                'matched': True,
                'phantom': False,
                # Prefer the engine-stamped exec_type on the alert; falls back
                # to empty string if missing (surfaces loudly rather than guessing
                # from the trigger name, which yielded things like "ut" for
                # "utbot_long_reset" — meaningless for execution routing).
                'exec_type': pending_entry.get('exec_type', '') or '',
                'data_source': 'raw_alerts',
                # Raw-alerts fallback: alert dicts carry `id` directly
                # (no separate live_execution record). Details drawer
                # (roadmap 9ad) looks these up to show webhook_deliveries.
                'entry_alert_id': pending_entry.get('id'),
                'exit_alert_id': alert.get('id'),
            })
            pending_entry = None

    if pending_entry is not None:
        entry_price = _safe_float(pending_entry.get('price'))
        stop_price = _safe_float(pending_entry.get('stop_price'))
        per_share_risk = abs(entry_price - stop_price) if stop_price and entry_price else 0
        quantity = int(risk_per_trade / per_share_risk) if per_share_risk > 0 else 1
        open_positions.append({
            'strategy_id': sid,
            'strategy_name': strategy_name,
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': pending_entry.get('timestamp', ''),
            'risk_per_trade': risk_per_trade,
            'quantity': quantity,
            'buying_power_used': quantity * entry_price,
            'stop_price': stop_price,
        })

    return trades, open_positions


def get_portfolio_alert_trades(portfolio: dict,
                                get_strategy_fn: Callable) -> dict:
    """Aggregate alert-based trades across all portfolio strategies.

    Primary source: live_executions on each strategy (has slippage data).
    Fallback: raw alerts from alerts DB/file, paired chronologically.

    Args:
        portfolio: Portfolio dict with 'strategies' list.
        get_strategy_fn: Callable(strategy_id) -> strategy dict or None.

    Returns dict with alert_trades, open_positions, strategies_with_data.
    """
    alert_trades = []
    open_positions = []
    strategies_with_data = set()
    strategies_needing_fallback = []

    for alloc in portfolio.get('strategies', []):
        sid = alloc.get('strategy_id')
        risk_per_trade = alloc.get('risk_per_trade', 100.0)
        strat = get_strategy_fn(sid)
        if strat is None:
            continue

        live_execs = strat.get('live_executions', [])
        stored = strat.get('stored_trades', [])

        if live_execs:
            # Primary path: use live_executions (has slippage data)
            strategies_with_data.add(sid)
            strategy_name = strat.get('name', f'Strategy {sid}')
            symbol = strat.get('symbol', '')
            direction = strat.get('direction', 'LONG')

            entry_execs = {}
            exit_execs = {}
            for ex in live_execs:
                tidx = ex.get('matched_trade_index')
                if tidx is None:
                    continue
                if ex.get('type') == 'entry':
                    entry_execs[tidx] = ex
                elif ex.get('type') == 'exit':
                    exit_execs[tidx] = ex

            for tidx in sorted(entry_execs.keys()):
                entry_ex = entry_execs[tidx]
                exit_ex = exit_execs.get(tidx)
                stored_trade = stored[tidx] if tidx < len(stored) else {}

                entry_price = entry_ex.get('alert_price', 0)
                theoretical_entry = entry_ex.get('theoretical_price', entry_price)
                stop_price = stored_trade.get('stop_price', 0)
                per_share_risk = abs(theoretical_entry - stop_price) if stop_price else 0
                entry_slip = entry_ex.get('slippage_r', 0)

                if exit_ex:
                    exit_price = exit_ex.get('alert_price', 0)
                    theoretical_exit = exit_ex.get('theoretical_price', exit_price)
                    exit_slip = exit_ex.get('slippage_r', 0)
                    stored_r = stored_trade.get('r_multiple', 0)
                    adjusted_r = stored_r - entry_slip - exit_slip
                    quantity = int(risk_per_trade / per_share_risk) if per_share_risk > 0 else 1

                    # Quantity taxonomy (2026-04-22 BP-cap work):
                    # rpt_quantity = risk-sized (historic `quantity` math).
                    # bp_quantity = None for live_execution rows until
                    #   match_alerts_to_trades propagates portfolio_context
                    #   (future follow-up — would be filled from
                    #   entry_alert.portfolio_context[pid].bp_quantity).
                    # executed_quantity = what the webhook actually carried;
                    #   for historical rows mirrors rpt (no cap applied).
                    rpt_quantity = quantity
                    bp_quantity = entry_ex.get('bp_quantity')  # None today
                    executed_quantity = entry_ex.get(
                        'executed_quantity', quantity)

                    alert_trades.append({
                        'strategy_id': sid,
                        'strategy_name': strategy_name,
                        'symbol': symbol,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'theoretical_entry': theoretical_entry,
                        'theoretical_exit': theoretical_exit,
                        'entry_time': entry_ex.get('alert_timestamp', ''),
                        'exit_time': exit_ex.get('alert_timestamp', ''),
                        'exit_reason': stored_trade.get('exit_reason', ''),
                        'r_multiple': adjusted_r,
                        'entry_slippage_r': entry_slip,
                        'exit_slippage_r': exit_slip,
                        'risk_per_trade': risk_per_trade,
                        'dollar_pnl': adjusted_r * risk_per_trade,
                        'planned_quantity': quantity,  # legacy alias
                        'quantity': executed_quantity,
                        'rpt_quantity': rpt_quantity,
                        'bp_quantity': bp_quantity,
                        'executed_quantity': executed_quantity,
                        'buying_power_used': quantity * entry_price,
                        'matched': True,
                        'phantom': False,
                        'exec_type': stored_trade.get('exec_type', 'C'),
                        'data_source': 'live_executions',
                        # Expose alert IDs so the Portfolio Trade History
                        # Details drawer (roadmap 9ad) can fetch the full
                        # alert row + webhook_deliveries per entry/exit.
                        'entry_alert_id': entry_ex.get('alert_id'),
                        'exit_alert_id': exit_ex.get('alert_id'),
                    })
                else:
                    quantity = int(risk_per_trade / per_share_risk) if per_share_risk > 0 else 1
                    open_positions.append({
                        'strategy_id': sid,
                        'strategy_name': strategy_name,
                        'symbol': symbol,
                        'direction': direction,
                        'entry_price': entry_price,
                        'entry_time': entry_ex.get('alert_timestamp', ''),
                        'risk_per_trade': risk_per_trade,
                        'quantity': quantity,
                        'buying_power_used': quantity * entry_price,
                        'stop_price': stop_price,
                    })
        else:
            # No live_executions — queue for raw alert fallback
            strategies_needing_fallback.append((alloc, strat))

    # Fallback: pair raw alerts for strategies without live_executions
    if strategies_needing_fallback:
        from alerts import get_alerts_for_portfolio
        pid = portfolio.get('id')
        raw_alerts = get_alerts_for_portfolio(pid, limit=10000) if pid else []

        for alloc, strat in strategies_needing_fallback:
            sid = alloc.get('strategy_id')
            risk_per_trade = alloc.get('risk_per_trade', 100.0)
            trades, opens = _pair_raw_alerts_for_strategy(
                raw_alerts, strat, risk_per_trade, portfolio_id=pid
            )
            if trades or opens:
                strategies_with_data.add(sid)
                alert_trades.extend(trades)
                open_positions.extend(opens)

    # Sort completed trades by exit_time
    alert_trades.sort(key=lambda t: t.get('exit_time', ''))
    for i, t in enumerate(alert_trades):
        t['trade_number'] = i + 1

    return {
        'alert_trades': alert_trades,
        'open_positions': open_positions,
        'strategies_with_data': strategies_with_data,
    }


def get_portfolio_combined_view(portfolio: dict,
                                 get_strategy_fn: Callable,
                                 get_trades_fn: Callable) -> dict:
    """Unified portfolio trade list with clear source tags.

    Merges stored_trades (backtest + forward test) with live alert trades and
    dedupes. Each row carries a ``data_source`` that the frontend uses to
    render status badges:

      - ``'backtest'``     stored_trade entered before the strategy's
                           ``forward_test_start`` — purely historical, no
                           live alerts expected.
      - ``'forward_test'`` stored_trade entered at/after ``forward_test_start``
                           with no matching live alert — the engine ran live
                           but the alert system didn't fire (or hasn't fired
                           yet for this bar).
      - ``'matched'``      live alert that paired with a forward-test
                           stored_trade — slippage data is real, webhooks
                           fired correctly.
      - ``'live'``         live alert without a matching stored_trade — the
                           engine and alert pipeline disagreed (phantom
                           candidate).

    When an alert matches a stored_trade we drop the stored_trade row and
    keep the alert row (richer: has slippage, actual prices). The equity
    curve stays backtest-based since alerts may be partial (no exit yet).

    Returns a dict with the same shape as ``get_portfolio_trades``:
        ``{'combined_trades': list[dict], 'equity_curve': Series,
           'daily_pnl': DataFrame, 'open_positions': list[dict]}``

    Note: unlike the other combined_trades producers here, this one returns
    a list[dict], not a DataFrame — the endpoint serializes it directly.
    """
    import pandas as _pd  # local alias to avoid shadowing module pd
    bt_data = get_portfolio_trades(portfolio, get_strategy_fn, get_trades_fn)
    bt_df = bt_data.get('combined_trades')

    alert_data = get_portfolio_alert_trades(portfolio, get_strategy_fn)
    alert_rows = alert_data.get('alert_trades', [])

    # Build (strategy_id, normalized_entry_time) keys for alerts that matched
    # a stored_trade — we dedupe these out of the backtest set.
    def _norm_ts(v):
        if v is None or v == '':
            return None
        if isinstance(v, str):
            return v[:19]  # ISO trim to second
        if hasattr(v, 'isoformat'):
            try:
                return v.isoformat()[:19] if not _pd.isna(v) else None
            except Exception:
                return v.isoformat()[:19]
        return str(v)[:19]

    matched_keys = set()
    for ar in alert_rows:
        if ar.get('matched'):
            key = (ar.get('strategy_id'), _norm_ts(ar.get('entry_time')))
            matched_keys.add(key)

    # Collect each strategy's forward_test_start for tagging
    fwd_starts = {}
    for ps in portfolio.get('strategies', []):
        sid = ps.get('strategy_id')
        strat = get_strategy_fn(sid)
        if strat and strat.get('forward_test_start'):
            try:
                fwd_starts[sid] = _pd.Timestamp(strat['forward_test_start'])
            except Exception:
                pass

    out_rows: list = []

    if bt_df is not None and len(bt_df) > 0:
        bt_records = bt_df.to_dict('records')
        for row in bt_records:
            sid = row.get('strategy_id')
            key = (sid, _norm_ts(row.get('entry_time')))
            if key in matched_keys:
                continue  # alert row will represent this trade
            fwd_start = fwd_starts.get(sid)
            is_forward = False
            if fwd_start is not None:
                try:
                    et = _pd.Timestamp(row.get('entry_time'))
                    # Align timezone awareness before comparing
                    if et.tz is None and fwd_start.tz is not None:
                        et = et.tz_localize(fwd_start.tz)
                    elif et.tz is not None and fwd_start.tz is None:
                        fwd_start = fwd_start.tz_localize(et.tz)
                    is_forward = et >= fwd_start
                except Exception:
                    is_forward = False
            row['data_source'] = 'forward_test' if is_forward else 'backtest'
            out_rows.append(row)

    for ar in alert_rows:
        ar = dict(ar)  # don't mutate caller's list
        if ar.get('phantom'):
            ar['data_source'] = 'live'
        elif ar.get('matched'):
            ar['data_source'] = 'matched'
        else:
            ar['data_source'] = 'live'
        out_rows.append(ar)

    # Sort by exit_time (strings and timestamps both work with this key)
    def _sort_key(r):
        et = r.get('exit_time')
        if et is None or et == '':
            return ''
        if isinstance(et, str):
            return et
        if hasattr(et, 'isoformat'):
            try:
                return et.isoformat() if not _pd.isna(et) else ''
            except Exception:
                return et.isoformat()
        return str(et)
    out_rows.sort(key=_sort_key)

    return {
        'combined_trades': out_rows,
        'equity_curve': bt_data.get('equity_curve'),
        'daily_pnl': bt_data.get('daily_pnl'),
        'open_positions': alert_data.get('open_positions', []),
    }


def compute_portfolio_benchmark(portfolio: dict,
                                 get_strategy_fn: Callable,
                                 get_trades_fn: Callable = None,
                                 filter_strategy_id: int = None,
                                 max_trades: int = 200) -> dict:
    """Compute the "Plan" line and confidence bands from backtest R distributions.

    The X-axis is trade number (not calendar date). Each trade's expected
    contribution is the weighted average of each strategy's backtest avg_r,
    weighted by trade frequency. Confidence bands widen as sqrt(N).

    Args:
        portfolio: Portfolio dict.
        get_strategy_fn: Callable(sid) -> strategy dict.
        get_trades_fn: Optional callable(strat) -> trades DataFrame (unused,
            kept for API consistency — we read stored_trades directly).
        filter_strategy_id: If set, compute benchmark for a single strategy only.
        max_trades: Maximum trade number for the X-axis.

    Returns dict with plan_cumulative_pnl, bands, per_strategy stats, etc.
    """
    per_strategy = {}

    for alloc in portfolio.get('strategies', []):
        sid = alloc.get('strategy_id')
        risk = alloc.get('risk_per_trade', 100.0)
        if filter_strategy_id is not None and sid != filter_strategy_id:
            continue

        strat = get_strategy_fn(sid)
        if strat is None:
            continue

        stored = strat.get('stored_trades', [])
        ft_start = strat.get('forward_test_start')
        dist = compute_strategy_r_distribution(stored, ft_start)

        if dist['n_trades'] == 0:
            continue

        # Estimate trade frequency (trades per trading day)
        # Use backtest trading days from stored trades
        bt_dates = set()
        ft_dt = None
        if ft_start:
            try:
                ft_dt = datetime.fromisoformat(ft_start)
                if ft_dt.tzinfo:
                    ft_dt = ft_dt.astimezone(timezone.utc).replace(tzinfo=None)
            except (ValueError, TypeError):
                ft_dt = None

        for t in stored:
            try:
                dt = datetime.fromisoformat(t.get('exit_time', t.get('entry_time', '')))
                if dt.tzinfo:
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                if ft_dt and dt >= ft_dt:
                    continue
                bt_dates.add(dt.date())
            except (ValueError, TypeError):
                continue

        trading_days = max(len(bt_dates), 1)
        trade_freq = dist['n_trades'] / trading_days

        per_strategy[sid] = {
            'avg_r': dist['avg_r'],
            'std_r': dist['std_r'],
            'var_r': dist['var_r'],
            'n_bt_trades': dist['n_trades'],
            'trade_frequency': trade_freq,
            'risk_per_trade': risk,
            'strategy_name': strat.get('name', f'Strategy {sid}'),
        }

    if not per_strategy:
        return {
            'plan_cumulative_pnl': [],
            'upper_1sd': [], 'lower_1sd': [],
            'upper_2sd': [], 'lower_2sd': [],
            'per_strategy': {},
            'total_expected_trades': 0,
        }

    # Compute combined expected step and variance per trade
    total_freq = sum(s['trade_frequency'] for s in per_strategy.values())
    if total_freq == 0:
        total_freq = 1.0

    expected_dollar_step = 0.0
    per_trade_dollar_var = 0.0
    for s in per_strategy.values():
        weight = s['trade_frequency'] / total_freq
        expected_dollar_step += s['avg_r'] * s['risk_per_trade'] * weight
        per_trade_dollar_var += (s['risk_per_trade'] ** 2) * s['var_r'] * weight

    # Build plan line and bands
    plan = []
    upper_1sd = []
    lower_1sd = []
    upper_2sd = []
    lower_2sd = []

    for n in range(1, max_trades + 1):
        cumulative_plan = n * expected_dollar_step
        cumulative_std = math.sqrt(n * per_trade_dollar_var) if per_trade_dollar_var > 0 else 0
        plan.append(cumulative_plan)
        upper_1sd.append(cumulative_plan + 1.0 * cumulative_std)
        lower_1sd.append(cumulative_plan - 1.0 * cumulative_std)
        upper_2sd.append(cumulative_plan + 2.0 * cumulative_std)
        lower_2sd.append(cumulative_plan - 2.0 * cumulative_std)

    return {
        'plan_cumulative_pnl': plan,
        'upper_1sd': upper_1sd,
        'lower_1sd': lower_1sd,
        'upper_2sd': upper_2sd,
        'lower_2sd': lower_2sd,
        'per_strategy': per_strategy,
        'total_expected_trades': max_trades,
        'expected_dollar_step': expected_dollar_step,
        'per_trade_dollar_var': per_trade_dollar_var,
    }


def classify_strategy_health(actual_trades: list,
                              benchmark: dict,
                              strategy_id: int,
                              correlation_matrix: pd.DataFrame = None) -> dict:
    """Classify a strategy's live performance health vs backtest expectations.

    Returns {status, message, deviation_sd, recommendation}.
    """
    strat_trades = [t for t in actual_trades if t.get('strategy_id') == strategy_id]
    n = len(strat_trades)

    strat_bench = benchmark.get('per_strategy', {}).get(strategy_id)
    if strat_bench is None:
        return {
            'status': 'no_benchmark',
            'message': 'No backtest data available',
            'deviation_sd': 0,
            'recommendation': '',
        }

    if n < 10:
        return {
            'status': 'insufficient_data',
            'message': f'Need 10+ alert trades ({n} so far)',
            'deviation_sd': 0,
            'recommendation': 'Keep monitoring — not enough data for assessment',
        }

    actual_cumulative_r = sum(t.get('r_multiple', 0) for t in strat_trades)
    expected_r = strat_bench['avg_r'] * n
    expected_std = strat_bench['std_r'] * math.sqrt(n) if strat_bench['std_r'] > 0 else 0

    deviation_sd = ((actual_cumulative_r - expected_r) / expected_std) if expected_std > 0 else 0

    # Check correlation with other strategies
    corr_note = ''
    if correlation_matrix is not None:
        strat_name = strat_bench.get('strategy_name', '')
        if strat_name in correlation_matrix.columns:
            for other_name in correlation_matrix.columns:
                if other_name == strat_name:
                    continue
                corr_val = correlation_matrix.loc[strat_name, other_name]
                if abs(corr_val) > 0.7:
                    corr_note = f'Highly correlated ({corr_val:.2f}) with {other_name}'
                    break

    if deviation_sd > 1.5:
        status = 'outperforming'
        message = f'{deviation_sd:+.1f} SD above expected'
        recommendation = 'Consider increasing risk per trade'
    elif deviation_sd < -1.5:
        status = 'underperforming'
        message = f'{deviation_sd:+.1f} SD below expected'
        recommendation = 'Review strategy — may be overfit or market regime has shifted'
    else:
        status = 'on_track'
        message = 'Performing within expected range'
        recommendation = 'No action needed'

    if corr_note:
        recommendation += f'. {corr_note}'

    return {
        'status': status,
        'message': message,
        'deviation_sd': round(deviation_sd, 2),
        'recommendation': recommendation,
    }


# =============================================================================
# PHASE 37E: CHANGE LOG & DAILY JOURNAL
# =============================================================================

MAX_CHANGE_LOG_ENTRIES = 500


def add_change_log_entry(portfolio: dict, change_type: str,
                          details: dict, description: str):
    """Append a change log entry to the portfolio.

    Args:
        portfolio: Portfolio dict (modified in-place).
        change_type: One of 'strategy_added', 'strategy_removed',
            'risk_adjusted', 'requirement_set_changed', 'portfolio_created'.
        details: Type-specific dict (e.g., strategy_id, old_value, new_value).
        description: Human-readable summary.
    """
    if 'change_log' not in portfolio:
        portfolio['change_log'] = []

    log = portfolio['change_log']

    entry_id = max((e.get('id', 0) for e in log), default=0) + 1

    log.append({
        'id': entry_id,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'change_type': change_type,
        'details': details,
        'description': description,
    })

    # Trim to max entries
    if len(log) > MAX_CHANGE_LOG_ENTRIES:
        portfolio['change_log'] = log[-MAX_CHANGE_LOG_ENTRIES:]


def compute_daily_journal(alert_trades: list,
                           change_log: list = None) -> list:
    """Generate daily summaries from alert trades and portfolio changes.

    Returns list of dicts sorted by date descending:
        {date, daily_pnl, n_trades, strategies_traded, changes}
    """
    from collections import defaultdict

    daily = defaultdict(lambda: {'pnl': 0, 'trades': 0, 'strategies': set(), 'changes': []})

    for t in alert_trades:
        exit_time = t.get('exit_time', '')
        if not exit_time:
            continue
        try:
            dt = datetime.fromisoformat(exit_time)
            date_str = dt.strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            continue
        daily[date_str]['pnl'] += t.get('dollar_pnl', 0)
        daily[date_str]['trades'] += 1
        daily[date_str]['strategies'].add(t.get('strategy_name', ''))

    if change_log:
        for entry in change_log:
            ts = entry.get('timestamp', '')
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts)
                date_str = dt.strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                continue
            daily[date_str]['changes'].append(entry)

    result = []
    for date_str, info in daily.items():
        result.append({
            'date': date_str,
            'daily_pnl': info['pnl'],
            'n_trades': info['trades'],
            'strategies_traded': sorted(info['strategies']),
            'changes': info['changes'],
        })

    result.sort(key=lambda x: x['date'], reverse=True)
    return result


# =============================================================================
# PHASE 37D: BUYING POWER TRACKER & ANOMALY DETECTION
# =============================================================================


def compute_alert_buying_power(alert_trades: list, open_positions: list,
                                account_balance: float) -> dict:
    """Compute intra-trade buying power timeline from alert-based trades.

    Uses the account balance (from Account tab ledger) as the authority,
    minus capital committed to open positions.

    Returns dict with timeline events, current buying power, and alerts.
    """
    events = []

    for t in alert_trades:
        bp_used = t.get('buying_power_used', 0)
        # Entry event — reduces buying power
        events.append({
            'time': t.get('entry_time', ''),
            'event_type': 'entry',
            'capital_change': bp_used,
            'strategy': t.get('strategy_name', ''),
            'symbol': t.get('symbol', ''),
        })
        # Exit event — restores buying power
        events.append({
            'time': t.get('exit_time', ''),
            'event_type': 'exit',
            'capital_change': -bp_used,
            'strategy': t.get('strategy_name', ''),
            'symbol': t.get('symbol', ''),
        })

    # Sort by time
    events.sort(key=lambda e: e.get('time', ''))

    # Build timeline
    timeline = []
    capital_deployed = 0
    realized_pnl = 0
    concurrent = 0
    max_deployed = 0
    max_concurrent = 0
    insufficient_events = []
    trade_idx = 0

    for ev in events:
        if ev['event_type'] == 'entry':
            capital_deployed += ev['capital_change']
            concurrent += 1
        else:
            capital_deployed -= abs(ev['capital_change'])
            concurrent -= 1
            # Add realized P&L from the trade
            if trade_idx < len(alert_trades):
                realized_pnl += alert_trades[trade_idx].get('dollar_pnl', 0)
                trade_idx += 1

        capital_deployed = max(capital_deployed, 0)
        concurrent = max(concurrent, 0)
        buying_power = account_balance + realized_pnl - capital_deployed
        max_deployed = max(max_deployed, capital_deployed)
        max_concurrent = max(max_concurrent, concurrent)

        timeline.append({
            'time': ev.get('time', ''),
            'buying_power': buying_power,
            'capital_deployed': capital_deployed,
            'concurrent_positions': concurrent,
            'event_type': ev['event_type'],
            'strategy': ev.get('strategy', ''),
        })

        if buying_power < 0:
            insufficient_events.append({
                'time': ev.get('time', ''),
                'shortfall': abs(buying_power),
                'strategy': ev.get('strategy', ''),
            })

    # Current state: account open positions
    open_capital = sum(p.get('buying_power_used', 0) for p in open_positions)
    current_bp = account_balance + realized_pnl - open_capital

    return {
        'timeline': timeline,
        'current_buying_power': current_bp,
        'insufficient_events': insufficient_events,
        'peak_capital_deployed': max_deployed,
        'max_concurrent_positions': max_concurrent,
        'open_capital_committed': open_capital,
    }


def detect_portfolio_anomalies(alert_trades: list, open_positions: list,
                                portfolio: dict,
                                get_strategy_fn: Callable = None) -> list:
    """Detect anomalous conditions in the portfolio's live trading.

    Returns list of anomaly dicts with type, severity, description, details.
    """
    anomalies = []

    # 1. Overexposure: multiple open positions on the same symbol
    symbol_positions = {}
    for pos in open_positions:
        sym = pos.get('symbol', '')
        if sym not in symbol_positions:
            symbol_positions[sym] = []
        symbol_positions[sym].append(pos)

    for sym, positions in symbol_positions.items():
        if len(positions) > 1:
            total_qty = sum(p.get('quantity', 0) for p in positions)
            strat_names = [p.get('strategy_name', '') for p in positions]
            anomalies.append({
                'type': 'overexposure',
                'severity': 'HIGH',
                'symbol': sym,
                'description': f"Multiple open positions on {sym} from {len(positions)} strategies",
                'details': {
                    'strategies': strat_names,
                    'total_quantity': total_qty,
                    'positions': positions,
                },
                'suggested_action': f"Review {sym} exposure across strategies",
            })

    # 2. Phantom trades: alert trades that didn't match forward test
    phantom_count = sum(1 for t in alert_trades if t.get('phantom', False))
    if phantom_count > 0:
        anomalies.append({
            'type': 'phantom_trade',
            'severity': 'MEDIUM',
            'symbol': '',
            'description': f"{phantom_count} phantom trade(s) detected — alerts fired without matching forward test trades",
            'details': {'count': phantom_count},
            'suggested_action': "Review alert matching and strategy configuration",
        })

    # 3. Long holds: open positions held much longer than expected
    for pos in open_positions:
        entry_time = pos.get('entry_time', '')
        if not entry_time:
            continue
        try:
            entry_dt = datetime.fromisoformat(entry_time)
            duration_hours = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600

            # Get strategy for expected hold time
            sid = pos.get('strategy_id')
            if get_strategy_fn and sid:
                strat = get_strategy_fn(sid)
                if strat:
                    from ralph_engine import TIMEFRAME_SECONDS
                    tf_seconds = TIMEFRAME_SECONDS.get(strat.get('timeframe', '1Min'), 60)
                    bar_count_exit = strat.get('bar_count_exit', 20)
                    expected_hours = (tf_seconds * bar_count_exit) / 3600
                    if duration_hours > expected_hours * 2:
                        anomalies.append({
                            'type': 'long_hold',
                            'severity': 'MEDIUM',
                            'symbol': pos.get('symbol', ''),
                            'description': f"{pos.get('strategy_name', '')}: {pos.get('symbol', '')} held for {duration_hours:.1f}h (expected max: {expected_hours:.1f}h)",
                            'details': {
                                'strategy_id': sid,
                                'duration_hours': duration_hours,
                                'expected_hours': expected_hours,
                            },
                            'suggested_action': "Check if exit signal was missed",
                        })
        except (ValueError, TypeError):
            continue

    return anomalies
