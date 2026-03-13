"""
Centralized database access layer for RoR Trader.

Wraps all Supabase/PostgreSQL operations with a USE_DB toggle for
incremental migration from JSON file storage. When USE_DB is false,
all code paths fall through to the existing JSON file operations.

Phase 22A: Foundation (client setup, toggle, transformation helpers)
Phase 22B: CRUD rewiring (strategies, portfolios, alerts, configs)
"""
import os
import json
import threading
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

# Toggle: "true" = use database, "false" = use JSON files
USE_DB = os.getenv("USE_DB", "false").lower() == "true"

# Supabase credentials from environment
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# ============================================================
# User Context (thread-local)
# Streamlit runs each session in its own thread, so we store
# the authenticated user's ID and JWT per-thread.
# ============================================================

_local = threading.local()


def set_current_user(user_id: str, access_token: str):
    """Set the current user context for this thread."""
    _local.user_id = user_id
    _local.access_token = access_token


def get_current_user_id() -> str:
    """Get the current user's UUID. Returns None if not authenticated."""
    return getattr(_local, 'user_id', None)


def get_current_token() -> str:
    """Get the current user's JWT access token."""
    return getattr(_local, 'access_token', None)


# ============================================================
# Supabase Client Factory
# ============================================================

_anon_client = None
_admin_client = None
_client_lock = threading.Lock()


def get_client():
    """Get a Supabase client authenticated as the current user.

    Uses the anon key. The user's JWT is set on the client so that
    Row Level Security policies are enforced server-side.
    """
    global _anon_client
    if not USE_DB:
        raise RuntimeError("get_client() called but USE_DB is false")

    from supabase import create_client

    # Create a fresh client per call to ensure correct JWT context.
    # supabase-py clients are lightweight — this is the recommended pattern
    # for multi-user server apps where different requests have different JWTs.
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    token = get_current_token()
    if token:
        client.postgrest.auth(token)
    return client


def get_admin_client():
    """Get a Supabase client with service role key (bypasses RLS).

    Used by the worker service only — never in the web app.
    The service role key has full access to all rows regardless of user_id.
    """
    global _admin_client
    if not USE_DB:
        raise RuntimeError("get_admin_client() called but USE_DB is false")

    with _client_lock:
        if _admin_client is None:
            from supabase import create_client
            _admin_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _admin_client


# ============================================================
# Strategy Transformation Helpers
# ============================================================

# Fields stored as dedicated database columns (not in config JSONB)
STRATEGY_COLUMN_FIELDS = {
    'id', 'user_id', 'name', 'symbol', 'direction', 'timeframe',
    'strategy_origin', 'forward_testing', 'forward_test_start',
    'alert_tracking_enabled', 'alert_tracking_reset_at',
    'kpis', 'stored_trades', 'equity_curve_data',
    'live_executions', 'discrepancies', 'discrepancies_dismissed_at',
    'data_refreshed_at', 'created_at', 'updated_at',
}


def _strategy_to_row(strat: dict) -> dict:
    """Split a flat strategy dict into column + JSONB format for the database.

    The app works with flat dicts where all fields are top-level keys.
    The database stores queryable fields as proper columns and everything
    else (triggers, confluence, risk params, lookback settings) in a
    single 'config' JSONB column.
    """
    row = {}
    config = {}
    for k, v in strat.items():
        if k in STRATEGY_COLUMN_FIELDS:
            row[k] = v
        else:
            config[k] = v
    row['config'] = config if config else {}
    return row


def _row_to_strategy(row: dict) -> dict:
    """Merge a database row back into a flat strategy dict.

    Unpacks the 'config' JSONB column and merges it with the column fields,
    producing the same flat dict shape that the rest of the app expects.
    """
    strat = {}
    for k, v in row.items():
        if k == 'config':
            if isinstance(v, str):
                v = json.loads(v)
            if v:
                strat.update(v)
        else:
            strat[k] = v
    return strat


# ============================================================
# Alert Transformation Helpers
# ============================================================

# Fields stored as dedicated alert columns (not in data JSONB)
ALERT_COLUMN_FIELDS = {
    'id', 'user_id', 'type', 'strategy_id', 'strategy_name',
    'symbol', 'direction', 'timeframe', 'source',
    'acknowledged', 'webhook_sent', 'timestamp',
}


def _alert_to_row(alert: dict) -> dict:
    """Split a flat alert dict into column + JSONB format for the database."""
    row = {}
    data = {}
    for k, v in alert.items():
        if k in ALERT_COLUMN_FIELDS:
            row[k] = v
        else:
            data[k] = v
    row['data'] = data if data else {}
    return row


def _row_to_alert(row: dict) -> dict:
    """Merge a database alert row back into a flat alert dict."""
    alert = {}
    for k, v in row.items():
        if k == 'data':
            if isinstance(v, str):
                v = json.loads(v)
            if v:
                alert.update(v)
        else:
            alert[k] = v
    return alert


# ============================================================
# Strategy CRUD (database path)
# ============================================================

def load_strategies_db() -> list:
    """Load all strategies for the current user from the database."""
    client = get_client()
    result = client.table('strategies') \
        .select('*') \
        .order('id') \
        .execute()
    return [_row_to_strategy(r) for r in result.data]


def get_strategy_by_id_db(strategy_id: int) -> dict | None:
    """Load a single strategy by ID for the current user."""
    client = get_client()
    result = client.table('strategies') \
        .select('*') \
        .eq('id', strategy_id) \
        .maybe_single() \
        .execute()
    if result and result.data:
        return _row_to_strategy(result.data)
    return None


def save_strategy_db(strategy: dict) -> dict:
    """Insert a new strategy for the current user. Returns the saved strategy."""
    user_id = get_current_user_id()
    strategy['user_id'] = user_id
    row = _strategy_to_row(strategy)
    # Remove 'id' so the database assigns the next SERIAL value
    row.pop('id', None)
    client = get_client()
    result = client.table('strategies').insert(row).execute()
    return _row_to_strategy(result.data[0])


def update_strategy_db(strategy_id: int, updated: dict) -> dict | None:
    """Update an existing strategy by ID. Returns the updated strategy."""
    row = _strategy_to_row(updated)
    row.pop('user_id', None)  # Don't allow changing user_id
    row.pop('id', None)       # Don't include PK in the SET clause
    client = get_client()
    result = client.table('strategies') \
        .update(row) \
        .eq('id', strategy_id) \
        .execute()
    if result.data:
        return _row_to_strategy(result.data[0])
    return None


def delete_strategy_db(strategy_id: int) -> bool:
    """Delete a strategy by ID. Returns True if deleted."""
    client = get_client()
    result = client.table('strategies') \
        .delete() \
        .eq('id', strategy_id) \
        .execute()
    return bool(result.data)


# ============================================================
# Portfolio CRUD (database path)
# ============================================================

def load_portfolios_db() -> list:
    """Load all portfolios for the current user."""
    client = get_client()
    result = client.table('portfolios') \
        .select('*') \
        .order('id') \
        .execute()
    return result.data


def get_portfolio_by_id_db(portfolio_id: int) -> dict | None:
    """Load a single portfolio by ID."""
    client = get_client()
    result = client.table('portfolios') \
        .select('*') \
        .eq('id', portfolio_id) \
        .maybe_single() \
        .execute()
    return result.data if result else None


def save_portfolio_db(portfolio: dict) -> dict:
    """Insert a new portfolio. Returns the saved portfolio."""
    portfolio['user_id'] = get_current_user_id()
    portfolio.pop('id', None)
    client = get_client()
    result = client.table('portfolios').insert(portfolio).execute()
    return result.data[0]


def update_portfolio_db(portfolio_id: int, updated: dict) -> dict | None:
    """Update a portfolio by ID."""
    updated.pop('user_id', None)
    updated.pop('id', None)
    client = get_client()
    result = client.table('portfolios') \
        .update(updated) \
        .eq('id', portfolio_id) \
        .execute()
    return result.data[0] if result.data else None


def delete_portfolio_db(portfolio_id: int) -> bool:
    """Delete a portfolio by ID."""
    client = get_client()
    result = client.table('portfolios') \
        .delete() \
        .eq('id', portfolio_id) \
        .execute()
    return bool(result.data)


# ============================================================
# Requirement Sets CRUD (database path)
# ============================================================

def load_requirements_db() -> list:
    """Load all requirement sets (own + built-in) for the current user."""
    client = get_client()
    result = client.table('requirement_sets') \
        .select('*') \
        .order('id') \
        .execute()
    return result.data


def save_requirement_set_db(req_set: dict) -> dict:
    """Insert a new requirement set."""
    req_set['user_id'] = get_current_user_id()
    req_set.pop('id', None)
    client = get_client()
    result = client.table('requirement_sets').insert(req_set).execute()
    return result.data[0]


def update_requirement_set_db(req_id: int, updated: dict) -> dict | None:
    """Update a requirement set by ID."""
    updated.pop('user_id', None)
    updated.pop('id', None)
    client = get_client()
    result = client.table('requirement_sets') \
        .update(updated) \
        .eq('id', req_id) \
        .execute()
    return result.data[0] if result.data else None


def delete_requirement_set_db(req_id: int) -> bool:
    """Delete a requirement set by ID."""
    client = get_client()
    result = client.table('requirement_sets') \
        .delete() \
        .eq('id', req_id) \
        .execute()
    return bool(result.data)


# ============================================================
# Alert CRUD (database path)
# ============================================================

def load_alerts_db(limit: int = 100) -> list:
    """Load recent alerts for the current user."""
    client = get_client()
    result = client.table('alerts') \
        .select('*') \
        .order('timestamp', desc=True) \
        .limit(limit) \
        .execute()
    return [_row_to_alert(r) for r in result.data]


def save_alert_db(alert: dict) -> dict:
    """Insert a new alert."""
    alert['user_id'] = get_current_user_id()
    row = _alert_to_row(alert)
    row.pop('id', None)
    client = get_client()
    result = client.table('alerts').insert(row).execute()
    return _row_to_alert(result.data[0])


def update_alert_db(alert_id: int, updates: dict) -> dict | None:
    """Update an alert by ID (e.g., acknowledge)."""
    client = get_client()
    result = client.table('alerts') \
        .update(updates) \
        .eq('id', alert_id) \
        .execute()
    return _row_to_alert(result.data[0]) if result.data else None


def get_alerts_for_strategy_db(strategy_id: int, limit: int = 100) -> list:
    """Load alerts for a specific strategy."""
    client = get_client()
    result = client.table('alerts') \
        .select('*') \
        .eq('strategy_id', strategy_id) \
        .order('timestamp', desc=True) \
        .limit(limit) \
        .execute()
    return [_row_to_alert(r) for r in result.data]


def delete_alerts_for_strategy_db(strategy_id: int) -> bool:
    """Delete all alerts for a strategy."""
    client = get_client()
    result = client.table('alerts') \
        .delete() \
        .eq('strategy_id', strategy_id) \
        .execute()
    return True


def clear_alerts_db() -> bool:
    """Delete all alerts for the current user."""
    client = get_client()
    # RLS ensures only current user's alerts are affected
    client.table('alerts').delete().neq('id', 0).execute()
    return True


# ============================================================
# Alert Config CRUD (database path)
# ============================================================

def load_alert_config_db() -> dict:
    """Load alert config for the current user."""
    client = get_client()
    result = client.table('alert_config') \
        .select('config') \
        .maybe_single() \
        .execute()
    if result and result.data:
        config = result.data['config']
        return json.loads(config) if isinstance(config, str) else config
    return {"global": {}, "strategies": {}, "portfolios": {}}


def save_alert_config_db(config: dict):
    """Upsert alert config for the current user."""
    user_id = get_current_user_id()
    client = get_client()
    client.table('alert_config').upsert({
        'user_id': user_id,
        'config': config,
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }, on_conflict='user_id').execute()


# ============================================================
# User Config CRUD (settings, confluence groups, packs)
# These are single-document-per-user tables.
# ============================================================

def _load_user_config(table: str, data_column: str, default):
    """Generic: load a single-row config for the current user."""
    client = get_client()
    result = client.table(table) \
        .select(data_column) \
        .maybe_single() \
        .execute()
    if result and result.data:
        val = result.data[data_column]
        return json.loads(val) if isinstance(val, str) else val
    return default


def _save_user_config(table: str, data_column: str, data):
    """Generic: upsert a single-row config for the current user."""
    user_id = get_current_user_id()
    client = get_client()
    client.table(table).upsert({
        'user_id': user_id,
        data_column: data,
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }, on_conflict='user_id').execute()


def load_settings_db() -> dict:
    return _load_user_config('user_settings', 'settings', {})

def save_settings_db(settings: dict):
    _save_user_config('user_settings', 'settings', settings)

def load_confluence_groups_db() -> list:
    return _load_user_config('confluence_groups', 'groups', [])

def save_confluence_groups_db(groups: list):
    _save_user_config('confluence_groups', 'groups', groups)

def load_general_packs_db() -> list:
    return _load_user_config('general_packs', 'packs', [])

def save_general_packs_db(packs: list):
    _save_user_config('general_packs', 'packs', packs)

def load_risk_management_packs_db() -> list:
    return _load_user_config('risk_management_packs', 'packs', [])

def save_risk_management_packs_db(packs: list):
    _save_user_config('risk_management_packs', 'packs', packs)


# ============================================================
# Webhook Templates CRUD (database path)
# ============================================================

def load_webhook_templates_db() -> list:
    """Load all webhook templates (own + defaults)."""
    client = get_client()
    result = client.table('webhook_templates') \
        .select('*') \
        .order('created_at') \
        .execute()
    return result.data


def save_webhook_template_db(template: dict) -> dict:
    """Insert a new webhook template."""
    template['user_id'] = get_current_user_id()
    client = get_client()
    result = client.table('webhook_templates').insert(template).execute()
    return result.data[0]


def update_webhook_template_db(template_id: str, updates: dict) -> dict | None:
    """Update a webhook template by ID."""
    updates.pop('user_id', None)
    updates.pop('id', None)
    client = get_client()
    result = client.table('webhook_templates') \
        .update(updates) \
        .eq('id', template_id) \
        .execute()
    return result.data[0] if result.data else None


def delete_webhook_template_db(template_id: str) -> bool:
    """Delete a webhook template by ID."""
    client = get_client()
    result = client.table('webhook_templates') \
        .delete() \
        .eq('id', template_id) \
        .execute()
    return bool(result.data)


# ============================================================
# Monitor Status CRUD (database path)
# ============================================================

def load_monitor_status_db() -> dict:
    """Load monitor status for the current user."""
    client = get_client()
    result = client.table('monitor_status') \
        .select('*') \
        .maybe_single() \
        .execute()
    if result and result.data:
        row = result.data
        status = row.get('status', {})
        if isinstance(status, str):
            status = json.loads(status)
        status['desired_state'] = row.get('desired_state', 'stopped')
        return status
    return {'desired_state': 'stopped'}


def save_monitor_status_db(status: dict):
    """Upsert monitor status for the current user."""
    user_id = get_current_user_id()
    desired_state = status.pop('desired_state', None)
    client = get_client()
    row = {
        'user_id': user_id,
        'status': status,
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }
    if desired_state:
        row['desired_state'] = desired_state
    client.table('monitor_status').upsert(
        row, on_conflict='user_id'
    ).execute()


def set_desired_state_db(desired_state: str):
    """Set the desired monitor state (called from web UI)."""
    user_id = get_current_user_id()
    client = get_client()
    client.table('monitor_status').upsert({
        'user_id': user_id,
        'desired_state': desired_state,
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }, on_conflict='user_id').execute()


# ============================================================
# Engine State CRUD (database path — used by Ralph worker)
# ============================================================

def load_engine_state_db(user_id: str = None) -> dict:
    """Load engine state (positions). Worker uses admin client + explicit user_id."""
    if user_id:
        client = get_admin_client()
        result = client.table('monitor_status') \
            .select('engine_state') \
            .eq('user_id', user_id) \
            .maybe_single() \
            .execute()
    else:
        client = get_client()
        result = client.table('monitor_status') \
            .select('engine_state') \
            .maybe_single() \
            .execute()
    if result and result.data:
        state = result.data.get('engine_state', {})
        return json.loads(state) if isinstance(state, str) else state
    return {'positions': {}}


def save_engine_state_db(state: dict, user_id: str = None):
    """Save engine state. Worker uses admin client + explicit user_id."""
    uid = user_id or get_current_user_id()
    client = get_admin_client() if user_id else get_client()
    client.table('monitor_status').upsert({
        'user_id': uid,
        'engine_state': state,
        'last_heartbeat': datetime.now(timezone.utc).isoformat(),
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }, on_conflict='user_id').execute()


# ============================================================
# Engine Audit Log (database path — used by Ralph worker)
# ============================================================

def append_audit_log_db(entry: dict, user_id: str = None):
    """Append a fidelity audit entry. Worker uses admin client."""
    uid = user_id or get_current_user_id()
    client = get_admin_client() if user_id else get_client()
    client.table('engine_audit_log').insert({
        'user_id': uid,
        'ts': entry.get('ts'),
        'symbol': entry.get('symbol'),
        'timeframe': entry.get('tf'),
        'bar_close': entry.get('bar_close'),
        'indicators': entry.get('indicators', {}),
        'triggers': entry.get('triggers', {}),
        'interpreters': entry.get('interpreters', {}),
        'positions': entry.get('positions', {}),
    }).execute()


# ============================================================
# Admin-only functions (worker service — bypasses RLS)
# ============================================================

def load_all_desired_states() -> list:
    """Load desired_state for all users. Worker polls this to start/stop engines."""
    client = get_admin_client()
    result = client.table('monitor_status') \
        .select('user_id, desired_state, updated_at') \
        .execute()
    return result.data


def load_alert_config_admin(user_id: str) -> dict:
    """Load alert config for a specific user (admin client)."""
    client = get_admin_client()
    result = client.table('alert_config') \
        .select('config, updated_at') \
        .eq('user_id', user_id) \
        .maybe_single() \
        .execute()
    if result and result.data:
        config = result.data['config']
        config = json.loads(config) if isinstance(config, str) else config
        config['_updated_at'] = result.data.get('updated_at', '')
        return config
    return {"global": {}, "strategies": {}, "portfolios": {}}


def load_strategies_admin(user_id: str) -> list:
    """Load all strategies for a specific user (admin client)."""
    client = get_admin_client()
    result = client.table('strategies') \
        .select('*') \
        .eq('user_id', user_id) \
        .order('id') \
        .execute()
    return [_row_to_strategy(r) for r in result.data]


def load_portfolios_admin(user_id: str) -> list:
    """Load all portfolios for a specific user (admin client)."""
    client = get_admin_client()
    result = client.table('portfolios') \
        .select('*') \
        .eq('user_id', user_id) \
        .order('id') \
        .execute()
    return result.data


def save_alert_admin(alert: dict, user_id: str) -> dict:
    """Insert an alert for a specific user (admin client)."""
    alert['user_id'] = user_id
    row = _alert_to_row(alert)
    row.pop('id', None)
    client = get_admin_client()
    result = client.table('alerts').insert(row).execute()
    return _row_to_alert(result.data[0])


def save_monitor_status_admin(user_id: str, status: dict):
    """Update monitor status for a specific user (admin client)."""
    now = datetime.now(timezone.utc).isoformat()
    desired_state = status.pop('desired_state', None)
    row = {
        'user_id': user_id,
        'status': status,
        'last_heartbeat': now,
        'updated_at': now,
    }
    if desired_state:
        row['desired_state'] = desired_state
    client = get_admin_client()
    client.table('monitor_status').upsert(
        row, on_conflict='user_id'
    ).execute()


def load_user_settings_admin(user_id: str) -> dict:
    """Load user settings for a specific user (admin client, for Alpaca keys)."""
    client = get_admin_client()
    result = client.table('user_settings') \
        .select('settings') \
        .eq('user_id', user_id) \
        .maybe_single() \
        .execute()
    if result and result.data:
        val = result.data['settings']
        return json.loads(val) if isinstance(val, str) else val
    return {}


def load_general_packs_admin(user_id: str) -> list:
    """Load general packs for a specific user (admin client)."""
    client = get_admin_client()
    result = client.table('general_packs') \
        .select('packs') \
        .eq('user_id', user_id) \
        .maybe_single() \
        .execute()
    if result and result.data:
        val = result.data['packs']
        return json.loads(val) if isinstance(val, str) else val
    return []


def load_confluence_groups_admin(user_id: str) -> list:
    """Load confluence groups for a specific user (admin client)."""
    client = get_admin_client()
    result = client.table('confluence_groups') \
        .select('groups') \
        .eq('user_id', user_id) \
        .maybe_single() \
        .execute()
    if result and result.data:
        val = result.data['groups']
        return json.loads(val) if isinstance(val, str) else val
    return []


def get_monitored_strategies_db(user_id: str) -> list:
    """Resolve which strategies to monitor for a user (admin client).

    A strategy is monitored if it belongs to any portfolio that has at
    least one enabled webhook in alert_config.
    """
    import logging
    _log = logging.getLogger("worker")

    config = load_alert_config_admin(user_id)
    strategies = load_strategies_admin(user_id)
    portfolios = load_portfolios_admin(user_id)

    _log.info("[%s] get_monitored: %d strategies, %d portfolios, "
              "config portfolio keys: %s",
              user_id[:8], len(strategies), len(portfolios),
              list(config.get('portfolios', {}).keys()))

    # Build set of strategy IDs in portfolios with active webhooks
    webhook_strategy_ids = set()
    for port in portfolios:
        pid = str(port['id'])
        pcfg = config.get('portfolios', {}).get(pid, {})
        webhooks = pcfg.get('webhooks', [])
        has_active_webhook = any(wh.get('enabled', True) for wh in webhooks)
        strat_ids_in_port = [a.get('strategy_id') for a in port.get('strategies', [])]
        _log.info("[%s]   Portfolio '%s' (id=%s): webhooks=%d, active=%s, strategies=%s",
                  user_id[:8], port.get('name', '?'), pid,
                  len(webhooks), has_active_webhook, strat_ids_in_port)
        if has_active_webhook:
            for alloc in port.get('strategies', []):
                webhook_strategy_ids.add(alloc.get('strategy_id'))

    monitored = []
    for strat in strategies:
        in_webhook_port = strat['id'] in webhook_strategy_ids
        has_confluence = 'entry_trigger_confluence_id' in strat
        if not in_webhook_port:
            _log.info("[%s]   SKIP '%s' (id=%s): not in webhook portfolio",
                      user_id[:8], strat.get('name', '?'), strat['id'])
            continue
        if not has_confluence:
            _log.info("[%s]   SKIP '%s' (id=%s): no entry_trigger_confluence_id",
                      user_id[:8], strat.get('name', '?'), strat['id'])
            continue
        monitored.append(strat)

    _log.info("[%s] get_monitored result: %d strategies", user_id[:8], len(monitored))
    return monitored
