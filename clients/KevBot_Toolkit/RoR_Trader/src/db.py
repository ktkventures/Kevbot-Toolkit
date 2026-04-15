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
    user_id = get_current_user_id()
    client = get_client()
    result = client.table('strategies') \
        .select('*') \
        .eq('user_id', user_id) \
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
    user_id = get_current_user_id()
    client = get_client()
    result = client.table('portfolios') \
        .select('*') \
        .eq('user_id', user_id) \
        .order('id') \
        .execute()
    return [_restore_portfolio_from_db(r) for r in result.data]


def get_portfolio_by_id_db(portfolio_id: int) -> dict | None:
    """Load a single portfolio by ID."""
    client = get_client()
    result = client.table('portfolios') \
        .select('*') \
        .eq('id', portfolio_id) \
        .maybe_single() \
        .execute()
    return _restore_portfolio_from_db(result.data) if result and result.data else None


# Fields stored in portfolio dicts but NOT as DB columns.
# These get nested inside the 'account' JSONB column for DB persistence.
_PORTFOLIO_NON_DB_FIELDS = {'change_log', 'journal_entries', 'buying_power_mode', 'webhook_group_id'}


def _prepare_portfolio_for_db(d: dict) -> dict:
    """Prepare portfolio dict for DB storage.

    Moves non-DB fields into the 'account' JSONB column so they persist
    without requiring DB schema changes. Uses deep copy to avoid mutating
    the original dict.
    """
    import copy as _copy
    payload = _copy.deepcopy(d)
    account = payload.get('account', {})
    if not isinstance(account, dict):
        account = {}

    # Nest non-DB fields inside account
    for field in _PORTFOLIO_NON_DB_FIELDS:
        if field in payload:
            account[f'_p37_{field}'] = payload.pop(field)

    payload['account'] = account
    return payload


def _restore_portfolio_from_db(row: dict) -> dict:
    """Restore non-DB fields from the account JSONB column back to top level."""
    if not row:
        return row
    account = row.get('account', {})
    # PostgREST may return JSONB as a string
    if isinstance(account, str):
        try:
            account = json.loads(account)
            row['account'] = account
        except (json.JSONDecodeError, TypeError):
            account = {}
    if isinstance(account, dict):
        for field in _PORTFOLIO_NON_DB_FIELDS:
            db_key = f'_p37_{field}'
            if db_key in account:
                row[field] = account.pop(db_key)
    return row


def save_portfolio_db(portfolio: dict) -> dict:
    """Insert a new portfolio. Returns the saved portfolio."""
    payload = _prepare_portfolio_for_db(portfolio)
    payload['user_id'] = get_current_user_id()
    payload.pop('id', None)
    client = get_client()
    result = client.table('portfolios').insert(payload).execute()
    return _restore_portfolio_from_db(result.data[0]) if result.data else {}


def update_portfolio_db(portfolio_id: int, updated: dict) -> dict | None:
    """Update a portfolio by ID."""
    payload = _prepare_portfolio_for_db(updated)
    payload.pop('user_id', None)
    payload.pop('id', None)
    client = get_client()
    result = client.table('portfolios') \
        .update(payload) \
        .eq('id', portfolio_id) \
        .execute()
    return _restore_portfolio_from_db(result.data[0]) if result.data else None


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
    user_id = get_current_user_id()
    client = get_client()
    result = client.table('alerts') \
        .select('*') \
        .eq('user_id', user_id) \
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
    user_id = get_current_user_id()
    client = get_client()
    result = client.table('alert_config') \
        .select('config') \
        .eq('user_id', user_id) \
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
    user_id = get_current_user_id()
    client = get_client()
    result = client.table(table) \
        .select(data_column) \
        .eq('user_id', user_id) \
        .maybe_single() \
        .execute()
    if result and result.data:
        val = result.data.get(data_column)
        if val is None:
            return default
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

def load_time_exit_packs_db() -> list:
    try:
        return _load_user_config('time_exit_packs', 'packs', [])
    except Exception:
        # Table may not exist yet — return empty so defaults are seeded
        return []

def save_time_exit_packs_db(packs: list):
    try:
        _save_user_config('time_exit_packs', 'packs', packs)
    except Exception:
        # Table may not exist yet — silently fail, defaults still work in-memory
        pass


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
    user_id = get_current_user_id()
    client = get_client()
    result = client.table('monitor_status') \
        .select('*') \
        .eq('user_id', user_id) \
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
        uid = get_current_user_id()
        client = get_client()
        result = client.table('monitor_status') \
            .select('engine_state') \
            .eq('user_id', uid) \
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


def get_strategy_by_id_admin(strategy_id: int, user_id: str) -> dict | None:
    """Load a single strategy by ID for a specific user (admin client)."""
    client = get_admin_client()
    result = client.table('strategies') \
        .select('*') \
        .eq('id', strategy_id) \
        .eq('user_id', user_id) \
        .maybe_single() \
        .execute()
    if result and result.data:
        return _row_to_strategy(result.data)
    return None


def update_strategy_admin(strategy_id: int, user_id: str, updates: dict) -> dict | None:
    """Update specific fields on a strategy (admin client, no JWT required).

    Used by the worker to persist algo trade records into stored_trades as
    they fire live. `updates` is a PARTIAL dict — pass only the fields you
    want to change (typically `stored_trades`, `live_executions`, `kpis`).

    CRITICAL: this function MUST NOT use `_strategy_to_row()` directly because
    that helper always emits `config = {}` when no config fields are present,
    which would overwrite the existing JSONB column and wipe trigger/stop/
    target settings. We split fields into column-vs-config buckets manually
    and only include `config` in the update payload when config fields are
    actually being changed (in which case the caller must pass a FULL config
    merge — partial JSONB updates aren't supported by PostgREST without RPC).
    """
    column_updates = {}
    config_updates = {}
    for k, v in updates.items():
        if k in STRATEGY_COLUMN_FIELDS:
            column_updates[k] = v
        else:
            config_updates[k] = v
    column_updates.pop('user_id', None)
    column_updates.pop('id', None)

    payload = dict(column_updates)
    if config_updates:
        # Caller is changing something inside config — they must have already
        # merged with the existing config. We can't do partial JSONB updates
        # via the supabase-py client, so the caller's responsibility.
        payload['config'] = config_updates

    if not payload:
        return None  # nothing to update

    client = get_admin_client()
    result = client.table('strategies') \
        .update(payload) \
        .eq('id', strategy_id) \
        .eq('user_id', user_id) \
        .execute()
    if result.data:
        return _row_to_strategy(result.data[0])
    return None


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

    M8.5 change: every strategy with the structural prerequisites (i.e. an
    ``entry_trigger_confluence_id``) is monitored by default, regardless of
    portfolio membership or webhook wiring. This lets alerts + forward-test
    data accumulate for every saved strategy, and lets the live chart fire
    broadcasts for any strategy the user opens on Strategy Detail.

    Webhook dispatch still respects portfolio + webhook-group config at alert
    time — monitoring here is strictly about "does the engine watch this?"
    """
    import logging
    _log = logging.getLogger("worker")

    strategies = load_strategies_admin(user_id)
    _log.info("[%s] get_monitored: scanning %d strategies",
              user_id[:8], len(strategies))

    monitored = []
    skipped_no_confluence = 0
    for strat in strategies:
        if 'entry_trigger_confluence_id' not in strat:
            skipped_no_confluence += 1
            continue
        monitored.append(strat)

    if skipped_no_confluence:
        _log.info("[%s]   %d strategies skipped (no entry_trigger_confluence_id — "
                  "legacy/unmigrated)", user_id[:8], skipped_no_confluence)
    _log.info("[%s] get_monitored result: %d strategies", user_id[:8], len(monitored))
    return monitored


# ============================================================
# Mass Search CRUD
# ============================================================

_MASS_SEARCHES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "config", "mass_searches.json")


def _load_mass_searches_file() -> list:
    if os.path.exists(_MASS_SEARCHES_FILE):
        try:
            with open(_MASS_SEARCHES_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            pass
    return []


def _save_mass_searches_file(searches: list):
    os.makedirs(os.path.dirname(_MASS_SEARCHES_FILE), exist_ok=True)
    with open(_MASS_SEARCHES_FILE, 'w') as f:
        json.dump(searches, f, indent=2, default=str)


def load_mass_searches() -> list:
    """Load all mass searches for the current user.

    Merges row-level fields (id, status, created_at, updated_at) with the
    config_data JSONB payload so callers see a unified dict. Row-level status
    is authoritative — config_data's status may lag behind.
    """
    if USE_DB:
        try:
            client = get_client()
            result = client.table('mass_searches') \
                .select('*') \
                .order('created_at', desc=True) \
                .execute()
            merged = []
            for r in (result.data or []):
                row = dict(r)
                cfg = row.pop('config_data', {}) or {}
                merged.append({**cfg, **row})
            return merged
        except Exception as e:
            logger.warning("load_mass_searches DB error (falling back to file): %s", e)
    return _load_mass_searches_file()


def get_mass_search(search_id) -> dict | None:
    """Get a single mass search by ID."""
    if USE_DB:
        try:
            client = get_client()
            result = client.table('mass_searches') \
                .select('*') \
                .eq('id', search_id) \
                .maybe_single() \
                .execute()
            if result.data:
                # Merge row-level fields with config_data for a unified view
                row = dict(result.data)
                cfg = row.pop('config_data', {}) or {}
                merged = {**cfg, **row}
                return merged
        except Exception as e:
            logger.warning("get_mass_search DB error: %s", e)
    for s in _load_mass_searches_file():
        if s.get('id') == search_id:
            return s
    return None


def save_mass_search(search: dict) -> str:
    """Save (upsert) a mass search. Returns the search ID."""
    search_id = search.get('id') or None
    now = datetime.now(timezone.utc).isoformat()
    search['updated_at'] = now
    if not search.get('created_at'):
        search['created_at'] = now

    if USE_DB:
        try:
            client = get_client()
            user_id = getattr(_local, 'user_id', None)
            row = {
                'user_id': user_id,
                'name': search.get('name', 'Untitled'),
                'status': search.get('status', 'pending'),
                'config_data': json.loads(json.dumps(search, default=str)),
                'created_at': search['created_at'],
                'updated_at': now,
            }
            if search_id:
                row['id'] = search_id
                client.table('mass_searches').upsert(row).execute()
                return search_id
            else:
                # Generate a unique integer ID (timestamp-based)
                import time as _time
                row['id'] = int(_time.time() * 1000) % (2**31)
                result = client.table('mass_searches').insert(row).execute()
                if result.data and len(result.data) > 0:
                    return result.data[0].get('id')
                return row['id']
        except Exception as e:
            logger.warning("save_mass_search DB error (falling back to file): %s", e)

    # Local JSON fallback
    searches = _load_mass_searches_file()
    existing = next((i for i, s in enumerate(searches) if s.get('id') == search_id), None)
    if existing is not None:
        searches[existing] = search
    else:
        searches.insert(0, search)
    _save_mass_searches_file(searches)
    return search_id


def update_mass_search(search_id, updates: dict):
    """Partial update of a mass search row in the DB."""
    if USE_DB:
        try:
            client = get_client()
            row = {'updated_at': datetime.now(timezone.utc).isoformat()}
            if 'status' in updates:
                row['status'] = updates['status']
            # Store results, progress, summary in config_data
            if 'results' in updates or 'progress' in updates or 'summary' in updates:
                # Read existing config_data, merge updates into it
                result = client.table('mass_searches').select('config_data').eq('id', search_id).maybe_single().execute()
                cfg = (result.data or {}).get('config_data', {}) if result and result.data else {}
                for key in ('results', 'progress', 'summary'):
                    if key in updates:
                        cfg[key] = updates[key]
                # Sanitize for JSON: convert numpy/pandas types
                import numpy as np
                def _sanitize(obj):
                    if isinstance(obj, (np.integer,)):
                        return int(obj)
                    if isinstance(obj, (np.floating,)):
                        return float(obj) if not np.isnan(obj) and not np.isinf(obj) else 0
                    if isinstance(obj, (np.bool_,)):
                        return bool(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, set):
                        return list(obj)
                    return str(obj)
                row['config_data'] = json.loads(json.dumps(cfg, default=_sanitize))
            client.table('mass_searches').update(row).eq('id', search_id).execute()
            return
        except Exception as e:
            logger.warning("update_mass_search DB error: %s", e)

    # File fallback
    existing = get_mass_search(search_id)
    if existing:
        existing.update(updates)
        save_mass_search(existing)


def delete_mass_search(search_id: str):
    """Delete a mass search by ID."""
    if USE_DB:
        try:
            client = get_client()
            client.table('mass_searches').delete().eq('id', search_id).execute()
            return
        except Exception as e:
            logger.warning("delete_mass_search DB error: %s", e)

    searches = _load_mass_searches_file()
    searches = [s for s in searches if s.get('id') != search_id]
    _save_mass_searches_file(searches)
