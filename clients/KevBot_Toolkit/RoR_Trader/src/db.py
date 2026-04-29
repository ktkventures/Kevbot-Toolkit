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
import contextvars
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
# User Context (contextvars)
#
# History: originally `threading.local` — fine for Streamlit where each
# session runs on one thread. Broke under FastAPI + uvicorn, where a
# sync dependency (`get_current_user`) and a sync endpoint run in
# anyio's threadpool and can land on DIFFERENT worker threads under
# load. Thread-local set on T1 was invisible on T2 → `user_id = None`,
# which postgrest stringified to the literal "None" and Supabase
# rejected as an invalid UUID (2026-04-22).
#
# ContextVar is async-safe: anyio's to_thread.run_sync() copies the
# current Context into the worker thread (via contextvars.copy_context),
# so the user_id set in the dependency propagates to the endpoint
# regardless of which threadpool worker runs it. Daemon threads we spawn
# manually (mass_builder worker) don't inherit the parent Context, but
# they always call set_admin_user_context / set_current_user themselves
# before any DB work — same as before.
# ============================================================

_user_id_var: contextvars.ContextVar = contextvars.ContextVar("user_id", default=None)
_access_token_var: contextvars.ContextVar = contextvars.ContextVar("access_token", default=None)
_admin_mode_var: contextvars.ContextVar = contextvars.ContextVar("admin_mode", default=False)


def set_current_user(user_id: str, access_token: str):
    """Set the current user context for this request/task."""
    _user_id_var.set(user_id)
    _access_token_var.set(access_token)
    _admin_mode_var.set(False)


def set_admin_user_context(user_id: str):
    """Set user context in ADMIN mode — for worker / service paths that
    need to invoke user-context-aware helpers (load_confluence_groups,
    load_general_packs, etc.) without a real user JWT. Under admin mode,
    get_client() returns the admin client so queries bypass RLS;
    get_current_user_id() still returns the given user_id so WHERE-clauses
    scope correctly.
    """
    _user_id_var.set(user_id)
    _access_token_var.set(None)
    _admin_mode_var.set(True)


def clear_current_user():
    """Clear user context. Safe to call even if nothing was set."""
    _user_id_var.set(None)
    _access_token_var.set(None)
    _admin_mode_var.set(False)


def get_current_user_id() -> str:
    """Get the current user's UUID. Returns None if not authenticated."""
    return _user_id_var.get()


def get_current_token() -> str:
    """Get the current user's JWT access token."""
    return _access_token_var.get()


def _is_admin_mode() -> bool:
    return _admin_mode_var.get()


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

    When the thread-local context is in admin mode
    (set_admin_user_context), returns the admin client instead so
    worker/service paths can run user-scoped queries without a JWT.
    """
    global _anon_client
    if not USE_DB:
        raise RuntimeError("get_client() called but USE_DB is false")

    if _is_admin_mode():
        return get_admin_client()

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
    # Strategy Health Badge (2026-04-27 migration: strategy_health_badge.sql)
    'data_source', 'kpis_stale_since', 'kpis_computed_at',
    # Auto-parity score (2026-04-28 migration: strategy_parity_status.sql)
    'parity_status',
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


def _row_to_strategy(row: dict, hydrate_trades: bool = True) -> dict:
    """Merge a database row back into a flat strategy dict.

    Unpacks the 'config' JSONB column and merges it with the column fields,
    producing the same flat dict shape that the rest of the app expects.

    Phase 40: when ``USE_TRADES_TABLE`` is ON and ``hydrate_trades=True``,
    populate ``stored_trades`` from the trades table so callers see the
    same DTO as under the JSONB-era. Pass ``hydrate_trades=False`` on the
    lite list path where we deliberately don't want the trades payload.
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
    if hydrate_trades:
        try:
            import trades_store as _trades_store
            if _trades_store._flag_on() and strat.get('id'):
                strat['stored_trades'] = _trades_store.load_trades_for_strategy(
                    strat['id'], strat.get('user_id'))
        except Exception:
            # Never let hydration failures break strategy loads — callers
            # that need trades can call hydrate_strategy_trades explicitly.
            pass
    return strat


# ============================================================
# Alert Transformation Helpers
# ============================================================

# Fields stored as dedicated alert columns (not in data JSONB).
# Trade_Timestamps_Spec (2026-04-17): 8 columns added 2026-04-20 in the
# primary migration; 3 more (hold_duration_s, behavior, webhook_deliveries)
# added in the follow-up ALTER, promoting everything out of JSONB drift.
ALERT_COLUMN_FIELDS = {
    'id', 'user_id', 'type', 'strategy_id', 'strategy_name',
    'symbol', 'direction', 'timeframe', 'source',
    'acknowledged', 'webhook_sent', 'timestamp',
    # Trade_Timestamps_Spec top-level columns (4-timestamp model + metadata)
    'event_type', 'side', 'trigger_ts', 'fill_ts',
    'exec_type', 'trigger_id', 'price', 'bar_time',
    'hold_duration_s', 'behavior',
    # actual_price: near-live market price at save moment (Ralph's last
    # per-second close). Gap vs `price` (theoretical fill) = price slippage.
    'actual_price',
    # Webhook delivery tracking — resolves pre-existing PGRST204 drift
    'webhook_deliveries',
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


# Columns returned by the lean list path. Includes `equity_curve_data` so
# frontend sparklines keep working (~70 KB typical), but EXCLUDES
# `stored_trades` (can hit 2.5 MB per strategy) and `live_executions`.
# Callers that genuinely need those — recompute paths, forward-test splits,
# strategy-detail page — use load_strategies_db() or get_strategy_by_id_db.
_STRATEGY_LIST_COLUMNS = (
    'id,user_id,name,symbol,direction,timeframe,'
    'config,kpis,equity_curve_data,'
    'alert_tracking_enabled,alert_tracking_reset_at,'
    'created_at,updated_at,data_refreshed_at,'
    'discrepancies,discrepancies_dismissed_at,'
    'forward_test_start,forward_testing,strategy_origin,'
    # Strategy Health Badge columns (2026-04-27 strategy_health_badge.sql)
    # Without these the list endpoint can't fully compute the badge —
    # data_source-based issues (rapid_test_data, no_hifi_run, kpis_time_stale)
    # silently skip.
    'data_source,kpis_stale_since,kpis_computed_at,'
    # Auto-parity score (2026-04-28 strategy_parity_status.sql) — used by
    # the strategy card to show a "live-executable" verdict at a glance.
    'parity_status'
)


def load_strategies_db_lite() -> list:
    """Load all strategies for the current user, skipping heavy JSONB.

    Excludes `stored_trades` and `live_executions`. Use for list endpoints,
    dashboard summaries, and any view that doesn't actually iterate trades.

    Reduces strategies-table read traffic by ~35× on fat strategies (one
    with 3000 trades observed at 2.5 MB stored_trades vs ~75 KB lean).

    Phase 40: hydrate_trades=False to skip the per-strategy trades-table
    fetch when flag ON — list views don't need trades and hydrating N
    strategies' worth would defeat the whole lite-loader purpose.
    """
    user_id = get_current_user_id()
    client = get_client()
    result = client.table('strategies') \
        .select(_STRATEGY_LIST_COLUMNS) \
        .eq('user_id', user_id) \
        .order('id') \
        .execute()
    return [_row_to_strategy(r, hydrate_trades=False) for r in result.data]


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

    # Alerts are always on for monitored strategies — there's no UI toggle
    # any more (removed 2026-04-24 after confirming the flag was never
    # actually checked in the worker's alert-save path). Force the column
    # default so downstream display paths stay consistent.
    strategy.setdefault('alert_tracking_enabled', True)
    if strategy.get('alert_tracking_enabled') and not strategy.get('alert_tracking_reset_at'):
        strategy['alert_tracking_reset_at'] = datetime.now(timezone.utc).isoformat()

    # Phase 40: when the trades-table is authoritative, pull stored_trades
    # off the payload so the strategies row stays lean, and re-insert them
    # into the trades table after the strategy row exists (the new row's
    # assigned id is needed as the FK target).
    pending_trades = None
    try:
        import trades_store as _trades_store
        if _trades_store._flag_on():
            pending_trades = strategy.pop('stored_trades', None) or []
    except Exception:
        _trades_store = None

    row = _strategy_to_row(strategy)
    # Remove 'id' so the database assigns the next SERIAL value
    row.pop('id', None)
    client = get_client()
    result = client.table('strategies').insert(row).execute()
    saved = _row_to_strategy(result.data[0])

    if pending_trades and _trades_store is not None and _trades_store._flag_on():
        try:
            _trades_store.replace_trades_for_strategy(
                saved['id'], saved.get('user_id') or user_id, pending_trades)
            # Echo the trades back onto the returned dict so callers see
            # the same shape they would have gotten from the JSONB path.
            saved['stored_trades'] = pending_trades
        except Exception as e:
            import logging as _l
            _l.getLogger(__name__).warning(
                "save_strategy_db: trades-table insert failed for "
                "strategy %s: %s", saved.get('id'), e)

    return saved


def update_strategy_db(strategy_id: int, updated: dict) -> dict | None:
    """Update an existing strategy by ID. Returns the updated strategy."""
    # Phase 40: redirect stored_trades writes to the trades table when flag ON.
    # Only touch the trades table when the caller actually included a
    # stored_trades key — partial updates that don't mention it must NOT
    # wipe trades (mirrors the feedback_jsonb_partial_updates rule).
    pending_trades = None
    try:
        import trades_store as _trades_store
        if _trades_store._flag_on() and 'stored_trades' in updated:
            pending_trades = updated.pop('stored_trades', None) or []
    except Exception:
        _trades_store = None

    row = _strategy_to_row(updated)
    row.pop('user_id', None)  # Don't allow changing user_id
    row.pop('id', None)       # Don't include PK in the SET clause

    # CRITICAL: partial-update safety. _strategy_to_row always emits a
    # `config` key (empty `{}` when no config-bucket fields are present).
    # On an UPDATE that sends `config: {}`, PostgREST overwrites the
    # JSONB column with `{}` and wipes trigger/confluence/stop settings.
    # This is the `feedback_jsonb_partial_updates` rule — strategies 70+71
    # got wiped on 2026-04-14, and strategies 129+130 got wiped on
    # 2026-04-24 via my own set_forward_test_start endpoint. Only include
    # `config` in the payload when the caller actually provided
    # config-bucket fields to change.
    caller_had_config_fields = any(
        k not in STRATEGY_COLUMN_FIELDS for k in updated.keys()
    )
    if not caller_had_config_fields:
        row.pop('config', None)

    if not row:
        return None  # nothing to update

    client = get_client()
    result = client.table('strategies') \
        .update(row) \
        .eq('id', strategy_id) \
        .execute()

    saved = _row_to_strategy(result.data[0]) if result.data else None

    if pending_trades is not None and _trades_store is not None and _trades_store._flag_on():
        try:
            uid = (saved.get('user_id') if saved else None) or get_current_user_id()
            _trades_store.replace_trades_for_strategy(
                strategy_id, uid, pending_trades)
            if saved is not None:
                saved['stored_trades'] = pending_trades
        except Exception as e:
            import logging as _l
            _l.getLogger(__name__).warning(
                "update_strategy_db: trades-table replace failed for "
                "strategy %s: %s", strategy_id, e)

    return saved


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
_PORTFOLIO_NON_DB_FIELDS = {'change_log', 'journal_entries', 'buying_power_mode'}


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


def get_alert_by_id_db(alert_id: int) -> dict | None:
    """Load a single alert row by id, scoped to the current user.

    Used by the Portfolio Trade History Details drawer (roadmap 9ad) to
    surface the full alert including its `webhook_deliveries` JSONB array.
    RLS enforces user scoping when called with a user-scoped client, but
    we still filter on user_id explicitly so this is safe under either
    get_client() or get_admin_client().
    """
    user_id = get_current_user_id()
    client = get_client()
    q = client.table('alerts').select('*').eq('id', alert_id)
    if user_id:
        q = q.eq('user_id', user_id)
    result = q.limit(1).execute()
    if not result.data:
        return None
    return _row_to_alert(result.data[0])


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
# Webhook Groups CRUD (database path) — Phase 39
# ============================================================
# Table created by migrations/phase39_webhook_groups.sql. RLS filters by
# auth.uid() = user_id, so user-scoped calls need a JWT. Worker calls are
# routed through get_client() which returns the admin client when
# _admin_mode_var is True (see set_admin_user_context).

def load_webhook_groups_db() -> list:
    """Load all webhook groups for the current user."""
    user_id = get_current_user_id()
    client = get_client()
    q = client.table('webhook_groups').select('*').order('created_at')
    if user_id:
        q = q.eq('user_id', user_id)
    result = q.execute()
    return result.data or []


def get_webhook_group_by_id_db(group_id: str) -> dict | None:
    """Load a single webhook group by ID. RLS enforces user scoping."""
    client = get_client()
    result = client.table('webhook_groups') \
        .select('*') \
        .eq('id', group_id) \
        .maybe_single() \
        .execute()
    return result.data if result and result.data else None


def save_webhook_group_db(group: dict) -> dict:
    """Insert a new webhook group. Uses the caller-supplied id."""
    payload = dict(group)
    payload['user_id'] = get_current_user_id()
    client = get_client()
    result = client.table('webhook_groups').insert(payload).execute()
    return result.data[0] if result.data else {}


def update_webhook_group_db(group_id: str, updates: dict) -> dict | None:
    """Update a webhook group by ID."""
    payload = dict(updates)
    payload.pop('id', None)
    payload.pop('user_id', None)
    client = get_client()
    result = client.table('webhook_groups') \
        .update(payload) \
        .eq('id', group_id) \
        .execute()
    return result.data[0] if result.data else None


def delete_webhook_group_db(group_id: str) -> bool:
    """Delete a webhook group by ID."""
    client = get_client()
    result = client.table('webhook_groups') \
        .delete() \
        .eq('id', group_id) \
        .execute()
    return bool(result.data)


# ============================================================
# Trades CRUD (database path) — Phase 40
# ============================================================
# Table created by migrations/phase40_trades_table.sql. RLS policies
# filter by auth.uid() = user_id. Worker + API callers need admin-mode
# context for service-role access; that's set at the request boundary
# the same way as for alerts/strategies.
#
# Hot columns (filtered/joined/aggregated heavily) + a single `data`
# JSONB for the long tail. Mirrors the alerts-table pattern.
TRADE_COLUMN_FIELDS = {
    'id', 'strategy_id', 'user_id',
    'entry_alert_id', 'exit_alert_id',
    'entry_trigger_ts', 'entry_fill_ts',
    'exit_trigger_ts', 'exit_fill_ts',
    'entry_price', 'exit_price',
    'r_multiple', 'dollar_pnl', 'executed_quantity',
    'direction', 'exec_type', 'exit_reason', 'data_source',
    'created_at',
}


def _trade_to_row(trade: dict) -> dict:
    """Split a flat trade dict into column + data JSONB format.

    Accepts legacy ``entry_time`` / ``exit_time`` aliases and maps them to
    the canonical ``entry_fill_ts`` / ``exit_fill_ts`` columns so older
    backfill payloads persist into the right fields. Mirrors the alerts
    _alert_to_row pattern.
    """
    row: dict = {}
    data: dict = {}
    legacy_entry = trade.get('entry_time')
    legacy_exit = trade.get('exit_time')
    for k, v in trade.items():
        if k in ('entry_time', 'exit_time'):
            continue  # handled via the canonical *_fill_ts below
        if k in TRADE_COLUMN_FIELDS:
            row[k] = v
        else:
            data[k] = v
    # Back-fill canonical timestamps from legacy aliases when needed
    if row.get('entry_fill_ts') in (None, '') and legacy_entry:
        row['entry_fill_ts'] = legacy_entry
    if row.get('exit_fill_ts') in (None, '') and legacy_exit:
        row['exit_fill_ts'] = legacy_exit
    # confluence_records can arrive as a set (trade record originals use
    # a set) — JSONB can't serialize sets; cast to a sorted list to keep
    # order deterministic across inserts.
    cr = data.get('confluence_records')
    if isinstance(cr, set):
        data['confluence_records'] = sorted(cr)
    row['data'] = data if data else {}
    return row


def _row_to_trade(row: dict) -> dict:
    """Reconstruct a flat trade dict from a trades table row.

    Merges the `data` JSONB back into the top-level dict so downstream
    consumers see the same shape as legacy stored_trades entries. Also
    emits the legacy ``entry_time`` / ``exit_time`` aliases so older
    consumers (Streamlit app.py, some services helpers) keep working
    without a bulk rename.
    """
    trade: dict = {}
    for k, v in row.items():
        if k == 'data':
            if isinstance(v, str):
                try:
                    v = json.loads(v)
                except Exception:
                    v = {}
            if isinstance(v, dict) and v:
                trade.update(v)
        else:
            trade[k] = v
    # Legacy aliases for consumers that haven't migrated to *_fill_ts.
    if trade.get('entry_fill_ts') and not trade.get('entry_time'):
        trade['entry_time'] = trade['entry_fill_ts']
    if trade.get('exit_fill_ts') and not trade.get('exit_time'):
        trade['exit_time'] = trade['exit_fill_ts']
    return trade


def load_trades_admin(strategy_id: int, user_id: str | None = None) -> list:
    """Load all trades for a strategy (admin client). Ordered by entry_fill_ts.

    user_id filter is optional — with the FK + CASCADE we know every
    trade's user_id matches the strategy's user_id, but we accept the
    parameter so callers can still supply it for defence-in-depth.

    PostgREST/Supabase caps every response at 1000 rows regardless of
    `limit` / `Range` headers (confirmed 2026-04-24). Strategies in this
    app can exceed 20k trades, so the loader paginates via `.range()`
    until a partial page comes back. Stops at a hard ceiling so a runaway
    query can't load the whole table.
    """
    client = get_admin_client()
    PAGE_SIZE = 1000
    MAX_TRADES = 200_000  # hard ceiling — flags a bug if ever hit
    all_rows: list = []
    offset = 0
    while offset < MAX_TRADES:
        q = client.table('trades') \
            .select('*') \
            .eq('strategy_id', strategy_id) \
            .order('entry_fill_ts') \
            .range(offset, offset + PAGE_SIZE - 1)
        if user_id:
            q = q.eq('user_id', user_id)
        result = q.execute()
        page = result.data or []
        all_rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return [_row_to_trade(r) for r in all_rows]


def insert_trade_admin(strategy_id: int, user_id: str, trade: dict) -> dict | None:
    """Insert a single trade row (admin client). Returns the saved row
    reconstructed into legacy flat shape, or None on conflict / failure.
    """
    row = _trade_to_row(trade)
    row['strategy_id'] = strategy_id
    row['user_id'] = user_id
    row.pop('id', None)
    row.pop('created_at', None)
    client = get_admin_client()
    try:
        result = client.table('trades').insert(row).execute()
    except Exception as e:
        # Unique-index collision on (strategy_id, entry_fill_ts, exit_fill_ts)
        # is not fatal — it's the idempotency guardrail. Other errors bubble.
        msg = str(e).lower()
        if 'duplicate' in msg or 'unique' in msg or 'conflict' in msg:
            return None
        raise
    return _row_to_trade(result.data[0]) if result.data else None


def delete_trades_for_strategy_admin(strategy_id: int) -> int:
    """Delete every trade row for a strategy. Returns deleted row count."""
    client = get_admin_client()
    result = client.table('trades') \
        .delete() \
        .eq('strategy_id', strategy_id) \
        .execute()
    return len(result.data) if result.data else 0


def delete_trade_placeholder_admin(strategy_id: int, entry_fill_ts) -> int:
    """Delete any open-position placeholder row for (strategy, entry_fill_ts).

    forward_test_service may emit a row with exit_fill_ts IS NULL while a
    position is still held. Once the exit fires, the worker wants to
    replace that placeholder with the closed trade rather than leave
    both in place. Returns deleted row count.
    """
    if entry_fill_ts is None or entry_fill_ts == '':
        return 0
    client = get_admin_client()
    result = client.table('trades') \
        .delete() \
        .eq('strategy_id', strategy_id) \
        .eq('entry_fill_ts', entry_fill_ts) \
        .is_('exit_fill_ts', 'null') \
        .execute()
    return len(result.data) if result.data else 0


def replace_trades_admin(
    strategy_id: int,
    user_id: str,
    trades: list[dict],
) -> int:
    """Atomically replace all trades for a strategy (admin client).

    DELETE followed by bulk INSERT. Used by the forward-test recompute
    and Mass Builder backtest-save paths. Returns inserted row count.
    """
    client = get_admin_client()
    client.table('trades').delete().eq('strategy_id', strategy_id).execute()
    if not trades:
        return 0
    rows = []
    for trade in trades:
        row = _trade_to_row(trade)
        row['strategy_id'] = strategy_id
        row['user_id'] = user_id
        row.pop('id', None)
        row.pop('created_at', None)
        rows.append(row)
    result = client.table('trades').insert(rows).execute()
    return len(result.data) if result.data else 0


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
    """Load all strategies for a specific user (admin client).

    Phase 40: hydrate_trades=False because Worker's "no strategies to
    monitor" diagnostic fallback only reads id/name/symbol-level fields;
    fetching trades per strategy here would spike Supabase load for a
    debug path. Callers that need trades should use
    ``get_strategy_by_id_admin`` for the specific strategy.
    """
    client = get_admin_client()
    result = client.table('strategies') \
        .select('*') \
        .eq('user_id', user_id) \
        .order('id') \
        .execute()
    return [_row_to_strategy(r, hydrate_trades=False) for r in result.data]


# Columns needed for Worker monitoring decisions (triggers, confluence,
# session, timeframe, alert tracking). Explicitly EXCLUDES the three
# JSONB columns that can grow into the megabytes per strategy:
#   - stored_trades         (trade history; single strategy observed at 2.56 MB)
#   - equity_curve_data     (per-trade cumulative-R points; ~70 KB typical)
#   - live_executions       (position-state mirror; grows with trade count)
# Ralph engine doesn't touch any of those three at monitoring time, and the
# one worker path that legitimately needs stored_trades (_persist_algo_trade)
# uses get_strategy_by_id_admin for a single row — not this bulk path.
# Update this list whenever a column is added to the strategies schema.
_STRATEGY_MONITOR_COLUMNS = (
    'id,user_id,name,symbol,direction,timeframe,'
    'config,kpis,'
    'alert_tracking_enabled,alert_tracking_reset_at,'
    'created_at,updated_at,data_refreshed_at,'
    'discrepancies,discrepancies_dismissed_at,'
    'forward_test_start,forward_testing,strategy_origin'
)


def load_strategies_monitoring_admin(user_id: str) -> list:
    """Load all strategies for a user, excluding heavy JSONB columns.

    Use this variant for the Worker's polling / hot-reload path where we
    only need monitoring metadata (triggers, confluence, timeframe). On
    a strategy with ~3000 trades, this is ~2.5 MB / strategy cheaper than
    ``load_strategies_admin`` — a big deal when the Worker re-polls every
    hot-reload cycle and Supabase statement-timeouts are the limiting
    factor (see roadmap 9an).

    If you need trade history, equity_curve_data, or live_executions, use
    ``load_strategies_admin`` or ``get_strategy_by_id_admin`` instead.
    """
    client = get_admin_client()
    result = client.table('strategies') \
        .select(_STRATEGY_MONITOR_COLUMNS) \
        .eq('user_id', user_id) \
        .order('id') \
        .execute()
    # hydrate_trades=False — monitor cycle never reads trades.
    return [_row_to_strategy(r, hydrate_trades=False) for r in result.data]


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

    # Phase 40: when the trades-table is authoritative, strip stored_trades
    # out of the strategies UPDATE payload and write to the trades table
    # instead. Only acts when 'stored_trades' is explicitly in the incoming
    # updates dict — omitting it means "don't touch trades" (preserves the
    # feedback_jsonb_partial_updates safety rule).
    pending_trades = None
    try:
        import trades_store as _trades_store
        if _trades_store._flag_on() and 'stored_trades' in column_updates:
            pending_trades = column_updates.pop('stored_trades', None) or []
    except Exception:
        _trades_store = None

    payload = dict(column_updates)
    if config_updates:
        # Caller is changing something inside config — they must have already
        # merged with the existing config. We can't do partial JSONB updates
        # via the supabase-py client, so the caller's responsibility.
        payload['config'] = config_updates

    if not payload and pending_trades is None:
        return None  # nothing to update

    saved = None
    if payload:
        client = get_admin_client()
        result = client.table('strategies') \
            .update(payload) \
            .eq('id', strategy_id) \
            .eq('user_id', user_id) \
            .execute()
        if result.data:
            saved = _row_to_strategy(result.data[0])

    if pending_trades is not None and _trades_store is not None and _trades_store._flag_on():
        try:
            _trades_store.replace_trades_for_strategy(
                strategy_id, user_id, pending_trades)
            if saved is not None:
                saved['stored_trades'] = pending_trades
        except Exception as e:
            import logging as _l
            _l.getLogger(__name__).warning(
                "update_strategy_admin: trades-table replace failed for "
                "strategy %s: %s", strategy_id, e)

    return saved


def load_portfolios_admin(user_id: str) -> list:
    """Load all portfolios for a specific user (admin client)."""
    client = get_admin_client()
    result = client.table('portfolios') \
        .select('*') \
        .eq('user_id', user_id) \
        .order('id') \
        .execute()
    # Apply the same JSONB → top-level restoration as the user-scoped path so
    # the worker sees change_log / journal_entries / buying_power_mode at the
    # top level, not buried inside account as `_p37_*` keys.
    return [_restore_portfolio_from_db(r) for r in result.data]


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

    # Use the lite variant (no stored_trades / equity_curve_data /
    # live_executions) — monitoring decisions only need config + trigger
    # fields, and the Worker polls this every hot-reload cycle.
    strategies = load_strategies_monitoring_admin(user_id)
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

    In USE_DB mode we RAISE on DB errors instead of silently falling back to
    the file-mode cache. On Railway there is no file, so the fallback used
    to return [] and the frontend would render "No saved searches yet" —
    indistinguishable from a user who genuinely has no searches. Raising
    lets the API return 500, which React Query treats as an error (keeping
    the previously-fetched data visible) rather than as an empty list.
    """
    if USE_DB:
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
            # Historical: this used to reference a module-level
            # threading.local() named `_local`. The user-context layer
            # migrated to ContextVar on 2026-04-22 (see top of this
            # module), but this reference was missed — stayed broken
            # silently because the except clause falls back to file
            # storage, which on Railway is the API container's ephemeral
            # filesystem. Mass Builder progress tracking then polls the
            # DB and never finds the search, so the UI hangs. Fix:
            # use get_current_user_id().
            user_id = get_current_user_id()
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
            # Store results, progress, summary, checkpoint in config_data.
            # `checkpoint` is the resume-from-checkpoint payload (Roadmap 9s):
            # completed_symbol_tfs + partial_results + diagnostics_so_far.
            _jsonb_keys = ('results', 'progress', 'summary', 'checkpoint')
            if any(k in updates for k in _jsonb_keys):
                # Read existing config_data, merge updates into it
                result = client.table('mass_searches').select('config_data').eq('id', search_id).maybe_single().execute()
                cfg = (result.data or {}).get('config_data', {}) if result and result.data else {}
                for key in _jsonb_keys:
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


# ============================================================
# User Pack Parity Status CRUD
# ============================================================
# Backs the user_pack_parity_status table — one row per
# (user_id, pack_slug) holding the last 4-quadrant parity-test
# result. See migrations/user_pack_parity_status.sql.

def load_pack_parity_status(pack_slug: str, user_id: str) -> dict | None:
    """Return the most recent saved parity-test result for this
    (user, pack) combo, or None if the user has never run the test."""
    try:
        client = get_admin_client()
        r = (client.table('user_pack_parity_status')
             .select('*')
             .eq('user_id', user_id)
             .eq('pack_slug', pack_slug)
             .limit(1)
             .execute())
        rows = r.data or []
        return rows[0] if rows else None
    except Exception as e:
        logger.warning(
            "load_pack_parity_status(%s, %s) failed: %s",
            pack_slug, user_id[:8] if user_id else '?', e)
        return None


def save_pack_parity_status(
    pack_slug: str,
    user_id: str,
    overall_verdict: str,
    summary: str,
    quadrants: dict,
    test_config: dict,
) -> dict | None:
    """Upsert the parity-test result for this (user, pack) combo.

    Idempotent: re-running overwrites the row. Tested_at is set to now
    on every upsert (independent of created_at).
    """
    from datetime import datetime, timezone
    if overall_verdict not in ('PASS', 'WARN', 'FAIL'):
        raise ValueError(
            f"overall_verdict must be PASS|WARN|FAIL, got {overall_verdict!r}")
    payload = {
        'user_id': user_id,
        'pack_slug': pack_slug,
        'overall_verdict': overall_verdict,
        'summary': summary or '',
        'quadrants': quadrants or {},
        'test_config': test_config or {},
        'tested_at': datetime.now(timezone.utc).isoformat(),
    }
    try:
        client = get_admin_client()
        # Upsert by (user_id, pack_slug) unique constraint
        r = (client.table('user_pack_parity_status')
             .upsert(payload, on_conflict='user_id,pack_slug')
             .execute())
        rows = r.data or []
        return rows[0] if rows else None
    except Exception as e:
        logger.warning(
            "save_pack_parity_status(%s, %s) failed: %s",
            pack_slug, user_id[:8] if user_id else '?', e)
        return None


def list_pack_parity_statuses(user_id: str) -> list:
    """Return all parity statuses for the user (one per pack). Used by
    the user_packs LIST endpoint to surface verdicts inline."""
    try:
        client = get_admin_client()
        r = (client.table('user_pack_parity_status')
             .select('*')
             .eq('user_id', user_id)
             .execute())
        return r.data or []
    except Exception as e:
        logger.warning(
            "list_pack_parity_statuses(%s) failed: %s",
            user_id[:8] if user_id else '?', e)
        return []
